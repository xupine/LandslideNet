import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataloaders.datasets import landslide_s, landslide_t, landslide_tv
from model.deeplab import *
from model.discriminator import FCDiscriminator
from model.sync_batchnorm.replicate import patch_replication_callback
from scripts.loss import Losses
from utils.loss import SegmentationLosses
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from utils.saver_source import Saver_source
from utils.saver_target import Saver_target
from utils.saver import Saver

from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver_source = Saver_source(args)
        self.saver_target = Saver_target(args)
        self.saver = Saver_target(args)


        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        self.summary_source = TensorboardSummary(self.saver_source.experiment_dir)
        self.writer_source = self.summary_source.create_summary()

        self.summary_target = TensorboardSummary(self.saver_target.experiment_dir)
        self.writer_target = self.summary_target.create_summary()

        # DATALOADERS
        # source
        self.source_train_set = landslide_s.Landslide(args, split='train', max_iters=args.max_iters * args.source_batch_size)
        self.source_val_set = landslide_s.Landslide(args, split='val', max_iters=None)
        self.source_num_class = self.source_train_set.NUM_CLASSES
        self.source_train_loader = DataLoader(self.source_train_set, batch_size=args.source_batch_size, shuffle=True, drop_last=True,
                                              num_workers=args.workers, pin_memory=True)
        self.source_val_loader = DataLoader(self.source_val_set, batch_size=args.source_batch_size, shuffle=False, drop_last=True,
                                            num_workers=args.workers, pin_memory=True)
        # target
        self.target_train_set = landslide_t.Landslide(args, split='train',max_iters=args.max_iters * args.target_batch_size)
        self.target_val_set = landslide_tv.Landslide(args, split='val', max_iters=None)
        self.target_num_class = self.target_train_set.NUM_CLASSES
        self.target_train_loader = DataLoader(self.target_train_set, batch_size=args.target_batch_size, shuffle=True, drop_last=True,
                                              num_workers=args.workers, pin_memory=True)
        self.target_val_loader = DataLoader(self.target_val_set, batch_size=args.target_batch_size, shuffle=False, drop_last=True,
                                            num_workers=args.workers, pin_memory=True)
        # SEGMENTATION model
        model = DeepLab(num_classes=self.source_num_class,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        # DISCRIMINATOR model
        # high_feature level
        d_h = FCDiscriminator(num_classes=self.source_num_class)
        #d_hc = FCDiscriminator(num_classes=1)
        # low_feature level
        d_l = FCDiscriminator(num_classes=self.source_num_class)
        #d_lc = FCDiscriminator(num_classes=1)

        # Define Optimizer
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        # discriminators' optimizers
        optimizer_d_h = optim.Adam(d_h.parameters(), lr=args.lr_D,
                                   betas=(0.9, 0.99))
        #optimizer_d_hc = optim.Adam(d_hc.parameters(), lr=args.lr_D,
                                    #betas=(0.9, 0.99))
        optimizer_d_l = optim.Adam(d_l.parameters(), lr=args.lr_D,
                                   betas=(0.9, 0.99))
        #optimizer_d_lc = optim.Adam(d_lc.parameters(), lr=args.lr_D,
                                    #betas=(0.9, 0.99))
        # Define Criterion
        self.segcriterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.criterion = Losses(num_class=self.source_num_class, weight=None, batch_average=True, ignore_index=255,
                           cuda=args.cuda, size_average=True)
        #self.seg = criterion.CrossEntropyLoss()
        #self.bce_loss = criterion.bce_loss()
        #self.symkl2d = criterion.Symkl2d_class()
        #self.domain_adv = criterion.bce_adv()

        self.model, self.d_h, self.d_l = model, d_h, d_l
        self.optimizer, self.optimizer_d_h, self.optimizer_d_l = optimizer, optimizer_d_h, optimizer_d_l

        # Define Evaluator
        self.evaluator_source = Evaluator(self.source_num_class)
        self.evaluator_target = Evaluator(self.target_num_class)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,args.epochs, args.num_steps)
        self.scheduler_D = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, args.num_steps)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            self.d_h = torch.nn.DataParallel(self.d_h, device_ids=self.args.gpu_ids)
            self.d_h = self.d_h.cuda()
            self.d_l = torch.nn.DataParallel(self.d_l, device_ids=self.args.gpu_ids)
            self.d_l = self.d_l.cuda()
            #self.d_hc = torch.nn.DataParallel(self.d_hc, device_ids=self.args.gpu_ids)
            #self.d_hc = self.d_hc.cuda()
            #self.d_lc = torch.nn.DataParallel(self.d_lc, device_ids=self.args.gpu_ids)
            #self.d_lc = self.d_lc.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_pred_source = 0.0
        self.best_pred_target = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            #self.best_pred_source = checkpoint['best_pred']
            #self.best_pred_target = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # labels for adversarial training
        self.source_label = 0
        self.target_label = 1

    def training(self, epoch):
        self.model.train()
        self.d_h.train()
        self.d_l.train()
        
        sourceloader_iter = enumerate(self.source_train_loader)
        targetloader_iter = enumerate(self.target_train_loader)
        for iter_i in tqdm(range(self.args.num_steps)):
            # reset optimizers
            self.optimizer.zero_grad()
            self.optimizer_d_h.zero_grad()
            self.optimizer_d_l.zero_grad()
            #self.optimizer_d_hc.zero_grad()

            # adapt LR if needed
            self.scheduler(self.optimizer, iter_i, epoch, self.best_pred)
            self.scheduler_D(self.optimizer_d_h, iter_i, epoch, self.best_pred)
            self.scheduler_D(self.optimizer_d_l, iter_i, epoch, self.best_pred)
            #self.model.adjust_learning_rate(self.args, self.optimizer, i)
            #self.d_h.adjust_learning_rate(self.args, self.optimizer_d_h, i)
            #self.d_hc.adjust_learning_rate(self.args, self.optimizer_d_hc, i)

            # UDA Training
            # only train segnet. Don't accumulate grads in disciminators
            for param in self.d_h.parameters():
                param.requires_grad = False
            for param in self.d_l.parameters():
                param.requires_grad = False
            #for param in self.d_hc.parameters():
                #param.requires_grad = False
            # train on source
            _, batch = sourceloader_iter.__next__()
            sample = batch
            images_source, labels_source = sample['image'], sample['label']
            if self.args.cuda:
                images_source, labels_source = images_source.cuda(), labels_source.cuda()
            pred_src_aux, pred_src_main = self.model(images_source)
            loss_seg_aux = self.segcriterion(pred_src_aux, labels_source)
            loss_seg_main = self.segcriterion(pred_src_main, labels_source)
            loss_seg = 1.0*loss_seg_main + 0.5*loss_seg_aux
            loss_seg.backward()

            # adversarial training ot fool the discriminator
            _, batch = targetloader_iter.__next__()
            sample = batch
            images_target = sample['image']
            pred_tar_aux, pred_tar_main = self.model(images_target)

            # calu. domai loss to fool discriminator
            dl_out_main = self.d_l(F.softmax(pred_tar_aux,dim=1))
            loss_adv_al = self.criterion.bce_adv(dl_out_main, self.source_label)

            dh_out_main = self.d_h(F.softmax(pred_tar_main,dim=1))
            loss_adv_ah = self.criterion.bce_adv(dh_out_main, self.source_label)
            loss_adv_t = 0.01*loss_adv_ah + 0.002*loss_adv_al
            loss_adv_t.backward()


            # Train discriminator networks
            # enable training mode on discriminator networks
            for param in self.d_h.parameters():
                param.requires_grad = True
            for param in self.d_l.parameters():
                param.requires_grad = True
            #for param in self.d_hc.parameters():
                #param.requires_grad = True

            # train witn source
            # domain
            pred_src_aux = pred_src_aux.detach()
            pred_src_main = pred_src_main.detach()

            dl_out_main = self.d_l(F.softmax(pred_src_aux,dim=1))
            loss_d_main_sl = self.criterion.bce_adv_DS(dl_out_main, self.source_label)
            loss_dl_sou = loss_d_main_sl / 2
            loss_dl_sou.backward()

            dh_out_main = self.d_h(F.softmax(pred_src_main,dim=1))
            loss_d_main_sh = self.criterion.bce_adv_DS(dh_out_main, self.source_label)
            loss_dh_sou = loss_d_main_sh / 2
            loss_dh_sou.backward()


            # train with target
            # domain
            pred_tar_aux = pred_tar_aux.detach()
            pred_tar_main = pred_tar_main.detach()

            dl_out_main = self.d_l(F.softmax(pred_tar_aux,dim=1))
            loss_d_main_tl = self.criterion.bce_adv_DT(dl_out_main, self.target_label)
            loss_dl_tar = loss_d_main_tl / 2
            loss_dl_tar.backward()
            dh_out_main = self.d_h(F.softmax(pred_tar_main,dim=1))
            loss_d_main_th = self.criterion.bce_adv_DT(dh_out_main, self.target_label)
            loss_dh_tar = loss_d_main_th / 2
            loss_dh_tar.backward()


            self.optimizer.step()
            self.optimizer_d_l.step()
            self.optimizer_d_h.step()

            #optimizer_d_hc.step()
            #output loss
            current_losses = {'loss_seg_main': loss_seg_main.item(),
                              'loss_seg_aux': loss_seg_aux.item(),
                              'loss_adv_t': loss_adv_t.item(),
                              'loss_dl_sou': loss_dl_sou.item(),
                              'loss_dh_sou': loss_dh_sou.item(),
                              'loss_dl_tar': loss_dl_tar.item(),
                              'loss_dh_tar': loss_dh_tar.item()}
            print(iter_i, current_losses)
            #Tensorboard
            self.writer.add_scalar('train/loss_seg_main', loss_seg_main.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_seg_aux', loss_seg_aux.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_adv_t', loss_adv_t.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_dl_sou', loss_dl_sou.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_dh_sou', loss_dh_sou.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_dl_tar', loss_dl_tar.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_dh_tar', loss_dh_tar.item(), iter_i + self.args.num_steps * epoch)
            sys.stdout.flush()


            #validation
            if iter_i % 1000 == 0 and iter_i != 0:
                #self.validation_source(iter_i,epoch)
                self.validation_target(iter_i,epoch)


        print(epoch, current_losses)
        
        #save
        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            filename1 = 'checkpoint_model.pth.tar'
            filename2 = 'checkpoint_d_h.pth.tar'
            filename3 = 'checkpoint_d_l.pth.tar'
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename1)

            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.d_h.state_dict(),
                'optimizer': self.optimizer_d_h.state_dict(),
            }, is_best, filename2)

            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.d_l.state_dict(),
                'optimizer': self.optimizer_d_l.state_dict(),
            }, is_best, filename3)

    def validation_source(self, i_iter, epoch):
        self.model.eval()
        self.evaluator_source.reset()
        tbar_source = tqdm(self.source_val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar_source):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                _, output = self.model(image)
            loss = self.criterion.CrossEntropyLoss(output, target)
            test_loss += loss.item()
            tbar_source.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator_source.add_batch(target, pred)
            
        Rec = self.evaluator_source.Pixel_Accuracy_ALLClass()
        Pre = self.evaluator_source.Pixel_Precision_ALLClass()
        F1 = self.evaluator_source.F1_ALLClass()
        F1_mean = self.evaluator_source.F1_MEANClass()
        IoU = self.evaluator_source.Class_Intersection_over_Union()
        Acc = self.evaluator_source.Pixel_Accuracy()
        Acc_class = self.evaluator_source.Pixel_Accuracy_Class()
        mIoU = self.evaluator_source.Mean_Intersection_over_Union()
        FWIoU = self.evaluator_source.Frequency_Weighted_Intersection_over_Union()
        self.writer_source.add_scalar('val/total_loss_epoch', test_loss, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/Rec[1]', Rec[1], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/Pre[1]', Pre[1], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/F1[0]', F1[0], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/F1[1]', F1[1], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/F1_mean', F1_mean, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/IoU[0]', IoU[0], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/IoU[1]', IoU[1], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/mIoU', mIoU, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/Acc', Acc, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/Acc_class', Acc_class, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val/fwIoU', FWIoU, i_iter + self.args.num_steps * epoch)
        print('Validation_source:')
        print("F1[0]:{}, F1[1]:{}, F1_mean: {}".format(F1[0], F1[1], F1_mean))
        print("IoU[0]:{}, IoU[1]:{}, mIoU: {}".format(IoU[0], IoU[1], mIoU))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print("Rec[1]:{}, Pre[1]:{}".format(Rec[1], Pre[1]))
        print('Loss: %.3f' % test_loss)
        filename = "./rec_source.txt"
        with open(filename,'a', encoding='utf-8') as f:
            f.writelines(str(Rec[1])+'\n')

        filename1 = "./pre_source.txt"
        with open(filename1,'a', encoding='utf-8') as f1:
            f1.writelines(str(Pre[1])+'\n')

        filename2 = "./miou_source.txt"
        with open(filename2,'a', encoding='utf-8') as f2:
            f2.writelines(str(IoU[1])+'\n')
  
        new_pred = mIoU1
        if new_pred > self.best_pred_source:
            is_best = True
            filename = 'checkpoint_model_source.pth.tar'
            self.best_pred_source = new_pred
            self.saver_source.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred_source,
            }, is_best, filename)

    def validation_target(self, i_iter, epoch):
        self.model.eval()
        self.evaluator_target.reset()
        tbar_target = tqdm(self.target_val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar_target):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                _, output = self.model(image)
            loss = self.criterion.CrossEntropyLoss(output, target)
            test_loss += loss.item()
            tbar_target.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator_target.add_batch(target, pred)

        Rec = self.evaluator_target.Pixel_Accuracy_ALLClass()
        Pre = self.evaluator_target.Pixel_Precision_ALLClass()
        F1 = self.evaluator_target.F1_ALLClass()
        F1_mean = self.evaluator_target.F1_MEANClass()
        IoU = self.evaluator_target.Class_Intersection_over_Union()
        Acc = self.evaluator_target.Pixel_Accuracy()
        Acc_class = self.evaluator_target.Pixel_Accuracy_Class()
        mIoU = self.evaluator_target.Mean_Intersection_over_Union()
        FWIoU = self.evaluator_target.Frequency_Weighted_Intersection_over_Union()
        self.writer_target.add_scalar('val/total_loss_epoch', test_loss, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/Rec[1]', Rec[1], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/Pre[1]', Pre[1], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/F1[0]', F1[0], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/F1[1]', F1[1], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/F1_mean', F1_mean, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/IoU[0]', IoU[0], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/IoU[1]', IoU[1], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/mIoU', mIoU, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/Acc', Acc, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/Acc_class', Acc_class, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val/fwIoU', FWIoU, i_iter + self.args.num_steps * epoch)
        print('Validation_target:')
        print("F1[0]:{}, F1[1]:{}, F1_mean: {}".format(F1[0], F1[1], F1_mean))
        print("IoU[0]:{}, IoU[1]:{}, mIoU: {}".format(IoU[0], IoU[1], mIoU))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print("Rec[1]:{}, Pre[1]:{}".format(Rec[1], Pre[1]))
        print('Loss: %.3f' % test_loss)
        filename = "./rec_target.txt"
        with open(filename,'a', encoding='utf-8') as f:
            f.writelines(str(Rec[1])+'\n')

        filename1 = "./pre_target.txt"
        with open(filename1,'a', encoding='utf-8') as f1:
            f1.writelines(str(Pre[1])+'\n')

        filename2 = "./miou_target.txt"
        with open(filename2,'a', encoding='utf-8') as f2:
            f2.writelines(str(IoU[1])+'\n')
        new_pred = IoU[1]
        if new_pred > self.best_pred_target:
            is_best = True
            filename = 'checkpoint_model_target.pth.tar'
            self.best_pred_target = new_pred
            self.saver_target.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred_target,
            }, is_best, filename)

def main():
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    # data
    parser.add_argument('--source-batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
								training (default: auto)')
    parser.add_argument('--target-batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
								training (default: auto)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')

    # seg net
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='BA',
                        choices=['ce', 'focal','CD','BA'],
                        help='loss func type (default: ce)')

    # optimizer
    parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_D', type=float, default=1e-4, metavar='LR_D',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    # LR
    parser.add_argument("--max-iters", type=int, default=2001, help="Max number of training steps.")
    parser.add_argument("--num-steps", type=int, default=2001, help="Number of training steps in every epoch.")

    # training  params
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')

    # GPU
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default="0",
                        help='use which gpu to train, must be a \
						comma-separated list of integers only (default=0,1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    #save checkpoint
    parser.add_argument("--save-pred-every", type=int, default=1000, help="Save summaries and checkpoint every often.")
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.checkname is None:
        args.checkname = 'class-gan-' + str(args.backbone)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)


if __name__ == "__main__":
    main()
