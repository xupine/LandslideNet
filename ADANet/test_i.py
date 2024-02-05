import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import time


from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary


from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import time

from PIL import Image
import matplotlib.image as mping
import cv2
from skimage import io
from dataloaders.utils import decode_segmap
import skimage
from skimage import util
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def nor(self,img):

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class DFPE_test(object):
    """docstring for DFPE"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    INPUT_SIZE = 512
    cuda = torch.cuda.is_available()
    gpu_ids = [0]
    def __init__(self,model_path):
        self.lr = 0.007
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        if DFPE_test.cuda and len(DFPE_test.gpu_ids) > 1:
            self.sync_bn = True
        else:
            self.sync_bn = False
        self.model = DeepLab(num_classes=2,backbone='resnet',output_stride=16,sync_bn=self.sync_bn,freeze_bn=False)
        self.train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        self.optimizer = torch.optim.SGD(self.train_params, momentum=0.9,weight_decay=5e-4, nesterov=False)
        if DFPE_test.cuda:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=DFPE_test.gpu_ids)
            patch_replication_callback(self.model)
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        if DFPE_test.cuda:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_pred = checkpoint['best_pred']
    def nor1(self,img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img
    def nor2(self,img):
        img -= self.mean
        img /= self.std
        return img
    def run(self,TEST_IMAGE_PATHS):
        self.model.eval()

        img1 = Image.open(TEST_IMAGE_PATHS1).convert('RGB')
        img1=self.nor1(img1)
        img1 = img1.transpose((2, 0, 1))

        # img2 = Image.open(TEST_IMAGE_PATHS2).convert('RGB')
        img2 = self.nor2(img2)
        img2 = img2.transpose((2, 0, 1))

        print(img1.shape)
        print(img2.shape)
        if img1.shape == img2.shape:
            n = 512
            height=img1.shape[1]
            width=img1.shape[2]
            if height%n != 0:
                n1=int(height/n)+1
            else:
                n1=int(height/n)
            if width%n != 0:
                n2=int(width/n)+1
            else:
                n2=int(width/n)
            imge1=[]
            imge2 = []
            seg=[]
            for i in range(0,n1*n2):
                imge1.append([])
                imge2.append([])
                seg.append([])
            count=0
            for i in range(n1):
                for j in range(n2):  
                    mm=((i+1)*n)
                    nn=((j+1)*n)
                    if ((i+1)*n)>height:
                        mm=height
                    if (j+1)*n>width:
                        nn=width
                    imge1[count]=img1[:,i*n:mm,j*n:nn]
                    image1=imge1[count]
                    image1=image1[np.newaxis,:,:,:]

                    imge2[count] = img2[:, i * n:mm, j * n:nn]
                    image2 = imge2[count]
                    image2 = image2[np.newaxis, :, :, :]

                    image1 = torch.from_numpy(image1)
                    image2 = torch.from_numpy(image2)

                    if DFPE_test.cuda:
                        image1 = image1.cuda()
                        image2 = image2.cuda()
                    with torch.no_grad():
                        seg_map1, seg_map2 = self.model(image1, image2)

                    seg_map1 = seg_map1.data.cpu().numpy()
                    seg_map1 = np.argmax(seg_map1, axis=1)
                    seg_map2 = seg_map2.data.cpu().numpy()
                    seg_map2 = np.argmax(seg_map2, axis=1)
                    seg_map = seg_map2 - seg_map1
                    seg_map[seg_map == -1] = 0
                    print("第%d次扫描"%count)
                    if count%n2==0:
                        seg[i]=seg_map
                        print("识别中：",seg[i].shape)
                    else:
                        seg[i]=np.concatenate((seg[i],seg_map),axis=2)
                        print("识别中：",seg[i].shape)
                    count+=1
            segm=seg[0]
            for i in range(0,n1-1):
                segm=np.concatenate((segm,seg[i+1]),axis=1)
            seg_maps=segm
            print(seg_maps.shape)
        return seg_maps
    def vis_segmentation(self, seg_map, m):
        # plt.figure(figsize=(30, 50))
        # grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
        # plt.subplot(grid_spec[0])
        # plt.imshow(image)
        # plt.axis('off')
        # plt.title('input image')
        # plt.subplot(grid_spec[1])

        seg_map = seg_map.squeeze()
        seg_map = np.array(seg_map).astype(np.uint8)
        # seg_image = decode_segmap(seg_map,dataset='landslide')
        # plt.imshow(seg_image)
        # plt.axis('off')
        # plt.title('segmentation map')
        #
        # plt.subplot(grid_spec[2])
        # plt.imshow(image)
        # plt.imshow(seg_image, alpha=0.7)
        # plt.axis('off')
        # plt.title('segmentation overlay')
        #
        #
        # #plt.show()
        # maxi = seg_image.max()
        # seg_image = seg_image / maxi * 255
        # seg_image = seg_image.astype(np.float32)
        seg_image = seg_map.astype(np.uint8)
        # image = Image.fromarray(image)
        seg_image=Image.fromarray(seg_image)
        seg_image.save(os.path.join('PATH/UDA_landslide/seg1','{}.png'.format(m)))

model_path ='PATH/checkpoint_model_target.pth.tar'
MODEL = DFPE_test(model_path)
PATH_TO_TEST_IMAGES_DIR = 'PATH/landslide_bhd'

TEST_IMAGE_PATHS1 = os.path.join(PATH_TO_TEST_IMAGES_DIR, '11.tif')
TEST_IMAGE_PATHS2 = os.path.join(PATH_TO_TEST_IMAGES_DIR, '2.tif')

for i in range(16):
    va = i * 0.01
    img2 = Image.open(TEST_IMAGE_PATHS2).convert('RGB')
    img2 = np.array(img2).astype(np.float32)
    img2 /= 255.0
    img2 = util.random_noise(img2, mode='s&p', seed=None, clip=True, amount=2*va)
    img2 = img2.astype(np.float32)
    start = time.time()
    seg_maps = MODEL.run(TEST_IMAGE_PATHS1, img2)
    end = time.time()
    print('time:', end-start)

    # mying=Image.open(TEST_IMAGE_PATHS2)
    # mying=mying.convert('RGB')
    # mying=np.array(mying, dtype=np.uint8)
    # print(mying.shape)
    MODEL.vis_segmentation(seg_maps, 2*va)

















        
        