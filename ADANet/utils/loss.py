import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=True):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'CD':
            return self.CDLoss
        elif mode == 'BA':
            return self.BALoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    def CDLoss(self, logit, target, eps=1e-7):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        ce_loss = criterion(logit, target.long())
        if self.batch_average:
            ce_loss /= n
        #dice loss
        logit = F.softmax(logit,dim=1)
        target2 = torch.zeros(n,h,w).cuda()
        num_classes = logit.shape[1]
        target = torch.where(target < num_classes+1.0, target, target2)
        target = target.long()
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logit)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logit, dim=1)
        true_1_hot = true_1_hot.type(logit.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = 1 - (2. * intersection / (cardinality + eps)).mean()
        loss = ce_loss + dice_loss

        return loss

    def gaussian_kernel(self, size, size_y=None, sigma=1.0):
        size = int(size)
        if not size_y:
            size_y = size
        else:
            size_y = int(size_y)
        x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
        g = np.exp(-(x ** 2 / (float(size) * sigma) + y ** 2 / (float(size_y) * sigma)))
        return g / g.sum()

    def gaussianize_image(self, inputs, filter_size, sigma=3.0):

        gaussian_filter_value = self.gaussian_kernel(int(filter_size[0] / 2), int(filter_size[1] / 2), sigma=sigma)
        gaussian_filter = torch.from_numpy(gaussian_filter_value)
        gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0).cuda().float()
        gaussian_image = F.conv2d(inputs, gaussian_filter, stride=1, padding=0)
        return gaussian_image

    def BALoss(self, logit, target, remain_rate=0.5, edge_multiplier=1.5):
        b, h, w = target.size()
        target = target.float()
        num_classes = logit.shape[1]
        target1 = target.unsqueeze(1)
        #true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
        #true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        edge_filter_value = np.array([[1, -0.5],[-0.5, 0]])
        edge_filter = torch.from_numpy(edge_filter_value)
        edge_filter = edge_filter.expand(1,1,2,2).cuda().float()
        fe = F.conv2d(target1, edge_filter, stride=1, padding=0)
        eg = torch.ge(fe,0.).float()
        filter_size = np.array([35, 35])
        gaussian_edge = self.gaussianize_image(eg, filter_size)
        gaussian_edge = F.interpolate(gaussian_edge, size=target1.size()[2:], mode='bilinear', align_corners=True)
        label_weight = torch.clamp(gaussian_edge* edge_multiplier + remain_rate,min=0.0,max=4.0)
        #loss
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduce=False)
        if self.cuda:
            criterion = criterion.cuda()
        loss1 = criterion(logit, target.long())
        label_weight = label_weight.squeeze()
        loss = torch.sum(label_weight*loss1)
        loss = loss/(b*h*w)
        return loss
if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(2, 3, 60, 60).cuda()
    b = torch.rand(2, 60, 60).cuda()
    print(loss.CDLoss(a, b))





