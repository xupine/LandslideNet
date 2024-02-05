import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Losses(object):
    def __init__(self, num_class, weight=None, batch_average=True, ignore_index=255, cuda=True, size_average=True):
        self.num_class = num_class
        self.weight = weight
        self.ignore_index = ignore_index
        self.batch_average = batch_average
        self.cuda = cuda
        self.size_average = size_average
    def CrossEntropyLoss(self, prediction, target):
        n, c, h, w = prediction.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(prediction, target.long())
        if self.batch_average:
            loss /= n
        return loss
    def Diff2d_class(self, inputs1, inputs2):
        loss = 0
        inputs1 = inputs1.view(inputs1.size(0),-1)
        inputs2 = inputs2.view(inputs2.size(0), -1)
        for i in range(self.num_class):
            loss  += torch.mean(torch.abs(inputs1[i]-inputs2[i]))
        return loss
    def Symkl2d_class(self, inputs1, inputs2):
        loss = 0
        #inputs1 = inputs1.view(inputs1.size(0), -1)
        #inputs2 = inputs2.view(inputs2.size(0), -1)
        for i in range(self.num_class):
            prob1 = F.softmax(inputs1[i],dim=1)
            prob2 = F.softmax(inputs2[i],dim=1)
            #prob = 0.5*(prob1+prob2)
            log_prob1 = F.log_softmax(prob1,dim=1)
            log_prob2 = F.log_softmax(prob2,dim=1)
            loss1 = 0.5 * (F.kl_div(log_prob1, prob2, size_average=self.size_average)
                          + F.kl_div(log_prob2, prob1, size_average=self.size_average))
            print(loss1)
            loss += loss1
        return loss

    def bce_loss(self, inputs1, inputs2):
        loss = 0
        truth_tensor = torch.FloatTensor(self.num_class, 1, 1, 16, 16).cuda()
        for i in range(self.num_class):
            truth_tensor[i] = torch.FloatTensor(inputs1[i].size())
            truth_tensor[i].fill_(inputs2)
            truth_tensor[i] = truth_tensor[i].cuda()
            loss1 = nn.BCEWithLogitsLoss()(inputs1[i].clone(), truth_tensor[i].clone())
            loss += loss1
        return loss

    def bce_adv(self, inputs1, inputs2):
        truth_tensor = torch.FloatTensor(inputs1.size())
        truth_tensor.fill_(inputs2)
        truth_tensor = truth_tensor.cuda()
        loss = nn.BCEWithLogitsLoss()(inputs1, truth_tensor)

        return loss
    def bce_loss_DS(self, inputs1, inputs2):
        loss = 0
        truth_tensor = torch.FloatTensor(self.num_class, 1, 1, 16, 16).cuda()
        for i in range(self.num_class):
            truth_tensor[i] = torch.FloatTensor(inputs1[i].size())
            label_fnoise = torch.FloatTensor(inputs1[i].size()).uniform_(0,.3)
            truth_tensor[i].fill_(inputs2)
            truth_tensor[i] = truth_tensor[i].cuda()+label_fnoise.cuda()
            loss1 = nn.BCEWithLogitsLoss()(inputs1[i].clone(), truth_tensor[i].clone())
            loss += loss1
        return loss

    def bce_adv_DS(self, inputs1, inputs2):
        truth_tensor = torch.FloatTensor(inputs1.size())
        label_fnoise = torch.FloatTensor(inputs1.size()).uniform_(0,.3)
        truth_tensor.fill_(inputs2)
        truth_tensor = truth_tensor.cuda()+label_fnoise.cuda()
        loss = nn.BCEWithLogitsLoss()(inputs1, truth_tensor)

        return loss
    def bce_loss_DT(self, inputs1, inputs2):
        loss = 0
        truth_tensor = torch.FloatTensor(self.num_class, 1, 1, 16, 16).cuda()
        for i in range(self.num_class):
            truth_tensor[i] = torch.FloatTensor(inputs1[i].size())
            label_fnoise = torch.FloatTensor(inputs1[i].size()).uniform_(-.3,0)
            truth_tensor[i].fill_(inputs2)
            truth_tensor[i] = truth_tensor[i].cuda()+label_fnoise.cuda()
            loss1 = nn.BCEWithLogitsLoss()(inputs1[i].clone(), truth_tensor[i].clone())
            loss += loss1
        return loss

    def bce_adv_DT(self, inputs1, inputs2):
        truth_tensor = torch.FloatTensor(inputs1.size())
        label_fnoise = torch.FloatTensor(inputs1.size()).uniform_(-.3,0)
        truth_tensor.fill_(inputs2)
        truth_tensor = truth_tensor.cuda()+label_fnoise.cuda()
        loss = nn.BCEWithLogitsLoss()(inputs1, truth_tensor)

        return loss


if __name__ == "__main__":
    loss = Losses(num_class=3)

    a = torch.rand(3, 3, 6, 6).cuda()

    #b = torch.rand(1, 7, 7).cuda()
    c = torch.rand(3, 3, 6, 6).cuda()
    print(loss.Symkl2d_class(a,c))


    #print(CrossEntropyLoss(a, b).item())
    #print(m)






