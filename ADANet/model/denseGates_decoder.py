import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class SEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEB, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1),
        	                      nn.BatchNorm2d(out_channels),
        	                      nn.ReLU())
        #self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x1, x2 = x
        return x1 * self.conv(x2)

class CA(nn.Module):
    def __init__(self, in_channel1, in_channel2, in_channel3, out_channels):
        super(CA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel1, out_channels, kernel_size=1, stride=1,padding=0),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel2, out_channels, kernel_size=1, stride=1,padding=0),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel3, out_channels, kernel_size=1, stride=1,padding=0),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())

        #self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        return x1,x2,x3



class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.relu(g+x)
        psi = self.psi(psi)
        x = x*psi
        return x
class Channel_block(nn.Module):
    def __init__(self,F_g,F_l):
        super(Channel_block,self).__init__()
        F_in = F_g + F_l
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.W = nn.Sequential(
            nn.Conv2d(F_in, F_l, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(F_l)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        x1 = torch.cat((x,g),dim=1)
        x1 = self.W(x1)
        x1 = self.avg_pool(x1)
        c_a = F.softmax(x1, dim=1)

        return c_a*x+x


class Self_attention(nn.Module):
    def __init__(self,F):
        super(Self_attention,self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(F, F//2, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(F//2)
            )     

        self.C2 = nn.Sequential(
            nn.Conv2d(F//2, 2, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        x1 = torch.cat((g,x),dim=1)
        x1 = self.C1(x1)
        x1 = self.relu(x1)
        a = self.C2(x1)
        indices1 = torch.LongTensor([0]).cuda()
        indices2 = torch.LongTensor([1]).cuda()
        a1 = torch.index_select(a,1,indices1)
        a2 = torch.index_select(a,1,indices2)
        g = a1*g
        x = a2*x
        return g,x

class Self_attentiontop(nn.Module):
    def __init__(self,F):
        super(Self_attentiontop,self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(F, F//2, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(F//2)
            )     

        self.C2 = nn.Sequential(
            nn.Conv2d(F//2, 1, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g):
        g1 = self.C1(g)
        g1 = self.relu(g1)
        a = self.C2(g1)
        g = a*g

        return g


class Gates_block(nn.Module):
    def __init__(self,x_l):
        super(Gates_block,self).__init__()
        self.gli = nn.Sequential(
            nn.Conv2d(x_l, 1, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self,x,g):
        gl = self.gli(g)
        p1 = (1-gl)*x
        p2 = (1+gl)*g
        p = torch.cat((p1,p2), dim=1)

        return p 
class SAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(SAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1),
            nn.MaxPool2d(kernel_size=(2,2))
            )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.MaxPool2d(kernel_size=(2,2))
            )
        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key = proj_key.view(m_batchsize, C//8, -1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        proj_value =proj_value.view(m_batchsize, C, -1)
        proj_value= self.relu(proj_value)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = out+x 
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU(inplace=True)

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X 1 X 1)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x1 = self.avg_pool(x)
        m_batchsize, C, height, width = x.size()
        proj_query = x1.view(m_batchsize, C, -1)
        proj_key = x1.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = out+x
        return out


       
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes1 = 64
            low_level_inplanes2 = 256
            low_level_inplanes3 = 512
        elif backbone == 'xception':
            low_level_inplanes1 = 64
            low_level_inplanes2 = 128
            low_level_inplanes3 = 256
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError



        self.Att3 = Attention_block(F_g=256,F_l=512,F_int=192)
        self.Con3 = Self_attention(F=576)
        self.CA3 = CA(in_channel1=256, in_channel2=256, in_channel3=64, out_channels=192)
        self.relu = nn.ReLU()

        #self.Gate3 = Gates_block(x_l=192)
        
        #self.conv2 = nn.Conv2d(low_level_inplanes2, 96, 1, bias=False)
        #self.bn2 = BatchNorm(96)
        self.Att2 = Attention_block(F_g=512,F_l=256,F_int=96)
        self.Con2 = Self_attention(F=288)
        self.CA2 = CA(in_channel1=256, in_channel2=512, in_channel3=64, out_channels=96)
        self.relu = nn.ReLU()
        #self.Gate2 = Gates_block(x_l=96)
        #self.conv3 = nn.Conv2d(low_level_inplanes1, 24, 1, bias=False)
        #self.bn3 = BatchNorm(24)
        self.Att1 = Attention_block(F_g=256,F_l=64,F_int=48)
        self.Con1 = Self_attentiontop(F=144)
        self.CA1 = CA(in_channel1=256, in_channel2=512, in_channel3=256, out_channels=48)
        self.relu = nn.ReLU()
        #self.Gate1 = Gates_block(x_l=24)

        self.relu = nn.ReLU()
        
        self.conv_3=nn.Sequential(nn.Conv2d(768, 256, kernel_size=1,stride=1,padding=0, bias=False),
                                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0, bias=False)
                                  )
        self.conv_2=nn.Sequential(nn.Conv2d(384, 128, kernel_size=1,stride=1,padding=0, bias=False),
                                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 384, kernel_size=1, stride=1, padding=0, bias=False)
                                  )
        self.conv_1=nn.Sequential(nn.Conv2d(192, 64, kernel_size=1,stride=1,padding=0, bias=False),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 192, kernel_size=1, stride=1, padding=0, bias=False)
                                  )


        self.four_conv=nn.Sequential(nn.Conv2d(1024,256, kernel_size=3,stride=1,padding=1,bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
        self.third_conv=nn.Sequential(nn.Conv2d(896,256, kernel_size=3,stride=1,padding=1,bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
        self.second_conv=nn.Sequential(nn.Conv2d(960,256, kernel_size=3,stride=1,padding=1,bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())

        self.domain_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
                                       
       

        self.last_conv = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat1, low_level_feat2, low_level_feat3):
        #x10 = self.AttS(x)
        #x11 = self.AttC(x)
        #x= x10+x11
        #media layers
        #x domain
        x_d = self.domain_conv(x)
        


        x1 = F.interpolate(x, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        c11 = self.Att3(g=x1,x=low_level_feat3)
        x11,x13,x14 = self.CA3(x1=x,x2=low_level_feat2,x3=low_level_feat1)
        x11 = F.interpolate(x11, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        x13 = F.interpolate(x13, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        x14 = F.interpolate(x14, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        a2 = torch.cat((x13,x14),dim=1)
        c12,c13 = self.Con3(g=x11,x=a2)
        x1 = torch.cat((c11,c12),dim=1)
        x1 = torch.cat((x1,c13),dim=1)
        x1 = x1+self.conv_3(x1)


        x2 = F.interpolate(low_level_feat3, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        c21 = self.Att2(g=x2,x=low_level_feat2)
        x21,x22,x24 = self.CA2(x1=x,x2=low_level_feat3,x3=low_level_feat1)
        x21 = F.interpolate(x21, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        x22 = F.interpolate(x22, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        x24 = F.interpolate(x24, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        a = torch.cat((x21,x22), dim=1)
        c22,c23 = self.Con2(g=a,x=x24)
        x2 = torch.cat((c21,c22),dim=1)
        x2 = torch.cat((x2,c23),dim=1)
        x2 = x2+self.conv_2(x2)

        x3 = F.interpolate(low_level_feat2, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        c31 = self.Att1(g=x3,x=low_level_feat1)
        x31,x32,x33 = self.CA1(x1=x,x2=low_level_feat3,x3=low_level_feat2)
        x31 = F.interpolate(x31, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        x32 = F.interpolate(x32, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        x33 = F.interpolate(x33, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        a = torch.cat((x31,x32), dim=1)
        a = torch.cat((a,x33), dim=1)
        c32 = self.Con1(g=a)
        x3 = torch.cat((c31,c32),dim=1)
        x3 = x3+self.conv_1(x3)

        #dense 
        x11 = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat((x11,x1),dim=1)
        x1 = self.four_conv(x1)

        x21 = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x22 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2 = torch.cat((x2,x21),dim=1)
        x2 = torch.cat((x2,x22),dim=1)
        x2 = self.third_conv(x2)

        x31 = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x32 = F.interpolate(x1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x33 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat((x3,x33),dim=1)
        x3 = torch.cat((x3,x32),dim=1)
        x3 = torch.cat((x3,x31),dim=1)
        x3 = self.second_conv(x3)

        x41 = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x42 = F.interpolate(x1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x43 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x3,x43),dim=1)
        x = torch.cat((x,x42),dim=1)
        x = torch.cat((x,x41),dim=1)
        x = self.last_conv(x)

        return x_d, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)