# Different segmentation frameworks
# 2024/09/20
# Duwei Dai

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
from torch.nn.parameter import Parameter

from math import floor
from functools import partial


"""
    Framework_1: Frame_1
    Framework_2: Frame_2
    Framework_3: Frame_3
    Framework_4: Frame_4
"""


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1) 
        output = self.bn(output)
        return F.relu(output)


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

      
class non_bottleneck_1d_eca(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        
        self.eca = eca_layer(chann, k_size=5)
        

    def forward(self, input):
        
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        output = self.eca(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, ratio):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//ratio),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//ratio),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out



class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels, ratio=8):
        super(AffinityAttention, self).__init__()
        print("ratio is: ", ratio)
        self.sab = SpatialAttentionBlock(in_channels, ratio)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        
        return out





class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        
        return x


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)



class Frame_1(nn.Module):
    # image_size=(224, 224), classes=2, channels=3
    def __init__(self, image_size=(224, 224), classes=2, channels=3):
        super(Frame_1, self).__init__()
        img_ch = channels 
        output_ch = classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)             

        x2 = self.Maxpool(x1)          
        x2 = self.Conv2(x2)            
        
        x3 = self.Maxpool(x2)          
        x3 = self.Conv3(x3)            

        x4 = self.Maxpool(x3)          
        x4 = self.Conv4(x4)            

        x5 = self.Maxpool(x4)          
        x5 = self.Conv5(x5)            

        # decoding + concat path
        d5 = self.Up5(x5)              
        d5 = torch.cat((x4,d5),dim=1)  
        
        d5 = self.Up_conv5(d5)         
        
        d4 = self.Up4(d5)             
        d4 = torch.cat((x3,d4),dim=1)  
        d4 = self.Up_conv4(d4)         

        d3 = self.Up3(d4)              
        d3 = torch.cat((x2,d3),dim=1)  
        d3 = self.Up_conv3(d3)         

        d2 = self.Up2(d3)              
        d2 = torch.cat((x1,d2),dim=1)  
        d2 = self.Up_conv2(d2)         

        d1 = self.Conv_1x1(d2)         
        d1 = F.sigmoid(d1)
        
        return d1




class Frame_2(nn.Module):

    def __init__(self, image_size=(224, 224), classes=2, channels=3, encoder=None):  #use encoder to pass pretrained encoder
        super(Frame_2, self).__init__()
        
        drop_1 = 0.03           # 0.03
        drop_2 = 0.3            # 0.3
        
        dim_1 = 128              # default 64
        dim_2 = 256             # 128 
        
        self.initial_block = DownsamplerBlock(3,16)
        
        self.down_1 = DownsamplerBlock(16,dim_1)
        
        self.encoder_1_1 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_2 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_3 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_4 = non_bottleneck_1d_eca(dim_1, drop_1, 1)

        self.down_2 = DownsamplerBlock(dim_1,dim_2)
        
        self.encoder_2_1 = non_bottleneck_1d_eca(dim_2, drop_2, 2)
        self.encoder_2_2 = non_bottleneck_1d_eca(dim_2, drop_2, 4)
        self.encoder_2_3 = non_bottleneck_1d_eca(dim_2, drop_2, 8)
        self.encoder_2_4 = non_bottleneck_1d_eca(dim_2, drop_2, 16)
        
        self.affinity_attention = AffinityAttention(dim_2, ratio=2)
        
        # start decoder #########################################
        
        self.up_1 = UpsamplerBlock(dim_2, dim_1)
        self.decoder_1_1 = non_bottleneck_1d_eca(dim_1, 0, 1)
        self.decoder_1_2 = non_bottleneck_1d_eca(dim_1, 0, 1)
        self.decoder_1_3 = non_bottleneck_1d_eca(dim_1, 0, 1)
        self.decoder_1_4 = non_bottleneck_1d_eca(dim_1, 0, 1)
        
        self.up_2 = UpsamplerBlock(dim_1, 16)
        self.decoder_2_1 = non_bottleneck_1d_eca(16, 0, 1)
        self.decoder_2_2 = non_bottleneck_1d_eca(16, 0, 1)
        self.decoder_2_3 = non_bottleneck_1d_eca(16, 0, 1)
        self.decoder_2_4 = non_bottleneck_1d_eca(16, 0, 1)
        
        self.output_conv = nn.ConvTranspose2d( 16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        
    def forward(self, input):
    
        e_0 = self.initial_block(input)           
        e_1 = self.down_1(e_0)                    
        
        e_1_1 = self.encoder_1_1(e_1)                 
        e_1_2 = self.encoder_1_2(e_1_1)                 
        e_1_3 = self.encoder_1_3(e_1_2)                 
        e_1_4 = self.encoder_1_4(e_1_3)                 
       
        e_2 = self.down_2(e_1_4)                    
        
        e_2_1 = self.encoder_2_1(e_2)                 
        e_2_2 = self.encoder_2_2(e_2_1)
        e_2_3 = self.encoder_2_3(e_2_2)
        e_2_4 = self.encoder_2_4(e_2_3)
        
        attention = self.affinity_attention(e_2_4)
        attention_fuse = e_2_4 + attention     
        
        # start decoder #####################################################################
        
        d_1_0 = self.up_1(attention_fuse)            
        d_1 = e_1_4 + d_1_0
        d_1_1 = self.decoder_1_1(d_1)         
        d_1_2 = self.decoder_1_2(d_1_1)         
        d_1_3 = self.decoder_1_3(d_1_2)
        d_1_4 = self.decoder_1_4(d_1_3)
    
        d_2_0 = self.up_2(d_1_4)              
        d_2 = e_0 + d_2_0
        d_2_1 = self.decoder_2_1(d_2)         
        d_2_2 = self.decoder_2_2(d_2_1)         
        d_2_3 = self.decoder_2_3(d_2_2)
        d_2_4 = self.decoder_2_4(d_2_3)

        logit = self.output_conv(d_2_4)     
        out = F.sigmoid(logit)

        return out

 
class Frame_3(nn.Module):

    def __init__(self, image_size=(224, 224), classes=2, channels=3, encoder=None):  #use encoder to pass pretrained encoder
        super(Frame_3, self).__init__()
        
        drop_1 = 0.03           # 0.03
        drop_2 = 0.3            # 0.3
        
        dim_1 = 128              # default 64
        dim_2 = 256             # 128 
        
        self.initial_block = DownsamplerBlock(3,16)
        
        self.down_1 = DownsamplerBlock(16,dim_1)
        
        self.encoder_1_1 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_2 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_3 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_4 = non_bottleneck_1d_eca(dim_1, drop_1, 1)

        self.down_2 = DownsamplerBlock(dim_1,dim_2)
        
        self.encoder_2_1 = non_bottleneck_1d_eca(dim_2, drop_2, 2)
        self.encoder_2_2 = non_bottleneck_1d_eca(dim_2, drop_2, 4)
        self.encoder_2_3 = non_bottleneck_1d_eca(dim_2, drop_2, 8)
        self.encoder_2_4 = non_bottleneck_1d_eca(dim_2, drop_2, 16)
        
        self.affinity_attention = AffinityAttention(dim_2, ratio=2)
        
        # start decoder #########################################
        
        self.up_1 = UpsamplerBlock(dim_2, dim_1)
        self.decoder_1_1 = non_bottleneck_1d_eca(dim_1, 0, 1)
        
        self.up_2 = UpsamplerBlock(dim_1, 16)
        self.decoder_2_1 = non_bottleneck_1d_eca(16, 0, 1)

        self.output_conv = nn.ConvTranspose2d( 16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        
    def forward(self, input):
    
        e_0 = self.initial_block(input)           
        e_1 = self.down_1(e_0)                    
        
        e_1_1 = self.encoder_1_1(e_1)                 
        e_1_2 = self.encoder_1_2(e_1_1)                 
        e_1_3 = self.encoder_1_3(e_1_2)                 
        e_1_4 = self.encoder_1_4(e_1_3)                 
       
        e_2 = self.down_2(e_1_4)                    
        
        e_2_1 = self.encoder_2_1(e_2)                 
        e_2_2 = self.encoder_2_2(e_2_1)
        e_2_3 = self.encoder_2_3(e_2_2)
        e_2_4 = self.encoder_2_4(e_2_3)
        
        attention = self.affinity_attention(e_2_4)
        attention_fuse = e_2_4 + attention     
        
        # start decoder #####################################################################
        
        d_1_0 = self.up_1(attention_fuse)            
        d_1 = e_1_4 + d_1_0
        d_1_1 = self.decoder_1_1(d_1)         

        d_2_0 = self.up_2(d_1_1)              
        d_2 = e_0 + d_2_0
        d_2_1 = self.decoder_2_1(d_2)         

        logit = self.output_conv(d_2_1)     
        out = F.sigmoid(logit)

        return out


class Frame_4(nn.Module):

    def __init__(self, image_size=(224, 224), classes=2, channels=3, encoder=None):  #use encoder to pass pretrained encoder
        super(Frame_4, self).__init__()
        
        drop_1 = 0.03           # 0.03
        drop_2 = 0.3            # 0.3
        
        dim_1 = 64              # default 64
        dim_2 = 96             # 128 
        
        self.initial_block = DownsamplerBlock(3,16)
        
        self.down_1 = DownsamplerBlock(16,dim_1)
        
        self.encoder_1_1 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_2 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_3 = non_bottleneck_1d_eca(dim_1, drop_1, 1)
        self.encoder_1_4 = non_bottleneck_1d_eca(dim_1, drop_1, 1)

        self.down_2 = DownsamplerBlock(dim_1,dim_2)
        
        self.encoder_2_1 = non_bottleneck_1d_eca(dim_2, drop_2, 2)
        self.encoder_2_2 = non_bottleneck_1d_eca(dim_2, drop_2, 4)
        self.encoder_2_3 = non_bottleneck_1d_eca(dim_2, drop_2, 8)
        self.encoder_2_4 = non_bottleneck_1d_eca(dim_2, drop_2, 16)
        
        self.affinity_attention = AffinityAttention(dim_2, ratio=2)
        
        # start decoder #########################################
        
        self.up_1 = UpsamplerBlock(dim_2, dim_1)
        self.decoder_1_1 = non_bottleneck_1d_eca(dim_1, 0, 1)
        
        self.up_2 = UpsamplerBlock(dim_1, 16)
        self.decoder_2_1 = non_bottleneck_1d_eca(16, 0, 1)

        self.output_conv = nn.ConvTranspose2d( 16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        
    def forward(self, input):
    
        e_0 = self.initial_block(input)           
        e_1 = self.down_1(e_0)                    
        
        e_1_1 = self.encoder_1_1(e_1)                 
        e_1_2 = self.encoder_1_2(e_1_1)                 
        e_1_3 = self.encoder_1_3(e_1_2)                 
        e_1_4 = self.encoder_1_4(e_1_3)                 
       
        e_2 = self.down_2(e_1_4)                    
        
        e_2_1 = self.encoder_2_1(e_2)                 
        e_2_2 = self.encoder_2_2(e_2_1)
        e_2_3 = self.encoder_2_3(e_2_2)
        e_2_4 = self.encoder_2_4(e_2_3)
        
        attention = self.affinity_attention(e_2_4)
        attention_fuse = e_2_4 + attention     
        
        # start decoder #####################################################################
        
        d_1_0 = self.up_1(attention_fuse)            
        d_1 = e_1_4 + d_1_0
        d_1_1 = self.decoder_1_1(d_1)         

        d_2_0 = self.up_2(d_1_1)              
        d_2 = e_0 + d_2_0
        d_2_1 = self.decoder_2_1(d_2)         

        logit = self.output_conv(d_2_1)     
        out = F.sigmoid(logit)

        return out

 
if __name__ == '__main__':
    input = torch.rand(8, 3, 224, 224)
    model = Frame_4()
    out = model(input)
    print(out.shape)
    print(torch.max(out))
    print(torch.min(out))

