import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from nets.resnet import resnet50
from nets.vgg import VGG16


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((512, 512))
        self.max_pool = nn.AdaptiveMaxPool2d((512, 512))
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1,  bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1,  bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y_avg = self.fc(self.avg_pool(x))
        y_max = self.fc(self.max_pool(x))
        
        out = y_avg + y_max

        return out
    
class PumpFourierNN(nn.Module):
    def __init__(self):
        super(PumpFourierNN, self).__init__()
                
        # 定义隐藏层之间的全连接层
        self.conv_hidden_layers1  = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0)
        self.conv_hidden_layers3  = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)       
        
        # 定义激活函数
        self.up     = nn.UpsamplingNearest2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)        

    def forward(self,inputs, x0):
        size = [int(s) for s in inputs.shape[2:]]

        x1 =  self.up(x0)
        x1 =  self.relu(x1)       
        # print(x1.shape)
        
        xf  = fft.fft(x1).real
        x11  =  self.conv_hidden_layers3(x1)
        x22  =  self.conv_hidden_layers3(x1)
     
        x33  =  self.conv_hidden_layers3(xf) 
                  
        x44  = torch.cat([x11,x22,x33], 1)
        x44  = self.conv_hidden_layers1(x44)
        
        x55 = F.interpolate(x44, size=size, mode='bilinear', align_corners=False)

        return x55*10

class ResFourierNN(nn.Module):
    def __init__(self):
        super(ResFourierNN, self).__init__()

        self.conv_hidden_layers1  = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv_hidden_layers2  = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
        
        # 定义激活函数
        self.up     = nn.UpsamplingNearest2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs, x0):
        # print(x0.shape)
        size   = [int(s) for s in inputs.shape[2:]]
         
        xi_up = self.up(x0)
        # print(xi_up.shape)
        xi_up_relu = self.relu(xi_up)
        xi_up_relu_conv1 = self.conv_hidden_layers1(xi_up_relu)
        xi_up_relu_conv2 = self.conv_hidden_layers1(xi_up_relu_conv1)
        xi_up_relu_conv2 = F.interpolate(xi_up_relu_conv2, size=size, mode='bilinear', align_corners=False)

        x66  = self.conv_hidden_layers2(xi_up_relu_conv2)
        x66 = self.relu(x66)
        # print(x66.shape)
        
        return x66

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class UNet(nn.Module):
    def __init__(self, num_classes = 2, pretrained = False, backbone = 'vgg'):
        super(UNet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone
        
        self.res   = ResFourierNN()
        self.Pump  = PumpFourierNN()
        self.ChannelAttention = ChannelAttention(6)
        
    def forward(self, x, stage_time, res, pump):
        
        res_new  = self.res(x, res.float())
        pump_new = self.Pump(x, pump.float())
        
        inputs = torch.cat([stage_time.float(), x, res_new, pump_new], 1)
        inputs = self.ChannelAttention(inputs.float())
    
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
