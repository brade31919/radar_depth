"""
This file is adapted from https://github.com/fangchangma/sparse-to-dense.pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3
import collections
import math


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]

        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_kaiming_leaky(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels//2)
        self.layer3 = self.UpProjModule(in_channels//4)
        self.layer4 = self.UpProjModule(in_channels//8)


def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # ipdb.set_trace()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x


class ResNet_pnp(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_pnp, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x

    #######################
    ## PnP-Depth forward ##
    #######################
    def pnp_forward_front(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x

    def pnp_forward_rear(self, x):
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x


class ResNet2(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet2, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1_d = conv_bn_relu(1, 64 // 4, kernel_size=3, stride=2, padding=1)
            self.conv1_img = conv_bn_relu(3, 64 * 3 // 4, kernel_size=3, stride=2, padding=1)

        self.output_size = output_size
        self.in_channels = in_channels

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        if self.in_channels == 3:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        else:
            x_d = self.conv1_d(x[:, 3:, :, :])
            x_img = self.conv1_img(x[:, :3, :, :])
            x = torch.cat((x_img, x_d), 1)
            # x = self.relu(x)
            # x = self.maxpool(x)

        # ipdb.set_trace()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)
        # ipdb.set_trace()
        x = self.conv3(x)
        x = self.bilinear(x)

        return x

    #######################
    ## PnP-Depth forward ##
    #######################
    def pnp_forward_front(self, x):
        # resnet
        if self.in_channels == 3:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        else:
            x_d = self.conv1_d(x[:, 3:, :, :])
            x_img = self.conv1_img(x[:, :3, :, :])
            x = torch.cat((x_img, x_d), 1)
            # x = self.relu(x)
            # x = self.maxpool(x)

        # ipdb.set_trace()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x

    def pnp_forward_rear(self, x):
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x


class ResNet_latefusion(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=4, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_latefusion, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        # Configurations required by resnet
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = 16
        self.groups = 1
        self.base_width = 16

        assert in_channels > 3
        ################
        ## RGB Branch ##
        ################
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        weights_init(self.conv1)
        weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        ##################
        ## Depth Branch ##
        ##################
        self.conv1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_depth = nn.BatchNorm2d(16)
        weights_init_kaiming_leaky(self.conv1)
        weights_init_kaiming(self.bn1)

        self.relu_depth = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool_depth = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_depth = self._make_layer(BasicBlock, 16, 2, stride=1, dilate=False)
        self.layer2_depth = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3_depth = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer4_depth = self._make_layer(BasicBlock, 128, 2, stride=2)

        # ToDo: If we need one more convolution to do the fusion
        # Define the fusion operator
        self.conv_fusion = nn.Conv2d(512 + 128, 512, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(512)

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    # Make layer function adapted from resnet
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        layers = nn.Sequential(*layers)

        # Explicitly initialize layers after construction
        for m in layers.modules():
            weights_init_kaiming(m)

        return layers

    def forward(self, x):
        x_img = x[:, :3, :, :]
        x_d = x[:, 3:, :, :]

        # ipdb.set_trace()
        # RGB
        x_img = self.conv1(x_img)
        x_img = self.bn1(x_img)
        x_img = self.relu(x_img)
        x_img = self.maxpool(x_img) # 113 x 200 x 64
        x_img = self.layer1(x_img)  # 113 x 200 x 64
        x_img = self.layer2(x_img)  # 57 x 100 x 128
        x_img = self.layer3(x_img)  # 29 x 50 x 256
        x_img = self.layer4(x_img)  # 15 x 25 x 512

        # Depth
        x_d = self.conv1_depth(x_d)
        x_d = self.bn1_depth(x_d)
        x_d = self.relu_depth(x_d)
        x_d = self.maxpool_depth(x_d) # 113 x 200 x 16
        x_d = self.layer1_depth(x_d)  # 113 x 200 x 16
        x_d = self.layer2_depth(x_d)  # 57 x 100 x 32
        x_d = self.layer3_depth(x_d)  # 29 x 50 x 64
        x_d = self.layer4_depth(x_d)  # 15 x 25 x 128

        x_fused = torch.cat((x_img, x_d), dim=1)
        x_fused = self.conv_fusion(x_fused)
        x_fused = self.bn_fusion(x_fused)

        x_fused = self.conv2(x_fused)
        x_fused = self.bn2(x_fused)

        # decoder
        x_fused = self.decoder(x_fused)
        x_fused = self.conv3(x_fused)
        x_fused = self.bilinear(x_fused)

        return x_fused

    #######################
    ## PnP-Depth forward ##
    #######################
    def pnp_forward_front(self, x):
        x_img = x[:, :3, :, :]
        x_d = x[:, 3:, :, :]

        # RGB
        x_img = self.conv1(x_img)
        x_img = self.bn1(x_img)
        x_img = self.relu(x_img)
        x_img = self.maxpool(x_img)
        x_img = self.layer1(x_img)
        x_img = self.layer2(x_img)
        x_img = self.layer3(x_img)
        x_img = self.layer4(x_img)

        # Depth
        x_d = self.conv1_depth(x_d)
        x_d = self.bn1_depth(x_d)
        x_d = self.relu_depth(x_d)
        x_d = self.maxpool_depth(x_d)
        x_d = self.layer1_depth(x_d)
        x_d = self.layer2_depth(x_d)
        x_d = self.layer3_depth(x_d)
        x_d = self.layer4_depth(x_d)

        x_fused = torch.cat((x_img, x_d), dim=1)
        x_fused = self.conv_fusion(x_fused)
        x_fused = self.bn_fusion(x_fused)

        x_fused = self.conv2(x_fused)
        x_fused = self.bn2(x_fused)

        return x_fused

    def pnp_forward_rear(self, x):
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x


class ResNet_multifusion(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=4, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_multifusion, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        # Configurations required by resnet
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = 16
        self.groups = 1
        self.base_width = 16

        assert in_channels > 3
        ################
        ## RGB Branch ##
        ################
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        weights_init(self.conv1)
        weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        ##################
        ## Depth Branch ##
        ##################
        self.conv1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_depth = nn.BatchNorm2d(16)
        weights_init_kaiming_leaky(self.conv1)
        weights_init_kaiming(self.bn1)

        self.relu_depth = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool_depth = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_depth = self._make_layer(BasicBlock, 16, 2, stride=1, dilate=False)
        self.layer2_depth = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3_depth = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer4_depth = self._make_layer(BasicBlock, 128, 2, stride=2)

        # ToDo: If we need one more convolution to do the fusion
        # Define the fusion operator
        self.conv_fusion1 = nn.Conv2d(64 + 16, 64, kernel_size=1, bias=False)
        self.bn_fusion1 = nn.BatchNorm2d(64)

        self.conv_fusion2 = nn.Conv2d(128 + 32, 128, kernel_size=1, bias=False)
        self.bn_fusion2 = nn.BatchNorm2d(128)

        self.conv_fusion3 = nn.Conv2d(256 + 64, 256, kernel_size=1, bias=False)
        self.bn_fusion3 = nn.BatchNorm2d(256)

        self.conv_fusion4 = nn.Conv2d(512 + 128, 512, kernel_size=1, bias=False)
        self.bn_fusion4 = nn.BatchNorm2d(512)

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv_fusion1.apply(weights_init_kaiming)
        self.conv_fusion2.apply(weights_init_kaiming)
        self.conv_fusion3.apply(weights_init_kaiming)
        self.conv_fusion4.apply(weights_init_kaiming)

        self.bn_fusion1.apply(weights_init_kaiming)
        self.bn_fusion2.apply(weights_init_kaiming)
        self.bn_fusion3.apply(weights_init_kaiming)
        self.bn_fusion4.apply(weights_init_kaiming)

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    # Make layer function adapted from resnet
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        layers = nn.Sequential(*layers)

        # Explicitly initialize layers after construction
        for m in layers.modules():
            weights_init_kaiming(m)

        return layers

    def forward(self, x):
        x_img = x[:, :3, :, :]
        x_d = x[:, 3:, :, :]

        # ipdb.set_trace()

        # RGB layer1
        x_img = self.conv1(x_img)
        x_img = self.bn1(x_img)
        x_img = self.relu(x_img)
        x_img = self.maxpool(x_img)
        x_img = self.layer1(x_img)

        # Depth layer1
        x_d = self.conv1_depth(x_d)
        x_d = self.bn1_depth(x_d)
        x_d = self.relu_depth(x_d)
        x_d = self.maxpool_depth(x_d)
        x_d = self.layer1_depth(x_d)

        # Fusion layer1
        x_fused1 = torch.cat((x_img, x_d), dim=1)
        x_fused1 = self.conv_fusion1(x_fused1)
        x_fused1 = self.bn_fusion1(x_fused1)

        # RGB layer2
        x_img = self.layer2(x_fused1)
        # Depth layer2
        x_d = self.layer2_depth(x_d)
        # Fusion layer2
        x_fused2 = torch.cat((x_img, x_d), dim=1)
        x_fused2 = self.conv_fusion2(x_fused2)
        x_fused2 = self.bn_fusion2(x_fused2)

        # RGB layer3
        x_img = self.layer3(x_fused2)
        # Depth layer3
        x_d = self.layer3_depth(x_d)
        # Fusion layer3
        x_fused3 = torch.cat((x_img, x_d), dim=1)
        x_fused3 = self.conv_fusion3(x_fused3)
        x_fused3 = self.bn_fusion3(x_fused3)

        # ipdb.set_trace()
        # RGB layer4
        x_img = self.layer4(x_fused3)
        # Depth layer4
        x_d = self.layer4_depth(x_d)
        # Fusion layer4
        x_fused4 = torch.cat((x_img, x_d), dim=1)
        x_fused4 = self.conv_fusion4(x_fused4)
        x_fused4 = self.bn_fusion4(x_fused4)

        x_fused = self.conv2(x_fused4)
        x_fused = self.bn2(x_fused)

        # decoder
        x_fused = self.decoder(x_fused)
        x_fused = self.conv3(x_fused)
        x_fused = self.bilinear(x_fused)

        return x_fused

    #######################
    ## PnP-Depth forward ##
    #######################
    def pnp_forward_front(self, x):
        x_img = x[:, :3, :, :]
        x_d = x[:, 3:, :, :]

        # ipdb.set_trace()

        # RGB layer1
        x_img = self.conv1(x_img)
        x_img = self.bn1(x_img)
        x_img = self.relu(x_img)
        x_img = self.maxpool(x_img)
        x_img = self.layer1(x_img)

        # Depth layer1
        x_d = self.conv1_depth(x_d)
        x_d = self.bn1_depth(x_d)
        x_d = self.relu_depth(x_d)
        x_d = self.maxpool_depth(x_d)
        x_d = self.layer1_depth(x_d)

        # Fusion layer1
        x_fused1 = torch.cat((x_img, x_d), dim=1)
        x_fused1 = self.conv_fusion1(x_fused1)
        x_fused1 = self.bn_fusion1(x_fused1)

        # RGB layer2
        x_img = self.layer2(x_fused1)
        # Depth layer2
        x_d = self.layer2_depth(x_d)
        # Fusion layer2
        x_fused2 = torch.cat((x_img, x_d), dim=1)
        x_fused2 = self.conv_fusion2(x_fused2)
        x_fused2 = self.bn_fusion2(x_fused2)

        # RGB layer3
        x_img = self.layer3(x_fused2)
        # Depth layer3
        x_d = self.layer3_depth(x_d)
        # Fusion layer3
        x_fused3 = torch.cat((x_img, x_d), dim=1)
        x_fused3 = self.conv_fusion3(x_fused3)
        x_fused3 = self.bn_fusion3(x_fused3)

        # ipdb.set_trace()
        # RGB layer4
        x_img = self.layer4(x_fused3)
        # Depth layer4
        x_d = self.layer4_depth(x_d)
        # Fusion layer4
        x_fused4 = torch.cat((x_img, x_d), dim=1)
        x_fused4 = self.conv_fusion4(x_fused4)
        x_fused4 = self.bn_fusion4(x_fused4)

        x_fused = self.conv2(x_fused4)
        x_fused = self.bn2(x_fused)

        return x_fused

    def pnp_forward_rear(self, x):
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        weights_init(m)

    return layers
