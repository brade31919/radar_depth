import os
import sys
sys.path.append("../")
sys.path.append("../results/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3
from model.models import Unpool, weights_init, weights_init_kaiming, weights_init_kaiming_leaky
from model.models import BasicBlock, Decoder, DeConv, UpConv, UpProj, choose_decoder
from model.models import ResNet_latefusion
from config.config_nuscenes import config_nuscenes as cfg
import collections
import math

################################
## Define the full model here ##
################################
# The multistage network
class ResNet_multistage(nn.Module):
    def __init__(self, layers, decoder, output_size, pretrained=True):
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        super(ResNet_multistage, self).__init__()

        # Define the model here
        self.stage1 = ResNet_latefusion2(layers, decoder, output_size, in_channels=4, pretrained=True)
        self.stage2 = ResNet_latefusion2(layers, decoder, output_size, in_channels=5, pretrained=True)
        self.filter_layer = Filter_layer()

        # ToDo: add method to load pretrained latefusion model
        if pretrained is True:
            # Get pretrained weights
            pretrained_path = os.path.join(cfg.PROJECT_ROOT, "pretrained/resnet18_latefusion.pth.tar")
            if not os.path.exists(pretrained_path):
                raise ValueError("[Error] Can't find pretrained latefusion model. "\
                    "Please follow the instructions in README.md to download the weights!")
            checkpoint = torch.load(pretrained_path)
            pretrain_weight = checkpoint["model_state_dict"]

            # Load state dict
            # Stage1 is the same so no problem
            self.stage1.load_state_dict(pretrain_weight)

            # Stage2 has some inconsistencies
            pretrain_weight_filtered = self.filter_state_dict(pretrain_weight, self.stage2.state_dict())
            self.stage2.load_state_dict(pretrain_weight_filtered, strict=False)

    def filter_state_dict(self, pretrain_dict, target_dict):
        # iterate throught all the pretrain element
        del_keys = []
        for key, value in pretrain_dict.items():
            if target_dict[key].shape != value.shape:
                del_keys.append(key)

        for key in del_keys:
            pretrain_dict.pop(key)

        return pretrain_dict

    def forward(self, x):
        # Fetch inputs from different dimensions
        x_img = x[:, :3, :, :]
        x_d = x[:, 3:, :, :]

        # Stage 1 inference
        depth_stage1 = self.stage1(x)

        # Perform filtering
        x_d_filtered, mask = self.filter_layer(x_d, depth_stage1)

        # Stage 2 inference
        x_stage2 = torch.cat((x_img, x_d_filtered, depth_stage1), dim=1)
        depth_stage2 = self.stage2(x_stage2)

        return {
            "stage1": depth_stage1,
            "stage2": depth_stage2,
            "mask": mask,
            "radar_filtered": x_d_filtered
        }


# Filter intermediate outputs
class Filter_layer(nn.Module):
    def __init__(self):
        super(Filter_layer, self).__init__()
        # Define some filter parameters
        self.alpha = torch.tensor(5.)
        self.beta = torch.tensor(18.)
        self.K = torch.tensor(100.)

    # Convert to SID depth threshold
    def sid_depth_thresh(self, input_depth):
        # Compute depth threshold
        depth_thresh = torch.exp(((input_depth * torch.log(self.beta / self.alpha)) / self.K) + torch.log(self.alpha))

        return depth_thresh

    # Compute valid mask
    def compute_valid_mask(self, sparse_depth, dense_depth):
        # Compute depth distance
        diff = torch.abs(dense_depth - sparse_depth)

        # Compute depth threshold
        depth_thresh = self.sid_depth_thresh(dense_depth)

        valid_mask = diff <= depth_thresh

        return valid_mask

    # Forward pass of the filtering
    def forward(self, sparse_depth, dense_depth):
        # Get valid mask
        mask = self.compute_valid_mask(sparse_depth, dense_depth).to(torch.float32)
        # pdb.set_trace()
        return sparse_depth * mask, mask


# The original latefusion model
class ResNet_latefusion2(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=4, pretrained=True):
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_latefusion2, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        # Configurations required by resnet
        self.in_channels = in_channels
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
        depth_input_dim = self.in_channels - 3
        self.conv1_depth = nn.Conv2d(depth_input_dim, 16, kernel_size=7, stride=2, padding=3, bias=False)
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
        assert x.shape[1] >= 4
        x_img = x[:, :3, :, :]

        if self.in_channels == 4:
            x_d = x[:, 3:, :, :]
        else:
            x_d_sparse = x[:, 3:4, :, :]
            x_d_dense = x[:, 4:5, :, :]
            x_d = torch.cat((x_d_sparse, x_d_dense), dim=1)

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


if __name__ == "__main__":
    # Create fake inputs
    inputs = torch.rand([16, 4, 450, 800]).to(torch.float32).cuda()

    # Create model
    model = ResNet_multistage(18, "upproj", [450, 800], True).cuda()

    # Run the inference
    outputs = model(inputs)