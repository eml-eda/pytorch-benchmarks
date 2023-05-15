# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

import copy
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _adapt_resnet18_statedict(pretrained_sd, model_sd, skip_inp=False):
    new_dict = copy.deepcopy(model_sd)
    for (item_pretr, item_mdl) in zip(pretrained_sd.items(), new_dict.items()):
        if skip_inp:
            skip_inp = False
            continue
        if 'fc' in item_pretr[0] and 'fc' in item_mdl[0]:
            continue
        new_dict[item_mdl[0]] = item_pretr[1]
    return new_dict


def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None
                        ) -> nn.Module:
    if model_name == 'resnet_18':
        pretrained = model_config.get('pretrained', True)
        state_dict = model_config.get('state_dict', None)
        std_head = model_config.get('std_head', False)
        model = ResNet18(std_head=std_head)
        if pretrained and state_dict is None:
            pretrained_model = torchvision.models.resnet18(pretrained=True)
            state_dict = _adapt_resnet18_statedict(
                pretrained_model.state_dict(), model.state_dict())
            model.load_state_dict(state_dict)
        elif pretrained and state_dict is not None:
            new_state_dict = _adapt_resnet18_statedict(
                state_dict, model.state_dict(), skip_inp=True)
            model.load_state_dict(new_state_dict, strict=False)
        return model
    else:
        raise ValueError(f"Unsupported model name {model_name}")


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1,
                                        stride=stride, bias=False)
            self.bn_ds = nn.BatchNorm2d(planes)
        else:
            self.downsample = None

    def forward(self, x):
        # if self.downsample is not None:
        #     residual = x
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.bn_ds(residual)

        out = out + residual

        # return out
        return F.relu(out)


class Backbone18(nn.Module):
    def __init__(self, std_head):
        super(Backbone18, self).__init__()
        self.std_head = std_head
        if std_head:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bb_1_0 = BasicBlock(64, 64, stride=1)
        self.bb_1_1 = BasicBlock(64, 64, stride=1)
        self.bb_2_0 = BasicBlock(64, 128, stride=2)
        self.bb_2_1 = BasicBlock(128, 128, stride=1)
        self.bb_3_0 = BasicBlock(128, 256, stride=2)
        self.bb_3_1 = BasicBlock(256, 256, stride=1)
        self.bb_4_0 = BasicBlock(256, 512, stride=2)
        self.bb_4_1 = BasicBlock(512, 512, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        if self.std_head:
            x = self.max_pool(x)
        x = self.bb_1_0(x)
        x = self.bb_1_1(x)
        x = self.bb_2_0(x)
        x = self.bb_2_1(x)
        x = self.bb_3_0(x)
        x = self.bb_3_1(x)
        x = self.bb_4_0(x)
        out = self.bb_4_1(x)
        out = self.avg_pool(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, input_size=64, num_classes=200, std_head=False, **kwargs):
        super(ResNet18, self).__init__()

        # ResNet18 parameters
        self.input_shape = [3, input_size, input_size]
        self.num_classes = 200
        self.inplanes = 64

        # Model
        if std_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False,
                                   padding=3)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False,
                                   padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.backbone = Backbone18(std_head=std_head)

        # Initialize bn and conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # Final classifier
        self.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
