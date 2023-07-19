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

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None,
                        ) -> nn.Module:
    # Unpack info from model_config
    classification = model_config.get('classification', True)
    win_size = model_config.get('win_size', 1)
    class_num = model_config.get('class_num', 4)
    out_ch_1 = model_config.get('out_ch_1', 64)
    out_ch_2 = model_config.get('out_ch_2', 64)
    use_pool = model_config.get('use_pool', False)
    use_2nd_conv = model_config.get('use_2nd_conv', True)
    use_2nd_lin = model_config.get('use_2nd_lin', True)

    model_zoo = ['simple_cnn', 'concat_cnn', 'cnn_tcn']

    assert (model_name in model_zoo), f'Select a model from {model_zoo}'

    if model_name == 'simple_cnn':
        model = SimpleCNN(classification, win_size, class_num,
                          out_ch_1, out_ch_2,
                          use_pool, use_2nd_conv, use_2nd_lin)
    elif model_name == 'concat_cnn':
        model = ConcatCNN(classification, win_size, class_num,
                          out_ch_1, out_ch_2,
                          use_pool, use_2nd_conv, use_2nd_lin)
    elif model_name == 'cnn_tcn':
        raise NotImplementedError

    return model


class SimpleCNN(nn.Module):
    def __init__(self, classification, win_size, class_num,
                 out_ch_1, out_ch_2,
                 use_pool, use_2nd_conv, use_2nd_lin):
        super(SimpleCNN, self).__init__()
        self.classification = classification
        self.win_size = win_size
        self.class_num = class_num
        self.use_pool = use_pool
        self.use_2nd_conv = use_2nd_conv
        self.use_2nd_lin = use_2nd_lin
        self.input_spat_dim = 8
        # Input convolution, always present
        self.conv1 = nn.Conv2d(in_channels=win_size, out_channels=out_ch_1,
                               kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch_1)
        self.spat_dim = self.input_spat_dim - 3 + 1
        self.ch_dim = out_ch_1
        # Optional pooling layer
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.spat_dim = self.spat_dim // 2
        # Optional 2nd conv layer
        if use_2nd_conv:
            msg = 'To use two conv specify the "out_ch_2" argument.'
            assert out_ch_2 is not None, msg
            self.conv2 = nn.Conv2d(in_channels=out_ch_1, out_channels=out_ch_2,
                                   kernel_size=3, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=out_ch_2)
            self.spat_dim = self.spat_dim - 3 + 1
            self.ch_dim = out_ch_2
        # Optional 2nd linear layer
        if use_2nd_lin:
            self.lin1 = nn.Linear(self.ch_dim * self.spat_dim**2, 64)
            self.spat_dim = 1
            self.ch_dim = 64  # Apparently this is hard-coded to 64
        # Output linear, always present
        if classification:
            self.lin2 = nn.Linear(self.ch_dim * self.spat_dim**2, class_num)
        else:
            self.lin2 = nn.Linear(self.ch_dim * self.spat_dim**2, 1)

    def forward(self, x):
        # Input convolution
        x = F.relu(self.bn1(self.conv1(x)))
        # Optional pooling
        if self.use_pool:
            x = self.pool(x)
        # Optional 2nd conv
        if self.use_2nd_conv:
            x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        # Optional 2nd linear
        if self.use_2nd_lin:
            x = F.relu(self.lin1(x))
        # Output linear
        x = self.lin2(x)
        if not self.classification:
            x = F.sigmoid(x)
        return x


class ConcatCNN(nn.Module):
    def __init__(self, classification, win_size, class_num,
                 out_ch_1, out_ch_2,
                 use_pool, use_2nd_conv, use_2nd_lin):
        super(ConcatCNN, self).__init__()
        self.classification = classification
        self.win_size = win_size
        self.class_num = class_num
        self.use_pool = use_pool
        self.use_2nd_conv = use_2nd_conv
        self.use_2nd_lin = use_2nd_lin
        self.input_spat_dim = 8
        # Input convolution, always present
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_ch_1,
                               kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch_1)
        self.spat_dim = self.input_spat_dim - 3 + 1
        self.ch_dim = out_ch_1
        # Optional pooling layer
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.spat_dim = self.spat_dim // 2
        # Optional 2nd conv layer
        if use_2nd_conv:
            msg = 'To use two conv specify the "out_ch_2" argument.'
            assert out_ch_2 is not None, msg
            self.conv2 = nn.Conv2d(in_channels=out_ch_1, out_channels=out_ch_2,
                                   kernel_size=3, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=out_ch_2)
            self.spat_dim = self.spat_dim - 3 + 1
            self.ch_dim = out_ch_2
        # Optional 2nd linear layer
        if use_2nd_lin:
            self.lin1 = nn.Linear(self.win_size * self.ch_dim * self.spat_dim**2, 64)
            self.spat_dim = 1
            self.ch_dim = 64  # Apparently this is hard-coded to 64
        # Output linear, always present
        if classification:
            self.lin2 = nn.Linear(self.ch_dim * self.spat_dim**2, class_num)
        else:
            self.lin2 = nn.Linear(self.ch_dim * self.spat_dim**2, 1)

    def forward(self, x):
        feature_list = []
        for x_i in x:
            # Input convolution
            x_i = F.relu(self.bn1(self.conv1(x_i)))
            # Optional pooling
            if self.use_pool:
                x_i = self.pool(x_i)
            # Optional 2nd conv
            if self.use_2nd_conv:
                x_i = F.relu(self.bn2(self.conv2(x_i)))
            feature_list.append(x_i)
        # Concat and flatten
        x = torch.cat(feature_list, dim=1)
        x = torch.flatten(x, 1)
        # Optional 2nd linear
        if self.use_2nd_lin:
            x = F.relu(self.lin1(x))
        # Output linear
        x = self.lin2(x)
        if not self.classification:
            x = F.sigmoid(x)
        return x


'''
# -------------------------------------------CNN+TCN+FC1-----------------------------------------------------
def build_CNN_TCN_FC1_pc(channel1:int, channel2:int, classification:bool, win_size:int,
class_number:int):
    # Build CNN architecture 3 model with 2 dense layers
    print("CNN3_TCN_FC1!")

    tcn_out = 32

    class CNN3_TCN_FC1_pc(nn.Module):
        def __init__(self, channel1, channel2, classification, win_size):

            #--------------------------------------------- feature extractor-----------------------
            super(CNN3_TCN_FC1_pc, self).__init__()
            # Pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1=6
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(channel1),
            kernel_size=3, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=int(channel1))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel),
            # the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2)
            # Pass (channel1,3,3) input to get output (channel2,1,1) with kernel (3,3) -> 3-3+1=1
            self.conv2 = nn.Conv2d(in_channels=channel1, out_channels=int(channel2), kernel_size=3,
            bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=int(channel2))
            self.sigmoid = nn.Sigmoid()

            #--------------------------------------------- tcn and fc layers---------------------
            self.tcn = nn.Conv1d(in_channels=channel2, out_channels=tcn_out, kernel_size=3)
            self.bn3 = nn.BatchNorm1d(num_features=int(tcn_out))

            self.classification = classification
            self.fc1 = nn.Linear(tcn_out, class_number if classification else 1)


        def forward(self, x1, x2, x3):
            feature_list = []

            x = F.relu(self.bn1(self.conv1(x1)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            out_i = torch.flatten(x, 1)
            out_to_cat = out_i.unsqueeze(-1)

            feature_list.append(out_to_cat)

            x = F.relu(self.bn1(self.conv1(x2)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            out_i = torch.flatten(x, 1)
            out_to_cat = out_i.unsqueeze(-1)

            feature_list.append(out_to_cat)

            x = F.relu(self.bn1(self.conv1(x3)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            out_i = torch.flatten(x, 1)
            out_to_cat = out_i.unsqueeze(-1)

            feature_list.append(out_to_cat)

            feature_out = torch.cat(feature_list, dim=-1)

            x = self.bn3(self.tcn(feature_out))
            x = F.relu(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))

            if self.classification:
                x = x
            else:
                  x = self.sigmoid(x)
            return x

    model = CNN3_TCN_FC1_pc(channel1, channel2, classification, win_size)

    return model
'''
