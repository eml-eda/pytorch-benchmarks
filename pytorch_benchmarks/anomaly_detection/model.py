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


def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None
                        ) -> nn.Module:
    if model_name == 'autoencoder':
        return AutoEncoder()
    else:
        raise ValueError(f"Unsupported model name {model_name}")


class NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.dense(input)
        return self.relu(self.bn(x))


class AutoEncoder(torch.nn.Module):
    def __init__(self, n_mels=128, frames=5):
        super().__init__()
        self.inputDims = n_mels * frames

        # AutoEncoder layers

        # 1st layer
        self.layer1 = NetBlock(in_channels=self.inputDims, out_channels=128)

        # 2nd layer
        self.layer2 = NetBlock(in_channels=128, out_channels=128)

        # 3d layer
        self.layer3 = NetBlock(in_channels=128, out_channels=128)

        # 4th layer
        self.layer4 = NetBlock(in_channels=128, out_channels=128)

        # 5h layer
        self.layer5 = NetBlock(in_channels=128, out_channels=8)

        # 6th layer
        self.layer6 = NetBlock(in_channels=8, out_channels=128)

        # 7th layer
        self.layer7 = NetBlock(in_channels=128, out_channels=128)

        # 8th layer
        self.layer8 = NetBlock(in_channels=128, out_channels=128)

        # 9th layer
        self.layer9 = NetBlock(in_channels=128, out_channels=128)

        # 10th layer
        self.out = nn.Linear(128, self.inputDims)

    def forward(self, input):

        # Input tensor shape    # [input dims]

        # 1st layer
        x = self.layer1(input)  # [128]

        # 2nd layer
        x = self.layer2(x)      # [128]

        # 3rd layer
        x = self.layer3(x)      # [128]

        # 4th layer
        x = self.layer4(x)      # [128]

        # 5th layer
        x = self.layer5(x)      # [8]

        # 6th layer
        x = self.layer6(x)      # [128]

        # 7th layer
        x = self.layer7(x)      # [128]

        # 8th layer
        x = self.layer8(x)      # [128]

        # 9th layer
        x = self.layer9(x)      # [128]

        # 10th layer
        x = self.out(x)         # [input dims]

        return x
