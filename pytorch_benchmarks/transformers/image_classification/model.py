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
# * Author:  Leonardo Tredese <s302294@studenti.polito.it>                     *
# *----------------------------------------------------------------------------*
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

def reset_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters() 

class ViTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes:int, is_encoder_frozen: bool, from_scratch: bool):
        super(ViTClassifier, self).__init__()
        if model_name == '':
            self.vit = ViTModel(ViTConfig())
        else:
            self.vit = ViTModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.vit.config.hidden_size, num_classes)

        for param in self.vit.parameters():
            param.requires_grad = not is_encoder_frozen

        if from_scratch:
            self.apply(reset_weights)


    def forward(self, x):
        # access the last hidden state of the [CLS] token
        x = self.vit(pixel_values=x, return_dict=True)['last_hidden_state'][:, 0]
        x = self.dropout(x)
        return self.fc(x)

def get_reference_model(model_name: str, num_classes: int, is_encoder_frozen: bool, from_scratch: bool):
    return ViTClassifier(model_name, num_classes, is_encoder_frozen, from_scratch)

