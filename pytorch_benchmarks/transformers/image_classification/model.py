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
from timm.models import create_model


def get_reference_model(num_classes: int, is_encoder_frozen: bool, from_scratch: bool):
    model = create_model('vit_base_patch16_224', pretrained=not from_scratch, drop_path_rate=0.1)

    if is_encoder_frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.reset_classifier(num_classes=num_classes)
    return model
