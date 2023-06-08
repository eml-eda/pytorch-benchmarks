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
import pytorch_benchmarks.transformers.image_classification as icl
from pytorch_benchmarks.utils import seed_all
import os

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Ensure deterministic execution
seed = seed_all(seed=42)

# impose dataset information
dataset_name = 'cifar10'
num_classes = 10

# Get the Data
datasets = icl.get_data(dataset_name, download=True)
dataloaders = icl.build_dataloaders(datasets, num_workers=os.cpu_count())
train_dl, val_dl, test_dl = dataloaders

# Get the Model
model = icl.get_reference_model(num_classes=num_classes, from_scratch=False, is_encoder_frozen=True)
if torch.cuda.is_available():
    model = model.cuda()

# Get Training Settings
criterion = icl.get_default_criterion()
optimizer = icl.get_default_optimizer(model)

# Training Loop
N_EPOCHS = 1
for epoch in range(N_EPOCHS):
    _ = icl.train_one_epoch(epoch, model, criterion, optimizer, train_dl, val_dl, device)
test_metrics = icl.evaluate(model, criterion, test_dl, device)

print("Test Set Loss:", test_metrics['loss'])
print("Test Set Accuracy:", test_metrics['acc'])
