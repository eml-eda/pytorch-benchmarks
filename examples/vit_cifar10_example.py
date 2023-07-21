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
batch_size = 64
update_frequency = 128 // batch_size
datasets = icl.get_data(dataset_name, download=True)
dataloaders = icl.build_dataloaders(datasets,
                                    batch_size=batch_size,
                                    num_workers=os.cpu_count(),
                                    seed=seed)
train_dl, val_dl, test_dl = dataloaders

# Get the Model
model_config = {
    'num_classes': num_classes,
    'is_encoder_frozen': False,
    'from_scratch': False
}
model = icl.get_reference_model(model_name='vit_small_patch16_384', model_config=model_config)
if torch.cuda.is_available():
    model = model.cuda()

# Get Training Settings
N_EPOCHS = 20
criterion = icl.get_default_criterion()
optimizer = icl.get_default_optimizer(model)
scheduler_config = {
        'epochs': N_EPOCHS,
        'update_frequency': update_frequency,
        'trainset_len': len(train_dl.dataset)
}
scheduler = icl.get_default_scheduler(optimizer, scheduler_config=scheduler_config)

# Training Loop
for epoch in range(N_EPOCHS):
    _ = icl.train_one_epoch(epoch, model, criterion, optimizer,
                            update_frequency, scheduler, train_dl, val_dl, device)
test_metrics = icl.evaluate(model, criterion, test_dl, device)

print("Test Set Loss:", test_metrics['loss'])
print("Test Set Accuracy:", test_metrics['acc'])
