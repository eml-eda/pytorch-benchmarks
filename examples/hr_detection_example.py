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

import torch
from pytorch_model_summary import summary
import pytorch_benchmarks.hr_detection as hrd
from pytorch_benchmarks.utils import seed_all, EarlyStopping

N_EPOCHS = 500

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Ensure deterministic execution
seed = seed_all(seed=42)

# Get the Model
model = hrd.get_reference_model('temponet')
if torch.cuda.is_available():
    model = model.cuda()

# Model Summary
input_example = torch.rand((1,) + model.input_shape)
print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = hrd.get_default_criterion()
optimizer = hrd.get_default_optimizer(model)

# Get the Data and perform cross-validation
mae_dict = dict()
data_gen = hrd.get_data()
for datasets in data_gen:
    train_ds, val_ds, test_ds = datasets
    test_subj = test_ds.test_subj
    dataloaders = hrd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders
    # Set earlystop
    earlystop = EarlyStopping(patience=20, mode='min')
    # Training Loop
    for epoch in range(N_EPOCHS):
        metrics = hrd.train_one_epoch(
            epoch, model, criterion, optimizer, train_dl, val_dl, device)
        if earlystop(metrics['val_MAE']):
            break
    test_metrics = hrd.evaluate(model, criterion, test_dl, device)
    print("Test Set Loss:", test_metrics['loss'])
    print("Test Set MAE:", test_metrics['MAE'])
    mae_dict[test_subj] = test_metrics['MAE']
print(f'MAE: {mae_dict}')
print(f'Average MAE: {sum(mae_dict.values()) / len(mae_dict)}')
