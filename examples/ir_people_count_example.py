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

from pathlib import Path

import numpy as np
from pytorch_model_summary import summary
import torch

import pytorch_benchmarks.ir_people_counting as irpc
from pytorch_benchmarks.utils import seed_all, EarlyStopping


DATA_DIR = Path('.')
N_EPOCHS = 500
WIN_SIZE = 1
CLASSIFICATION = True
CLASS_NUM = 4
MAJORITY_WIN = 5

# No need to use GPU on this task
device = torch.device("cpu")
print("Training on:", device)

# Ensure deterministic execution
seed = seed_all(seed=42)

# Get the data
ds_linaige_cv = irpc.get_data(win_size=WIN_SIZE,
                              confindence='easy',
                              remove_frame=True,
                              classification=CLASSIFICATION,
                              session_number=None,
                              test_split=None,
                              majority_win=MAJORITY_WIN,
                              seed=seed)

# Build model_config dict
model_config = {'classification': CLASSIFICATION,
                'win_size': WIN_SIZE,
                'class_num': CLASS_NUM,
                'out_ch_1': 64,
                'out_ch_2': 64,
                'use_pool': True,
                'use_2nd_conv': True,
                'use_2nd_lin': True,
                }

# Build lists where to store results
loss_list = []
bas_list = []
acc_list = []
roc_list = []
f1_list = []
mse_list = []
mae_list = []

# Cross-val loop
for data in ds_linaige_cv:
    # Get the model
    model = irpc.get_reference_model('simple_cnn', model_config)

    # Get class weights
    class_weight = data[-1]

    # Model summary
    input_example = torch.unsqueeze(data[0][0][0], 0)
    print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

    # Get Training Settings
    criterion = irpc.get_default_criterion(classification=CLASSIFICATION,
                                           class_weight=class_weight)
    optimizer = irpc.get_default_optimizer(model)
    reduce_lr = irpc.get_default_scheduler(optimizer)
    earlystop = EarlyStopping(patience=10, mode='min')

    # Build dataloaders
    train_dl, test_dl = irpc.build_dataloaders(data, seed=seed)

    # Training loop
    for epoch in range(N_EPOCHS):
        metrics = irpc.train_one_epoch(epoch, model, criterion, optimizer, train_dl,
                                       device, CLASSIFICATION, CLASS_NUM)
        reduce_lr.step(metrics['loss'])
        if earlystop(metrics['loss']):
            break
    # Compute test metrics
    test_metrics = irpc.evaluate(model, criterion, test_dl,
                                 device, CLASSIFICATION, CLASS_NUM,
                                 MAJORITY_WIN)
    # Unpack and save metrics for current fold
    loss_list.append(test_metrics['loss'])
    bas_list.append(test_metrics['BAS'])
    acc_list.append(test_metrics['ACC'])
    roc_list.append(test_metrics['ROC'])
    f1_list.append(test_metrics['F1'])
    mse_list.append(test_metrics['MSE'])
    mae_list.append(test_metrics['MAE'])
    # Log metrics
    print(f"Test Set Loss: {test_metrics['loss']}")
    print(f"Test Set BAS: {test_metrics['BAS']}")
    print(f"Test Set ACC: {test_metrics['ACC']}")
    print(f"Test Set ROC: {test_metrics['ROC']}")
    print(f"Test Set F1: {test_metrics['F1']}")
    print(f"Test Set MSE: {test_metrics['MSE']}")
    print(f"Test Set MAE: {test_metrics['MAE']}")
# Final Summary
print(f"Test Set loss: {loss_list} ({np.mean(loss_list)}+/-{np.std(loss_list)})")
print(f"Test Set BAS: {bas_list} ({np.mean(bas_list)}+/-{np.std(bas_list)})")
print(f"Test Set ACC: {acc_list} ({np.mean(acc_list)}+/-{np.std(acc_list)})")
print(f"Test Set ROC: {roc_list} ({np.mean(roc_list)}+/-{np.std(roc_list)})")
print(f"Test Set F1: {f1_list} ({np.mean(f1_list)}+/-{np.std(f1_list)})")
print(f"Test Set MSE: {mse_list} ({np.mean(mse_list)}+/-{np.std(mse_list)})")
print(f"Test Set MAE: {mae_list} ({np.mean(mae_list)}+/-{np.std(mae_list)})")
