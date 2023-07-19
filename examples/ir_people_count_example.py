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

import math
from pathlib import Path

import numpy as np
from pytorch_model_summary import summary
import torch

import pytorch_benchmarks.ir_people_counting as irpc
from pytorch_benchmarks.utils import seed_all, EarlyStopping


DATA_DIR = Path('.')
MODEL = 'concat_cnn'  # One of ['simple_cnn', 'concat_cnn', 'cnn_tcn']
N_EPOCHS = 500
WIN_SIZE = 5  # One of [1, 3, 5, 7, 9]
CLASSIFICATION = True
CLASS_NUM = 4
MAJORITY_WIN = None  # One of [None, 3, 5, 7, 9]

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
# TODO: add out_fc (now hardcoded to 64)
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
fold_samples = []
for data in ds_linaige_cv:
    # Get the model
    model = irpc.get_reference_model(MODEL, model_config)

    # Get class weights
    class_weight = data[-1]

    # Store the number of samples in the test-set of current fold
    fold_samples.append(len(data[1]))

    # Model summary
    input_example = torch.unsqueeze(data[0][0][0], 0).to(device)
    if type(model).__name__ in ['ConcatCNN', 'CNN_TCN']:
        input_example = [input_example[:, i, :, :].unsqueeze(1)
                         for i in range(input_example.shape[1])]
    print(summary(model, input_example, show_input=False, show_hierarchical=True))

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
# Compute fold weights
fold_weights = [s / sum(fold_samples) for s in fold_samples]
# Final Summary
avg_loss = np.average(loss_list, weights=fold_weights)
std_loss = math.sqrt(np.average((loss_list-avg_loss)**2, weights=fold_weights))
print(f"Test Set loss: {loss_list} ({avg_loss} +/- {std_loss})")
avg_bas = np.average(bas_list, weights=fold_weights)
std_bas = math.sqrt(np.average((bas_list-avg_bas)**2, weights=fold_weights))
print(f"Test Set BAS: {bas_list} ({avg_bas} +/- {std_bas})")
avg_acc = np.average(acc_list, weights=fold_weights)
std_acc = math.sqrt(np.average((acc_list-avg_acc)**2, weights=fold_weights))
print(f"Test Set ACC: {acc_list} ({avg_acc} +/- {std_acc})")
avg_roc = np.average(roc_list, weights=fold_weights)
std_roc = math.sqrt(np.average((roc_list-avg_roc)**2, weights=fold_weights))
print(f"Test Set ROC: {roc_list} ({avg_roc} +/- {std_roc})")
avg_f1 = np.average(f1_list, weights=fold_weights)
std_f1 = math.sqrt(np.average((f1_list-avg_f1)**2, weights=fold_weights))
print(f"Test Set F1: {f1_list} ({avg_f1} +/- {std_f1})")
avg_mse = np.average(mse_list, weights=fold_weights)
std_mse = math.sqrt(np.average((mse_list-avg_mse)**2, weights=fold_weights))
print(f"Test Set MSE: {mse_list} ({avg_mse} +/- {std_mse})")
avg_mae = np.average(mae_list, weights=fold_weights)
std_mae = math.sqrt(np.average((mae_list-avg_mae)**2, weights=fold_weights))
print(f"Test Set MAE: {mae_list} ({avg_mae} +/- {std_mae})")
