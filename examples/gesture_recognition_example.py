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
# * Author:  Alessio Burrello <alessio.burrello@polito.it>                     *
# *----------------------------------------------------------------------------*

import torch
from pytorch_model_summary import summary
import pytorch_benchmarks.gesture_recognition as gr
from pytorch_benchmarks.utils import seed_all, EarlyStopping

N_EPOCHS = 500

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Ensure deterministic execution
seed = seed_all(seed=42)

# Get the Model
model = gr.get_reference_model('bioformer')
if torch.cuda.is_available():
    model = model.cuda()

# Model Summary
input_example = torch.rand((1,) + model.input_shape)
print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = gr.get_default_criterion()
optimizer = gr.get_default_optimizer(model)

# Get the Data and perform cross-validation
Acc_dict = dict()
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for sub in subjects:
    data_gen = gr.get_data(subjects=[sub])
    train_loader, val_loader, test_loader = gr.build_dataloaders(data_gen, seed=seed)
    # Set earlystop
    earlystop = EarlyStopping(patience=20, mode='min')
    # Training Loop
    for epoch in range(N_EPOCHS):
        metrics = gr.train_one_epoch(epoch, model, criterion, optimizer, train_loader, device)
        if earlystop(metrics['loss']):
            break
        test_metrics = gr.evaluate(model, criterion, test_loader, device)
        print("Test Set Loss:", test_metrics['loss'])
        print("Test Set Acc:", test_metrics['Acc'])
    test_metrics = gr.evaluate(model, criterion, test_loader, device)
    Acc_dict[f"{sub}"] = test_metrics['Acc']
print(f'Acc: {Acc_dict}')
print(f'Average Acc: {sum(Acc_dict.values()) / len(Acc_dict)}')
