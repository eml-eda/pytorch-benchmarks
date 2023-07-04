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
import pytorch_benchmarks.visual_wake_words as vww
from pytorch_benchmarks.utils import seed_all

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Ensure deterministic execution
seed = seed_all(seed=42)

# Get the Data
datasets = vww.get_data(seed=seed)
# tmp #
test_data = datasets[2]
test_label = [test_data[i][1].item() for i in range(len(test_data))]
import pickle
with open('label_billie.pkl', 'wb') as f:
    pickle.dump(test_label, f)
# tmp #
dataloaders = vww.build_dataloaders(datasets)
train_dl, val_dl, test_dl = dataloaders

# Get the Model
model = vww.get_reference_model('mobilenet')
if torch.cuda.is_available():
    model = model.cuda()

# Model Summary
input_example = torch.unsqueeze(datasets[0][0][0], 0)
print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = vww.get_default_criterion()
optimizer = vww.get_default_optimizer(model)
scheduler = vww.get_default_scheduler(optimizer)

# Training Loop
N_EPOCHS = 1
for epoch in range(N_EPOCHS):
    _ = vww.train_one_epoch(epoch, model, criterion, optimizer, train_dl, val_dl, device)
    scheduler.step()
test_metrics = vww.evaluate(model, criterion, test_dl, device)

print("Test Set Loss:", test_metrics['loss'])
print("Test Set Accuracy:", test_metrics['acc'])
