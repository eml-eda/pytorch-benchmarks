import torch
from pytorch_model_summary import summary
import numpy as np
import pytorch_benchmarks.anomaly_detection as amd

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Get the Data
datasets = amd.get_data()
dataloaders = amd.build_dataloaders(datasets)
train_dl, val_dl, test_dl = dataloaders

# Get the Model
model = amd.get_reference_model('autoencoder')
if torch.cuda.is_available():
    model = model.cuda()

# Model Summary
input_example = torch.unsqueeze(datasets[0][0], 0)
print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = amd.get_default_criterion()
optimizer = amd.get_default_optimizer(model)

# Training Loop
N_EPOCHS = 100
for epoch in range(N_EPOCHS):
    _ = amd.train_one_epoch(epoch, model, criterion, optimizer, train_dl, val_dl, device)

# Testing model
results = amd.test_model(test_dl, model)
performance = []
print("\nTest results:")
for k, v in results.items():
    print('machine id={}, accuracy={:.5f}, precision/accuracy={:.5f}, auc={:.5f}, p_auc={:.5f}'
          .format(k, v['acc'], v['pr_acc'], v['auc'], v['p_auc']))
    performance.append([v['auc'], v['p_auc']])
# calculate averages for AUCs and pAUCs
averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
print("Average AUC: ", averaged_performance[0])
print("Average pAUC: ", averaged_performance[1])
