import torch
from pytorch_model_summary import summary
import pytorch_benchmarks.keyword_spotting as kws
from pytorch_benchmarks.utils import seed_all

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Ensure determinstic execution
seed = seed_all(seed=42)

# Get the Data
# use commented line for TCResNet14
# datasets = kws.get_data(one_dim=True)
datasets = kws.get_data()
dataloaders = kws.build_dataloaders(datasets)
train_dl, val_dl, test_dl = dataloaders

# Get the Model
# use commented line for TCResNet14
# model = kws.get_reference_model('tc_resnet_14')
model = kws.get_reference_model('ds_cnn')
if torch.cuda.is_available():
    model = model.cuda()

# Model Summary
input_example = torch.unsqueeze(datasets[0][0][0], 0)
print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = kws.get_default_criterion()
optimizer = kws.get_default_optimizer(model)
scheduler = kws.get_default_scheduler(optimizer)

# Training Loop
N_EPOCHS = 36
for epoch in range(N_EPOCHS):
    _ = kws.train_one_epoch(epoch, model, criterion, optimizer, train_dl, val_dl, device)
    scheduler.step()
test_metrics = kws.evaluate(model, criterion, test_dl, device)

print("Test Set Loss:", test_metrics['loss'])
print("Test Set Accuracy:", test_metrics['acc'])
