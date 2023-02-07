import torch
from pytorch_model_summary import summary
import pytorch_benchmarks.tiny_imagenet as tin
from pytorch_benchmarks.utils import seed_all, EarlyStopping

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Ensure deterministic execution
seed = seed_all(seed=42)

# Get the Data with 224pixel resolution as full-imagenet
datasets = tin.get_data(inp_res=224)
dataloaders = tin.build_dataloaders(datasets)
train_dl, val_dl, test_dl = dataloaders

# Get pretrained Model on full imagenet with standard ResNet18 head
config = {'pretrained': True, 'std_head': True}
model = tin.get_reference_model('resnet_18', model_config=config)
if torch.cuda.is_available():
    model = model.cuda()

# Model Summary
input_example = torch.unsqueeze(datasets[0][0][0], 0)
print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = tin.get_default_criterion()
optimizer = tin.get_default_optimizer(model)
scheduler = tin.get_default_scheduler(optimizer)

# Training Loop
N_EPOCHS = 50
# Set earlystop
earlystop = EarlyStopping(patience=10, mode='max')
for epoch in range(N_EPOCHS):
    metrics = tin.train_one_epoch(epoch, model, criterion, optimizer, train_dl, val_dl, device)
    if earlystop(metrics['val_acc']):
        break
    scheduler.step()
test_metrics = tin.evaluate(model, criterion, test_dl, device)

print("Test Set Loss Pretraining:", test_metrics['loss'])
print("Test Set Accuracy Pretraining:", test_metrics['acc'])

# Get the Data with 64pixel resolution as full-imagenet
datasets = tin.get_data(inp_res=64)
dataloaders = tin.build_dataloaders(datasets)
train_dl, val_dl, test_dl = dataloaders

# Now, get Model with reduced ResNet18 head
config = {'pretrained': True,
          'state_dict': model.state_dict(),  # state-dict of previously trained model
          'std_head': False}
tiny_model = tin.get_reference_model('resnet_18', model_config=config)
if torch.cuda.is_available():
    tiny_model = tiny_model.cuda()

# Model Summary
input_example = torch.unsqueeze(datasets[0][0][0], 0)
print(summary(tiny_model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = tin.get_default_criterion()
optimizer = tin.get_default_optimizer(tiny_model)
scheduler = tin.get_default_scheduler(optimizer)

# Training Loop
N_EPOCHS = 50
# Set earlystop
earlystop = EarlyStopping(patience=10, mode='max')
for epoch in range(N_EPOCHS):
    metrics = tin.train_one_epoch(epoch, tiny_model, criterion, optimizer, train_dl, val_dl, device)
    if earlystop(metrics['val_acc']):
        break
    scheduler.step()
test_metrics = tin.evaluate(tiny_model, criterion, test_dl, device)

print("Test Set Loss:", test_metrics['loss'])
print("Test Set Accuracy:", test_metrics['acc'])
