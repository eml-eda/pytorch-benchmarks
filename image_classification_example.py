import torch
from pytorch_model_summary import summary
import pytorch_benchmarks.image_classification as icl

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Get the Data
datasets = icl.get_data()
dataloaders = icl.build_dataloaders(datasets)
train_dl, val_dl, test_dl = dataloaders

# Get the Model
model = icl.get_reference_model('resnet_8')
if torch.cuda.is_available():
    model = model.cuda()

# Model Summary
input_example = torch.unsqueeze(datasets[0][0][0], 0)
print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

# Get Training Settings
criterion = icl.get_default_criterion()
optimizer = icl.get_default_optimizer(model)
scheduler = icl.get_default_scheduler(optimizer)

# Training Loop
N_EPOCHS = 500
for epoch in range(N_EPOCHS):
    _ = icl.train_one_epoch(epoch, model, criterion, optimizer, train_dl, val_dl, device)
    scheduler.step()
test_metrics = icl.evaluate(model, criterion, test_dl, device)

print("Test Set Loss:", test_metrics['loss'])
print("Test Set Accuracy:", test_metrics['acc'])
