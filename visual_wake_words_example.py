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
datasets = vww.get_data()
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
N_EPOCHS = 50
for epoch in range(N_EPOCHS):
    _ = vww.train_one_epoch(epoch, model, criterion, optimizer, train_dl, val_dl, device)
    scheduler.step()
test_metrics = vww.evaluate(model, criterion, test_dl, device)

print("Test Set Loss:", test_metrics['loss'])
print("Test Set Accuracy:", test_metrics['acc'])
