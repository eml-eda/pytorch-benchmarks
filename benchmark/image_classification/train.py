from cifar_library import *


# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Define configuration variables
config = {
  # data
  "batch_size": 32,
  "num_workers": 2,
  "val_split": 0.2,
  # training
  "n_epochs": 500,
  "lr": 0.001
}

# Import benchmark dataset
train_val_set, test_set, labels = get_benchmark()

# Define training, validation and test dataloader
trainLoader, valLoader, testLoader = get_dataloaders(config, train_val_set, test_set)

# Define the model
net = ResnetV1Eembc()
if torch.cuda.is_available():
  net = net.cuda()

# Define the optimizer, the loss and the number of epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=1e-4)
checkpoint = CheckPoint('./checkpoints', net, optimizer, 'max')

# Training loop
for epoch in range(config['n_epochs']):
  metrics = train_one_epoch(epoch, net, criterion, optimizer, trainLoader, valLoader, device)
  checkpoint(epoch, metrics['val_acc'])

# Retrieve best checkpoint and test the model
checkpoint.load_best()
checkpoint.save('final_best.ckp')
test_loss, test_acc = evaluate(net, criterion, testLoader, device)
print("Test Set Accuracy:", test_acc.get())