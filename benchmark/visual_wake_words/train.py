from coco_library import *


# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Define configuration variables
config = {
  # data
  "batch_size": 32,
  "num_workers": 2,
  "val_split": 0,
  # training
  "n_epochs": [20, 10, 20],
  "lr": [0.001, 0.0005, 0.00025]
}

# Import benchmark dataset
train_val_set, test_set, labels = get_benchmark()

# Define training, validation and test dataloader
trainLoader, valLoader, testLoader = get_dataloaders(config, train_val_set, test_set)

for i in range(len(config['n_epochs'])):
  print("training #{} with {} epochs and learning rate = {}".format(i, config['n_epochs'][i], config['lr'][i]))

  # Define the model
  net = MobilenetV1()
  if torch.cuda.is_available():
    net = net.cuda()

  # Define the optimizer, the loss and the number of epochs
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=config['lr'][i], weight_decay=1e-4)
  checkpoint = CheckPoint('./checkpoints{}'.format(i), net, optimizer, 'max')

  # Training loop
  for epoch in range(config['n_epochs'][i]):
    metrics = train_one_epoch(epoch, net, criterion, optimizer, trainLoader, valLoader, device)
    if len(valLoader)>0:
      checkpoint(epoch, metrics['val_acc'])
    else:
      checkpoint(epoch, metrics['acc'])

  # Retrieve best checkpoint and test the model
  checkpoint.load_best()
  checkpoint.save('final_best.ckp')
  test_loss, test_acc = evaluate(net, criterion, testLoader, device)
  print("Test Set Accuracy:", test_acc.get())