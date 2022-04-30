from toycar_library import *


# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Define configuration variables
config = {
  # directories
  "dev_directory": "./dev_data",
  "eval_directory": "./eval_data",
  "model_directory": "./model",
  "result_directory": "./result",
  "result_file": "result.csv",

  # audio parameters
  'n_mels': 128,
  'frames': 5,
  'n_fft': 1024,
  'hop_length': 512,
  'power': 2.0,

  # data
  "batch_size": 512,
  "num_workers": 2,
  "val_split": 0.1,

  # training
  "n_epochs": 100,
  "lr": 0.001,
  "max_fpr" : 0.1
}

# check mode
# "development": mode == True
# "evaluation": mode == False
mode = com.command_line_chk()
if mode is None:
  sys.exit(-1)


# make output directory
os.makedirs(config["model_directory"], exist_ok=True)

# load base_directory list for development
dirs = com.select_dirs(param=config, mode=mode)

# loop of the base directory
for idx, target_dir in enumerate(dirs):
  print("\n===========================")
  print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

  # set path
  machine_type = os.path.split(target_dir)[1]
  model_file_path = "{model}/model_{machine_type}.hdf5".format(model=config["model_directory"],machine_type=machine_type)

  if os.path.exists(model_file_path):
    com.logger.info("model exists")
    continue

  # generate dataset
  print("============== DATASET_GENERATOR ==============")

  # Import benchmark dataset
  train_val_set = get_benchmark(target_dir, config)

  # Define training, validation and test dataloader
  trainLoader, valLoader = get_dataloaders(config, train_val_set)

  # train model
  print("============== MODEL TRAINING ==============")
  net = AutoEncoder(config["n_mels"]*config["frames"])
  if torch.cuda.is_available():
    net = net.cuda()

  # Define the optimizer, the loss and the number of epochs
  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters())
  checkpoint = CheckPoint('./checkpoints', net, optimizer, 'max')

  # Training loop
  for epoch in range(config['n_epochs']):
    results = train_one_epoch(epoch, net, criterion, optimizer, trainLoader, valLoader, device)
    checkpoint(epoch, results['val_loss'])

  #net.save(model_file_path)
  #com.logger.info("save_model -> {}".format(model_file_path))
  #print("============== END TRAINING ==============")


# load base_directory list for testing
dirs = com.select_dirs(param=config, mode=mode)

# initialize lines in csv for AUC and pAUC
csv_lines = []

# loop of the base directory
for idx, target_dir in enumerate(dirs):
  print("\n===========================")
  print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))
  machine_type = os.path.split(target_dir)[1]

  # make output result directory
  os.makedirs(config["result_directory"], exist_ok=True)

  if mode:
    # results by type
    csv_lines.append([machine_type])
    csv_lines.append(["id", "AUC", "pAUC"])
    performance = []

  machine_id_list = com.get_machine_id_list_for_test(target_dir)

  for id_str in machine_id_list:
    # load test file
    test_files, y_true = com.test_file_list_generator(target_dir, id_str, mode)

    # setup anomaly score file path
    anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                             result=config["result_directory"],
                                                                             machine_type=machine_type,
                                                                             id_str=id_str)
    anomaly_score_list = []

    print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
    y_pred = [0. for k in test_files]
    for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
        try:
          data = com.file_to_vector_array(file_path,
                                          n_mels=config["n_mels"],
                                          frames=config["frames"],
                                          n_fft=config["n_fft"],
                                          hop_length=config["hop_length"],
                                          power=config["power"])
          #print("data", type(data), data)
          data = data.astype('float32')
          #print("data float32", type(data), data.shape, data)
          data = torch.from_numpy(data)
          #print("tensor data", type(data), data)
          pred = net(data)
          #print("prediction", pred)

          data = data.cpu().detach().numpy()
          pred = pred.cpu().detach().numpy()
          #print("pred numpy", pred.shape, pred)
          diff = data - pred
          #print("difference", diff)
          #print("square", numpy.square(data - pred))
          errors = numpy.mean(numpy.square(data - pred), axis=1)
          #print(errors)
          y_pred[file_idx] = numpy.mean(errors)
          #print(y_pred)
          anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
          #print(anomaly_score_list)
        except Exception as e:
          com.logger.error("file broken!!: {}, {}".format(file_path, e))

    # save anomaly score
    com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
    com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

    if mode:
        # append AUC and pAUC to lists
        auc = metrics.roc_auc_score(y_true, y_pred)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config["max_fpr"])
        csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
        performance.append([auc, p_auc])
        com.logger.info("AUC : {}".format(auc))
        com.logger.info("pAUC : {}".format(p_auc))

        acc_eembc = eval_functions_eembc.calculate_ae_accuracy(y_pred, y_true)
        pr_acc_eembc = eval_functions_eembc.calculate_ae_pr_accuracy(y_pred, y_true)
        auc_eembc = eval_functions_eembc.calculate_ae_auc(y_pred, y_true, "dummy")
        com.logger.info("EEMBC Accuracy: {}".format(acc_eembc))
        com.logger.info("EEMBC Precision/recall accuracy: {}".format(pr_acc_eembc))
        com.logger.info("EEMBC AUC: {}".format(auc_eembc))

    print("\n============ END OF TEST FOR A MACHINE ID ============")

  if mode:
    # calculate averages for AUCs and pAUCs
    averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
    csv_lines.append(["Average"] + list(averaged_performance))
    csv_lines.append([])

if mode:
  # output results
  result_path = "{result}/{file_name}".format(result=config["result_directory"], file_name=config["result_file"])
  com.logger.info("AUC and pAUC results -> {}".format(result_path))
  com.save_csv(save_file_path=result_path, save_data=csv_lines)