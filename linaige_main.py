import torch
from pytorch_model_summary import summary
# import pytorch_benchmarks.LINAIGE_Kaggle as lk
from  pytorch_benchmarks.LINAIGE_Kaggle import model as model_module
from  pytorch_benchmarks.LINAIGE_Kaggle import  data as data_module
from  pytorch_benchmarks.LINAIGE_Kaggle import  train as train_module
from pytorch_benchmarks.utils import seed_all, EarlyStopping
from pathlib import Path
import numpy as np

N_EPOCHS = 500

# Check CUDA availability
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Training on:", device)

# Ensure deterministic execution
seed = seed_all(seed=42)

# Listing the parameters
win_size_list = [1,3,5,8]
confindence_list = ['easy', 'all']
remove_frame_list = [True, False]
classification_list = [True, False]
channel1_list = [8, 16, 32, 64]
channel2_list = [8, 16, 32, 64]  # for cnn3
model_name_list = ['cnn1', 'cnn1_dense', 'cnn2', 'cnn2_dense', 'cnn3', 'cnn3_dense']

win_size = win_size_list[1]
confindence = confindence_list[0]
remove_frame = remove_frame_list[0]
classification=classification_list[0]
data_dir=None
model_name=model_name_list[0]
channel1=channel1_list[0]
channel2=channel2_list[0]

parameters = f'''******************Parameters******************: 
win_size={win_size} 
confindence = {confindence} 
remove_frame = {remove_frame} 
classification = {classification}
data_dir = {data_dir}
model_name = {model_name}
channel1 = {channel1}
channel2 = {channel2}
'''
print(parameters)

# Getting the linaige dataset genrator using cross validation
print('***Data Preparation***')
ds_linaige_cv, class_number = data_module.get_data( win_size=win_size,
                                                    confindence=confindence,
                                                    remove_frame=remove_frame,
                                                    classification=classification,
                                                    data_dir=None)


Loss_list = []
BAS_list = []
ACC_list = []
ROC_list = []
F1_list = []
MSE_list = []
MAE_list  = []


# Exporting generated datasets from generator due to cross validation
print('***Starting Main Loop***')
for dataset in ds_linaige_cv:
    # Getting cnn model
    model_base = model_module.get_reference_model(  model_name=model_name,
                                                    channel1=channel1,
                                                    channel2=channel2,
                                                    classification=classification,
                                                    win_size=win_size,
                                                    class_number=class_number)

    # Extracting all datasets from generator
    x_train, y_train, x_test, y_test, class_weight = dataset
    # Passing y_train to get_default_criterion(classification, class_weight=None) funtion to get 'crossEntropy' bassed on class_weights
    criterion = train_module.get_default_criterion(classification=classification, class_weight=class_weight)
    # criterion = train_module.get_default_criterion(classification=classification)
    # Setting the optimizer
    optimizer = train_module.get_default_optimizer(model_base)
    # Setting learning_rate and early_stop
    earlystop = EarlyStopping(patience=10, mode='min')
    reduce_lr = train_module.get_default_scheduler(optimizer)
    # Calling dataloader to creat batched datasets
    ds_train, ds_test = data_module.build_dataloaders(dataset)
    # Starting traning loop
    for epoch in range(N_EPOCHS):
        # Running one epoch for train and getting metric dict as {'loss': avgloss, 'ACC': avgacc, 'ROC': avgroc}

        # metrics = train_module.train_one_epoch(epoch, model_base, criterion, optimizer, ds_train, device, classification, class_number)
        metrics = train_module.train_one_epoch(epoch, model_base, criterion, optimizer, ds_train, device, classification, class_number)

        # Checking if there is not any imporvement in metrics to reduce the learning rate, the patient is 5 epochs
        reduce_lr.step(metrics['loss'])
        # Checking if there is not any imporvement in metrics, the patient is 10 epochs
        if earlystop(metrics['loss']):
            break
    
    '''Two options to save the model apart from 'tqdm' in train.py'''
    # pretrained_weights = 'pretrained_weights.pth'
    # torch.save(model_base.state_dict(), pretrained_weights)
    
    # with tempfile.NamedTemporaryFile(suffix='.pt') as f:
    #     pretrained_weights = f.name
    #     torch.save(model_base.state_dict(), pretrained_weights)
        
    '''Note that we set session as test_set from the last one to the first one except session one'''
    test_metrics = train_module.evaluate(model_base, criterion, ds_test, device, classification, class_number)

    
    Loss_list.append(test_metrics['loss'])
    BAS_list.append(test_metrics['BAS'])
    ACC_list.append(test_metrics['ACC'])
    ROC_list.append(test_metrics['ROC'])
    F1_list.append(test_metrics['F1'])
    MSE_list.append(test_metrics['MSE'])
    MAE_list.append(test_metrics['MAE'])
    
    print("Test Set Loss:", test_metrics['loss'])
    print("Test Set BAS:", test_metrics['BAS'])
    print("Test Set ACC:", test_metrics['ACC'])
    print("Test Set ROC:", test_metrics['ROC'])
    print("Test Set F1:", test_metrics['F1'])
    print("Test Set MSE:", test_metrics['MSE'])
    print("Test Set MAE:", test_metrics['MAE'])


print("Test Set Loss_list:", Loss_list, 'Mean is:', np.mean(Loss_list), 'std is:', np.std(Loss_list))
print("Test Set BAS_list:", BAS_list, 'Mean is:', np.mean(BAS_list), 'std is:', np.std(BAS_list))
print("Test Set ACC_list:", ACC_list, 'Mean is:', np.mean(ACC_list), 'std is:', np.std(ACC_list))
print("Test Set ROC_list:", ROC_list, 'Mean is:', np.mean(ROC_list), 'std is:', np.std(ROC_list))
print("Test Set F1_list:", F1_list, 'Mean is:', np.mean(F1_list), 'std is:', np.std(F1_list))
print("Test Set MSE_list:", MSE_list, 'Mean is:', np.mean(MSE_list), 'std is:', np.std(MSE_list))
print("Test Set MAE_list:", MAE_list, 'Mean is:', np.mean(MAE_list), 'std is:', np.std(MAE_list))  

    