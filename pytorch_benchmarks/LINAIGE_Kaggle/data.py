import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
import pickle
import os
import opendatasets as od
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

'''
Read carefully the steps of the link below to download the datasets.
https://github.com/JovianHQ/opendatasets/blob/master/README.md#kaggle-credentials
Briefly, follow these steps to find your API credentials:

1.Go to https://kaggle.com/me/account (sign in if required).

2.Scroll down to the "API" section and click "Create New API Token". This will download a file kaggle.json with the following contents:

3.{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}, When you run opendatsets.download, you will be asked to enter your username & Kaggle API, which you can get from the file downloaded in step 2.

Note that you need to download the kaggle.json file only once. You can also place the kaggle.json file in the same directory as the Jupyter notebook, and the credentials will be read automatically.

'''

LINAIGE_URL = 'https://www.kaggle.com/datasets/francescodaghero/linaige/download?datasetVersionNumber=3'


def get_class_weight(y):
    # If ‘balanced’, class weights will be given by n_samples / (n_classes * np.bincount(y))
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    return class_weights

def read_files(directory_LINAIGE):
    data=list()
    for dirname, _, filenames in os.walk(directory_LINAIGE):
        for filename in sorted(filenames):
            full_path = os.path.join(dirname, filename)
            session_name = filename.split("_")[0]
            session_id = int(session_name.replace("Session",""))
            # Read the file
            file_data = pd.read_csv(full_path)
            file_data["session"] = session_id
            data.append(file_data)
    data = pd.concat(data)
    return data

class Linaige_set(Dataset):
    def __init__(self, samples, targets):
        super(Linaige_set).__init__()
        self.samples = samples
        self.targets = targets

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.samples[idx]
        target = self.targets[idx]
        return sample, target

    def __len__(self):
        return len(self.samples)


def dataset_cross_validation(data:pd.DataFrame, windowing:int, confindence:str, remove_frame:bool, classification:bool):
    
    remove_frame_index = 8
    if windowing < 2: # sinle frame implementation
        
        # Checking for eleminating records with hard confidence
        if confindence == 'easy':
            data = data[data["confidence"]=='e'] 
        else:
            pass
        
        # Changing the target to a binary classification problem
        if classification == False:
            data.loc[:,"target"] = (data["people_number"]>1).astype(int) 
        else:
            data.loc[:,"target"] = (data["people_number"]).astype(int)
        
        # Finding how many distinct sessions exist
        session_numbers = set(data.loc[:, 'session'])
        
        '''
        # Reshaping vector(64,) of pixels to (1,8,8)
        # Storing tuple of (images, labels) of each session in sessions_image_label
        # Seperating data for each specific session 
        '''
        sessions_image_label = list()
        for sn in session_numbers:
            pixel_df = data[data["session"]==sn]
            labels = pixel_df.values[:, -1]
            images = pixel_df.values[:,2:66].reshape((pixel_df.shape[0],1,8,8)).astype(np.float32)
            sessions_image_label.append((images,labels))
        
        '''
        # Starting "cross_validation" part and removing first frames
        # 'remove_frame' can be either true or false to remove first 8 frames of test session
        # Set last index to remove first and be as test set
        '''
        # Determining the first session to be as test set (which is the last session here)
        remove_seesion = len(sessions_image_label) - 1
        
        # Loop for creating train and test sets, session 1 is always in the train set by this condition
        while remove_seesion >= 1:
            
            x_train = np.array([])
            y_train = np.array([])
                
            '''
            # Preparing train sets
            # Putting images and lebals of all sessions together,respectively.
            # sessions_image_label[i][0] stores image session of (8,8,1) and sessions_image_label[i][1] stores corresponding labels
            # At last, x_train includes all images of (8,8,1) from e.g. sessions 1 to 4, and y_train includes corresponding labels
            '''
            for i in range(len(sessions_image_label)):
                if i != remove_seesion:   
                    # To vstack an empty array with another array (n_D) we need to check the array exists and then align the deminasion for concat
                    x_train = np.vstack([x_train, sessions_image_label[i][0]]) if x_train.size else sessions_image_label[i][0]
                    # We can't use vstack for 1_D array so 'append' is used here to avoid misalignment
                    y_train = np.append(y_train, sessions_image_label[i][1])

            # class_weight = get_class_weight(y_train)
            
            if classification:
                class_weights = get_class_weight(y_train)
            else:
                class_weights = get_class_weight(y_train)[1] 
            
            '''
            # Preparing test sets
            # Remove first frames if it is needed for comparison
            '''
            if remove_frame:
                x_test = sessions_image_label[remove_seesion][0][remove_frame_index:]
                y_test = sessions_image_label[remove_seesion][1][remove_frame_index:]   
            else:
                x_test = sessions_image_label[remove_seesion][0]
                y_test = sessions_image_label[remove_seesion][1]
            
            '''Shuffling both train and test datasets'''
            
            # from sklearn.model_selection import train_test_split
            
            # x_train, y_train, _, _ = train_test_split(x_train, y_train, test_size=0, shuffle=True)
            # x_test, y_test, _, _ = train_test_split(x_test, y_test, test_size=0, shuffle=True)
            test_indices = np.arange(x_test.shape[0])
            np.random.shuffle(test_indices)

            x_test = x_test[test_indices]
            y_test = y_test[test_indices]
            
            train_indices = np.arange(x_train.shape[0])
            np.random.shuffle(train_indices)

            x_train = x_train[train_indices]
            y_train = y_train[train_indices]
            
            
            pickle_dict = {'removed_session': remove_seesion,'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
            with open('./pickle_file_single', 'wb') as f:
                 pickle.dump(pickle_dict, f, pickle.HIGHEST_PROTOCOL)
            
            yield x_train, y_train, x_test, y_test, class_weights
            remove_seesion -= 1
    
    else: # windowing implementation
        
        # Changing the target to a binary classification problem
        if classification == False:
            data.loc[:,"target"] = (data["people_number"]>1).astype(int) 
        else:
            data.loc[:,"target"] = (data["people_number"]).astype(int)
        
        # Finding how many distinct sessions exist
        session_numbers = set(data.loc[:, 'session'])
        
        '''
        # Reshaping vector(64,) of pixels to (1,8,8)
        # Storing tuple of (images, labels, confidence) of each session in sessions_image_label_confidence
        # Seperating data for each specific session 
        '''
        sessions_image_label_confidence = list()
        for sn in session_numbers:
            pixel_df = data[data["session"]==sn]
            labels = pixel_df.values[:, -1]
            images = pixel_df.values[:,2:66].reshape((pixel_df.shape[0],1,8,8)).astype(np.float32)
            conf = pixel_df.values[:, -3]
            sessions_image_label_confidence.append((images,labels, conf))
            
        '''
        # Preparing train sets
        # First, performing "cross_validation" to have train sets back2back e.g. session 1 to 4
        # Second, performing "windowing" (considering "confidence") by using 'concatenate' of 'window_size' images with dimension (1,8,8) to have one output as (window_size,8,8)  
        # In case "confindence=easy", the confidence of the last frame for each window is check whether it is 'e' or not
        # If it is 'h', we avoid to build this window and skip it to the next windowing step
        # N.B. no need to remove first frames for train set
        '''
        # Determining the first session to be as test set (which is the last session here)
        remove_seesion = len(sessions_image_label_confidence) - 1
        
        # Loop for creating train and test sets, session 1 is always in the train set by this condition
        while remove_seesion >= 1:
            x_train = np.array([])
            y_train = np.array([])
            conf_train = np.array([])
            

            # Preparing train sets
            # Putting images and lebals of all sessions together,respectively.
            # sessions_image_label[i][0] stores image session of (8,8,1) and sessions_image_label[i][1] stores corresponding labels sessions_image_label[i][2] stores corresponding conf
            # At last, x_train includes all images of (8,8,1) from e.g. sessions 1 to 4, y_train includes corresponding labels and conf_train includes corresponding conf
            
            for i in range(len(sessions_image_label_confidence)):
                if i != remove_seesion:   
                    # To vstack an empty array with another array (n_D) we need to check the array exists and then align the deminasion for concat
                    x_train = np.vstack([x_train, sessions_image_label_confidence[i][0]]) if x_train.size else sessions_image_label_confidence[i][0]
                    # We can't use vstack for 1_D array so append is used here to avoid misalignment
                    y_train = np.append(y_train, sessions_image_label_confidence[i][1])
                    conf_train = np.append(conf_train, sessions_image_label_confidence[i][2])

            # class_weight = get_class_weight(y_train)
            
            if classification:
                class_weights = get_class_weight(y_train)
            else:
                class_weights = get_class_weight(y_train)[1] 
               
            # Performing 'windowing'
            flag_easy = True if confindence == 'easy' else False
            
            '''################# Train Part#################'''
            x_train_w = np.array([])
            y_train_w_list = list()
            
            for i in range(len(x_train) - windowing + 1): # Total number of element after windowing is len() - w + 1
                
                if flag_easy and conf_train[i + windowing - 1] == 'h': # If we want just 'easy' frames and skip 'hard' ones ???? is it needed for train part????
                    pass
                else:
                    x_combined = x_train[i, :]
                    for j in range(windowing - 1): # For e.g. w=3, we need to concatenate i, i+1, i+2, so j+1 would 1 and 2 to produce i+1 and i+2
                        # Concatenating images (1,8,8) to get (windowing,8,8)
                        x_combined = np.concatenate((x_combined, x_train[i + j + 1, :]), axis=0)

                    # To convert (1,8,8) to (1,1,8,8)
                    x_combined_reshape = x_combined[np.newaxis, :]
                    # To vstack an empty array with another array (n_D) we need to check the array exists and then align the deminasion for concat to have (x_train_w.shape[0],windowing,8,8)
                    x_train_w = np.vstack([x_train_w, x_combined_reshape]) if x_train_w.size else x_combined_reshape
                    # Appending labels with size of 'windowing' as a list, then converting to numpy array
                    y_train_w_list.append(y_train[(i + windowing -1)])

            # To have numpy array of labels with shape (y_train_w.shape[0])
            y_train_w = np.array(y_train_w_list)
            
            '''################# Test Part #################''' 
            
            x_test = sessions_image_label_confidence[remove_seesion][0]
            y_test = sessions_image_label_confidence[remove_seesion][1]
            conf_test = sessions_image_label_confidence[remove_seesion][2]
            
            x_test_w = np.array([])
            y_test_w_list = list()
            
            for i in range(len(x_test) - windowing + 1): # Total number of element after windowing is len() - w + 1
                
                if flag_easy and conf_test[i + windowing - 1] == 'h': # If we want just 'easy' frames and skip 'hard' ones
                    pass
                else:
                    x_combined = x_test[i, :]
                    for j in range(windowing - 1): # For e.g. w=3, we need to concatenate i, i+1, i+2, so j+1 would 1 and 2 to produce i+1 and i+2
                        # Concatenating images (1,8,8) to get (windowing,8,8)
                        x_combined = np.concatenate((x_combined, x_test[i + j + 1, :]), axis=0)

                    # To convert (1,8,8) to (1,1,8,8)
                    x_combined_reshape = x_combined[np.newaxis, :] 
                    # To vstack an empty array with another array (n_D) we need to check the array exists and then align the deminasion for concat to have (x_train_w.shape[0],windowing,8,8)
                    x_test_w = np.vstack([x_test_w, x_combined_reshape]) if x_test_w.size else x_combined_reshape
                    # Appending labels with size of 'windowing' as a list, then converting to numpy array
                    y_test_w_list.append(y_test[(i + windowing -1)])
            
            # To have numpy array of labels with shape (y_test_w.shape[0])
            y_test_w = np.array(y_test_w_list)
            
            '''
            # Preparing test sets
            # Remove first frames if it is needed for comparison
            '''
            if remove_frame and  windowing <= remove_frame_index: # if window size is more than 8, we can not remove any thing since we need the eighth element for our first comparison
                cutting_index = remove_frame_index - windowing + 1 # By this index, we can reach the first element for comparison and remove previous ones
                x_test_w = x_test_w[cutting_index:]
                y_test_w = y_test_w[cutting_index:] 
                
            '''Shuffling both train and test datasets'''
            test_indices = np.arange(x_test_w.shape[0])
            np.random.shuffle(test_indices)

            x_test = x_test_w[test_indices]
            y_test = y_test_w[test_indices]
            
            train_indices = np.arange(x_train_w.shape[0])
            np.random.shuffle(train_indices)

            x_train = x_train_w[train_indices]
            y_train = y_train_w[train_indices]
            
            pickle_dict = {'removed_session': remove_seesion,'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
            with open('./pickle_file_window', 'wb') as f:
                pickle.dump(pickle_dict, f, pickle.HIGHEST_PROTOCOL)
            
            yield x_train, y_train, x_test, y_test, class_weights
            remove_seesion -= 1
        
            
   
def get_data(win_size:int, confindence:str, remove_frame:bool, classification:bool, data_dir:str=None) -> Tuple[Dataset, ...]:
    
    # Creating name for directory 
    if data_dir is None:
        data_dir = Path('.').absolute() / 'LINAIGE_Folder'
    
    # Creating the directory with created name before
    if not data_dir.exists():
        print('Downloading...')
        data_dir.mkdir()
        od.download(LINAIGE_URL, data_dir)
        
    data = read_files(data_dir)
    '''
    # Identifying how many distinct numbers exist for people_number column as different class
    # Optional to pass it to the model for classification task
    '''
    if classification:
      class_number = len(set(data.loc[:, 'people_number'])) 
    else:
      class_number = len(set(data.loc[:, 'people_number']>1))
    
    '''
    # data: 'DataFrame' of whole dataset
    # windowing: the size of window which can be from 1 (no windowing) to any meaningful integer number
    # confidence: can be 'easy' or 'all'
    # remove_frame: can be 'True' or 'False' to remove first 8 frames of test set for making comparison fair
    # classification: can be 'True' or 'False' to have classification or regression (binary classification)
    '''
    dataset_cv = dataset_cross_validation(data, win_size, confindence, remove_frame, classification)
    
    return dataset_cv, class_number


def build_dataloaders(datasets: Tuple[Dataset, ...], batch_size=128, num_workers=2) -> Tuple[DataLoader, DataLoader]:
    
    # Extracting datasets from get_data function
    x_train, y_train, x_test, y_test, class_weight = datasets
    

    train_set = Linaige_set(x_train, y_train)
    test_set = Linaige_set(x_test, y_test)
    
    # Feeding the dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader
