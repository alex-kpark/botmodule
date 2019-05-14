import preprocessor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import itertools
import sklearn as sk
import os

#feature
feature_list = [
    'total_cash',
    'cash_in_account',
    'cash_in_bank',
    'cash_in_mail',
    'cash_in_vendor',
    'evaluated_asset_value',
    'item_number',
    'total_agency_default_price',
    'total_mail_default_price',
    'asset_value_in_bank',
    'asset_value_in_account_bank',
    'account_ratio_cash', #### 터짐
    'bank_ratio_cash',
    'mail_ratio_cash',
    'vendor_ratio_cash',
    'asset_per_item',
    'asset_per_cash',
    'gap_btw_cash_asset'
]


#Parameters
seq_length = 48
num_feature = len(feature_list)

#Flatten 파일 가져오기
def flatten_to_numpy(dir_path, file):

    file_path = dir_path + file
    temp = []
    for i in range(0,864):
        temp.append(str(i))

    data = pd.read_csv(file_path, names=temp) #, names=feature_list)
    dataframe = pd.DataFrame(data)
    #print(file_path)
    #print(dataframe.shape)

    bot_dataset = []
    
    #data 개수에 맞춰서 len(dataframe.index)
    for i in range(len(dataframe.index)):        
        daily_data = dataframe.loc[i].tolist()
        np_data = np.array([np.array(daily_data).astype(np.float32).reshape(seq_length,num_feature)])
        bot_dataset.append(np_data[0])

    #list로 리턴    
    return bot_dataset

#[optional]필요없는 피쳐들 지워주기 위한 기능
def dataset_cleaner(target_np):
    axis = 1 #col
    delete_idx = [11,12,13,14,15,16,17]
    deleted_np = np.delete(target_np, delete_idx, axis)
    return deleted_np

#후에 Testing 시 필요
def metrics_generator(classification, true_label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    i = 0
    for i in range(len(classification)-1):
        i = i + 1
        #True - True
        if classification[i] == 1 and true_label[i] == 1:
            TP = TP + 1
        #False Positive
        if classification[i] == 1 and true_label[i] == 0:
            FP = FP + 1
        #False Negative
        if classification[i] == 0 and true_label[i] == 1:
            FN = FN + 1
        #True Negative
        if classification[i] == 0 and true_label[i] == 0:
            TN = TN + 1
            
    metrics_dict = {}
    metrics_dict['TP'] = TP
    metrics_dict['FP'] = FP
    metrics_dict['TN'] = TN
    metrics_dict['FN'] = FN
    
    return metrics_dict

def bot_generator(bot_dir_path, file_list):
    
    bot_list = []
    
    for csv in file_list: #files : 폴더 리스트
        try:
            managed_file = flatten_to_numpy(bot_dir_path ,csv) #np들이 들어있는 List return
            bot_list.append(managed_file)
        except Exception as e:
            print("[Error] Failed to Load")
            print(e)
            pass
    
    merged_bots = list(itertools.chain.from_iterable(bot_list))
    merged_bots = np.asarray(merged_bots)

    return merged_bots

def user_generator(user_dir_path, file_list):
    
    user_list = []

    for csv in file_list:
        try:
            managed_file = flatten_to_numpy(user_dir_path, csv)
            user_list.append(managed_file)
        
        except Exception as e:
            print("[Error] Failed to Load")
            print(e)
            pass
        
    merged_user = list(itertools.chain.from_iterable(user_list))
    merged_user = np.asarray(merged_user)

    return merged_user


