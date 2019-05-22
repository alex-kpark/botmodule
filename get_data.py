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
    'total_cash', #0
    'cash_in_account', #1
    'cash_in_bank', #2
    'cash_in_mail', #3
    'cash_in_vendor', #4
    'evaluated_asset_value', #5
    'item_number', #6
    'total_agency_default_price', #7
    'total_mail_default_price', #8
    'asset_value_in_bank', #9
    'asset_value_in_account_bank', #10
    'account_ratio_cash', #11 터짐
    'bank_ratio_cash', #12
    'mail_ratio_cash', #13
    'vendor_ratio_cash', #14
    'asset_per_item', #15
    'asset_per_cash', #16
    'gap_btw_cash_asset' #17
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
    delete_idx = [3,2,7,10,11,12,13,14,15,16,17] #2,3일 때 0.76
    deleted_np = np.delete(target_np, delete_idx, axis)

    
    #Stat add modeul
    mean = np.mean(deleted_np, axis=0)
    std = np.std(deleted_np, axis=0)
    median = np.median(deleted_np, axis=0)
    
    deleted_np = np.vstack((deleted_np, mean))
    deleted_np = np.vstack((deleted_np, std))
    deleted_np = np.vstack((deleted_np, median))
    
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
            managed_file = flatten_to_numpy(bot_dir_path, csv) #np들이 들어있는 List return
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


