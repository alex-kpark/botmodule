import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import itertools
import sklearn as sk
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import get_data
import model

from collections import Counter

#전처리한 데이터들 놓기
bot_dir_path = 'D:/AION_DATA/Featured_bot_Dataset/' 
bot_file_list = os.listdir(bot_dir_path)

user_dir_path = 'D:/AION_DATA/Featured_user_Dataset/'
user_file_list = os.listdir(user_dir_path)

#hidden_dim : 80이 SOTA

#하이퍼파라미터

#output_dim = 1
#iterations = 1000

seq_length = 51 #뉴런의 개수 + 통계값 수
data_dim = 8 #피쳐 
n_class = 2 #Binary Classification
total_epochs = 10000

hidden_dim = 30 #hidden layer의 개수 #80
learning_rate = 0.01
batch_size = 11000 #Batch 없이 하는게 효율적이지만, GPU 쓸 때 메모리 터짐 #현재 70


'''
데이터 불러오기
'''
bots = get_data.bot_generator(bot_dir_path, bot_file_list) 
users = get_data.user_generator(user_dir_path, user_file_list)

print(len(bots))
print(len(users))