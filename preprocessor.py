import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
import math
import csv

#String으로 된 시간 문자열을 Datetime으로 변환해주는 역할
def string_to_datetime(target_df):
    log_time = target_df['log_time'] #log_time 열을 잡고
    target_df['processed_log_time'] = 0 #processed_log_time이라는 열에 값을 0을 넣어서 추가
    
    #for문 돌면서 str을 time data로 변환
    i = 0
    for str_log in log_time:
        deleted_log = str_log[:-4] #뒤 초단위는 모두 자르고
        timized_data = datetime.strptime(deleted_log, '%Y-%m-%d %H:%M:%S') #datetime으로 바꾸고
        target_df['processed_log_time'][i] = timized_data
        print(target_df['processed_log_time'][i])
        i = i + 1
        
    return target_df