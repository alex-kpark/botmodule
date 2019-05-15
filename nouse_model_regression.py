'''
USELESS MODEL
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import itertools
import sklearn as sk
import matplotlib.pyplot as plt

def metrics_generator(classified_result, test_Y): #list, list in list
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    i = 0
    for i in range(len(test_Y)-1):
        i = i + 1
        if test_Y[i][0] == 1 and classified_result[i] == 1:
            TP = TP + 1
        if test_Y[i][0] == 1 and classified_result[i] == 0:
            FN = FN + 1
        if test_Y[i][0] == 0 and classified_result[i] == 1:
            FP = FP + 1
        if test_Y[i][0] == 0 and classified_result[i] == 0:
            TN = TN + 1
            
    metrics_dict = {}
    metrics_dict['TP'] = TP
    metrics_dict['FP'] = FP
    metrics_dict['TN'] = TN
    metrics_dict['FN'] = FN
    
    return metrics_dict

def threshold(test_predict_list, numeric):
    for i in test_predict_list:
            if i[0] > float(numeric):
                classified_result.append(1)
            else:
                classified_result.append(0)

    result = metrics_generator(classified_result, test_label)
    
    TP = result['TP']
    TN = result['TN']
    FP = result['FP']
    FN = result['FN']
    
    accuracy = float((TP+TN) / (TP+TN+FN+FP))
    precision = float(TP/(TP+FP))
    recall = float(TP/(TP+FN))
    f1_score = float((2*precision*recall)/(precision+recall))
    Wrong_to_Bot = float(FN/(TP+FN)) #봇인데 일반유저라고 판단한 경우
    Wrong_to_User = float(FP/(FP+TN)) #일반유저인데 봇이라고 판단한 경우
    
    print("Accuracy: {}".format(accuracy))
    print("RMSE: {}".format(rmse_val))
    print("Recall: {}".format(recall))
    print("Precision: {}".format(precision))
    print("F1_score: {}".format(f1_score))
    print("Wrong_to_Bot: {}".format(Wrong_to_Bot))
    print("Wrong_to_User: {}".format(Wrong_to_User))
    
    #Graphical Display
    np_classified = np.asarray(classified_result)

    plt.plot(loss_list, color='black')
    plt.figsize=(800,100)
    plt.show()

    plt.plot(test_predict, 'bo', color='red')
    plt.plot(test_label, 'bo', color='blue')
    plt.figsize=(800,100)
    plt.show()

    plt.plot(np_classified, 'bo', color='red')
    plt.plot(test_label, 'bo', color='blue')
    plt.figsize=(800,100)
    plt.show()


def lstm_regression(train_x, train_label, test_x, test_label, seq_length, data_dim, output_dim,
                    hidden_dim, batch_size, n_class, learning_rate, total_epochs, iterations):
    
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim]) #Batch_size는 0으로 줌
    Y = tf.placeholder(tf.float32, [None, 1])

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,
                                        state_is_tuple=True,
                                        activation=tf.tanh) #tanh로 hidden layer 만들고    

    outputs, _states = tf.nn.dynamic_rnn(cell,
                                        X,
                                        dtype=tf.float32)

    Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1],
                                           output_dim,
                                           activation_fn=tf.nn.relu) #activation fn은 None


    cost = tf.reduce_sum(tf.square(Y_pred - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(cost) #train 시켜야 할 가설

    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    with tf.Session() as sess:
        
        #init
        init = tf.global_variables_initializer()
        #init = tf.contrib.layers.xavier_initializer()
        sess.run(init)

        total_batch = int(len(train_x)/batch_size)

        
        for epoch in range(total_epochs):
            #iteration
            loss_list = []
            
            print('### [Epoch:{}] has started ###'.format(epoch))
            #for i in range(iterations):
            for batch_idx in range(total_batch):
                batch_xs = train_x[(batch_idx*batch_size) : (batch_idx+1)*batch_size]
                batch_ys = train_label[(batch_idx*batch_size) : (batch_idx+1)*batch_size]

                _, step_loss = sess.run([train, cost],feed_dict={
                    X : batch_xs,
                    Y : batch_ys
                })
                
                loss_list.append(step_loss)
                print('[step: {}] cost: {}'.format(batch_idx, step_loss))
                
                
        ##################Testing
        test_predict = sess.run(Y_pred, feed_dict = {X: test_x}) #예측치
            
        rmse_val = sess.run(rmse, feed_dict={
            targets : test_label, #정답
            predictions: test_predict #예측치
        })

        ##################classified_result : 1과 0으로 분류된 그래프 그리기
        classified_result = [] #결과값
        test_predict_list = test_predict.tolist() #np.array값을 list로 만들어주고

        print("Threshold: 0.5")
        threshold(test_predict_list, 0.5)
        print('\n')

        print("Threshold: 0.6")
        threshold(test_predict_list, 0.6)
        print('\n')

        print("Threshold: 0.7")
        threshold(test_predict_list, 0.7)
        print('\n')


        '''
        for i in test_predict_list:
            if i[0] > float(0.5):
                classified_result.append(1)
            else:
                classified_result.append(0)

        result = metrics_generator(classified_result, test_label)
        
        TP = result['TP']
        TN = result['TN']
        FP = result['FP']
        FN = result['FN']
        
        accuracy = float((TP+TN) / (TP+TN+FN+FP))
        precision = float(TP/(TP+FP))
        recall = float(TP/(TP+FN))
        f1_score = float((2*precision*recall)/(precision+recall))
        Wrong_to_Bot = float(FN/(TP+FN)) #봇인데 일반유저라고 판단한 경우
        Wrong_to_User = float(FP/(FP+TN)) #일반유저인데 봇이라고 판단한 경우
        
        print("Accuracy: {}".format(accuracy))
        print("RMSE: {}".format(rmse_val))
        print("Recall: {}".format(recall))
        print("Precision: {}".format(precision))
        print("F1_score: {}".format(f1_score))
        print("Wrong_to_Bot: {}".format(Wrong_to_Bot))
        print("Wrong_to_User: {}".format(Wrong_to_User))
        
        #Graphical Display
        np_classified = np.asarray(classified_result)

        plt.plot(loss_list, color='black')
        plt.figsize=(800,100)
        plt.show()

        plt.plot(test_predict, 'bo', color='red')
        plt.plot(test_label, 'bo', color='blue')
        plt.figsize=(800,100)
        plt.show()

        plt.plot(np_classified, 'bo', color='red')
        plt.plot(test_label, 'bo', color='blue')
        plt.figsize=(800,100)
        plt.show()
        '''