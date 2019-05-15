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


def lstm(train_x, train_label, test_x, test_label, seq_length, data_dim, hidden_dim, batch_size,
        n_class, learning_rate, total_epochs):
    
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim]) #배치크기 x 뉴런 수 x 입력차원(피쳐)
    Y = tf.placeholder(tf.float32, [None, n_class])

    W = tf.Variable(tf.random_normal([hidden_dim, n_class]))
    b = tf.Variable(tf.random_normal([n_class]))    

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim,
                                        state_is_tuple=True,
                                        activation=tf.tanh) #Hidden Layer의 Activation

    print("Info \n")
    print(X)
    print(Y)
    print(W)
    print(b)

    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    #outputs = tf.transpose(outputs, [1,0,2])
    #outputs = outputs[-1] #Many-to-One Model이므로 마지막 값만 사용
    #model = tf.matmul(outputs, W) + b

    y_pred = tf.contrib.layers.fully_connected(outputs[:,-1],
                                                2, #output_dim
                                                activation_fn=tf.nn.softmax)
    print(y_pred)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred,
                                                                    labels=Y))
    
    #cost = tf.reduce_sum(tf.square(y_pred - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:

        init_g = tf.global_variables_initializer()

        sess.run(init_g)

        total_batch = int(len(train_x)/batch_size)

        print("[Notice] Training Starts..." + '\n')

        for epoch in range(total_epochs):
            total_cost = 0
            batch_xs = train_x
            batch_ys = train_label

            for batch_idx in range(total_batch):
                batch_xs = train_x[(batch_idx*batch_size) : (batch_idx+1)*batch_size]
                batch_ys = train_label[(batch_idx*batch_size) : (batch_idx+1)*batch_size]

                _, cost_val = sess.run([optimizer, cost],
                                        feed_dict={
                                            X: batch_xs,
                                            Y: batch_ys
                                        })
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1),
                    'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
        
        print("Optimize Finish \n")
        
        test_xs = test_x
        test_ys = test_label
    
        prediction = tf.argmax(y_pred, 1)
        label = tf.argmax(Y, 1)
        

        print("Softmax Value ##################")
        soft = sess.run(y_pred, feed_dict={
                                X: test_xs,
                                Y: test_ys
                            })
                            
        print("Prediction Value ################")
        res = sess.run(prediction, feed_dict={
                                X: test_xs #,
                                #Y: test_ys
                        })        
        print(res)
        print('\n')

        print("Label Value ################")
        res2 = sess.run(label, feed_dict={
                                #X: test_xs,
                                Y: test_ys
                        })
        print(res2)


        '''
        is_correct = tf.equal(prediction, label)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        precision = tf.metrics.precision(label, prediction)
        recall = tf.metrics.recall(label, prediction)
        #f1_score = tf.contrib.metrics.f1_score(label, prediction) #label, prediction
        
        

        init_l = tf.local_variables_initializer()


        print("Accuracy is \n", sess.run(accuracy,
                                        feed_dict={
                                            X: test_xs,
                                            Y: test_ys
                                        }))
        sess.run(init_l) #학습하기 전에 Initialize 해주어야 함
        print("Precision is \n", sess.run(precision,
                                        feed_dict={
                                            X: test_xs,
                                            Y: test_ys
                                        }))
        sess.run(init_l)
        print("Recall is \n", sess.run(recall,
                                        feed_dict={
                                            X: test_xs,
                                            Y: test_ys
                                        }))
        
        print("F1 Score is \n", sess.run(f1_score,
                                        feed_dict={
                                            X: test_xs,
                                            Y: test_ys
                                        }))
        '''

