import numpy as np
import tensorflow as tf
import math
import logging
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/lirunzhuo/Desktop/cw/iris.data', header=None,names=["a","b","c","d","e"])
df = df.join(pd.get_dummies(df.e))

features=np.array(df[["a","b","c","d"]])
labels=np.array(df[["Iris-setosa","Iris-versicolor","Iris-virginica"]])


features_standard = preprocessing.StandardScaler().fit_transform(features)

features_train, features_test, lables_train, labels_test=train_test_split(features_standard,labels,test_size=0.33)

features_standard.shape

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 4
n_output = 3

#Learning parameters
learning_constant = 10
number_epochs = 1000
batch_size = 1000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES

#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))

#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))

#Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))

#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))

#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))

#Weights connecting second layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))


def multilayer_perceptron(input_d):
    # Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    # Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    # Task of neurons of output layer
    out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_2, w3), b3))

    return out_layer


# Create model
neural_network = multilayer_perceptron(X)

# Define loss an d optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))

# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(neural_network, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # Training epoch
    for epoch in range(number_epochs):
        _, acc_train = sess.run([optimizer, accuracy], feed_dict={X: features_train, Y: lables_train})
        # Display the epoch
        if epoch % 10 == 0:
            acc_test = sess.run(accuracy, feed_dict={X: features_test, Y: labels_test})
            print("Epoch %d,trainning accuracy %f,testing accarycy %f" % (epoch, acc_train, acc_test))



#tf.keras.estimator

