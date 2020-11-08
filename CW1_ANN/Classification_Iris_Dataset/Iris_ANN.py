import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
# %matplotlib inline
# %tensorflow_version 1.x

import tensorflow as tf
print(tf.__version__)

#Function Definition
def add_layer(inputs, in_size, out_size, activation_function=None):
  Weights = tf.Variable(tf.random_normal([in_size, out_size])) 
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) 
  Wx_plus_b = tf.matmul(inputs, Weights) + biases
  if activation_function is None:
    outputs = Wx_plus_b
  else:
    outputs = activation_function(Wx_plus_b)
  return outputs

# 1、Load Data and Do Preprocessing
dataset = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Machine Learning/Iris_Dataset.csv')
dataset = pd.get_dummies(dataset, columns=['Species']) # One Hot Encoding
values = list(dataset.columns.values) # Get the column names into a list

y = dataset[values[-3:]]
y = np.array(y, dtype='float32') # Get labels

X = dataset[values[1:-3]]
X = np.array(X, dtype='float32') # Get features

# Shuffle Data
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]

# Create A Training Set, A Validation Set And A Testing Set
test_size = 30
validation_size = 20

X_train = X_values[: -(test_size + validation_size)]
y_train = y_values[: -(test_size + validation_size)]

X_validation = X_values[-(test_size + validation_size) : -test_size]
y_validation = y_values[-(test_size + validation_size) : -test_size]

X_test = X_values[-test_size:]
y_test = y_values[-test_size:]

# 2. Build Artificial Neural Network

#Parameter Settings
interval = 50
epoch = 1000
hidden_layer_nodes = 8

# Initialize Input Layer (placeholders)
xs = tf.placeholder(shape=[None, 4], dtype=tf.float32)
ys = tf.placeholder(shape=[None, 3], dtype=tf.float32)

# Create Hidden Layers
l1 = add_layer(xs, 4, 8, activation_function=tf.nn.relu)

# Create Output Layer
prediction = add_layer(l1, 8, 3, activation_function=tf.nn.softmax)

# Cost Function
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction + 1e-10), axis=0)) #1e-10 is used to avoid NaN

# Optimizer & Train Step
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# Global Variables Initializer
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

### Do Some Testing BEFORE Training The ANN ###
sess.run(prediction, feed_dict={xs:X_test})
pred = tf.argmax(prediction, axis=1)
sess.run(pred, feed_dict={xs:X_test})

# 3. Train Artificial Neural Network
for i in range(epoch+1):
  # training
  sess.run(train_step, feed_dict={xs:X_train, ys: y_train})
  if i % interval == 0:
    # to see the improvement
    pred = tf.argmax(prediction, axis=1)
    actual = tf.argmax(y_validation, axis=1)
    correct = tf.cast(tf.equal(pred, actual), dtype=tf.int32)
    correct = tf.reduce_sum(correct)
    correctNum = sess.run(correct, feed_dict={xs:X_validation})
    accuracy = float(correctNum) / float(validation_size)
    print("Epoch %d, loss: %f, evaluation accuracy: %f." % (i, sess.run(loss, feed_dict={xs: X_train, ys: y_train}), accuracy))

# 4. Testing
pred = tf.argmax(prediction, axis=1)
actual = tf.argmax(y_test, axis=1)
correct = tf.cast(tf.equal(pred, actual), dtype=tf.int32)
correct = tf.reduce_sum(correct)
correctNum = sess.run(correct, feed_dict={xs:X_test})
accuracy = float(correctNum) / float(test_size)
print("The accuracy of testing set is %f" % accuracy)