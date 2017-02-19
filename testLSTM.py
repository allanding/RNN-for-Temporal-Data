# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:50:45 2016

@author: ading
"""
import SupportDoc as sd
import random as rp
import tensorflow as tf
from sklearn.metrics import roc_auc_score


class LoadData(object):
    
    def __init__(self, X, y, rlist):
        self.data = X[rlist,:,:]
        self.labels = y[rlist,:]
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])

        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
        
"""
The main function here
"""   
## reading data and split to training & testing sets
X, y = sd.preprocess()

#rp.seed(0)
m, d1, d2 = X.shape
ran_a = range(m)
ran_b = rp.sample(ran_a, m)

trainset = LoadData(X = X, y = y, rlist = ran_b[1:m/5*4])

testset = LoadData(X = X, y = y, rlist = ran_b[1+m/5*4:m])

Xtr = trainset.data
ytr = trainset.labels


Xts = testset.data
yts = testset.labels


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 256
display_step = 10

# Network Parameters
n_input = 6 
n_steps = 57 
n_hidden = 64 # hidden layer num of features
n_classes = 2 # total classes

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        ## use the whole training data
        ## I will use batch instead soon
        #batch_x, batch_y = Xtr, ytr
        batch_x, batch_y = trainset.next(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for testing set

    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: Xts, y: yts})
    
    # Calculate the predicted labels for Xts
    predic = sess.run(pred, feed_dict={x: Xts})

    # Calculate the probabilty 
    prob = sd.CalProb(pLabel = predic)

    # plot the AUC value
    print "AUC Score:", \
        roc_auc_score(yts[:,1],prob)
