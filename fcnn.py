
# Description: simple Full Connect Neural Network(FCNN) model
# Remark: refer to "tutorials" and "how to" of tensorflow     

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


# full connected nn with dropout
def fcnn_layer(input_tensor, input_dim, output_dim, keep_prob, layer_name):
  with tf.name_scope(layer_name):
    weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1.0 / math.sqrt(float(input_dim))), name='weights')
    biases  = tf.Variable(tf.zeros([output_dim]), name='biases')
    relu    = tf.nn.relu(tf.matmul(input_tensor, weights) + biases, name='relu')
    dropout = tf.nn.dropout(relu, keep_prob, name='dropout')
    return dropout

# model
def inference(input_layer, keep_prob, input_units=2048, hidden1_units=1024, hidden2_units=512, hidden3_units=256, hidden4_units=128, output_units=2):
  hidden1 = fcnn_layer(input_layer, input_units,   hidden1_units,  keep_prob, "hidden_layer1")
  hidden2 = fcnn_layer(hidden1,     hidden1_units, hidden2_units,  keep_prob, "hidden_layer2")
  hidden3 = fcnn_layer(hidden2,     hidden2_units, hidden3_units,  keep_prob, "hidden_layer3")
  hidden4 = fcnn_layer(hidden3,     hidden3_units, hidden4_units,  keep_prob, "hidden_layer4")
  # softmax_linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([128, output_units], stddev=1.0 / math.sqrt(float(128))), name='weights')
    biases  = tf.Variable(tf.zeros([output_units]), name='biases')
    softmax = tf.nn.softmax(tf.matmul(hidden4, weights) + biases, name='softmax')
  return softmax

# loss
def loss(softmax, labels):
  # cross entropy
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1]))
  return cross_entropy

# train
def train(total_loss, global_step):
  # learning rate exponential_decay(start_learning_rate = 0.1, global_step, decay_step = 1000, decay_rate = 0.7)
  # learning rate = start_learning_rate * 0.7 ^ (global_step / 1000)
  learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.7)
  # Optimizer
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
  return train_op 

# sensitivity, specificity, matthews correlation coefficient(mcc) 
def compute_performance(label, prediction):
  assert label.shape[0] == prediction.shape[0], "label number should be equal to prediction number"
  N = label.shape[0]
  APP = sum(prediction)
  ATP = sum(label)
  TP = sum(prediction * label)
  FP = APP - TP
  FN = ATP - TP
  TN = N - TP - FP - FN
  SEN = TP / (ATP) 
  SPE = TN / (N - ATP)
  MCC = (TP * TN - FP * FN) / (math.sqrt((N - APP) * (N - ATP) * APP * ATP)) if not (N - APP) * (N - ATP) * APP * ATP == 0 else 0.0
  return TP,TN,FP,FN,SEN,SPE,MCC

