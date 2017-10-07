
# Description: train the model, print the performance upon the train dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
from datetime import datetime

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
"""
import time
import tensorflow as tf

import os
import sys
sys.path.append('/home/xiaotaw/tools')
import fcnn

def train(compds, labels, ckpt_dir = "/tmp/train", max_step = 1000):
  """ train the model """
  with tf.Graph().as_default():

    input_placeholder = tf.placeholder(tf.float32, shape = (None, 2048))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))

    global_step = tf.Variable(0, trainable=False)

    # build a Graph that computes the softmax predictions from the
    # inference model.
    softmax = fcnn.inference(input_placeholder, 0.8)

    # compute loss.
    loss = fcnn.loss(softmax, label_placeholder)

    # compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, 1), tf.argmax(label_placeholder, 1)), tf.float32))

    # build a Graph that trains the model with a batch of examples and
    # updates the model parameters.
    train_op = fcnn.train(loss, global_step)

    # create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep = None)

    # start running operations on the Graph.
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    print("  step    TP    FN    TN    FP    SEN    SPE    ACC    MCC   min/step")
    # train with max step
    for step in xrange(max_step):
      start_time = time.time()
      _, loss_value, ACC, prediction, label_dense = sess.run(
        [train_op, loss, accuracy, tf.argmax(softmax, 1), tf.argmax(labels, 1)], 
        feed_dict = {
          input_placeholder: compds,
          label_placeholder: labels
        }
      )
      duration = float(time.time() - start_time)
      
      # compute performance
      if step % 30 ==0 or (step + 1) == max_step:
        TP,TN,FP,FN,SEN,SPE,MCC = fcnn.compute_performance(label_dense, prediction)
        
        format_str = "%6d %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %6.3f"
        print(format_str % (step, TP, FN, TN, FP, SEN, SPE, ACC, MCC, duration))      


      # save the model checkpoint periodically.
      if step % 30 == 0 or (step + 1) == max_step:
        checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

