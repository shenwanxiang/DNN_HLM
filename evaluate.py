
# Description: evaluate the model, print the performance upon the test dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import os
"""
import sys
sys.path.append('/home/xiaotaw/tools')
"""
import fcnn

def evaluate(compds, labels, ckpt_dir = "/tmp/train"):
  """ evaluate the model """
  with tf.Graph().as_default():

    input_placeholder = tf.placeholder(tf.float32, shape = (None, 2048))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))

    # build a Graph that computes the softmax predictions from the
    # inference model.
    softmax = fcnn.inference(input_placeholder, 1.0)

    # compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, 1), tf.argmax(label_placeholder, 1)), tf.float32))

    # create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Start running operations on the Graph.
    with tf.Session() as sess:
      """
      # Restores variables from checkpoint
      ckpt = tf.train.get_checkpoint_state(ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/tmp/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
      """
      # get checkpoint file list from file named with "checkpoint" 
      infile = open(ckpt_dir + "/checkpoint", 'r')
      ckpt_list = list()
      for line in infile:
        ckpt_list.append(ckpt_dir + "/" + line.split()[1][1:-1])

      print("  step    TP    FN    TN    FP    SEN    SPE    ACC    MCC")
      # for each checkpoint file, restore the variables, and evaluate the model 
      for i in range(1,len(ckpt_list)):
        saver.restore(sess, ckpt_list[i])
        global_step = ckpt_list[i].split('/')[-1].split('-')[-1]

        # evaluate the model
        ACC, prediction, label_dense = sess.run(
          [accuracy, tf.argmax(softmax, 1), tf.argmax(labels, 1)], 
          feed_dict = {
            input_placeholder: compds,
            label_placeholder: labels})

        # compute performance
        TP,TN,FP,FN,SEN,SPE,MCC = fcnn.compute_performance(label_dense, prediction)
        format_str = "%6d %5d %5d %5d %5d %5.3f %5.3f %5.3f %5.3f"
        print(format_str % (int(global_step), TP, FN, TN, FP, SEN, SPE, ACC, MCC))  





