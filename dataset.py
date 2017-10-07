# prepare input data

import tensorflow as tf
import numpy
import math

class DataSet(object):
  def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=tf.float32):
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]
      if dtype == tf.float32:
        images = images.astype(numpy.float32)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, num_classes=2):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * images.shape[1]
      if self.one_hot:
        fake_label = [1] + [0] * (num_classes-1)
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes=2):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel().astype(int)] = 1
  # print(labels_one_hot)
  return labels_one_hot


def multiply_positive(data, fold=3):
  # compute positive data number
  positive_num = 0
  for l in data[:,-1]:
    if l == 1:
      positive_num += 1
  # get positive data
  positive_data = numpy.zeros((positive_num, data.shape[1]))
  index = 0
  for l in data[:]:
    if l[-1] == 1:
      positive_data[index] = l
      index += 1
  # multiply positive data
  new_data = numpy.zeros([positive_num * (fold-1) + data.shape[0], data.shape[1]])
  for i in range(fold):
    new_data[i*positive_num : (i+1)*positive_num] = positive_data
  new_data[(i+1)*positive_num :] = data

  # shulffe the data
  perm = numpy.arange(new_data.shape[0])
  numpy.random.shuffle(perm)
  new_data = new_data[perm]
  
  return new_data





