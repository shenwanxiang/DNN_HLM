import os
import numpy
import train
import dataset

# get data set from file
data = numpy.genfromtxt("../data/HLM_ECFP4_CODE", dtype = 'float32')

# split data into train and cross_validation
cv_set_num = data.shape[0] / 10
train_set = data[cv_set_num:]

# create tmp checkpoint dir
if not os.path.exists("./tmp"):
  os.mkdir("./tmp")

# split train data into compd and label
train_compd = train_set[:,:-1]
train_label_dense = train_set[:,-1]
train_label = dataset.dense_to_one_hot(train_label_dense)

# train fcnn model
train.train(train_compd, train_label, ckpt_dir = "./tmp", max_step = 30000)

