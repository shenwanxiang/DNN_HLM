import os
import numpy
import evaluate
import dataset

# get data set from file
data = numpy.genfromtxt("../data/HLM_ECFP4_CODE", dtype = 'float32')

# split data into train and cross_validation
cv_set_num = data.shape[0] / 10
cv_set = data[:cv_set_num]

# create tmp checkpoint dir
if not os.path.exists("./tmp"):
  os.mkdir("./tmp")

# split cross_validation data into compd and label
cv_compd = cv_set[:,:-1]
cv_label_dense = cv_set[:,-1]
cv_label = dataset.dense_to_one_hot(cv_label_dense)

# evaluate fcnn model
evaluate.evaluate(cv_compd, cv_label, ckpt_dir = "./tmp")
