#!/usr/bin/python

# Description: Extract molecular fingerprint from .sd or .sdf file

import numpy
import math
import sys, getopt

def extractMoFp(filename, fp_name, label_name = False, cluster_name = False):
  """
  # extractMoFp: extract molecular fingerprint from .sd or .sdf file
  # filename: input file should be a .sdf or .sd file
  # fp: usually "ECFP_4", "ECFP_6", etc
  # label_name: label column, usually "label", "Label", "LABEL"
  # cluster_name: cluster column, usually "cluster", "Cluster", "CLUSTER"
  """
  print("extract " + fp_name + " from " + filename + ", please wait...")
  
  FLAG_fp = False              # fingerprint flag
  if label_name:
    FLAG_label = False         # label flag
  if cluster_name:
    FLAG_cluster = False       # cluster flag

  fp_list = []                 # tmp fingerprint list of each molecule
  molecule = {}                # tmp molecule, include fp, label and cluster(if exsist)
  molecule_list = []           # all molecule 

  infile = open(filename, 'r') 
  for line in infile:
    """
    # pseudo-code for extract something
    # if '> <' + something_name + '>' in line:
    #   FLAG_something = True
    #   continue 
    # if FLAG_something:
    #   read something
    #   if read finished:
    #     continue
    """
    # extract molecular fingerprint
    if '> <' + fp_name + '>' in line:
      FLAG_fp = True
      continue
    if FLAG_fp:
      if line == '\n':
        FLAG_fp = False
        continue
      fp_list.append(line.split()[0])
    # extract label
    if label_name:             
      if '> <' + label_name + '>' in line:
        FLAG_label = True
        continue
      if FLAG_label:
        label = line.split()[0]
        FLAG_label = False
        continue
    # extract cluster
    if cluster_name:
      if '> <' + cluster_name + '>' in line:
        FLAG_cluster = True
        continue
      if FLAG_cluster:
        cluster = line.split()[0]
        FLAG_cluster = False
        continue
    # '$$$$' marks the end of a molecule
    if '$$$$' in line:
      molecule["fp"] = fp_list
      if label_name:
        molecule["label"] = label
      if cluster_name:
        molecule["cluster"] = cluster
      molecule_list.append(molecule)
      # reset fp_list
      fp_list = []
      molecule = {}
  
  infile.close()
  print("    length of mole_list: %d" % len(molecule_list))
  print("    finished")
  return molecule_list


def encode(molecule_list, fp_list = False, label = False):
  """
  # encode: generate code from molecule fingerprint
  # molecule_list: a list of molecule, which is a dict {"fp": fp_list, ...}
  # fp_list: code is generated according to fp_list. if fp_list is not given,
  #          it will be generated from all the fingerprint appeared in the molecule_list
  # label: default is False, then the label will be get from the molecule_list
  #        if label is given as a single integer, then all the molecule will be labeled with the same given label
  #        if label is given as a list, then each molecule will be labeled with the given label accroding to their order. 
  """
  print("generate code(with label) according to fingerprint list, please wait...")

  # generate fp_list if necessary
  fp_list_generated = list()
  fp_set = set()
  if not fp_list:
    for mole in molecule_list:
      fp_set.update(mole["fp"])
    fp_list_generated = list(fp_set)
  else:
    fp_set = set(fp_list)
    assert len(fp_list) == len(fp_set), "fp_list should be a set."
    fp_list_generated = list(fp_list)

  fp_dict = dict()
  for i in xrange(len(fp_list_generated)):
    fp_dict[fp_list_generated[i]] = i

  # generate labels_code if necessary
  labels_code = numpy.zeros(len(molecule_list), dtype = 'int32')
  if label == False:
    for i in range(len(molecule_list)):
      labels_code[i] = molecule_list[i]["label"]
  else:
    labels_code = labels_code + label
    
  # generate code
  fp_code = numpy.zeros([len(molecule_list), len(fp_list_generated)], dtype = 'int32')
  for i in range(len(molecule_list)):
    for fp in molecule_list[i]["fp"]:
      fp_code[i][fp_dict[fp]] = 1
  print("    fp_code.shape: " + str(fp_code.shape))
  print("    finished")
  return (fp_code, labels_code) if fp_list else (fp_code, labels_code, fp_list_generated)

# fold the code
def fold(code, out_len=2048):
  """ 
  # code: the fingerprint code to be folded. 
  #       A ndarray with shape = (molecule_number, fingerprint_code_length) 
  # out_len: the target code length, default is 2048
  """
  print("fold the code into a length of %d, please wait..." % out_len)

  # compute fold times
  fold_times = int(math.ceil( code.shape[1] / float(out_len)))

  # extend code length to the multiple of out_len
  code_extend = numpy.zeros([code.shape[0], fold_times * out_len])
  code_extend[:,:code.shape[1]] = code[:,:]

  # fold the extended code
  code_out = code_extend.reshape(code.shape[0], fold_times, out_len).sum(axis = 1)

  # use ndarray.clip to replace any number greater than 1 with 1
  code_out.clip(0, 1, out = code_out)

  print("    code.shape: " + str(code_out.shape))
  print("    finished")
  return code_out


if __name__ == '__main__':

  # default value
  work_dir = "/home/xiaotaw/tools/test/"
  input_filename = work_dir + "fingerprint_test.sdf"
  output_filename = work_dir + "fingerprint_test.out"
  fp_list_out_filename = work_dir + "fingerprint_list_test.out"
  fp_name = "ECFP_6"
  label_name = "label"
  cluster_name = False

  # get options argvs from command line
  opts, args = getopt.getopt(sys.argv[1:], "i:f:l:c:o:", ["fp_list="])
  for op, value in opts:
    if op == "-i":
      input_filename = value
    elif op == "-f":
      fp_name = value
    elif op == "-l":
      label_name = value
    elif op == "-c":
      cluster_name = value
    elif op == "-o":
      output_filename = value
    elif op == "--fp_list":
      fp_list_out_filename = value

  # extract molecular fingerprint
  molecule_list = extractMoFp(input_filename, fp_name, label_name, cluster_name)
  
  # encode fingerprint code and label code
  fp_code, labels_code, fp_list = encode(molecule_list)

  # fold into a length of 2048
  fp_code_2048 = fold(fp_code)


  print("write fp_code and labels_code into file: " + output_filename + ", please wait...")
  # merge fp_code with labels_code
  fp_label = numpy.hstack([fp_code_2048, numpy.array([labels_code]).T])

  # shuffle the data
  perm = numpy.arange(fp_label.shape[0])
  numpy.random.shuffle(perm)
  fp_label = fp_label[perm]

  # print to a file
  outfile = open(output_filename,'w')
  for i in xrange(fp_code_2048.shape[0]):
    fp_label[i].tofile(outfile, sep = "\t", format = "%d")
    outfile.write('\n')
  outfile.close()
  print("    finished.")

  # print fp_list to a file
  fp_file = open(fp_list_out_filename, 'w')
  for fp in fp_list:
    fp_file.write("%s\n" % fp)
  fp_file.close()

