
# coding: utf-8

# # Predicting Breast Cancer Proliferation Scores with Apache Spark and Apache SystemML
#
# ## Machine Learning
# ---

# # Setup

# In[1]:

import os
import subprocess

import numpy as np
from pyspark.sql.functions import col, max
import systemml  # pip3 install systemml
from systemml import MLContext, dml


# In[2]:

ml = MLContext(sc)


# # Read in train & val data

# In[3]:

# Settings
size=256
grayscale = False
c = 1 if grayscale else 3
p = 0.01
folder = "data_new"


# In[4]:

if p < 1:
  tr_filename = os.path.join(folder, "train_{}_sample_{}{}.parquet".format(p, size, "_grayscale" if grayscale else ""))
  val_filename = os.path.join(folder, "val_{}_sample_{}{}.parquet".format(p, size, "_grayscale" if grayscale else ""))
else:
  tr_filename = os.path.join(folder, "train_{}{}.parquet".format(size, "_grayscale" if grayscale else ""))
  val_filename = os.path.join(folder, "val_{}{}.parquet".format(size, "_grayscale" if grayscale else ""))
train_df = spark.read.load(tr_filename)
val_df = spark.read.load(val_filename)
train_df, val_df


# In[5]:

tc = train_df.count()
vc = val_df.count()
tc, vc, tc + vc


# In[6]:

train_df.select(max(col("__INDEX"))).show()
train_df.groupBy("tumor_score").count().show()
val_df.groupBy("tumor_score").count().show()


# # Extract X and Y matrices

# In[7]:

# Note: Must use the row index column, or X may not
# necessarily correspond correctly to Y
X_df = train_df.select("__INDEX", "sample")
X_val_df = val_df.select("__INDEX", "sample")
y_df = train_df.select("__INDEX", "tumor_score")
y_val_df = val_df.select("__INDEX", "tumor_score")
X_df, X_val_df, y_df, y_val_df


# # Convert to SystemML Matrices
# Note: This allows for reuse of the matrices on multiple
# subsequent script invocations with only a single
# conversion.  Additionally, since the underlying RDDs
# backing the SystemML matrices are maintained, any
# caching will also be maintained.

# In[8]:

script = """
# Scale images to [-1,1]
X = X / 255
X_val = X_val / 255
X = X * 2 - 1
X_val = X_val * 2 - 1

# One-hot encode the labels
num_tumor_classes = 3
n = nrow(y)
n_val = nrow(y_val)
Y = table(seq(1, n), y, n, num_tumor_classes)
Y_val = table(seq(1, n_val), y_val, n_val, num_tumor_classes)
"""
outputs = ("X", "X_val", "Y", "Y_val")
script = dml(script).input(X=X_df, X_val=X_val_df, y=y_df, y_val=y_val_df).output(*outputs)
X, X_val, Y, Y_val = ml.execute(script).get(*outputs)
X, X_val, Y, Y_val


# # Trigger Caching (Optional)
# Note: This will take a while and is not necessary, but doing it
# once will speed up the training below. Otherwise, the cost of
# caching will be spread across the first full loop through the
# data during training.

# In[ ]:

# script = """
# # Trigger conversions and caching
# # Note: This may take a while, but will enable faster iteration later
# print(sum(X))
# print(sum(Y))
# print(sum(X_val))
# print(sum(Y_val))
# """
# script = dml(script).input(X=X, X_val=X_val, Y=Y, Y_val=Y_val)
# ml.execute(script)


# # Save Matrices (Optional)

# In[ ]:

# script = """
# write(X, "data/X_"+p+"_sample_binary", format="binary")
# write(Y, "data/Y_"+p+"_sample_binary", format="binary")
# write(X_val, "data/X_val_"+p+"_sample_binary", format="binary")
# write(Y_val, "data/Y_val_"+p+"_sample_binary", format="binary")
# """
# script = dml(script).input(X=X, X_val=X_val, Y=Y, Y_val=Y_val, p=p)
# ml.execute(script)


# ---

# # Softmax Classifier

# ## Sanity Check: Overfit Small Portion

# In[ ]:

script = """
source("breastcancer/softmax_clf.dml") as clf

# Hyperparameters & Settings
lr = 1e-2  # learning rate
mu = 0.9  # momentum
decay = 0.999  # learning rate decay constant
batch_size = 32
epochs = 500
log_interval = 1
n = 200  # sample size for overfitting sanity check

# Train
[W, b] = clf::train(X[1:n,], Y[1:n,], X[1:n,], Y[1:n,], lr, mu, decay, batch_size, epochs, log_interval)
"""
outputs = ("W", "b")
script = dml(script).input(X=X, Y=Y, X_val=X_val, Y_val=Y_val).output(*outputs)
W, b = ml.execute(script).get(*outputs)
W, b


# ## Train

# In[ ]:

script = """
source("breastcancer/softmax_clf.dml") as clf

# Hyperparameters & Settings
lr = 5e-7  # learning rate
mu = 0.5  # momentum
decay = 0.999  # learning rate decay constant
batch_size = 32
epochs = 1
log_interval = 10

# Train
[W, b] = clf::train(X, Y, X_val, Y_val, lr, mu, decay, batch_size, epochs, log_interval)
"""
outputs = ("W", "b")
script = dml(script).input(X=X, Y=Y, X_val=X_val, Y_val=Y_val).output(*outputs)
W, b = ml.execute(script).get(*outputs)
W, b


# ## Eval

# In[ ]:

script = """
source("breastcancer/softmax_clf.dml") as clf

# Eval
probs = clf::predict(X, W, b)
[loss, accuracy] = clf::eval(probs, Y)
probs_val = clf::predict(X_val, W, b)
[loss_val, accuracy_val] = clf::eval(probs_val, Y_val)
"""
outputs = ("loss", "accuracy", "loss_val", "accuracy_val")
script = dml(script).input(X=X, Y=Y, X_val=X_val, Y_val=Y_val, W=W, b=b).output(*outputs)
loss, acc, loss_val, acc_val = ml.execute(script).get(*outputs)
loss, acc, loss_val, acc_val


# ---

# # LeNet-like ConvNet

# ## Sanity Check: Overfit Small Portion

# In[ ]:

script = """
source("breastcancer/convnet.dml") as clf

# Hyperparameters & Settings
lr = 1e-2  # learning rate
mu = 0.9  # momentum
decay = 0.999  # learning rate decay constant
lambda = 0  #5e-04
batch_size = 32
epochs = 300
log_interval = 1
dir = "models/lenet-cnn/sanity/"
n = 200  # sample size for overfitting sanity check

# Train
[Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2] =
  clf::train(X[1:n,], Y[1:n,], X[1:n,], Y[1:n,], C, Hin, Win, lr, mu, decay, lambda, batch_size,
             epochs, log_interval, dir)
"""
outputs = ("Wc1", "bc1", "Wc2", "bc2", "Wc3", "bc3", "Wa1", "ba1", "Wa2", "ba2")
script = (dml(script).input(X=X, X_val=X_val, Y=Y, Y_val=Y_val,
                            C=c, Hin=size, Win=size)
                     .output(*outputs))
Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2 = ml.execute(script).get(*outputs)
Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2


# ## Hyperparameter Search

# In[ ]:

script = """
source("breastcancer/convnet.dml") as clf

dir = "models/lenet-cnn/hyperparam-search/"

# TODO: Fix `parfor` so that it can be efficiently used for hyperparameter tuning
j = 1
while(j < 2) {
#parfor(j in 1:10000, par=6) {
  # Hyperparameter Sampling & Settings
  lr = 10 ^ as.scalar(rand(rows=1, cols=1, min=-7, max=-1))  # learning rate
  mu = as.scalar(rand(rows=1, cols=1, min=0.5, max=0.9))  # momentum
  decay = as.scalar(rand(rows=1, cols=1, min=0.9, max=1))  # learning rate decay constant
  lambda = 10 ^ as.scalar(rand(rows=1, cols=1, min=-7, max=-1))  # regularization constant
  batch_size = 32
  epochs = 1
  log_interval = 10
  trial_dir = dir + "j/"

  # Train
  [Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2] = clf::train(X, Y, X_val, Y_val, C, Hin, Win, lr, mu, decay, lambda, batch_size, epochs, log_interval, trial_dir)

  # Eval
  #probs = clf::predict(X, C, Hin, Win, Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2)
  #[loss, accuracy] = clf::eval(probs, Y)
  probs_val = clf::predict(X_val, C, Hin, Win, Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2)
  [loss_val, accuracy_val] = clf::eval(probs_val, Y_val)

  # Save hyperparams
  str = "lr: " + lr + ", mu: " + mu + ", decay: " + decay + ", lambda: " + lambda + ", batch_size: " + batch_size
  name = dir + accuracy_val + "," + j  #+","+accuracy+","+j
  write(str, name)
  j = j + 1
}
"""
script = (dml(script).input(X=X, X_val=X_val, Y=Y, Y_val=Y_val, C=c, Hin=size, Win=size))
ml.execute(script)


# ## Train

# In[9]:

ml.setStatistics(True)
ml.setExplain(True)
ml.setExplainLevel("recompile_hops")


# In[ ]:

# sc.setLogLevel("OFF")


# In[ ]:

# remove previous checkpoints
subprocess.run(["hdfs", "dfs", "-rm", "-r", "models/lenet-cnn/train"])

script = """
source("breastcancer/convnet_distrib_sgd.dml") as clf

# Hyperparameters & Settings
lr = 0.00205  # learning rate
mu = 0.632  # momentum
decay = 0.99  # learning rate decay constant
lambda = 0.00385
batch_size = 32
parallel_batches = 19
epochs = 1
log_interval = 1
dir = "models/lenet-cnn/train"
#n = 50  #1216  # limit on number of samples (for debugging)
#X = X[1:n,]
#Y = Y[1:n,]
#X_val = X_val[1:n,]
#Y_val = Y_val[1:n,]

# Train
[Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2] =
    clf::train(X, Y, X_val, Y_val, C, Hin, Win, lr, mu, decay,
               lambda, batch_size, parallel_batches, epochs,
               log_interval, dir)
"""
outputs = ("Wc1", "bc1", "Wc2", "bc2", "Wc3", "bc3",
           "Wa1", "ba1", "Wa2", "ba2")
script = (dml(script).input(X=X, X_val=X_val, Y=Y, Y_val=Y_val,
                            C=c, Hin=size, Win=size)
                     .output(*outputs))
outs = ml.execute(script).get(*outputs)
Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2 = outs
Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2


# In[14]:

script = """
source("breastcancer/convnet_distrib_sgd.dml") as clf

# Hyperparameters & Settings
lr = 0.00205  # learning rate
mu = 0.632  # momentum
decay = 0.99  # learning rate decay constant
lambda = 0.00385
batch_size = 1
parallel_batches = 19
epochs = 1
log_interval = 1
dir = "models/lenet-cnn/train/"

# Dummy data
[X, Y, C, Hin, Win] = clf::generate_dummy_data(50)  #1216)
[X_val, Y_val, C, Hin, Win] = clf::generate_dummy_data(100)

# Train
[Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2] =
    clf::train(X, Y, X_val, Y_val, C, Hin, Win, lr, mu, decay,
               lambda, batch_size, parallel_batches, epochs,
               log_interval, dir)
"""
outputs = ("Wc1", "bc1", "Wc2", "bc2", "Wc3", "bc3",
           "Wa1", "ba1", "Wa2", "ba2")
script = dml(script).output(*outputs)
outs = ml.execute(script).get(*outputs)
Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2 = outs
Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2


# ## Eval

# In[ ]:

script = """
source("breastcancer/convnet_distrib_sgd.dml") as clf

# Eval
probs = clf::predict(X, C, Hin, Win, Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2)
[loss, accuracy] = clf::eval(probs, Y)
probs_val = clf::predict(X_val, C, Hin, Win, Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2)
[loss_val, accuracy_val] = clf::eval(probs_val, Y_val)
"""
outputs = ("loss", "accuracy", "loss_val", "accuracy_val")
script = (dml(script).input(X=X, X_val=X_val, Y=Y, Y_val=Y_val,
                            C=c, Hin=size, Win=size,
                            Wc1=Wc1, bc1=bc1,
                            Wc2=Wc2, bc2=bc2,
                            Wc3=Wc3, bc3=bc3,
                            Wa1=Wa1, ba1=ba1,
                            Wa2=Wa2, ba2=ba2)
                     .output(*outputs))
loss, acc, loss_val, acc_val = ml.execute(script).get(*outputs)
loss, acc, loss_val, acc_val


# In[ ]:

script = """
source("breastcancer/convnet_distrib_sgd.dml") as clf

# Dummy data
[X, Y, C, Hin, Win] = clf::generate_dummy_data(1216)
[X_val, Y_val, C, Hin, Win] = clf::generate_dummy_data(100)

# Eval
probs = clf::predict(X, C, Hin, Win, Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2)
[loss, accuracy] = clf::eval(probs, Y)
probs_val = clf::predict(X_val, C, Hin, Win, Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2)
[loss_val, accuracy_val] = clf::eval(probs_val, Y_val)
"""
outputs = ("loss", "accuracy", "loss_val", "accuracy_val")
script = (dml(script).input(Wc1=Wc1, bc1=bc1,
                            Wc2=Wc2, bc2=bc2,
                            Wc3=Wc3, bc3=bc3,
                            Wa1=Wa1, ba1=ba1,
                            Wa2=Wa2, ba2=ba2)
                     .output(*outputs))
loss, acc, loss_val, acc_val = ml.execute(script).get(*outputs)
loss, acc, loss_val, acc_val


# ---

# In[ ]:

# script = """
# N = 102400  # num examples
# C = 3  # num input channels
# Hin = 256  # input height
# Win = 256  # input width
# X = rand(rows=N, cols=C*Hin*Win, pdf="normal")
# """
# outputs = "X"
# script = dml(script).output(*outputs)
# thisX = ml.execute(script).get(*outputs)
# thisX


# In[ ]:

# script = """
# f = function(matrix[double] X) return(matrix[double] Y) {
#   if (1==1) {}
#   a = as.scalar(rand(rows=1, cols=1))
#   Y = X * a
# }
# Y = f(X)
# """
# outputs = "Y"
# script = dml(script).input(X=thisX).output(*outputs)
# thisY = ml.execute(script).get(*outputs)
# thisY


# In[ ]:



