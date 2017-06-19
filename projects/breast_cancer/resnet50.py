
# coding: utf-8

# # Imports

# In[ ]:

import math
import multiprocessing as mp
import os

import keras
import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import VarianceScaling
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, merge
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# After move to Keras 2.0 API, need to check if this can still be used.
#from preprocessing.image import ImageDataGenerator  # multiprocessing ImageDataGenerator

plt.rcParams['figure.figsize'] = (10, 10)


# # Settings

# In[ ]:

# os.environ['CUDA_VISIBLE_DEVICES'] = ""
size = 224
channels = 3
data_format = 'channels_last'  # channels_first is too slow, prob due to unnecessary conversions
classes = 3
p = 1
val_p = 0.01
num_gpus = 4
batch_size = 32 * num_gpus  # for 2 GPUs, 32/GPU has 1.2x systems speedup over 16/GPU
train_dir = "train_updated_norm_v3"
val_dir = "val_updated_norm_v3"
new_run = True
experiment_template = "resnet50-{p}%-{num_gpus}-gpu-{batch_size}-batch-size-{train_dir}-data-{val_p}%-val-NEW-ADAM"
experiment = experiment_template.format(p=int(p*100), val_p=int(val_p*100), num_gpus=num_gpus,
                                        batch_size=batch_size, train_dir=train_dir)
print(experiment)


# In[ ]:

K.set_image_data_format(data_format)
if data_format == 'channels_first':
  input_shape = (channels, size, size)
else:
  input_shape = (size, size, channels)


# # Setup experiment directory

# In[ ]:

def get_run_dir(path, new_run):
  """Create a directory for this training run."""
  os.makedirs(path, exist_ok=True)
  num_experiments = len(os.listdir(path))
  if new_run:
    run = num_experiments  # run 0, 1, 2, ...
  else:
    run = min(0, num_experiments - 1)  # continue training
  run_dir = os.path.join(path, str(run))
  os.makedirs(run_dir, exist_ok=True)
  return run_dir

def get_experiment_dir(experiment, new_run):
  """Create an experiment directory for this experiment."""
  base_dir = os.path.join("experiments", "keras", experiment)
  exp_dir = get_run_dir(base_dir, new_run)
  return exp_dir

exp_dir = get_experiment_dir(experiment, new_run=new_run)
print(exp_dir)


# # Create train & val data generators

# In[ ]:

def preprocess_input(x):
  """
  Preprocesses a tensor encoding a batch of images.

  Adapted from keras/applications/imagenet_utils.py

  # Arguments
      x: input Numpy tensor, 4D of shape (N, H, W, C).
  # Returns
      Preprocessed tensor.
  """
  # Zero-center by subtracting mean pixel value per channel
  # based on means from a 50%, evenly-distributed sample.
  # Means: updated-data norm v3, norm, no-norm original
  x[:, :, :, 0] -= 183.36777842  #189.54944625  #194.27633667
  x[:, :, :, 1] -= 138.81743141  #152.73427159  #145.3067627
  x[:, :, :, 2] -= 166.07406199  #176.89543273  #181.27861023
  x = x[:, :, :, ::-1]  # 'RGB'->'BGR'
  return x

# Multi-GPU exploitation
def split(x, num_splits):
  """Split batch into K equal-sized batches."""
  # Split tensors evenly, even if it means throwing away a few examples.
  samples = math.floor(len(x) / num_splits)
  x_splits = [arr[:samples] for arr in np.array_split(x, num_splits)]
  return x_splits

def gen_preprocessed_batch(batch_generator, num_gpus):
  """Yield preprocessed batches of x,y data."""
  for xs, ys in batch_generator:
#     yield split(preprocess_input(xs), num_gpus), split(ys, num_gpus)
    yield split(xs, num_gpus), split(ys, num_gpus)  # for tf aug experiments


# In[ ]:

K.image_data_format()


# In[ ]:

train_save_dir = "images/{stage}/{p}".format(stage=train_dir, p=p)
val_save_dir = "images/{stage}/{p}".format(stage=val_dir, p=val_p)
print(train_save_dir, val_save_dir)


# In[ ]:

# Create train & val image generators
#try:
#  # For interactive work, kill any existing pool.
#  pool.terminate()
#except:
#  pass
#pool = mp.Pool(processes=8)
#train_datagen = ImageDataGenerator(pool=pool, horizontal_flip=True, vertical_flip=True,
#                                   rotation_range=180, shear_range=0.1, fill_mode='reflect')
#val_datagen = ImageDataGenerator(pool=pool)

train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, samplewise_center=True)
                                   #rotation_range=180, shear_range=0.1, fill_mode='reflect')
val_datagen = ImageDataGenerator()
train_generator_orig = train_datagen.flow_from_directory(train_save_dir, batch_size=batch_size, target_size=(size, size))
val_generator_orig = val_datagen.flow_from_directory(val_save_dir, batch_size=batch_size, target_size=(size, size))


# In[ ]:

# Create train & val preprocessed generators
train_generator = gen_preprocessed_batch(train_generator_orig, num_gpus)
val_generator = gen_preprocessed_batch(val_generator_orig, num_gpus)


# ## Get number of batches

# In[ ]:

# Number of examples.
tc = train_generator_orig.samples
vc = val_generator_orig.samples

# Number of batches for multi-GPU exploitation.
# Note: Multi-GPU exploitation for data parallelism splits mini-batches
# into a set of micro-batches to be run in parallel on each GPU, but
# Keras will view the set of micro-batches as a single batch with
# multiple sources of inputs (i.e. Keras will view a set of examples
# being run in parallel as a single example with multiple sources of
# inputs).
train_batches = int(math.ceil(tc/batch_size))
val_batches = int(math.ceil(vc/batch_size))

# Class counts (just for information)
train_class_counts = np.bincount(train_generator_orig.classes)
val_class_counts = np.bincount(val_generator_orig.classes)

print(tc, vc)
print(train_batches, val_batches)
print(train_class_counts / np.sum(train_class_counts), val_class_counts / np.sum(val_class_counts))


# ## Generate class weights for training

# In[ ]:

class_counts = np.bincount(train_generator_orig.classes)
class_weights = dict(zip(range(classes), min(class_counts) / class_counts))
print(class_counts)
print(class_weights)


# In[ ]:

next(train_generator_orig)[0].shape


# ## Plot random images (Optional)

# In[ ]:

def show_random_image(save_dir):
  c = np.random.randint(1, 4)
  class_dir = os.path.join(save_dir, str(c))
  files = os.listdir(class_dir)
  i = np.random.randint(0, len(files))
  fname = os.path.join(class_dir, files[i])
  print(fname)
  img = Image.open(fname)
  plt.imshow(img)

# show_random_image(train_save_dir)


# In[ ]:

def plot(gen):
  r, c = 6, 6
  fig, ax = plt.subplots(r, c)
  plt.setp(ax, xticks=[], yticks=[])
  plt.tight_layout()
  x, y = next(gen)
  batch_size = x.shape[0]
  for i in range(r):
    for j in range(c):
      if i*c + j < batch_size:
        im = x[i*c + j].astype(np.uint8)
        if K.image_data_format() == 'channels_first':
          im = im.transpose(1,2,0)  # (C,H,W) -> (H,W,C)
        ax[i][j].imshow(im)
        ax[i][j].set_xlabel(y[i*c + j])

#plot(train_generator_orig)
#plot(val_generator_orig)


# # Training
# 1. Setup ResNet50 pretrained model with new input & output layers.
# 2. Train new output layers (all others frozen).
# 3. Fine tune [some subset of the] original layers.
# 4. Profit.

# ## Setup training metrics & callbacks

# In[ ]:

# Setup training metrics & callbacks
# Careful, TensorBoard callback could OOM with large validation set
# TODO: Add input images to TensorBoard output (maybe as a separate callback)
# TODO: Monitor size of input queues with callbacks
model_filename = os.path.join(exp_dir, "{val_loss:.2f}-{epoch:02d}.hdf5")
checkpointer = ModelCheckpoint(model_filename)
tensorboard = TensorBoard(log_dir=exp_dir, write_graph=False)
callbacks = [checkpointer, tensorboard]
metrics = ['accuracy'] #, fmeasure, precision, recall]


# ## Setup ResNet50 model

# In[ ]:

## Color augmentation
## TODO: Visualize this in TensorBoard with custom callback every ~100 iterations
#def preprocess(x):
#  # import these inside this function so that future model loads
#  # will not complain about `tf` not being defined
#  import tensorflow as tf
#  import keras.backend as K
#
#  def augment(img):
#    img = tf.image.random_brightness(img, max_delta=64/255)
#    img = tf.image.random_saturation(img, lower=0, upper=0.25)
#    img = tf.image.random_hue(img, max_delta=0.04)
#    img = tf.image.random_contrast(img, lower=0, upper=0.75)
#    return img
#
#  # Fix dimensions for tf.image ops
#  if K.image_data_format() == 'channels_first':
#    x = tf.transpose(x, [0,2,3,1])  # (N,C,H,W) -> (N,H,W,C)
#
#  # Augment during training.
#  x = K.in_train_phase(tf.map_fn(augment, x, swap_memory=True), x)
#
#  # Zero-center by subtracting mean pixel value per channel
#  # based on means from a 50%, evenly-distributed sample.
#  # Means: updated-data norm v3, norm, no-norm original
#  x = x - [183.36777842, 138.81743141, 166.07406199]
#  x = tf.reverse(x, axis=[-1])
#
#  if K.image_data_format() == 'channels_first':
#    x = tf.transpose(x, [0,3,1,2])  # (N,H,W,C) -> (N,C,H,W)
#  return x


# In[ ]:

K.clear_session()

# Create model by replacing classifier of ResNet50 model with new
# classifier specific to the breast cancer problem.
with tf.device("/cpu"):
  inputs = Input(shape=input_shape)
  x = inputs  # Lambda(preprocess)(inputs)
  resnet50_base = ResNet50(include_top=False, input_shape=input_shape, input_tensor=x, weights=None)
  x = Flatten()(resnet50_base.output)  # could also use GlobalAveragePooling2D since output is (None, 1, 1, 2048)
  x = Dropout(0.5)(x)
  # init Dense weights with Gaussian scaled by sqrt(1/fan_in)
  preds = Dense(classes, kernel_initializer=VarianceScaling(), activation="softmax")(x)
#   resnet50 = Model(input=resnet50_base.input, output=preds, name="resnet50")
  resnet50 = Model(inputs=inputs, outputs=preds, name="resnet50")

# Multi-GPU exploitation via a linear combination of GPU loss functions.
ins = []
outs = []
for i in range(num_gpus):
  with tf.device("/gpu:{}".format(i)):
    x = Input(shape=input_shape)  # split of batch
    out = resnet50(x)  # run split on shared model
    ins.append(x)
    outs.append(out)
model = Model(inputs=ins, outputs=outs)  # multi-GPU, data-parallel model

# Freeze all pre-trained ResNet layers.
for layer in resnet50_base.layers:
  layer.trainable = False

# Compile model.
#optim = SGD(lr=0.1, momentum=0.9, decay=0.99, nesterov=True)
#optim = keras.optimizers.RMSprop(lr=0.05)
optim = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optim, loss="categorical_crossentropy",
              loss_weights=[1/num_gpus]*num_gpus, metrics=metrics)


# In[ ]:

# Explore model
for x in model.inputs + model.outputs + model.metrics_tensors + model.targets:
  print(x.name, x.device)  # check that tensor devices exploit multi-GPU

# for i, layer in enumerate(resnet50.layers):
#   print(i, layer.name, layer.input_shape, layer.output_shape)

# print(model.summary())
#print(resnet50.summary())


# In[ ]:

# Visualize Model
# from IPython.display import SVG
# from keras.utils.visualize_util import model_to_dot
# SVG(model_to_dot(resnet50).create(prog='dot', format='svg'))


# ## Train new softmax classifier

# In[ ]:

# Dual-GPU speedup: ~1.7-1.8x
# Keras device placement improvements (metrics, losses) (no val or callbacks, full model):
#   batch_size=32,  2 gpus, 100 iters, no keras changes: 128s, 108s, 107s
#   batch_size=32,  2 gpus, 100 iters, w/ keras changes: 94s, 75s, 75s
#   batch_size=32,  1 gpu,  100 iters, w/ keras changes: 148s, 133s, 133s
#   batch_size=64,  2 gpus,  50 iters, w/ keras changes: 93s, 74s, 75s
#   batch_size=128, 2 gpus,  25 iters, w/ keras changes: 90s, 73s, 74s
epochs = 2
hist1 = model.fit_generator(train_generator, steps_per_epoch=train_batches,
                            validation_data=val_generator, validation_steps=val_batches,
                            epochs=epochs, class_weight=class_weights, callbacks=callbacks) #,
                            #max_q_size=8, nb_worker=1, pickle_safe=False)


# ## Fine-tune model

# Explore model
# for x in model.inputs + model.outputs + model.metrics_tensors + model.targets:
#   print(x.name, x.device)  # check that tensor devices exploit multi-GPU

for i, layer in enumerate(resnet50_base.layers):
  print(i, layer.name, layer.input_shape, layer.output_shape)

print(model.summary())
#print(model.get_layer("resnet50").summary())


# In[ ]:

# Unfreeze some subset of the model and fine-tune by training slowly with low lr.
for layer in resnet50_base.layers[154:]:  #[164:]:  # unfreeze final 2 residual blocks + exit flow ([154:])
  layer.trainable = True
#   if hasattr(layer, 'W_regularizer'):
#     layer.W_regularizer = l2(1e-4)

#optim = SGD(lr=0.0001, momentum=0.9)
optim = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=optim, loss="categorical_crossentropy",
              loss_weights=[1/num_gpus]*num_gpus, metrics=metrics)

# In[ ]:

# model.load_weights(os.path.join("experiments/keras/resnet50-100%-2-gpu-64-batch-size/0", "5.08-08.hdf5"))


# In[ ]:

initial_epoch = epochs
epochs = initial_epoch + 20
hist2 = model.fit_generator(train_generator, steps_per_epoch=train_batches,
                            validation_data=val_generator, validation_steps=val_batches,
                            epochs=epochs, initial_epoch=initial_epoch,
                            class_weight=class_weights, callbacks=callbacks) #,
                            #max_q_size=8, nb_worker=1, pickle_safe=False)

# Unfreeze some subset of the model and fine-tune by training slowly with low lr.
for layer in resnet50_base.layers[141:]:  #[164:]:  # unfreeze final 2 residual blocks + exit flow ([154:])
  layer.trainable = True
#   if hasattr(layer, 'W_regularizer'):
#     layer.W_regularizer = l2(1e-4)

#optim = SGD(lr=0.0001, momentum=0.9)
optim = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=optim, loss="categorical_crossentropy",
              loss_weights=[1/num_gpus]*num_gpus, metrics=metrics)

# In[ ]:

# model.load_weights(os.path.join("experiments/keras/resnet50-100%-2-gpu-64-batch-size/0", "5.08-08.hdf5"))


# In[ ]:

initial_epoch = epochs
epochs = initial_epoch + 20
hist2 = model.fit_generator(train_generator, steps_per_epoch=train_batches,
                            validation_data=val_generator, validation_steps=val_batches,
                            epochs=epochs, initial_epoch=initial_epoch,
                            class_weight=class_weights, callbacks=callbacks) #,
                            #max_q_size=8, nb_worker=1, pickle_safe=False)


# ## Evaluate model on training set

# In[ ]:

raw_metrics = model.evaluate_generator(train_generator, steps=val_batches) #,
                                       #max_q_size=8, nb_worker=1, pickle_safe=False)
labeled_metrics = list(zip(model.metrics_names, raw_metrics))
losses = [v for k,v in labeled_metrics if k == "loss"]
accuracies = [v for k,v in labeled_metrics if k.endswith("acc")]
loss = sum(losses) / num_gpus
acc = sum(accuracies) / num_gpus
final_metrics = {"loss": loss, "acc": acc}
print(labeled_metrics)
print(final_metrics)

# ## Evaluate model on validation set

# In[ ]:

raw_metrics = model.evaluate_generator(val_generator, steps=val_batches) #,
                                       #max_q_size=8, nb_worker=1, pickle_safe=False)
labeled_metrics = list(zip(model.metrics_names, raw_metrics))
losses = [v for k,v in labeled_metrics if k == "loss"]
accuracies = [v for k,v in labeled_metrics if k.endswith("acc")]
loss = sum(losses) / num_gpus
acc = sum(accuracies) / num_gpus
final_metrics = {"loss": loss, "acc": acc}
print(labeled_metrics)
print(final_metrics)


# ## Save model

# In[ ]:

filename = "{acc:.5}_acc_{loss:.5}_loss_model.hdf5".format(**final_metrics)
fullpath = os.path.join(exp_dir, filename)
model.save(fullpath)
print("Saved model file to {}".format(fullpath))


# # Cleanup

# In[ ]:

# # Stop processes cleanly.  Otherwise, zombie processes will
# # persist and hold onto GPU memory.
# try:
#     pool.terminate()
# except:
#     pass
# for p in mp.active_children():
#   p.terminate()
# mp.active_children()

