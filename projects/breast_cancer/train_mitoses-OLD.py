"""Training - mitosis detection"""
import argparse
from datetime import datetime
import math
import os

import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import VarianceScaling
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf


class MyTensorBoard(TensorBoard):
  """Better Tensorboard basic visualizations.

  # Arguments
      log_dir: the path of the directory where to save the log
          files to be parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation
          and weight histograms for the layers of the model. If set to 0,
          histograms won't be computed. Validation data (or split) must be
          specified for histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard.
          The log file can become quite large when
          write_graph is set to True.
      write_grads: whether to visualize gradient histograms in TensorBoard.
          `histogram_freq` must be greater than 0.
      batch_size: size of batch of inputs to feed to the network
          for histograms computation.
      write_images: whether to write model weights to visualize as
          image in TensorBoard.
      embeddings_freq: frequency (in epochs) at which selected embedding
          layers will be saved.
      embeddings_layer_names: a list of names of layers to keep eye on. If
          None or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name
          in which metadata for this embedding layer is saved. See the
          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
          about metadata files format. In case if the same metadata file is
          used for all embedding layers, string can be passed.
  """

  def __init__(self, log_dir='./logs',
               histogram_freq=0,
               batch_size=32,
               write_graph=True,
               write_grads=False,
               write_images=False,
               embeddings_freq=0,
               embeddings_layer_names=None,
               embeddings_metadata=None):
      super(MyTensorBoard, self).__init__(log_dir, histogram_freq, batch_size, write_graph,
          write_grads, write_images, embeddings_freq, embeddings_layer_names, embeddings_metadata)

      # KEY CHANGE:
      # use train and val writers so that plots can be on same graph
      self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
      self.val_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "val"))


  def on_epoch_end(self, epoch, logs=None):
      # basically keep everything here the same except for adding the train & val writers
      logs = logs or {}

      if not self.validation_data and self.histogram_freq:
          raise ValueError("If printing histograms, validation_data must be "
                           "provided, and cannot be a generator.")
      if self.validation_data and self.histogram_freq:
          if epoch % self.histogram_freq == 0:

              val_data = self.validation_data
              tensors = (self.model.inputs +
                         self.model.targets +
                         self.model.sample_weights)

              if self.model.uses_learning_phase:
                  tensors += [K.learning_phase()]

              assert len(val_data) == len(tensors)
              val_size = val_data[0].shape[0]
              i = 0
              while i < val_size:
                  step = min(self.batch_size, val_size - i)
                  if self.model.uses_learning_phase:
                      # do not slice the learning phase
                      batch_val = [x[i:i + step] for x in val_data[:-1]]
                      batch_val.append(val_data[-1])
                  else:
                      batch_val = [x[i:i + step] for x in val_data]
                  assert len(batch_val) == len(tensors)
                  feed_dict = dict(zip(tensors, batch_val))
                  result = self.sess.run([self.merged], feed_dict=feed_dict)
                  summary_str = result[0]
                  self.writer.add_summary(summary_str, epoch)
                  i += self.batch_size

      if self.embeddings_freq and self.embeddings_ckpt_path:
          if epoch % self.embeddings_freq == 0:
              self.saver.save(self.sess,
                              self.embeddings_ckpt_path,
                              epoch)

      for name, value in logs.items():
          if name in ['batch', 'size']:
              continue
          summary = tf.Summary()
          summary_value = summary.value.add()
          summary_value.simple_value = value.item()
          # KEY CHANGE:
          # use train and val writers so that plots can be on same graph
          if name.startswith("val"):
            name = name[4:]  # strip 'val_' from the front of the name
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
          else:
            summary_value.tag = name
            self.train_writer.add_summary(summary, epoch)
      self.writer.flush()
      self.train_writer.flush()
      self.val_writer.flush()

  def on_train_end(self, _):
      self.writer.close()
      self.train_writer.close()
      self.val_writer.close()


def train(train_path, val_path, exp_path, batch_size, patch_size, clf_epochs, finetune_epochs):
  """Train a model.

  Args:
    train_path: String path to the generated training image patches.
      This should contain folders for each class.
    val_path: String path to the generated validation image patches.
      This should contain folders for each class.
    exp_path: String path in which to store the model checkpoints, logs,
      etc. for this experiment
    batch_size: Integer batch size.
    patch_size: Integer length to which the square patches will be
      resized.
    clf_epochs: Integer number of epochs for which to training the new
      classifier layers.
    clf_epochs: Integer number of epochs for which to fine-tune the
      model.
  """
  # TODO: break this out into:
  #   * data gen func
  #   * inference func
  #     * model creation, loss, and compilation
  #   * train func

  # data
  # TODO: inject the multi-GPU datagen code from the larger example if needed
  train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
      samplewise_center=True, samplewise_std_normalization=True) #preprocessing_function=preprocess_input)
  val_datagen = ImageDataGenerator(
      samplewise_center=True, samplewise_std_normalization=True) #preprocessing_function=preprocess_input)
  train_generator = train_datagen.flow_from_directory(train_path, batch_size=batch_size,
      target_size=(patch_size, patch_size), class_mode="binary")
  val_generator = val_datagen.flow_from_directory(val_path, batch_size=batch_size,
      target_size=(patch_size, patch_size), class_mode="binary")

  input_shape = (patch_size, patch_size, 3)

  # number of examples
  tc = train_generator.samples
  vc = val_generator.samples

  # number of batches in one epoch [for multi-GPU exploitation]
  # Note: Multi-GPU exploitation for data parallelism splits mini-batches
  # into a set of micro-batches to be run in parallel on each GPU, but
  # Keras will view the set of micro-batches as a single batch with
  # multiple sources of inputs (i.e. Keras will view a set of examples
  # being run in parallel as a single example with multiple sources of
  # inputs).
  train_batches = int(math.ceil(tc/batch_size))
  val_batches = int(math.ceil(vc/batch_size))

  # generate class weights for training
  class_counts = np.bincount(train_generator.classes)
  num_classes = len(class_counts)
  class_weights = dict(zip(range(num_classes), min(class_counts) / class_counts))

  # create in-memory validation set:
  val_x = []
  val_y = []
  for _ in range(val_batches):
    batch_x, batch_y = next(val_generator)
    val_x.extend(batch_x)
    val_y.extend(batch_y)
  #val_data = [next(val_generator) for _ in range(val_batches)]
  val_data = (np.array(val_x), np.array(val_y))


  # Setup training metrics & callbacks
  # Careful, TensorBoard callback could OOM with large validation set
  # TODO: Add input images to TensorBoard output (maybe as a separate callback)
  # TODO: Monitor size of input queues with callbacks
  model_filename = os.path.join(exp_path, "{val_loss:.2f}-{epoch:02d}.hdf5")
  checkpointer = ModelCheckpoint(model_filename)
  tensorboard = TensorBoard(log_dir=exp_path, write_graph=True)
  #tensorboard = TensorBoard(log_dir=exp_path, histogram_freq=1, write_graph=True,
  #    write_grads=True, write_images=True)
  tensorboard = MyTensorBoard(log_dir=exp_path, histogram_freq=1, write_graph=True,
      write_grads=True, write_images=True)
  callbacks = [checkpointer, tensorboard]
  # NOTE: metrics are pretty awful in Keras.  instead, we could create a custom callback that
  # calls `preds = model.predict_generator(val_generator, steps=val_batches)` for train and val
  # datasets and then plot them with TensorBoard.  it will take a bit of extra time, but it will be
  # much more accurate, and we could plot any number of items.  then, we could use the default
  # TensorBoard callback, and set the below metrics to an empty list, and then write to TensorBoard
  # within our custom callback.
  metrics = ['accuracy'] #, fmeasure, precision, recall]

  # Models

  # Softmax classifier
  inputs = Input(shape=input_shape)
  x = Flatten()(inputs)  # get rid of this
  # init Dense weights with Gaussian scaled by sqrt(1/fan_in)
  preds = Dense(1, kernel_initializer=VarianceScaling(), activation="sigmoid",
      kernel_regularizer=keras.regularizers.l2(0.01))(x)
  model_tower = Model(inputs=inputs, outputs=preds, name="model")
  model_base = keras.models.Sequential()  # dummy since we aren't fine-tuning this model

  ## Create model by replacing classifier of VGG16 model with new
  ## classifier specific to the breast cancer problem.
  ##with tf.device("/cpu"):
  #inputs = Input(shape=input_shape)
  #model_base = VGG16(include_top=False, input_shape=input_shape, input_tensor=inputs)
  #x = Flatten()(model_base.output)  # could also use GlobalAveragePooling2D since output is (None, 1, 1, 2048)
  #x = Dropout(0.5)(x)
  ## init Dense weights with Gaussian scaled by sqrt(1/fan_in)
  #preds = Dense(1, kernel_initializer=VarianceScaling(), activation="sigmoid")(x)
  #model_tower = Model(inputs=inputs, outputs=preds, name="model")

  # TODO: add this when it's necessary, and move to a separate function
  ## Multi-GPU exploitation via a linear combination of GPU loss functions.
  #ins = []
  #outs = []
  #for i in range(num_gpus):
  #  with tf.device("/gpu:{}".format(i)):
  #    x = Input(shape=input_shape)  # split of batch
  #    out = resnet50(x)  # run split on shared model
  #    ins.append(x)
  #    outs.append(out)
  #model = Model(inputs=ins, outputs=outs)  # multi-GPU, data-parallel model
  model = model_tower

  # Freeze all pre-trained model layers.
  for layer in model_base.layers:
    layer.trainable = False

  # Compile model.
  optim = keras.optimizers.Adam(lr=0.001)
  model.compile(optimizer=optim, loss="binary_crossentropy", metrics=metrics)
                #loss_weights=[1/num_gpus]*num_gpus, metrics=metrics)

  # train new softmax classifier
  hist1 = model.fit_generator(train_generator, steps_per_epoch=train_batches,
                              #validation_data=val_generator, validation_steps=val_batches,
                              validation_data=val_data,
                              epochs=clf_epochs, class_weight=class_weights, callbacks=callbacks)


  # fine-tune model
  # Unfreeze some subset of the model and fine-tune by training slowly with low lr.
  for layer in model_base.layers:
    if layer.name.startswith("block5"):
      layer.trainable = True

  optim = keras.optimizers.SGD(lr=0.0001, momentum=0.9)
  model.compile(optimizer=optim, loss="binary_crossentropy", metrics=metrics)
                #loss_weights=[1/num_gpus]*num_gpus, metrics=metrics)

  initial_epoch = clf_epochs
  epochs = initial_epoch + finetune_epochs
  hist2 = model.fit_generator(train_generator, steps_per_epoch=train_batches,
                              #validation_data=val_generator, validation_steps=val_batches,
                              validation_data=val_data,
                              epochs=epochs, initial_epoch=initial_epoch,
                              class_weight=class_weights, callbacks=callbacks) #,
                              #max_q_size=8, nb_worker=1, pickle_safe=False)




if __name__ == "__main__":
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--patches_path", default=os.path.join("data", "mitoses", "patches"),
      help="path to the generated image patches containing `train` & `val` folders \
            (default: %(default)s)")
  parser.add_argument("--exp_path", default=os.path.join("experiments", "mitoses", "sanity",
      datetime.strftime(datetime.today(), "%y-%m-%d_%H:%M:%S")),
      help="path in which to store the model checkpoints, logs, etc. for this experiment \
            (default: %(default)s)")
  parser.add_argument("--batch_size", type=int, default=32,
      help="batch size (default: %(default)s)")
  parser.add_argument("--patch_size", type=int, default=64,
      help="integer length to which the square patches will be resized (default: %(default)s)")
  parser.add_argument("--clf_epochs", type=int, default=1,
      help="number of epochs for which to training the new classifier layers \
           (default: %(default)s)")
  parser.add_argument("--finetune_epochs", type=int, default=1,
      help="number of epochs for which to fine-tune the model (default: %(default)s)")
  args = parser.parse_args()

  # set any other defaults
  train_path = os.path.join(args.patches_path, "train")
  val_path = os.path.join(args.patches_path, "val")

  # make experiment folder (TODO: fail if it already exists)
  os.makedirs(args.exp_path, exist_ok=True)
  print("experiment directory: {}".format(args.exp_path))

  # train!
  train(train_path, val_path, args.exp_path, args.batch_size, args.patch_size,
      args.clf_epochs, args.finetune_epochs)


