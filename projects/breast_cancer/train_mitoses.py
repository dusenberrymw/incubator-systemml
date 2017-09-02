"""Training - mitosis detection"""
import argparse
from datetime import datetime
import math
import os

import keras
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import VarianceScaling
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf


def get_label(filename):
  """Get label from filename.

  Args:
    filename: String in format
      "**/train|val/mitosis|normal/{lab}_{case}_{region}_{row}_{col}_{suffix}.{ext}",
      where the label is either "mitosis" or "normal".

  Returns:
    TensorFlow float binary label equal to 1 for mitosis or 0 for
      normal.
  """
  # note file name format:
  # lab is a single digit, case and region are two digits with padding if needed
  # "**/train|val/mitosis|normal/{lab}_{case}_{region}_{row}_{col}_{suffix}.{ext}"
  splits = tf.string_split([filename], "/")
  label_str = splits.values[-2]
  # check that label string is valid
  is_valid = tf.logical_or(tf.equal(label_str, 'normal'), tf.equal(label_str, 'mitosis'))
  assert_op = tf.Assert(is_valid, [label_str])
  with tf.control_dependencies([assert_op]):  # test for correct label extraction
    #label = tf.to_int32(tf.equal(label_str, 'mitosis'))
    label = tf.to_float(tf.equal(label_str, 'mitosis'))
    return label


def get_image(filename, patch_size):
  """Get image from filename.

  Args:
    filename: String filename of an image.
    patch_size: Integer length to which the square image will be
      resized.

  Returns:
    TensorFlow tensor containing the decoded and resized image.
  """
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_resized = tf.image.resize_images(image_decoded, [patch_size, patch_size])
  return image_resized


def preprocess(filename, patch_size):
  """Get image and label from filename.

  Args:
    filename: String filename of an image.
    patch_size: Integer length to which the square image will be
      resized.

  Returns:
    Tuple of TensorFlow image tensor, binary label, and filename.
  """
  #  return image_resized, label
  image = get_image(filename, patch_size)
  label = get_label(filename)
  return image, label, filename


def create_reset_metric(metric, scope, **metric_kwargs):  # prob safer to only allow kwargs
  """Create a resettable metric.

  Args:
    metric: A tf.metrics metric function.
    scope: A String scope name to enclose the metric variables within.
    metric_kwargs:  Kwargs for the metric.

  Returns:
    The metric op, the metric update op, and a metric reset op.
  """
  # originally from https://github.com/tensorflow/tensorflow/issues/4814
  with tf.variable_scope(scope) as scope:
    metric_op, update_op = metric(**metric_kwargs)
    scope_name = tf.contrib.framework.get_name_scope()  # in case nested name/variable scopes
    local_vars = tf.contrib.framework.get_variables(scope_name,
        collection=tf.GraphKeys.LOCAL_VARIABLES)  # get all local variables in this scope
    reset_op = tf.variables_initializer(local_vars)
  return metric_op, update_op, reset_op


def train(train_path, val_path, exp_path, batch_size, patch_size, clf_epochs, finetune_epochs,
    clf_lr, finetune_lr, log_interval, threads):
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
    finetune_epochs: Integer number of epochs for which to fine-tune the
      model.
    clf_lr: Float learning rate for training the new classifier layers.
    finetune_lr: Float learning rate for fine-tuning the model.
    log_interval: Integer number of steps between logging during
      training.
    threads: Integer number of threads for dataset buffering.
  """
  # TODO: break this out into:
  #   * data gen func
  #   * inference func
  #     * model creation, loss, and compilation
  #   * train func
  sess = K.get_session()

  # debugger
  #from tensorflow.python import debug as tf_debug
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  # data
  with tf.name_scope("data"):
    # TODO: add data augmentation function
    train_dataset = (tf.contrib.data.Dataset.list_files('{}/*/*.jpg'.format(train_path))
        .shuffle(10000)
        .map(lambda x: preprocess(x, patch_size), num_threads=threads,
          output_buffer_size=100*batch_size)
        .batch(batch_size)
        )
    val_dataset = (tf.contrib.data.Dataset.list_files('{}/*/*.jpg'.format(val_path))
        .map(lambda x: preprocess(x, patch_size), num_threads=threads,
          output_buffer_size=100*batch_size)
        .batch(batch_size)
        )

    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
    images, labels, filenames = iterator.get_next()
    labels = tf.expand_dims(labels, -1)  # tf sucks
    actual_batch_size = tf.shape(images)[0]
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

  # models
  with tf.name_scope("model"):
    # Softmax classifier
    input_shape = (patch_size, patch_size, 3)
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)  # get rid of this
    # init Dense weights with Gaussian scaled by sqrt(1/fan_in)
    logits = Dense(1, kernel_initializer=VarianceScaling(),
        kernel_regularizer=keras.regularizers.l2(0.01))(x)
    model_tower = Model(inputs=inputs, outputs=logits, name="model")
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

    # call model on dataset images to compute logits and predictions
    logits = model(images)
    preds = tf.round(tf.nn.sigmoid(logits), name="preds")  # implicit threshold at 0.5

  # loss
  with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

  # freeze all pre-trained model layers.
  for layer in model_base.layers:
    layer.trainable = False

  # optim
  with tf.name_scope("optim"):
    #global_step = tf.Variable(0, trainable=False, name='global_step')
    opt = tf.train.AdamOptimizer(clf_lr)
    grads_and_vars = opt.compute_gradients(loss, var_list=model.trainable_weights)
    #train_op = opt.minimize(loss, var_list=model_base.trainable_weights)
    apply_grads_op = opt.apply_gradients(grads_and_vars) #, global_step=global_step)
    model_update_ops = model.updates
    train_op = tf.group(apply_grads_op, *model_update_ops)

  # metrics
  with tf.name_scope("metrics"):
    mean_loss, mean_loss_update_op, mean_loss_reset_op = create_reset_metric(tf.metrics.mean,
        'mean_loss', values=loss)
    acc, acc_update_op, acc_reset_op = create_reset_metric(tf.metrics.accuracy, 'acc',
        labels=labels, predictions=preds)
    ppv, ppv_update_op, ppv_reset_op = create_reset_metric(tf.metrics.precision,
        'ppv', labels=labels, predictions=preds)
    recall, recall_update_op, recall_reset_op = create_reset_metric(tf.metrics.recall,
        'recall', labels=labels, predictions=preds)
    f1 = 2 * (ppv * recall) / (ppv + recall)

    # combine all reset & update ops
    metric_update_ops = tf.group(mean_loss_update_op, acc_update_op, ppv_update_op,
        recall_update_op)
    metric_reset_ops = tf.group(mean_loss_reset_op, acc_reset_op, ppv_reset_op,
        recall_reset_op)

  # tensorboard
  # - minibatch summaries
  images_summary = tf.summary.image("images", images) #, max_outputs=10)
  actual_batch_size_summary = tf.summary.scalar("batch_size", actual_batch_size)
  minibatch_loss_summary = tf.summary.scalar("minibatch_loss", loss)
  minibatch_summaries = tf.summary.merge([minibatch_loss_summary]) #, actual_batch_size_summary,
      #images_summary])
  # - epoch summaries
  epoch_loss_summary = tf.summary.scalar("epoch_avg_loss", mean_loss)
  epoch_acc_summary = tf.summary.scalar("epoch_acc", acc)
  epoch_ppv_summary = tf.summary.scalar("epoch_ppv", ppv)
  epoch_recall_summary = tf.summary.scalar("epoch_recall", recall)
  epoch_f1_summary = tf.summary.scalar("epoch_f1", f1)
  epoch_summaries = tf.summary.merge([epoch_loss_summary, epoch_acc_summary,
    epoch_ppv_summary, epoch_recall_summary, epoch_f1_summary])
  #all_summaries = tf.summary.merge_all()

  # use train and val writers so that plots can be on same graph
  writer = tf.summary.FileWriter(exp_path, sess.graph)
  train_writer = tf.summary.FileWriter(os.path.join(exp_path, "train"))
  val_writer = tf.summary.FileWriter(os.path.join(exp_path, "val"))

  # initialize stuff
  global_init_op = tf.global_variables_initializer()
  local_init_op = tf.local_variables_initializer()
  sess.run([global_init_op, local_init_op])

  # classifier train loop
  # TODO: encapsulate this into a function for reuse during fine-tuning, probably into a
  # lightweight class with `forward`, `loss`, `train`, `metrics`, etc.
  i = 0
  for e in range(clf_epochs):
    # training
    sess.run(train_init_op)
    while True:
      try:
        if log_interval > 0 and i % log_interval == 0:
          # train, update metrics, & log stuff
          _, _, loss_val, summary_str, mean_loss_val = sess.run([train_op, metric_update_ops, loss,
              minibatch_summaries, mean_loss], feed_dict={K.learning_phase(): 1})
          train_writer.add_summary(summary_str, i)
          print("train", e, i, loss_val, mean_loss_val)
        else:
          # train & update metrics
          _, _, loss_val = sess.run([train_op, metric_update_ops, minibatch_loss_summary],
              feed_dict={K.learning_phase(): 1})
          train_writer.add_summary(summary_str, i)
        i += 1
      except tf.errors.OutOfRangeError:
        break
    # log average training metrics for epoch & reset
    print("---epoch {}, train average loss: ".format(e), sess.run(mean_loss))
    train_writer.add_summary(sess.run(epoch_summaries), e)
    sess.run(metric_reset_ops)

    # validation
    sess.run(val_init_op)
    while True:
      try:
        # evaluate & update metrics
        _, loss_val, mean_loss_val, acc_val = sess.run([metric_update_ops, loss, mean_loss, acc],
            feed_dict={K.learning_phase(): 0})
        print("val", e, loss_val, mean_loss_val, acc_val)
      except tf.errors.OutOfRangeError:
        break
    # log average validation metrics for epoch & reset
    print("---epoch {}, val average loss: ".format(e), sess.run(mean_loss))
    val_writer.add_summary(sess.run(epoch_summaries), e)
    sess.run(metric_reset_ops)

    val_writer.flush()
    #train_writer.flush()


#  # fine-tune model
#  # Unfreeze some subset of the model and fine-tune by training slowly with low lr.
#  for layer in model_base.layers:
#    if layer.name.startswith("block5"):
#      layer.trainable = True
#
#  optim = keras.optimizers.SGD(lr=0.0001, momentum=0.9)
#  model.compile(optimizer=optim, loss="binary_crossentropy", metrics=metrics)
#                #loss_weights=[1/num_gpus]*num_gpus, metrics=metrics)
#
#  initial_epoch = clf_epochs
#  epochs = initial_epoch + finetune_epochs
#  hist2 = model.fit_generator(train_generator, steps_per_epoch=train_batches,
#                              #validation_data=val_generator, validation_steps=val_batches,
#                              validation_data=val_data,
#                              epochs=epochs, initial_epoch=initial_epoch,
#                              class_weight=class_weights, callbacks=callbacks) #,
#                              #max_q_size=8, nb_worker=1, pickle_safe=False)
#



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
  parser.add_argument("--clf_lr", type=float, default=0.001,
      help="learning rate for training the new classifier layers \
           (default: %(default)s)")
  parser.add_argument("--finetune_lr", type=int, default=1e-5,
      help="learning rate for fine-tuning the model (default: %(default)s)")
  parser.add_argument("--log_interval", type=int, default=100,
      help="number of steps between logging during training")
  parser.add_argument("--threads", type=int, default=5,
      help="number of threads for dataset buffering")
  args = parser.parse_args()

  # set any other defaults
  train_path = os.path.join(args.patches_path, "train")
  val_path = os.path.join(args.patches_path, "val")

  # make experiment folder (TODO: fail if it already exists)
  os.makedirs(args.exp_path, exist_ok=True)
  print("experiment directory: {}".format(args.exp_path))

  # train!
  train(train_path, val_path, args.exp_path, args.batch_size, args.patch_size,
      args.clf_epochs, args.finetune_epochs, args.clf_lr, args.finetune_lr, args.log_interval,
      args.threads)


# ---
# tests
# TODO: eventually move these to a separate file.
# `py.test train_mitoses.py`

def test_get_label():
  import pytest
  sess = tf.Session()

  # mitosis
  filename = "train/mitosis/1_03_05_713_348.jpg"
  label_op = get_label(filename)
  label = sess.run(label_op)
  assert label == 1

  # normal
  filename = "train/normal/1_03_05_713_348.jpg"
  label_op = get_label(filename)
  label = sess.run(label_op)
  assert label == 0

  # wrong label name
  with pytest.raises(tf.errors.InvalidArgumentError):
    filename = "train/unknown/1_03_05_713_348.jpg"
    label_op = get_label(filename)
    label = sess.run(label_op)


def test_resettable_metric():
  x = tf.placeholder(tf.int32, [None, 1])
  x1 = np.array([1,0,0,0]).reshape(4,1)
  x2 = np.array([0,0,0,0]).reshape(4,1)

  with tf.name_scope("something"):  # testing nested name/variable scopes
    mean_op, update_op, reset_op = create_reset_metric(tf.metrics.mean, 'mean_loss', values=x)

  sess = K.get_session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x1})
  assert np.allclose([out_up], [1/4])
  assert np.allclose([sess.run(mean_op)], [1/4])
  assert np.allclose([sess.run(mean_op)], [1/4])

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x1})
  assert np.allclose([out_up], [2/8])
  assert np.allclose([sess.run(mean_op)], [2/8])
  assert np.allclose([sess.run(mean_op)], [2/8])

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x2})
  assert np.allclose([out_up], [2/12])
  assert np.allclose([sess.run(mean_op)], [2/12])
  assert np.allclose([sess.run(mean_op)], [2/12])

  sess.run(reset_op)  # make sure this works!

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x2})
  assert out_up == 0
  assert sess.run(mean_op) == 0

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x1})
  assert np.allclose([out_up], [1/8])
  assert np.allclose([sess.run(mean_op)], [1/8])
  assert np.allclose([sess.run(mean_op)], [1/8])

