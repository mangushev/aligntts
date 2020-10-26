
#This code contain some BERT code from https://github.com/google-research/bert, please see LICENSE-BERT


import numpy as np
np.set_printoptions(edgeitems=25, linewidth=10000, precision=4, suppress=True)

import collections
import re
import argparse
import sys
import os
import tensorflow as tf

from model import AlignttsModel, get_shape_list
from utils import alphabet, ix_to_char

FLAGS = None

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

def make_input_fn(filename, is_training, drop_reminder):
  """Returns an `input_fn` for train and eval."""

  def input_fn(params):
    def parser(serialized_example):
      example = tf.io.parse_single_example(
          serialized_example,
          features={
              "input": tf.io.FixedLenFeature([FLAGS.max_input_length], tf.int64),
              "input_length": tf.io.FixedLenFeature((), tf.int64),
              "input_mask": tf.io.FixedLenFeature([FLAGS.max_input_length], tf.int64),
              "input_durations": tf.io.FixedLenFeature([FLAGS.max_input_length], tf.int64),
              "mel": tf.io.FixedLenFeature([FLAGS.max_mel_length, FLAGS.num_mels], tf.float32),
              "mel_length": tf.io.FixedLenFeature((), tf.int64),
              "guid": tf.io.FixedLenFeature((), tf.int64),
          })
      
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.to_int32(t)
        example[name] = t

      return example

    dataset = tf.data.TFRecordDataset(
      filename, buffer_size=FLAGS.dataset_reader_buffer_size)
    
    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        parser, batch_size=params["batch_size"],
        num_parallel_batches=8,
        drop_remainder=drop_reminder))
    return dataset

  return input_fn

def model_fn_builder(init_checkpoint, learning_rate, num_train_steps, use_tpu):

  def model_fn(features, labels, mode, params):

    input = features["input"]
    input_length = features["input_length"]
    input_mask = features["input_mask"]
    input_durations = features["input_durations"]
    mel = features["mel"]
    mel_length = features["mel_length"]
    guid = features["guid"]

    alpha=params["alpha"]
    if mode == tf.estimator.ModeKeys.TRAIN:
      alpha = 1.0

    is_trainable = True if mode == tf.estimator.ModeKeys.TRAIN else False

    model = AlignttsModel(input,
      input_length,
      input_mask,
      input_durations,
      mel,
      mel_length,
      hidden_size=params["hidden_size"],
      num_hidden_layers=params["num_hidden_layers"],
      num_attention_heads=params["num_attention_heads"],
      filter_width=params["filter_width"],
      duration_predictor_hidden_layers=params["duration_predictor_hidden_layers"],
      duration_predictor_attention_heads=params["duration_predictor_attention_heads"],
      duration_predictor_hidden_size=params["duration_predictor_hidden_size"],
      num_mix_density_hidden_layers=params["num_mix_density_hidden_layers"], #as in DEEP MIXTURE DENSITY NETWORKS GOOGLE Paper
      mix_density_hidden_size=params["mix_density_hidden_size"],
      alphabet_size=params["alphabet_size"],
      initializer_range=params["initializer_range"],
      activation_fn=tf.nn.relu,
      alpha=alpha,
      dropout_prob=params["dropout_prob"],
      use_durations=params["use_durations"],
      is_trainable=is_trainable)

    if mode == tf.estimator.ModeKeys.TRAIN:

      tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

      if params["training_task"] == "alignment_loss" or params["training_task"] == "fixed_encoder":
        calculated_learning_rate = tf.math.pow(tf.cast(params["hidden_size"], dtype=tf.float32), -0.5)*tf.math.minimum(tf.math.pow(tf.cast(tf.compat.v1.train.get_global_step()+1, dtype=tf.float32), -0.5), tf.cast(tf.compat.v1.train.get_global_step()+1, dtype=tf.float32)*tf.math.pow(4000.0, -1.5))
        effective_learning_rate = learning_rate
        #effective_learning_rate = tf.Print(calculated_learning_rate, [calculated_learning_rate], "Calculated learning rate")
      else:
        effective_learning_rate = learning_rate

      if params["training_task"] == "alignment_loss":

        #step 1: training for durations
        #we adopt the samelearning rate schedule in [18] with 40K training steps in thefirst two training stages

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "input_embeddings")
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "input_positions"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "encoder_ttf"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "mix_density_network"))

        loss = tf.math.reduce_mean(model.per_example_alignment_loss, keepdims=False, name="mean_loss")
      elif params["training_task"] == "fixed_encoder":
        #step 2: all network training with fixed encoder with durations from input (precalculated by mix) 
        #we adopt the samelearning rate schedule in [18] with 40K training steps in thefirst two training stages
        #mean square error (MSE)loss between the predicted and target mel-spectrum.

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "mel_positions")
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "decoder_ttf"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "mel_spectrograms"))

        loss = tf.compat.v1.losses.mean_squared_error(mel, model.mel_spectrograms)

      elif params["training_task"] == 'joint_fft_mix_density':
        #step 3: all network training with durations from mix network
        #fixed learning rate of10−4with80K training steps in fine-tuning the parameters of the wholemodel
        #mean square error (MSE)loss between the predicted and target mel-spectrum.

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "input_embeddings")
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "input_positions"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "encoder_ttf"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "mix_density_network"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "mel_positions"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "decoder_ttf"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "mel_spectrograms"))

        loss = tf.compat.v1.losses.mean_squared_error(mel, model.mel_spectrograms)
      elif params["training_task"] == 'duration_predictor':
        #step 4: Train predictor with durations from input (precalculated by mix)
        #the duration predictor is trained with afixed learning rate of10−4and 10K training steps.
        #mean square error (MSE)loss between the predicted and target mel-spectrum.

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "duration_embeddings")
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "duration_positions"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "duration_ttf"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "nominal_durations"))

        loss = tf.compat.v1.losses.mean_squared_error(input_durations, model.nominal_durations)

      for i, v in enumerate(tvars):
        tf.logging.info("{}: {}".format(i, v))

      grads = tf.gradients(loss, tvars, name='gradients')

      if (FLAGS.clip_gradients > 0):
        gradients, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
      else:
        gradients = grads

      #Adam optimizer with β1=  0.9,β2=  0.98,ε=  10−9.
      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=effective_learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-09)
      if FLAGS.use_tpu:
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

      train_op = optimizer.apply_gradients(zip(gradients, tvars), global_step=tf.compat.v1.train.get_global_step())

      training_hooks = None
      if not FLAGS.use_tpu:
        logging_hook = tf.train.LoggingTensorHook({"loss": loss, "step": tf.train.get_global_step()}, every_n_iter=1)
        training_hooks = [logging_hook]

      return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode, predictions=None, loss=loss, train_op=train_op, eval_metrics=None,
        export_outputs=None, scaffold_fn=scaffold_fn, host_call=None, training_hooks=training_hooks,
        evaluation_hooks=None, prediction_hooks=None)

    else:

      if params["prediction_task"] == "durations":
        spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={'mix_durations': model.mix_durations,
                       'guid': guid
                      })

      elif params["prediction_task"] == "alpha_values":
        spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={'alpha': model.log_alpha,
                       'input': input,
                       'input_length': input_length,
                       'mel_length': mel_length,
                       'guid': guid
                      })
      else:
        spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={'mel_durations': model.mel_durations,
                       'mel_spectrograms': model.mel_spectrograms,
                       'guid': guid
                      })
      return spec 

  return model_fn   

def get_alpha_durations(probabilities):
  o_len = probabilities.shape[0]
  s_len = probabilities.shape[1]

  best = np.zeros((o_len, s_len), dtype=np.float)

  b = s_len - 1
  best[o_len - 1, b] = probabilities[o_len - 1, b]

  for t in range(o_len - 2, -1, -1):
    if b == 0:
      b = 0
    elif probabilities[t, b - 1] > probabilities[t, b]:
      b = b - 1
    else:
      b = b
    best[t, b] = probabilities[t, b]

  return best

def get_durations(probabilities):
  o_len = probabilities.shape[0]
  s_len = probabilities.shape[1]

  delta = np.zeros((o_len, s_len), dtype=np.float)
  path = np.zeros((o_len, s_len), dtype=np.int)
  best = np.zeros((o_len, s_len), dtype=np.int)

  for t in range(1, o_len):
    for j in range(s_len):
      m = np.argmax(delta[t - 1])
      if (j == m):
        delta[t, j] = delta[t - 1, m] + probabilities[t - 1, m]
      elif (j - 1 == m):
        delta[t, j] = delta[t - 1, m] + probabilities[t - 1, m]
      else:
        delta[t, j] = 0
      path[t, j] = m

  b = np.argmax(delta[-1])

  for t in range(o_len - 1, 0, -1):
    best[t, b] = 1
    b = path[t, b]

  return best, delta

def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  tpu_cluster_resolver = None

  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=None,
      job_name='worker',
      coordinator_name=None,
      coordinator_address=None,
      credentials='default', service=None,
      discovery_url=None
    )

  tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
    iterations_per_loop=FLAGS.iterations_per_loop, 
    num_cores_per_replica=FLAGS.num_tpu_cores,
    per_host_input_for_training=True 
  )

  run_config = tf.compat.v1.estimator.tpu.RunConfig(
    tpu_config=tpu_config,
    evaluation_master=None,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    master=None,
    cluster=tpu_cluster_resolver,
    **{
      'save_checkpoints_steps': FLAGS.save_checkpoints_steps,
      'tf_random_seed': FLAGS.random_seed,
      'model_dir': FLAGS.output_dir, 
      'keep_checkpoint_max': FLAGS.keep_checkpoint_max,
      'log_step_count_steps': FLAGS.log_step_count_steps
    }
  )

  #Use duration from: 0 - input, 1 - mix network, 2 - duration predictor
  #mel prediction require 2, alpha or durations does not matter
  use_durations = 2

  if FLAGS.action == 'TRAIN':
    if FLAGS.training_task == 'fixed_encoder':
      use_durations = 0
    elif FLAGS.training_task == 'joint_fft_mix_density':
      use_durations = 1

  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    model_fn=model_fn_builder(FLAGS.init_checkpoint, FLAGS.learning_rate, FLAGS.num_train_steps, FLAGS.use_tpu),
    use_tpu=FLAGS.use_tpu,
    train_batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.batch_size,
    predict_batch_size=FLAGS.batch_size,
    config=run_config,
    params={
        "hidden_size": FLAGS.hidden_size,
        "num_hidden_layers": FLAGS.num_hidden_layers,
        "num_attention_heads": FLAGS.num_attention_heads,
        "filter_width": FLAGS.filter_width,
        "duration_predictor_hidden_layers": FLAGS.duration_predictor_hidden_layers,
        "duration_predictor_attention_heads": FLAGS.duration_predictor_attention_heads,
        "duration_predictor_hidden_size": FLAGS.duration_predictor_hidden_size,
        "num_mix_density_hidden_layers": FLAGS.num_mix_density_hidden_layers,
        "mix_density_hidden_size": FLAGS.mix_density_hidden_size,
        "alphabet_size": len(alphabet),
        "initializer_range": FLAGS.initializer_range,
        "alpha": FLAGS.alpha,
        "num_mels": FLAGS.num_mels,
        "dropout_prob": FLAGS.dropout_prob,
        "use_tpu": FLAGS.use_tpu,
        "use_durations": use_durations,
        "training_task": FLAGS.training_task,
        "prediction_task": FLAGS.prediction_task
    })

  if FLAGS.action == 'TRAIN':
    estimator.train(input_fn=make_input_fn(FLAGS.train_file, is_training=True, drop_reminder=True), max_steps=FLAGS.num_train_steps)
  
  if FLAGS.action == 'PREDICT':
    predict_drop_remainder = True if FLAGS.use_tpu else False
    results = estimator.predict(input_fn=make_input_fn(FLAGS.test_file, is_training=False, drop_reminder=predict_drop_remainder))

    if FLAGS.prediction_task == 'durations':
      output_predict_file = os.path.join(FLAGS.output_dir, "durations.csv")
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        for (i, prediction) in enumerate(results):
          writer.write("LJ{:03d}-{:04d}|".format(int(prediction["guid"]/10000), int(prediction["guid"]%10000)))
          writer.write(",".join([str(i) for i in prediction["mix_durations"]]) + "\n")

    elif FLAGS.prediction_task == 'alpha_values':
      for i, prediction in enumerate(results):
        best1 = get_alpha_durations(prediction["alpha"][:prediction["mel_length"], :prediction["input_length"]])
      
        print ("LJ{:03d}-{:04d}".format(int(prediction["guid"]/10000), int(prediction["guid"]%10000)))
        print ('@' + '__'.join([ix_to_char[prediction["input"][i]] for i in range(prediction["input_length"])]))
        print ('@' + ' '.join(["{:02d}".format(np.count_nonzero(best1, axis=0)[i]) for i in range(prediction["input_length"])]))

        print ("LAST:" , prediction["alpha"][prediction["mel_length"]-1, prediction["input_length"]-1])

        print ("BEST START:")
        for j, row in enumerate(best1):
          print ("B", j, ":", " ".join(["{:1.0e}".format(x) for x in row]))
        file_name = "best-LJ{:03d}-{:04d}.npy".format(int(prediction["guid"]/10000), int(prediction["guid"]%10000))
        np.save(file_name, best1, allow_pickle=True, fix_imports=True)

        print ("ALPHA START:")
        for j, row in enumerate(prediction["alpha"][:prediction["mel_length"]]):
          print ("A", j, ":", " ".join(["{:2.0e}".format(p) for p in row[:prediction["input_length"]]]))

        file_name = "alpha-LJ{:03d}-{:04d}.npy".format(int(prediction["guid"]/10000), int(prediction["guid"]%10000))
        np.save(file_name, prediction["alpha"][:prediction["mel_length"], :prediction["input_length"]], allow_pickle=True, fix_imports=True)

        file_name = "log_alpha-LJ{:03d}-{:04d}.npy".format(int(prediction["guid"]/10000), int(prediction["guid"]%10000))
        np.save(file_name, prediction["alpha"][:prediction["mel_length"], :prediction["input_length"]], allow_pickle=True, fix_imports=True)
        if (i >= 10):
          break
    else:
      for prediction in results:
        file_name = "LJ{:03d}-{:04d}.mel".format(int(prediction["guid"]/10000), int(prediction["guid"]%10000))
        mel_spectrograms = prediction["mel_spectrograms"]
        mel_length = np.sum(prediction["mel_durations"])
        data = np.array(mel_spectrograms[:mel_length], 'float32')
        fid = open(file_name, 'wb')
        data.tofile(fid)
        fid.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='gs://speech_synthesis/aligntts/output',
            help='Model directrory in google storage.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
            help='This will be checkpoint from previous training phase.')
    parser.add_argument('--train_file', type=str, default='gs://speech_synthesis/aligntts/data/train.tfrecords',
            help='Train file location in google storage.')
    parser.add_argument('--test_file', type=str, default='gs://speech_synthesis/aligntts/data/test.tfrecords',
            help='Test file location in google storage.')
    parser.add_argument('--max_input_length', type=int, default=200,
            help='Max length of input strings in characters will shorter strings filled with zeros.')
    parser.add_argument('--max_mel_length', type=int, default=1024,
            help='Length of the autio signal in frames. It is defined in feature preparation tool.')
    parser.add_argument('--num_mels', type=int, default=80,
            help='dimension of the output is 160 (80 dimensions for the meanand 80 dimensions for variance of the gaussian distribution).')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
            help='As in FastSpeech article.')
    parser.add_argument('--num_train_steps', type=int, default=140000,
            help='Number of steps to run trainer.')
    parser.add_argument('--iterations_per_loop', type=int, default=1000,
            help='Number of iterations per TPU training loop.')
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--log_step_count_steps', type=int, default=1000,
            help='Number of step to write logs.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
    parser.add_argument('--dataset_reader_buffer_size', type=int, default=100,
            help='input pipeline is I/O bottlenecked, consider setting this parameter to a value 1-100 MBs.')
    parser.add_argument('--shuffle_buffer_size', type=int, default=12500,
            help='Items are read from this buffer.')
    parser.add_argument('--use_tpu', default=False, action='store_true',
            help='Train on TPU.')
    parser.add_argument('--tpu', type=str, default='node-1-15-2',
            help='TPU instance name.')
    parser.add_argument('--num_tpu_cores', type=int, default=8,
            help='Number of cores on TPU.')
    parser.add_argument('--tpu_zone', type=str, default='us-central1-c',
            help='TPU instance zone location.')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
            help='Optimizer learning rate.')
    parser.add_argument('--clip_gradients', type=float, default=-1.,
            help='Clip gradients to deal with explosive gradients.')
    parser.add_argument('--random_seed', type=int, default=1234,
            help='Random seed to initialize values in a grath. It will produce the same results only if data and grath did not change in any way.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--action', default='PREDICT', choices=['TRAIN','EVALUATE','PREDICT'],
            help='An action to execure.')
    parser.add_argument('--training_task', choices=['alignment_loss', 'fixed_encoder', 'joint_fft_mix_density', 'duration_predictor'],
            help='Training phase.')
    parser.add_argument('--prediction_task', default='mel_values', choices=['durations', 'alpha_values', 'mel_values'],
            help='Values to predict.')
    parser.add_argument('--restore', default=False, action='store_true',
            help='Restore last checkpoint.')
    parser.add_argument('--hidden_size', type=int, default=768,
            help='dimension of each network in the Feed-Forward Transformer is all set to 768.')
    parser.add_argument('--num_hidden_layers', type=int, default=6,
            help='Feed-Forward  Transformer  contains  6  FFT  blocks.')
    parser.add_argument('--num_attention_heads', type=int, default=2,
            help='number of attention head is set to 2 in all FFT block.')
    parser.add_argument('--filter_width', type=int, default=3,
            help='kernel size of 1D convolution is set to 3 in all FFT block')
    parser.add_argument('--duration_predictor_hidden_layers', type=int, default=2,
            help='duration predictor includes 2 FFT blocks.')
    parser.add_argument('--duration_predictor_attention_heads', type=int, default=2,
            help='number of attention head is set to 2 in all FFT block.')
    parser.add_argument('--duration_predictor_hidden_size', type=int, default=128,
            help='.')
    parser.add_argument('--num_mix_density_hidden_layers', type=int, default=4,
            help='DEEP MIXTURE DENSITY NETWORKS GOOGLE Paper.')
    parser.add_argument('--mix_density_hidden_size', type=int, default=256,
            help='hidden size of the linear layer in the mix network is set to 256.')
    parser.add_argument('--initializer_range', type=float, default=0.02,
            help='.')
    parser.add_argument('--alpha', type=float, default=1.0,
            help='adjust thevoice speed from 0.5x to 1.5x.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
