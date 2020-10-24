
import tensorflow as tf
import numpy as np
import os
import argparse
import sys
import random
import logging
import librosa
from audio_utils import melspectrogram

from utils import char_to_ix

FLAGS = None

np.set_printoptions(edgeitems=12, linewidth=10000, precision=4, suppress=True)

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def get_mel(path):
    sig, fs = librosa.load(path, sr=FLAGS.sample_rate, mono=True)
    norm_log_mel = melspectrogram(sig)

    logger.debug ("spectrogram shape: {}".format(norm_log_mel.shape))

    logger.debug ("spectrogram:{} {}".format("\n", norm_log_mel))

    mel_len = norm_log_mel.shape[0]

    if (mel_len<FLAGS.max_mel_length):
        for i in range(FLAGS.max_mel_length-mel_len):
            norm_log_mel = np.concatenate((norm_log_mel,np.zeros((1, FLAGS.num_mels))), axis=0)

    logger.debug ("spectrogram final shape: {}".format(norm_log_mel.shape))

    return norm_log_mel, mel_len #sequence with 80 banks each items

def audio_example(name, input, input_length, input_mask, input_durations, mel, mel_length):
    #LJ001-0002
    guid = int(name[2:5])*10000 + int(name[6:10])

    record = {
        'input': tf.train.Feature(int64_list=tf.train.Int64List(value=input)),
        'input_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_length])),
        'input_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=input_mask)),
        'input_durations': tf.train.Feature(int64_list=tf.train.Int64List(value=input_durations)),
        'mel': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(mel, [-1]))),
        'mel_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[mel_length])),
        'guid': tf.train.Feature(int64_list=tf.train.Int64List(value=[guid]))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

def create_records(metadata_file, audio_files, durations_file, tfrecords_file):
    max_input_len = -1
    max_mel_len = -1
    input_long = 0
    mel_long = 0
    record_count = 0

    writer = [None]*3
    writer[0] = tf.io.TFRecordWriter(tfrecords_file.format("train"))
    writer[1] = tf.io.TFRecordWriter(tfrecords_file.format("val"))
    writer[2] = tf.io.TFRecordWriter(tfrecords_file.format("test"))

    indicator = np.empty((13100), dtype=int)
    indicator[0:12300] = 0
    indicator[12300:13040] = 1
    indicator[13040:] = 2
    random.shuffle(indicator)

    durations = {}
    if durations_file:
      f = open(durations_file)

      for line in f:
        name, durations_list = line.split('|')
        durations[name] = [int(i) for i in durations_list.split(",")]

    f = open(metadata_file)

    for line in f:
        name, _, transcript = line.split('|')

        logger.debug ("--START-- IN: {}".format(transcript))

        transcript = transcript.replace("“", " ")
        transcript = transcript.replace("”", " ")
        transcript = transcript.replace(".", " ")
        transcript = transcript.rstrip()
        transcript = " " + transcript + "."

        logger.debug ("OUT: {}".format(transcript))

        max_input_len = max(max_input_len, len(transcript))

        if audio_files:
            mel, mel_len = get_mel(os.path.join(audio_files, name + ".wav"))

            max_mel_len = max(max_mel_len, mel_len)

            logger.debug ("name={} len={} mel len={} records={}".format(name, len(transcript), mel_len, record_count))
        
            if (mel_len > FLAGS.max_mel_length):
                logger.info ('{} skipped mel len={}'.format(name, mel_len))
                mel_long = mel_long + 1
                continue

            mel = np.float32(mel)
        else:
            mel = np.zeros((FLAGS.max_mel_length, FLAGS.num_mels))
            mel_len = FLAGS.max_mel_length

        try:
            input_tensor = [char_to_ix[ch] for ch in transcript.lower()]
        except Exception as e:
            logger.info ("{} length: {} skipped error {}".format(transcript, len(transcript), str(e)))
            input_long = input_long + 1
            continue

        input_mask = [1] * len(transcript)

        while len(input_tensor) < FLAGS.max_input_length:
          input_tensor.append(0)
          input_mask.append(0)

        logger.debug ("input tensor: {}".format(input_tensor))

        if len(durations) != 0:
          if name in durations:
            input_durations = np.array(durations[name])
            if input_durations.shape[0] < FLAGS.max_input_length:
              input_durations = np.pad(input_durations, FLAGS.max_input_length - len(durations[name]), 'constant')
          else:
            logger.info ('{} skipped since missing duration'.format(name))
            continue
        else:
          input_durations = np.zeros((FLAGS.max_input_length), dtype=int)

        tf_example = audio_example(name, input_tensor, len(transcript), input_mask, input_durations, mel, mel_len)
        writer[indicator[record_count]].write(tf_example.SerializeToString())
        record_count = record_count + 1

    return record_count, input_long, mel_long, max_input_len, max_mel_len

def main():
    record_count, input_long_count, mel_long_count, max_input_len, max_mel_len = create_records(FLAGS.metadata_file, FLAGS.audio_files, FLAGS.durations_file, FLAGS.tfrecords_file)

    logging.info ("record_count {} inputr_long {} label_long {} max_input_len {} max_mel_len {}".format(record_count, input_long_count, mel_long_count, max_input_len, max_mel_len))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_mel_length', type=int, default=1024,
            help='Length of the autio signal in frames. Shorter signals will be complemented with zero filled frames, longer will be cut.')
    parser.add_argument('--sample_rate', type=int, default=22050,
            help='Signal will be resampled to this rate.')
    parser.add_argument('--num_mels', type=int, default=80,
            help='This is number of mel filter banks as per Deep Speech 1 article.')
    parser.add_argument('--winlen', type=float, default=0.020,
            help='Audio frame window size as per Deep Speech 1 article.')
    parser.add_argument('--winstep', type=float, default=0.010,
            help='Audio frame sliding as per Deep Speech 1 article.')
    parser.add_argument('--max_input_length', type=int, default=200,
            help='Max length of output strings in characters will shorter strings filled with zeros.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--metadata_file', type=str, default='/work/datasets/LJSpeech-1.1/metadata.csv',
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--durations_file', type=str, default=None,
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--audio_files', type=str, default=None,
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--tfrecords_file', type=str, default='data/{}.tfrecords',
            help='tfrecords output file. It will be used as a prefix if split.')

    FLAGS, unparsed = parser.parse_known_args()

    logger.setLevel(FLAGS.logging)

    logger.debug ("Running with parameters: {}".format(FLAGS))

    main()

