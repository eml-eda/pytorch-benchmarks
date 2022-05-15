#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.platform import gfile

import functools

import matplotlib.pyplot as plt
import numpy as np
import os, pickle

import kws_util
import keras_model as models
import speech_dscnn as modelsdscnn
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import time
torch.set_printoptions(precision=8)
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa

word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right",
               "Stop", "Up", "Yes", "Silence", "Unknown"]

def convert_to_int16(sample_dict):
  audio = sample_dict['audio']
  label = sample_dict['label']
  audio16 = tf.cast(audio, 'int16')
  return audio16, label

def cast_and_pad(sample_dict):
  audio = sample_dict['audio']
  label = sample_dict['label']
  paddings = [[0, 16000-tf.shape(audio)[0]]]
  audio = tf.pad(audio, paddings)
  audio16 = tf.cast(audio, 'int16')
  return audio16, label

def convert_dataset(item):
  """Puts the mnist dataset in the format Keras expects, (features, labels)."""
  audio = item['audio']
  label = item['label']
  return audio, label


def get_preprocess_audio_func(model_settings,is_training=False,background_data = []):
  def prepare_processing_graph(next_element):
    """Builds a TensorFlow graph to apply the input distortions.
    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.
    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:
      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.
    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings['desired_samples']
    background_frequency = model_settings['background_frequency']
    background_volume_range_= model_settings['background_volume_range_']

    wav_decoder = tf.cast(next_element['audio'], tf.float32)
    if model_settings['feature_type'] != "td_samples":
      wav_decoder = wav_decoder/tf.reduce_max(wav_decoder)
    else:
      wav_decoder = wav_decoder/tf.constant(2**15,dtype=tf.float32)
    #Previously, decode_wav was used with desired_samples as the length of array. The
    # default option of this function was to pad zeros if the desired samples are not found
    wav_decoder = tf.pad(wav_decoder,[[0,desired_samples-tf.shape(wav_decoder)[-1]]]) 
    # Allow the audio sample's volume to be adjusted.
    foreground_volume_placeholder_ = tf.constant(1,dtype=tf.float32)
    
    scaled_foreground = tf.multiply(wav_decoder,
                                    foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    time_shift_padding_placeholder_ = tf.constant([[2,2]], tf.int32)
    time_shift_offset_placeholder_ = tf.constant([2],tf.int32)
    scaled_foreground.shape
    padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples])
    
    if model_settings['feature_type'] == 'mfcc':
      stfts = tf.signal.stft(sliced_foreground, frame_length=model_settings['window_size_samples'], 
                         frame_step=model_settings['window_stride_samples'], fft_length=None,
                         window_fn=tf.signal.hann_window
                         )
      spectrograms = tf.abs(stfts)
      num_spectrogram_bins = stfts.shape[-1]
      # default values used by contrib_audio.mfcc as shown here
      # https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
      linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins,
                                                                           model_settings['sample_rate'],
                                                                           lower_edge_hertz, upper_edge_hertz)
      mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
      mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
      # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
      log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
      # Compute MFCCs from log_mel_spectrograms and take the first 13.
      mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :model_settings['dct_coefficient_count']]
      mfccs = tf.reshape(mfccs,[model_settings['spectrogram_length'], model_settings['dct_coefficient_count'], 1])
      next_element['audio'] = mfccs
      #next_element['label'] = tf.one_hot(next_element['label'],12)
      
    return next_element
  
  return prepare_processing_graph


def prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME):
  """Searches a folder for background noise audio, and loads it into memory.
  It's expected that the background audio samples will be in a subdirectory
  named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
  the sample rate of the training data, but can be much longer in duration.
  If the '_background_noise_' folder doesn't exist at all, this isn't an
  error, it's just taken to mean that no background noise augmentation should
  be used. If the folder does exist, but it's empty, that's treated as an
  error.
  Returns:
    List of raw PCM-encoded audio samples of background noise.
  Raises:
    Exception: If files aren't found in the folder.
  """
  background_data = []
  background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
  if not os.path.exists(background_dir):
    return background_data
  search_path = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME,'*.wav')
  for wav_path in gfile.Glob(search_path):
    raw_audio = tf.io.read_file(wav_path)
    audio = tf.audio.decode_wav(raw_audio)
    background_data.append(audio[0])
  if not background_data:
    raise Exception('No background wav files were found in ' + search_path)
  return background_data


def get_training_data(Flags):
  
  label_count=12
  model_settings = modelsdscnn.prepare_model_settings(label_count, Flags)

  bg_path=Flags.bg_path
  BACKGROUND_NOISE_DIR_NAME='_background_noise_' 
  background_data = prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME)

  splits = ['train', 'test', 'validation']
  (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits, data_dir=Flags.data_dir, with_info=True)

  # change output from a dictionary to a feature,label tuple
  ds_train = ds_train.map(convert_dataset)
  ds_test = ds_test.map(convert_dataset)
  ds_val = ds_val.map(convert_dataset)

  return ds_train, ds_test, ds_val, model_settings

def preprocess_pytorch(tmp_audio, tmp_label, model_settings):

  model_settings = model_settings
  desired_samples = model_settings['desired_samples']
  tmp_audio_post = []

  with tqdm(total=len(tmp_audio), unit="batch") as tepoch:
    tepoch.set_description(f"preprocess in pytorch: ")
    for audio in tmp_audio:
      tepoch.update(1)
      audio = (audio * 1.0) / (2 ** 15 * 1.0)
      audio = torch.tensor(audio, dtype=torch.float32)
      length = len(audio)
      audio = np.pad(audio, (0, desired_samples - length), 'constant', constant_values=0)
      audio = audio * 1.0

      time_shift_padding_placeholder_ = (2, 2)
      time_shift_offset_placeholder_ = 2

      audio = np.pad(audio, time_shift_padding_placeholder_, 'constant', constant_values=0)
      audio = audio[time_shift_offset_placeholder_:desired_samples + time_shift_offset_placeholder_]
      audio = torch.tensor(audio)
      audio = torch.squeeze(audio)

      n_mfcc = 10
      n_mels = 40
      n_fft = 512
      win_length = 480
      hop_length = 320
      fmin = 20
      fmax = 4000
      sr = 16000

      melkwargs = {"n_fft": n_fft, "n_mels": n_mels, "win_length": win_length, "hop_length": hop_length,
                   "f_min": fmin, "f_max": fmax, "center": False, "norm":None}
      mfcc_torch_log = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                                                  dct_type=2, norm='ortho', log_mels=True,
                                                  melkwargs=melkwargs)(torch.from_numpy(audio.numpy()))
      mfcc_torch_log = np.moveaxis(mfcc_torch_log.numpy(), 0, -1)
      mfcc_torch_log = np.expand_dims(mfcc_torch_log, axis=0)
      tmp_audio_post.append(mfcc_torch_log)

  return tmp_audio_post, tmp_label

class SpeechCommands(torch.utils.data.Dataset):
    def __init__(self, audio, label):
      super().__init__()
      # images with person and train in the filename
      self.audio = audio
      self.label = label

    def __getitem__(self, index):
      speech = self.audio[index]
      command = self.label[index]
      return speech, command

    def __len__(self):
      return len(self.audio)


def get_benchmark(ds_train, ds_val, ds_test, model_settings):
  tmp_audio = []
  tmp_label = []

  with tqdm(total=len(ds_train), unit="batch") as tepoch:
    tepoch.set_description(f"audio/labels from ds_train: ")
    for i in ds_train:
      tepoch.update(1)
      tmp_audio.append(i[0].numpy())
      tmp_label.append(i[1].numpy())

  tmp_audio, tmp_label = preprocess_pytorch(tmp_audio, tmp_label, model_settings)
  train_set = SpeechCommands(tmp_audio, tmp_label)

  tmp_audio = []
  tmp_label = []

  with tqdm(total=len(ds_val), unit="batch") as tepoch:
    tepoch.set_description(f"audio/labels from ds_val: ")
    for i in ds_val:
      tepoch.update(1)
      tmp_audio.append(i[0].numpy())
      tmp_label.append(i[1].numpy())

  tmp_audio, tmp_label = preprocess_pytorch(tmp_audio, tmp_label, model_settings)
  val_set = SpeechCommands(tmp_audio, tmp_label)

  tmp_audio = []
  tmp_label = []

  with tqdm(total=len(ds_test), unit="batch") as tepoch:
    tepoch.set_description(f"audio/labels from ds_test: ")
    for i in ds_test:
      tepoch.update(1)
      tmp_audio.append(i[0].numpy())
      tmp_label.append(i[1].numpy())

  tmp_audio, tmp_label = preprocess_pytorch(tmp_audio, tmp_label, model_settings)
  test_set = SpeechCommands(tmp_audio, tmp_label)

  return train_set, val_set, test_set

def get_dataloaders(config, train_set, val_set, test_set):
      train_loader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=config['batch_size'],
                                                 shuffle=True,
                                                 num_workers=config['num_workers'])
      val_loader = torch.utils.data.DataLoader(val_set,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=config['num_workers'])
      test_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=config['batch_size'],
                                                shuffle=False,
                                                num_workers=config['num_workers'])
      return train_loader, val_loader, test_loader

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()
  ds_train, ds_test, ds_val = get_training_data(Flags)

  for dat in ds_train.take(1):
    print("One element from the training set has shape:")
    print(f"Input tensor shape: {dat[0].shape}")
    print(f"Label shape: {dat[1].shape}")
