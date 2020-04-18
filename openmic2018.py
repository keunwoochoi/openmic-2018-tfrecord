"""Entrypoint to prepare openmic-2018 dataset. See scripts/data_creation/openmicn-2018.sh for details"""
import numpy as np
import tensorflow as tf
import pandas as pd
import absl.flags, absl.app
import os
import multiprocessing
import librosa
from collections import defaultdict

absl.flags.DEFINE_string('openmic_dir', None, 'directory to load raw openmic dataset from')
absl.flags.DEFINE_string('tfrecord_dir', None, 'directory to save the tfrecord files')
absl.flags.DEFINE_integer('sampling_rate', 22050, 'target sampling rate of the created dataset')
absl.flags.DEFINE_integer('n_cpu', 0, 'Number of cpu to use')

FLAGS = absl.flags.FLAGS


def float_feature(values):
    if values is None:
        values = []
    if type(values) is float:
        values = [values]
    values = [v for v in values if v is not None]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
    if values is None:
        values = []
    if type(values) is bytes:
        values = [values]
    values = [v for v in values if v is not None]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def int64_feature(values):
    if values is None:
        values = []
    if type(values) is int:
        values = [values]
    values = [v for v in values if v is not None]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def id_to_audio_filename(sample_key: str, ext='ogg'):
    return os.path.join('audio', sample_key[:3], '%s.%s' % (sample_key, ext))


def load_audio(path, sr, mono=True):
    src, _ = librosa.load(path, sr=sr, mono=mono)

    # reshape it to (length, channel)
    ndim = src.ndim
    if ndim == 1:
        src = np.expand_dims(src, 1)
    elif ndim == 2:
        src = src.T
    return src


class OpenmicDataset:
    def __init__(self, path, train=True):
        """
        Args:
            path (str): e.g., gs://whatever/datasets/openmic/openmic-2018,
                one that has `audio/`, `partitions/`, etc.
            train (bool): if train set (or test set)

        """
        self.path = path
        self.train = train
        fn_id_csv = 'split01_train.csv' if train else 'split01_test.csv'
        self.df_id = pd.read_csv(os.path.join(path, 'partitions', fn_id_csv), header=None)

        df_label = pd.read_csv(os.path.join(path, 'openmic-2018-aggregated-labels.csv'), header=0)

        self.insts = sorted(df_label.instrument.unique().tolist())
        self.inst_to_idx = {val: key for key, val in enumerate(self.insts)}
        self.n_inst = df_label.instrument.nunique()

        self.df_label = df_label[df_label.sample_key.isin(self.df_id[0])]

    def __len__(self):
        return len(self.df_id)  # number of tracks

    def __getitem__(self, item):
        """
        Args:
            item (int):

        Returns:
            id (str): id of the track
            audio path (str): path to load the audio from
            target (1D numpy array of float32): target n-hot vector
            inst_idxs (list): list of inst index
        """
        id = self.df_id[0][item]  # e.g., '000321_218880'

        target, inst_idxs = self._id_to_target_vector(id)
        audio_path = os.path.join(self.path, id_to_audio_filename(id))
        return id, audio_path, target, inst_idxs  # str, str, numpy arrays, list

    def _id_to_target_vector(self, id):
        target = np.zeros([self.n_inst,], dtype=np.float32)
        inst_idxs = []
        for label in self.df_label[self.df_label.sample_key == id].instrument.to_numpy():
            inst_idx = self.inst_to_idx[label]

            inst_idxs.append(inst_idx)
            target[inst_idx] = 1.0

        return target, inst_idxs


def openmic_item_to_tfexample(id, audio_path, label, inst_idxs, sr):
    """
    Args:
        id (str): id of track
        audio_path (str):
        label (np.array): n-hot-vector e.g., [0.0, 1.0, 1.0, 0.0. 0.0. 0.0.. ]
        inst_idxs (list): list of inst index.  e.g., [1, 2]
        sr (int): sampling rate
    """
    audio = load_audio(audio_path, sr)

    feature = {
        'track_id': bytes_feature(id.encode('utf-8')),
        'audio': bytes_feature(
            [tf.audio.encode_wav(audio, sample_rate=sr).numpy()]
        ),
        'nhot_vector': float_feature(
            label.tolist()
        ),
        'inst_idxs': int64_feature(inst_idxs),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(args):
    """
    Write a tfrecord file with tfrecords that are specified by the given arguments

    Args:
         args (tuple): a tuple that is unpacked to
            audio_paths (list of str): paths of audio files to load
            targets (list of 1D numpy arrays): list of 20-len n-hot-vectors
            inst_idxss (list of list): list of lists, each is instrument indices (int)
            tfr_path (str): path to save the resulting tfrecord file
            sr_save (int): target sampling rate
    """
    audio_paths, targets, inst_idxss, tfr_path, sr_save = args

    with tf.io.TFRecordWriter(tfr_path) as out:
        for audio_path, target, inst_idxs in zip(audio_paths, targets, inst_idxss):
            example = openmic_item_to_tfexample(audio_path, target, inst_idxs, sr_save)
            out.write(example.SerializeToString())

    print('Writing %s: done' % tfr_path)


def main(_):
    openmic_dir = FLAGS.openmic_dir
    tfrecord_dir = FLAGS.tfrecord_dir
    sr_save = FLAGS.sampling_rate

    set_names = ['train', 'test']

    items_per_file = 200  # with sr=22050, each tfrecord file will be about 167MB

    n_cpu = FLAGS.n_cpu if FLAGS.n_cpu else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpu)
    zipped_args = []

    for set_name in set_names:
        dataset = OpenmicDataset(openmic_dir, train=set_name == 'train')
        num_tracks = len(dataset)
        num_tfrecords = int(np.ceil(num_tracks / items_per_file))
        file_splits = defaultdict(list)

        for track_idx in range(len(dataset)):
            tfrecord_idx = track_idx % num_tfrecords
            file_splits[tfrecord_idx].append(track_idx)

        for tfrecord_idx, track_idxs in file_splits.items():
            tfr_path = os.path.join(tfrecord_dir, '%s_%d.tfrecords' % (set_name, tfrecord_idx))
            audio_paths, targets, inst_idxss = [], [], []

            for track_idx in track_idxs:
                audio_path, target, inst_idxs = dataset[track_idx]
                audio_paths.append(audio_path)
                targets.append(target)
                inst_idxss.append(inst_idxs)

            zipped_args.append((audio_paths, targets, inst_idxss, tfr_path, sr_save))

    print('Multiprocessing job is going to launched for %d tfrecord files...' % len(zipped_args))
    pool.map(write_tfrecord, zipped_args)


if __name__ == '__main__':
    absl.app.run(main)
