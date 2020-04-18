import os
import tensorflow as tf

_AUTOTUNE = tf.data.experimental.AUTOTUNE
_FIXED_SEED = 135


class Openmic:
    def __init__(self, path, train=True):
        """
        Args:
            path (str): where tfrecords are stored
            train (bool): if it's train set (or test set)
        """
        self.path = path
        self.train = train
        set_name = 'train' if train else 'test'
        self.pattern = '{}_*.tfrecords'.format(set_name)

    def init_dataset(self):
        def preprocess_example(example):
            audio, sr = tf.audio.decode_wav(example['audio'])
            return {'track_id': example['track_id'],
                    'audio': audio,
                    'sample_rate': sr,
                    'nhot_vector': example['nhot_vector'],
                    }

        seed = None if self.train else _FIXED_SEED

        ds = tf.data.Dataset.list_files(
            os.path.join(self.path, self.pattern), shuffle=False, seed=seed
        )
        ds = ds.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=_AUTOTUNE,
            num_parallel_calls=_AUTOTUNE,
        )

        options = tf.data.Options()
        options.experimental_deterministic = not self.train

        ds = ds.with_options(options)

        ds = ds.map(preprocess_example, num_parallel_calls=_AUTOTUNE)
        return ds

    def get_dataset(self, batch_size, shuffle=True, repeats=-1):
        """Read dataset.

        Args:
            batch_size: Size of batch.
            shuffle: Whether to shuffle the examples.
            repeats: Number of times to repeat dataset. -1 for endless repeats.

        Returns:
            A batched tf.data.Dataset.
        """
        dataset = self.init_dataset(shuffle)
        dataset = dataset.repeat(repeats)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=_AUTOTUNE)
        return dataset
