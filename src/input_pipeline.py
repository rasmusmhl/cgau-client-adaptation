import tensorflow as tf
import audio_utilities
import params

import sys

import scipy
import numpy as np
import numpy.matlib

class InputPipeline:
    def __init__(self, batch_size, buffer_size, *args, **kwargs):
        self.feature_description = {
            'sound_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        self.labels_cough = None
        self.labels_dry = None
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
    
    def setup_paths(self, paths):
        self.paths = paths
        
    def setup_labels_cough(self, labels_cough):
        self.labels_cough = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(list(labels_cough.keys())),
                    values=tf.constant(list(labels_cough.values())),
                ),
                default_value=tf.constant(-1),
                name="cough_class"
            )

    def parse_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        out = tf.io.parse_single_example(example_proto, self.feature_description)
        yid = out['id']
        out = tf.io.decode_raw(out['sound_raw'], tf.int16)
        out = tf.math.divide(tf.dtypes.cast(out, tf.float32), 32768)
        
        # TODO: fix this, but it's needed to circumvent dataloss error with fsd...
        if len(out) > 160000:
          out = out[:160000]
        
        return (out, yid), self.labels_cough.lookup(yid)
      
    def shuffle_repeat_prefetch_map(self, ds, buffer_size=100, repeat=None):
        ds = (ds.shuffle(buffer_size, seed=1)
              .repeat(repeat)
              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
              .map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
             )
        return ds

    def balance_batching(self, ds):
        neg_filter = lambda X, y: tf.reduce_any(y==0)
        pos_filter = lambda X, y: tf.reduce_any(y==1)
        ds = tf.data.Dataset.zip((ds.filter(neg_filter), ds.filter(pos_filter)))

        map_datasets = lambda neg, pos: tf.data.Dataset.from_tensors(neg).concatenate(tf.data.Dataset.from_tensors(pos))                         
        ds = ds.flat_map(map_datasets)
        return ds

    def validation_split(self, ds, validation_percentage=0.05):
        dataset_size = ds.reduce(0, lambda x, _: x + 1)
        ds = ds.shuffle(dataset_size.numpy(), reshuffle_each_iteration=False)

        val_size = int(float(dataset_size) * validation_percentage)
        train_size = dataset_size-val_size

        ds_val = ds.take(val_size)
        ds_train = ds.skip(val_size)

        return [ds_train, ds_val], [train_size, val_size]

    def make_datasets(self):
        drop_remainder = True
        print(f'Drop remainder: {drop_remainder}')
        
        datasets = {}
        
	##############################################
	# Part related to private data removed here. #
	# Code might not run/run as intended.	     #
	##############################################

        ds_fsd_train = tf.data.TFRecordDataset(filenames = [self.paths['fsd_tfrec_train']])

        [ds_fsd_train, ds_fsd_val], [fsd_train_size, fsd_val_size] = self.validation_split(ds_fsd_train)

        
        fsd_padding_shape = (([160000,], []), [])

        print(f'FSD dataset padding (cropping if needed!): {fsd_padding_shape}')
        
        fsd_train_size = ds_fsd_train.reduce(0, lambda x, _: x + 1)
        ds_fsd_train = self.shuffle_repeat_prefetch_map(ds_fsd_train)
        ds_fsd_train = self.balance_batching(ds_fsd_train) 
        #ds_fsd_train = ds_fsd_train.batch(1) 
        ds_fsd_train = ds_fsd_train.padded_batch(self.batch_size, 
                                                 padded_shapes=fsd_padding_shape,
                                                 drop_remainder=drop_remainder)

        ds_fsd_val = self.shuffle_repeat_prefetch_map(ds_fsd_val)
        ds_fsd_val = self.balance_batching(ds_fsd_val) 
        #ds_fsd_val = ds_fsd_val.batch(1)
        ds_fsd_val = ds_fsd_val.padded_batch(self.batch_size, 
                                             padded_shapes=fsd_padding_shape,
                                             drop_remainder=drop_remainder)

        ds_fsd_test = tf.data.TFRecordDataset(filenames = [self.paths['fsd_tfrec_test']])
        fsd_test_size = ds_fsd_test.reduce(0, lambda x, _: x + 1)
        ds_fsd_test = self.shuffle_repeat_prefetch_map(ds_fsd_test, repeat=1)

        # This is potentially problematic, since we're "test-time augmenting zeros", maybe add mask?:
        ds_fsd_test = ds_fsd_test.padded_batch(self.batch_size, 
                                               padded_shapes=fsd_padding_shape, 
                                               drop_remainder=drop_remainder)

        datasets['fsd'] = {}
        datasets['fsd']['train'] = {'ds': ds_fsd_train, 'size': fsd_train_size}
        datasets['fsd']['val'] = {'ds': ds_fsd_val, 'size': fsd_val_size} 
        datasets['fsd']['test'] = {'ds': ds_fsd_test, 'size': fsd_test_size}

        self.datasets = datasets

    def remove_fids(self, ds):
        return ds.map(lambda X_and_fid, y: (X_and_fid[0],y))

    def make_single_file_ds(self, file_id=None):
	##############################################
	# Part related to private data removed here. #
	# Code might not run/run as intended.	     #
	##############################################
        ds = shuffle_repeat_prefetch_map(ds)
        ds = ds.filter(lambda X_and_fid, y: X_and_fid[1]==file_id')     
        return ds
