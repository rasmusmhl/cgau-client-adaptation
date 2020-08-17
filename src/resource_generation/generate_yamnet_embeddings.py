import sys
import os
import importlib
import pickle

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib as mpl 
prop = fm.FontProperties(fname='Roboto-Thin.ttf', size=30)
import seaborn as sns
sns.set()

import IPython # !pip install IPython==7.12 #Factory reset runtime

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

import audio_utilities

# Load the paths to data, labels and models
from paths import paths

print(tf.__version__)
print(IPython.__version__)
ipd = IPython.display
physical_devices = tf.config.list_physical_devices('GPU') 
num_gpus = len(physical_devices)
for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True) 
print(f'Number of physical devices: {num_gpus}')

##############################################
# Part related to private data removed here. #
# Code might not run/run as intended.	     #
##############################################

labels_cough = {}


df_train_labels = pd.read_csv(paths['fsd_train_labels_path'])
df_test_labels = pd.read_csv(paths['fsd_test_labels_path'])
for fname, label in zip(list(df_train_labels["fname"]), list(df_train_labels["label"])):
    fsdid = fname.split('.')[0]
    labels_cough[fsdid] = int(label == 'Cough')
for fname, label in zip(list(df_test_labels["fname"]), list(df_test_labels["label"])):
    fsdid = fname.split('.')[0]
    labels_cough[fsdid] = int(label == 'Cough')
    
import importlib
import input_pipeline
importlib.reload(input_pipeline)

ip = input_pipeline.InputPipeline(batch_size=32, buffer_size=100)
ip.setup_paths(paths)
ip.setup_labels_cough(labels_cough)
ip.make_datasets()

import params
import yamnet as yamnet_model
import importlib
importlib.reload(yamnet_model)
import tflite_compat
importlib.reload(tflite_compat)

params.BATCH_SIZE = ip.batch_size
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights(paths['yamnet_weights'])
yamnet_classes = yamnet_model.class_names(paths['yamnet_classes'])

embeddings = {}
# This loops in a slighly stupid way, because the yamnet 
# requires fixed length and fixed batch size - for tflite
# compatibility. So, it repeats until all ids have randomly
# not been dropped in making fixed batch_sizes. Also, since
# the length has to match the 10s of audioset, the FSD data
# is either croppoed to 10s or padded with zeros. 
for ds in ['fsd']:
    embeddings[ds] = {}
    print(f'{ds}')
    for split in ['train', 'test', 'val']:
        print(f'\t{split}')
        embeddings[ds][split] = {}
        n_processed = 0
        for (Xs, fids), ys in ip.datasets[ds][split]['ds']:
            pred, feats = yamnet.predict(Xs)
            for i, fid in enumerate(fids):
                fid = fid.numpy().decode()
                embeddings[ds][split][fid] = feats[i, ...]
                n_processed += 1
            progress = n_processed / int(ip.datasets[ds][split]['size'])
            print(f'\t\t {progress}')
            condition = len(embeddings[ds][split]) == int(ip.datasets[ds][split]['size'])
            if condition:
                break

import pickle

with open('./resources/yamnet_embeddings.pickle', 'wb') as file:
    pickle.dump(embeddings, file)
