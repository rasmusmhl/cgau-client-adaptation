import pickle
import collections
import sys

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from paths import paths
from utilities import tsprint

def load_data(config):
  tsprint('Begin load_data: loading yamnet embeddings and determine labels.', 1)
  

  with open(paths['cifar10_embeddings'], 'rb') as file:
    embeddings = pickle.load(file)
  with open(paths['cifar10_labels'], 'rb') as file:
    labels = pickle.load(file)


  data = {}
  X = []; y = []; p = [];
  for partition in ['train', 'val','test']:


    X_current = embeddings[partition]

    y_current =to_categorical(labels[partition],num_classes =10)
    p_current = np.asarray(len(y_current)*[partition])
    X.append(X_current);y.append(y_current);p.append(p_current);
  X = np.vstack(X);y = np.vstack(y);p = np.hstack(p)

  sample_client_id = simulate_clients(X, y, config)
  
  
  data = {}
  for partition in np.unique(p):
    client_data_dict = {}
    for current_client in np.unique(sample_client_id):
      client_filter = (sample_client_id == current_client)
      partition_filter = (p == partition)
      idx = np.logical_and(client_filter, partition_filter)
      
      client_data_dict[str(current_client)] = {'features': X[idx, :], 'label': y[idx]}
      if config.global_conditioning:
        batch_client_id = np.ones(shape=(X[idx, :].shape[0],), dtype=int)*current_client
        batch_client_id = tf.one_hot(batch_client_id, depth=config.num_clients)
        client_data_dict[str(current_client)]['client_id'] = batch_client_id
    data[partition] = tff.simulation.FromTensorSlicesClientData(client_data_dict)
  
  example_dataset = data['train'].create_tf_dataset_for_client(data['train'].client_ids[0])
  preprocessed_example_dataset = preprocess(config, example_dataset)
  
  tsprint('Done load_data: returning data dict.', 1)
  return data, preprocessed_example_dataset

def simulate_clients(X, y, config):
  num_samples = len(y)
  
  if config.simulated_client_dim_reduction == 'pca':
    dimensionality_reduction = PCA(n_components=2)
  elif config.simulated_client_dim_reduction == 'tsne':
    dimensionality_reduction = TSNE(n_components=2, random_state=config.seed)
                                                                
  X_reduced = dimensionality_reduction.fit_transform(X)
  
  sample_client_id = np.zeros(num_samples) # an ID that specifies which client each sample should go to
  client_ids = np.arange(config.num_clients, dtype=int) 
  
  for current_class in range(config.num_classes):
    idx = (y[:,current_class] == 1)
    x = X_reduced[idx, :]
    num_samples_class = x.shape[0]
    if num_samples_class < config.num_clients:
      tsprint(f'Number of samples {num_samples_class} is smaller than number of clients {config.num_clients} for class {config.num_samples_class}. Randomly assigning samples to clients.', 3)
      sample_client_id[idx] = np.random.choice(client_ids, size=num_samples_class)
    else:  
      tsprint(f'Making simulated client dataset with {config.num_clients} clients.', 3)
      kmeans = KMeans(n_clusters=config.num_clients, random_state=config.seed).fit(x)
      sample_client_id[idx] = kmeans.labels_
                                                 
  tsprint(f'Randomly shuffle with percentage {config.label_shuffle_percentage}.', 2)
  num_to_shuffle = int(num_samples*config.label_shuffle_percentage)
  choices = np.random.choice(np.arange(num_samples), size=num_to_shuffle, replace=False)
  choices_permute = np.random.permutation(choices)
  sample_client_id[choices] = sample_client_id[choices_permute]
  return sample_client_id

def preprocess(config, dataset, final_eval=False):
  def batch_format_fn(element):
    if config.global_conditioning:
      x = (tf.reshape(element['features'], [-1, config.embedding_size]), tf.reshape(element['client_id'], [-1, config.num_clients]))
    else:
      x = tf.reshape(element['features'], [-1, config.embedding_size])
    y = tf.reshape(tf.cast(element['label'], tf.int32), [-1, 10])
    
    return collections.OrderedDict(x = x, y = y)    

  ds = dataset.batch(config.batch_size).map(batch_format_fn)
  if final_eval:
    ds = ds.repeat(1)
  else:
    ds = ds.shuffle(config.shuffle_buffer).take(config.num_local_steps)
    
  return ds.prefetch(config.prefetch_buffer)
  
def subsample_clients(config, client_ids):
  return np.random.choice(client_ids, config.num_sampled_clients)

def make_federated_data(config, client_data, client_ids, final_eval=False):
  return [
      preprocess(config, client_data.create_tf_dataset_for_client(x), final_eval)
      for x in client_ids
  ]