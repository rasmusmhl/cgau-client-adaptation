import pickle
import collections
import sys

import tensorflow as tf
import tensorflow_federated as tff

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from paths import paths
from utilities import tsprint
from evaluation import frechet_distance

import wandb

def load_data(config):
  tsprint('Begin load_data: loading yamnet embeddings and determine labels.', 1)
  
  with open(paths['yamnet_embeddings'], 'rb') as file:
    embeddings = pickle.load(file)

  labels = {}
  df_train_labels = pd.read_csv(paths['fsd_train_labels_path'])
  df_test_labels = pd.read_csv(paths['fsd_test_labels_path'])
  for df in [df_train_labels, df_test_labels]:
    for fname, label in zip(list(df["fname"]), list(df["label"])):
      fsdid = fname.split('.')[0]
      labels[fsdid] = int(label == 'Cough')

    
  X = []; y = []; p = [];
  for partition, data in embeddings['fsd'].items():
    for k, v in data.items():
      X.append(v); y.append(labels[k]); p.append(partition);
  X = np.asarray(X); y = np.asarray(y); p = np.asarray(p)
  sample_client_id = simulate_clients(X, y, p, config)

  # Determine the Frechét distances between clients and rest of data
  client_frechet_distances = []
  frechet_dict = {}
  for current_client in np.unique(sample_client_id):
    client_filter = (sample_client_id == current_client)
    not_client_filter = (sample_client_id != current_client)
      
    distance = frechet_distance(X[not_client_filter, :], X[client_filter, :])
    client_frechet_distances.append(distance)
    
    frechet_dict[str(int(current_client))] = distance

  mu = np.mean(client_frechet_distances)
  std = np.std(client_frechet_distances)
  tsprint(f'Client average Frechét distances: {mu:.02f}) +/- {std:.02f}', 1)
  tsprint(f'Client Frechét distances: {client_frechet_distances}', 1)
  
  client_average_frechet = mu
  
  data = {}
  for partition in np.unique(p):
    client_data_dict = {}
    for current_client in np.unique(sample_client_id):
      client_filter = (sample_client_id == current_client)
      partition_filter = (p == partition)
      idx = np.logical_and(client_filter, partition_filter)
      
      label_distribution = [np.sum(y[idx] == 0), np.sum(y[idx] == 1)]
      tsprint(f'Client {current_client}, {partition} label distribution: {label_distribution}.',1)
      client_id = str(int(current_client))
      client_data_dict[client_id] = {'features': X[idx, :], 'label': y[idx]}
      if config.global_conditioning:
        batch_client_id = np.ones(shape=(X[idx, :].shape[0],), dtype=int)*current_client
        batch_client_id = tf.one_hot(batch_client_id, depth=config.num_clients)
        client_data_dict[client_id]['client_id'] = batch_client_id

    data[partition] = tff.simulation.FromTensorSlicesClientData(client_data_dict)
  
  example_dataset = data['train'].create_tf_dataset_for_client(data['train'].client_ids[0])
  preprocessed_example_dataset = preprocess(config, example_dataset)
  
  tsprint('Done load_data: returning data dict.', 1)
  return data, preprocessed_example_dataset, client_average_frechet

def simulate_clients(X, y, p, config):
  num_samples = len(y)
  
  if config.simulated_client_dim_reduction == 'pca':
    dimensionality_reduction = PCA(n_components=2)
  elif config.simulated_client_dim_reduction == 'tsne':
    dimensionality_reduction = TSNE(n_components=2, random_state=config.seed)

  train_filter = (p == 'train')
  X_train = X[train_filter, :]
  dimensionality_reduction.fit(X_train)
  X_reduced = dimensionality_reduction.transform(X)
  
  sample_client_id = np.zeros(num_samples) # an ID that specifies which client each sample should go to
  client_ids = np.arange(config.num_clients, dtype=int) 
  
  for current_class in range(config.num_classes):
    class_filter = (y == current_class)
    class_train_filter = np.logical_and(class_filter, train_filter)
    num_train_samples_class = np.sum(class_train_filter)
    if num_train_samples_class < config.num_clients:
      tsprint(f'Number of training samples {num_train_samples_class} is smaller than number of clients {config.num_clients} for class {current_class}.', 3)
      tsprint(f'Randomly assigning samples to clients.', 3)
      sample_client_id[class_filter] = np.random.choice(client_ids, size=np.sum(class_filter))
    else:  
      X_reduced_train_current_class = X_reduced[class_train_filter, :]
      tsprint(f'Making simulated client dataset with {config.num_clients} clients.', 3)
      kmeans = KMeans(n_clusters=config.num_clients, random_state=config.seed).fit(X_reduced_train_current_class)
      sample_client_id[class_filter] = kmeans.predict(X_reduced[class_filter, :])
                                                 
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
    y = tf.reshape(tf.cast(element['label'], tf.int32), [-1, 1])
    
    return collections.OrderedDict(x = x, y = y)    

  ds = dataset.batch(config.batch_size).map(batch_format_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
  #ds = dataset.map(batch_format_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
  
  if final_eval:
    ds = ds.repeat(1)
  else:
    ds = ds.shuffle(config.shuffle_buffer).take(config.num_local_steps)
    #ds = ds.shuffle(config.shuffle_buffer).repeat(1) 
    
  return ds.prefetch(tf.data.experimental.AUTOTUNE)
  #return ds.batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  
def subsample_clients(config, client_ids):
  return np.random.choice(client_ids, config.num_sampled_clients)

def make_federated_data(config, client_data, client_ids, final_eval=False):
  return [
      preprocess(config, client_data.create_tf_dataset_for_client(x), final_eval)
      for x in client_ids
  ]