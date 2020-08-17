import sys
import os
import collections
import datetime
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
tfk = tf.keras
tfkl = tfk.layers
tf.compat.v1.enable_v2_behavior()

from utilities import dotdict, tsprint, log_metrics_train_val, log_metrics
from paths import paths
from data import load_data, subsample_clients, make_federated_data
from model import Model 

import wandb
from wandb.keras import WandbCallback

# Constrain number of CPU threads
tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv('OMP_NUM_THREADS')))
tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv('OMP_NUM_THREADS')))

def main(config):
  tsprint(f'Using seed: {config.seed}.')
  tsprint('Loading data and simulating clients.')
  data, preprocessed_example_dataset = load_data(config)

  
  tsprint('Setting up model definition based on sample dataset.')
  client_optimizer_fn = tf.keras.optimizers.SGD(learning_rate=config.client_learning_rate,
                                                momentum=config.client_learning_rate_momentum,
                                                decay=config.client_learning_rate_decay)
  m = Model(config, preprocessed_example_dataset)
  print("***************************\n MODEL DONE")
  tsprint('Setup federated learning process.')
  iterative_process = tff.learning.build_federated_averaging_process(
    m.get_tff_model,
    client_optimizer_fn=lambda: client_optimizer_fn)
  state = iterative_process.initialize()
  evaluation = tff.learning.build_federated_evaluation(m.get_tff_model)
  
  
  tsprint('Beginning federated training.', 1)
  # This is overwritten at each step, if the number of sampled clients is smaller than
  # the total number of clients.
  federated_train_data = make_federated_data(config, data['train'], data['train'].client_ids)
  federated_val_data = make_federated_data(config, data['val'], data['val'].client_ids)

  global_step = 0
  best_loss = 1e6
  for _ in range(config.max_num_fl_rounds):
    global_step += 1

    if config.num_sampled_clients < config.num_clients:
      sample_clients = subsample_clients(config, data['train'].client_ids)
      federated_train_data = make_federated_data(config, data['train'], sample_clients)
      federated_val_data = make_federated_data(config, data['val'], sample_clients)

    state, train_metrics = iterative_process.next(state, federated_train_data) 
    val_metrics = evaluation(state.model, federated_val_data)
    log_metrics_train_val(train_metrics, val_metrics, global_step)

    if val_metrics.loss < best_loss:
      tsprint(f'Updating best state (previous best: {best_loss}, new best: {val_metrics.loss}.', 2)
      best_loss, best_state, best_step = val_metrics.loss, state, global_step
      
    for split_str, metrics in zip(('train','validation'), (train_metrics, val_metrics)):
      for metric_str, metric in zip(('loss','accuracy', 'auc'), (metrics.loss,  metrics.categorical_accuracy, metrics.auc)):
        wandb.log({f'{split_str}/{metric_str}': metric}, step=global_step)
        
  tsprint('Begin final evaluation.', 1)
  for (split_str, ds) in data.items():
    federated_data = make_federated_data(config, ds, ds.client_ids, final_eval=True)
    metrics = evaluation(best_state.model, federated_data)
    log_metrics(split_str, metrics, global_step)
    for metric_str, metric in zip(('loss','accuracy', 'auc'), (metrics.loss,  metrics.categorical_accuracy, metrics.auc)):
      wandb.log({f'final_{split_str}/{metric_str}': metric}, step=global_step)
    

    
if __name__ == '__main__':  

  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_group', type=str)
  
  parser.add_argument('--label_shuffle_percentage', type=float)
  parser.add_argument('--num_local_steps', type=int)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--all_client_data_centrally', type=bool)
  parser.add_argument('--global_conditioning', type=bool)

  parser.add_argument('--client_learning_rate', type=float)
  parser.add_argument('--client_learning_rate_decay', type=float)
  parser.add_argument('--client_learning_rate_momentum', type=float)
  args = parser.parse_args()


  # Use Weights&Biases to keep track of experiments, 
  # TODO: something is fishy with the logging, though.
  # To turn of logging, do 'wandb off' from the terminal
  # in the federated_cough_detector folder.
  wandb.init(project="[FILL IN]", 
             entity="[FILL IN]", 
             group=args.__dict__['experiment_group'])
  
  for k,v in args.__dict__.items():
    if v is not None:
      tsprint(f'Changeing parameter {k} from default to: {v}')
      wandb.config.update({k: v}, allow_val_change=True) 
  
  config = wandb.config
  
  
  # Get and use random seed
  np.random.seed(config.seed)
  tf.random.set_seed(config.seed)

  # The config dotdict stems from the config-defaults.yaml, which
  # is automatically loaded by W&B; do not change the name of the 
  # YAML file.
  tsprint('Final config:')
  tsprint(config)
  
  main(config)
