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

import simple_fedavg_tff

# Constrain number of CPU threads
tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv('OMP_NUM_THREADS')))
tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv('OMP_NUM_THREADS')))

def main(config):
  tsprint(f'Using seed: {config.seed}.')
  tsprint('Loading data and simulating clients.')
  data, preprocessed_example_dataset, client_average_frechet = load_data(config)
  
  tsprint('Setting up model definition based on sample dataset.')
  #client_optimizer_fn = tf.keras.optimizers.SGD(learning_rate=config.client_learning_rate,
  #                                              momentum=config.client_learning_rate_momentum,
  #                                              decay=config.client_learning_rate_decay)
  m = Model(config, preprocessed_example_dataset)
  
  tsprint('Setup federated learning process.')
  #iterative_process = tff.learning.build_federated_averaging_process(
  #  m.get_tff_model,
  #  client_optimizer_fn=lambda: client_optimizer_fn)
  #state = iterative_process.initialize()
  #evaluation = tff.learning.build_federated_evaluation(m.get_tff_model)
  def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=1.0)

  def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=config.client_learning_rate,
                                   momentum=config.client_learning_rate_momentum,
                                   decay=config.client_learning_rate_decay)
  
  def keras_evaluate(model, test_data, metrics):
    metrics_collective, metrics_clients = metrics
    results_collective = {}
    results_clients = {}
    for client_id in metrics_clients.keys():
      results_clients[client_id] = {}
    
    for key, metric in metrics_collective.items():
      metric.reset_states()
      for client_id, client_ds in zip(metrics_clients.keys(), test_data):
        metrics_clients[client_id][key].reset_states()
        for batch in client_ds: 
          preds = model(batch['x'], training=False)
          metric(batch['y'], preds)
          metrics_clients[client_id][key](batch['y'], preds)
        results_clients[client_id][key] = metrics_clients[client_id][key].result()
      results_collective[key] = metric.result()
    
    return dotdict(results_collective), results_clients
  
  iterative_process = simple_fedavg_tff.build_federated_averaging_process(
      m.get_simplefedavg_tff_model, server_optimizer_fn, client_optimizer_fn)
  state = iterative_process.initialize()
  
  
  tsprint('Beginning federated training.', 1)
  # This is overwritten at each step, if the number of sampled clients is smaller than
  # the total number of clients.
  federated_train_data = make_federated_data(config, data['train'], data['train'].client_ids)
  federated_val_data = make_federated_data(config, data['val'], data['val'].client_ids, final_eval=True)

  global_step = 0
  best_loss = 1e6
  monitoring_metrics_fns = {'loss': tf.keras.metrics.BinaryCrossentropy,}
  monitoring_metrics = {}
  monitoring_metrics_clients = {}
  for c in data['train'].client_ids:
      monitoring_metrics_clients[c] = {}
  
  for k, v in monitoring_metrics_fns.items():
    monitoring_metrics[k] = v()
    for c in data['train'].client_ids:
      monitoring_metrics_clients[c][k] = v()
      
  model = m.get_simplefedavg_tff_model()
  for _ in range(config.max_num_fl_rounds):
    global_step += 1

    if config.num_sampled_clients < config.num_clients:
      sample_clients = subsample_clients(config, data['train'].client_ids)
      federated_train_data = make_federated_data(config, data['train'], sample_clients)
      federated_val_data = make_federated_data(config, data['val'], sample_clients, final_eval=True)

    state, train_metrics = iterative_process.next(state, federated_train_data) 

    train_metrics_loss = train_metrics
    model.from_weights(state.model_weights)
    val_metrics, val_metrics_clients = keras_evaluate(model.keras_model, 
                                                      federated_val_data, 
                                                      [monitoring_metrics, monitoring_metrics_clients])
    
    tsprint(f'Round {global_step} loss: {train_metrics_loss} \t {val_metrics.loss}', 0)

    wandb.log({f'train/loss': train_metrics_loss}, step=global_step)
    wandb.log({f'validation/loss': val_metrics.loss}, step=global_step)
    
    if val_metrics.loss < best_loss:
      tsprint(f'Updating best state (previous best: {best_loss}, new best: {val_metrics.loss}.', 2)
      best_loss, best_state, best_step = val_metrics.loss, state, global_step
    
        
  tsprint('Begin final evaluation.', 1)
  wandb.log({f'client_average_frechet': client_average_frechet}, step=global_step)
  
  
  final_eval_metrics_fns = {'loss': tf.keras.metrics.BinaryCrossentropy, 
             'accuracy': tf.keras.metrics.BinaryAccuracy, 
             'auc': tf.keras.metrics.AUC}
  final_eval_metrics = {}
  final_eval_metrics_clients = {}
  for c in data['train'].client_ids:
      final_eval_metrics_clients[c] = {}
  

  for k, v in final_eval_metrics_fns.items():
    final_eval_metrics[k] = v()
    for c in data['train'].client_ids:
      final_eval_metrics_clients[c][k] = v()
  
  for (split_str, ds) in data.items():
    federated_data = make_federated_data(config, ds, ds.client_ids, final_eval=True)
    
    model.from_weights(best_state.model_weights)
    cur_metrics, cur_metrics_clients = keras_evaluate(model.keras_model,
                                 federated_data, 
                                 [final_eval_metrics, final_eval_metrics_clients])
    
    
    for k,v in cur_metrics_clients.items():
      tsprint(f'{k}: {v}', 1)
    
    for metric_str, metric in zip(('loss','auc','accuracy'), (cur_metrics.loss, cur_metrics.auc, cur_metrics.accuracy)):
      wandb.log({f'final_{split_str}/{metric_str}': metric}, step=global_step)
      tsprint(f'final_{split_str}/{metric_str}: {metric}')
    
  tsprint('Done.')
    
if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_group', type=str)
  
  parser.add_argument('--label_shuffle_percentage', type=float)
  parser.add_argument('--seed', type=int, default=1)
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
