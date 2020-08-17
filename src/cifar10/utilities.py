import datetime
import numpy as np
import tensorflow as tf

def tsprint(string, level=0):
  """Print a string with a UTC timestamp."""
  level_str = ''
  if level:
    for _ in range(level):
      level_str += '\t'
  print(f'{datetime.datetime.utcnow()}: {level_str} {string}.')

class dotdict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  
  
metrics_to_ignore = ['keras_training_time_client_sum_sec']
def log_metrics_train_val(train_metrics, val_metrics, step):
  '''Function for printing and logging metrics from training across a train
  and validation split somewhat neatly...'''

  mt = train_metrics._asdict()
  mv = val_metrics._asdict()
  metric_str = ''
  for name, value in mt.items():
    train_value = value
    val_value = mv[name]
    if name in metrics_to_ignore:
      continue
    if name == 'num_examples':
      metric_str += f' {name:<5}: {train_value:.0f}/{val_value:.0f}'
    else:
      metric_str += f' {name:<5}: {train_value:.3f}/{val_value:.3f}'
  
  tsprint(f'Global step {step:2d} | Metrics: {metric_str}', 0)
  
def log_metrics(split, metrics, step):
  '''Function for printing and logging metrics.'''
  string = f'{split}'
  metric_str = ''
  for name, value in metrics._asdict().items():
    if name in metrics_to_ignore:
      continue
    metric_str += f'{name}: {value:.4f} \t'
  tsprint(f'{string:<5} \t | Global step {step:2d} \t | Metrics: {metric_str}', 0)