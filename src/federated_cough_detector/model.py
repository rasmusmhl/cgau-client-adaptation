import collections 
from utilities import dotdict

import tensorflow as tf
import tensorflow_federated as tff

import simple_fedavg_tf

tfk = tf.keras
tfkl = tfk.layers
tf.compat.v1.enable_v2_behavior()

class Model():
  def __init__(self, config, example_dataset):
    self.config = config
    self.element_spec = example_dataset.element_spec
  
  def create_keras_model(self):
    input_features = tfkl.Input(shape=(self.config.embedding_size,))

    if self.config.global_conditioning:
      one_hot_client_id = tfkl.Input(shape=(10,), dtype=tf.int32)
      
    x = input_features

    if self.config.global_conditioning:
      for _ in range(self.config.num_layers):
        x = tfkl.Dropout(self.config.dropout)(x)
        x1 = tfkl.Dense(self.config.hidden_layer_size)(x) 
        x1 += tfkl.Dense(self.config.hidden_layer_size)(one_hot_client_id)
        x2 = tfkl.Dense(self.config.hidden_layer_size)(x) 
        x2 += tfkl.Dense(self.config.hidden_layer_size)(one_hot_client_id)
        x = tfkl.Activation('tanh')(x1) * tfkl.Activation('sigmoid')(x2)
      x = tfkl.Dense(1)(x)
      x += tfkl.Dense(1)(one_hot_client_id)
      x = tfkl.Activation('sigmoid')(x)
      output = x
    else:
      for _ in range(self.config.num_layers):
        x = tfkl.Dropout(self.config.dropout)(x)
        x = tfkl.Dense(self.config.hidden_layer_size)(x) 
        #if self.config.global_conditioning: x += tfkl.Dense(self.config.hidden_layer_size, use_bias=False)(one_hot_client_id)
        x = tfkl.Activation('relu')(x)
      x = tfkl.Dense(1)(x)
      #if self.config.global_conditioning: x += tfkl.Dense(1, use_bias=False)(one_hot_client_id)
      x = tfkl.Activation('sigmoid')(x)
      output = x

    inputs = [input_features]
    if self.config.global_conditioning: inputs += [one_hot_client_id] #[client_id]

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
  
  def get_tff_model(self):
    keras_model = self.create_keras_model()
    class ConstantWeightLoss(tfk.losses.Loss):
      def call(self, y_true, y_pred):
        weights = tf.cast(y_true, tf.float32) * 49.0 + 1.0 
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred, weights)
    loss = ConstantWeightLoss()
    
    return tff.learning.from_keras_model(
      keras_model,
      input_spec=self.element_spec,
      loss=loss,
      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
  
  
  def get_simplefedavg_tff_model(self):
    keras_model = self.create_keras_model()
    class ConstantWeightLoss(tfk.losses.Loss):
      def call(self, y_true, y_pred):
        weights = tf.cast(y_true, tf.float32) * 49.0 + 1.0 
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred, weights)
    loss = ConstantWeightLoss()
    
    # Todo, get metrics in there...
    return simple_fedavg_tf.KerasModelWrapper(keras_model,
                                              self.element_spec, loss)