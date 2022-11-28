import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
def eng_class(x):
  if x <= 0:
    return 0
  else:
    return 1

def sampling_k_elements(group, k=103202):
  if len(group) < k:
    return group
  return group.sample(k)

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
      fnn_layers.append(layers.BatchNormalization())
      fnn_layers.append(layers.Dropout(dropout_rate))
      fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)
