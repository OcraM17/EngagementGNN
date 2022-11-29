from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from utils import create_ffn


def MLP(input_shape, hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(input_shape,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block.
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
        # Add skip connection.
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
    # Compute logits.
    logits = layers.Dense(num_classes, name="logits")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logits, name="baseline")


def create_MLP(input_shape, hidden_units, num_classes, dropout_rate):
    model = MLP(input_shape, hidden_units, num_classes, dropout_rate)
    return model
