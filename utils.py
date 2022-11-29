import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import networkx as nx
from keras.utils import to_categorical

from Conv1D import create_Conv1D
from GAT import create_GAT
from GCN import create_GCN
from MLP import create_MLP


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


def normalize(df):
    df["user_followers"] = np.log10(df["user_followers"] + 1e-5)
    df["user_ntweet"] = np.log10(df["user_ntweet"] + 1e-5)
    df = df.drop(["hashtag", "text", "time", "screen_name", "favorite", "engagement", "retweet", "id"], axis=1)
    for col in df.columns:
        if not isinstance(df[col].values[0], str):
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def extract_graph(g, df):
    mapping_graph = {k: v for v, k in enumerate(g.nodes)}
    g = nx.relabel_nodes(g, mapping_graph)
    edges = np.array(list(g.edges)).T
    edges_weight = [x[2]["weight"] for x in g.edges(data=True)]
    features_names = set(df.columns) - {"n_emojis", "user_following", "official_source", "class"}
    node_features = tf.cast(
        df.sort_index()[features_names].to_numpy(), dtype=tf.dtypes.float32
    )
    graph_info = (node_features, edges, edges_weight)
    return graph_info


def select_params(Model_type, X_train, y_train, X_test, y_test, df, g, num_classes=2, num_epochs=300):
    num_classes = num_classes
    num_epochs = num_epochs
    dropout_rate = None
    num_layers = None
    num_heads = None
    if Model_type == 'GCN':
        hidden_units = [16]
        dropout_rate = 0.3
        learning_rate = 0.1
        batch_size = 256
        input = np.array(X_train.index)
        target = to_categorical(y_train)
        loss = keras.losses.CategoricalCrossentropy
        optimizer = keras.optimizers.Adam
        input_test = np.array(X_test.index)
        target_test = y_test
        graph_info = extract_graph(g, df)
        model = create_GCN(graph_info, num_classes, hidden_units, dropout_rate)
    if Model_type == 'MLP':
        hidden_units = [32, 32]
        learning_rate = 0.01
        dropout_rate = 0.5
        batch_size = 256
        loss = keras.losses.CategoricalCrossentropy
        input = X_train
        target = to_categorical(y_train)
        input_test = X_test
        target_test = y_test
        optimizer = keras.optimizers.Adam
        model = create_MLP(X_train.shape[1], hidden_units, num_classes, dropout_rate)
    if Model_type == 'Conv1D':
        hidden_units = 64
        learning_rate = 0.1  # Original 0.1
        batch_size = 256
        model = create_Conv1D(num_classes, hidden_units, X_train.shape[1])
        input = X_train.values.reshape(-1, X_train.shape[1], 1)
        loss = keras.losses.CategoricalCrossentropy
        target = to_categorical(y_train)
        optimizer = keras.optimizers.Adam
        input_test = X_test
        target_test = y_test
    if Model_type == 'GAT':
        hidden_units = 100
        num_heads = 2
        num_layers = 1
        batch_size = 64
        learning_rate = 1e-2
        graph_info = extract_graph(g, df)
        input = np.array(X_train.index)
        target = to_categorical(y_train)
        model = create_GAT(graph_info[0], graph_info[1].T, hidden_units, num_heads, num_layers, num_classes)
        loss = keras.losses.CategoricalCrossentropy
        optimizer = keras.optimizers.SGD
        input_test = np.array(X_test.index)
        target_test = y_test
    return hidden_units, num_classes, learning_rate, num_epochs, dropout_rate, batch_size, num_layers, num_heads, input, target, loss, optimizer, input_test, target_test, model
