import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import networkx as nx


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
