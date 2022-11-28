import networkx as nx
import pandas as pd
from utils import eng_class, sampling_k_elements
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import gc
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from GCN import create_GCN
from MLP import create_MLP
from Conv1D import create_Conv1D
from GAT import create_GAT
from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from Training import run_experiment
from Evaluation import evaluate
from utils import eng_class

def main(COMPUTE_BERT=False,EXTRACT_BERT=False, PCA = False, Model_Type = 'MLP' ):
    g = nx.read_gpickle('./fist_week.pickle')
    print("POST:", len(g.nodes))
    print("ARCS:", len(g.edges))
    print("COMPONENTS:", nx.number_connected_components(g))
    if COMPUTE_BERT:
        df = pd.read_csv("./first_week.csv", lineterminator="\n")
        df["class"] = df["engagement"].apply(lambda x: eng_class(x))
        df = df.groupby('class').apply(sampling_k_elements).reset_index(drop=True)
        model = SentenceTransformer('efederici/sentence-bert-base')
        emb = model.encode(df["text"])
        if PCA:
            pca = PCA(n_components=48)
            pca.fit(emb)
            pca_emb = pca.transform(emb)
            df = pd.concat([df, pd.DataFrame(emb)], axis=1)
        df["user_followers"] = np.log10(df["user_followers"] + 1e-5)
        df["user_ntweet"] = np.log10(df["user_ntweet"] + 1e-5)
        df = df.drop(["hashtag", "text", "time", "screen_name", "favorite", "engagement", "retweet", "id"], axis=1)
        for col in df.columns:
            if not isinstance(df[col].values[0], str):
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        del emb,model
    elif EXTRACT_BERT:
        df = pd.read_csv("./first_week_posts_bert.csv")
    else:
        df = pd.read_csv("./first_week.csv", lineterminator="\n")
        df["class"] = df["engagement"].apply(lambda x: eng_class(x))
        df = df.groupby('class').apply(sampling_k_elements).reset_index(drop=True)
        df["user_followers"] = np.log10(df["user_followers"] + 1e-5)
        df["user_ntweet"] = np.log10(df["user_ntweet"] + 1e-5)
        df = df.drop(["hashtag", "text", "time", "screen_name", "favorite", "engagement", "retweet", "id"], axis=1)
        for col in df.columns:
            if not isinstance(df[col].values[0], str):
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    gc.collect()
    mapping_graph = {k: v for v, k in enumerate(g.nodes)}
    g = nx.relabel_nodes(g, mapping_graph)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["class"], axis=1), df["class"], test_size=0.2,
                                                        random_state=42, stratify=df["class"])
    #edges = np.array(list(g.edges)).T
    #edges_weight = [x[2]["weight"] for x in g.edges(data=True)]
    #features_names = set(df.columns) - {"n_emojis", "user_following", "official_source", "class"}
    #node_features = tf.cast(
    #    df.sort_index()[features_names].to_numpy(), dtype=tf.dtypes.float32
    #)
    #graph_info = (node_features, edges, edges_weight)
    if Model_Type == 'GCN':
        num_classes = 2
        hidden_units=[16]
        dropout_rate = 0.3
        learning_rate = 0.1
        num_epochs = 300
        batch_size = 256
        input = np.array(X_train.index)
        target = to_categorical(y_train)
        model = create_GCN(graph_info, num_classes, hidden_units, dropout_rate)
        loss = keras.losses.CategoricalCrossentropy
        optimizer = keras.optimizers.Adam
        input_test = np.array(X_test.index)
        target_test = y_test
    if Model_Type == 'MLP':
        hidden_units = [32, 32]
        learning_rate = 0.01
        dropout_rate = 0.5
        num_epochs = 300
        batch_size = 256
        num_classes = 2
        model = create_MLP(X_train.shape[1], hidden_units, num_classes, dropout_rate)
        loss = keras.losses.CategoricalCrossentropy
        input = X_train
        target = to_categorical(y_train)
        input_test = X_test
        target_test = y_test
        optimizer = keras.optimizers.Adam
    if Model_Type== 'Conv1D':
        hidden_units=64
        num_classes = 2
        learning_rate = 0.1  # Original 0.1
        num_epochs = 300
        batch_size = 256
        model = create_Conv1D(num_classes, hidden_units, X_train.shape[1])
        input=X_train.values.reshape(-1, X_train.shape[1], 1)
        loss = keras.losses.CategoricalCrossentropy
        target=to_categorical(y_train)
        optimizer = keras.optimizers.Adam
        input_test = X_test
        target_test = y_test
    if Model_Type=='GAT':
        hidden_units = 100
        num_heads = 2
        num_layers = 1
        num_classes = 2
        num_epochs = 100
        batch_size = 64
        learning_rate = 1e-2
        input = np.array(X_train.index)
        target = to_categorical(y_train)
        model=create_GAT(graph_info[0], graph_info[1].T, hidden_units, num_heads, num_layers, num_classes)
        loss = keras.losses.CategoricalCrossentropy
        optimizer = keras.optimizers.SGD
        input_test = np.array(X_test.index)
        target_test = y_test
    #del edges, node_features, edges_weight, df, g
    gc.collect()
    run_experiment(model, input, target, learning_rate, loss, num_epochs, batch_size, optimizer)
    evaluate(model, input_test, target_test)

if __name__=='__main__':
    main()
