import pandas as pd
from sentence_transformers import SentenceTransformer
import gc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from Training import run_experiment, run_experiment_XGB
from Evaluation import evaluate, evaluate_XGB
from utils import normalize, eng_class, sampling_k_elements, extract_graph, eng_mult_class
import numpy as np
import networkx as nx
from tensorflow import keras
from keras.utils import to_categorical
import random
from models.Xgboost import create_XGB
from models.Conv1D import create_Conv1D
from models.GAT import create_GAT
from models.GCN import create_GCN
from models.MLP import create_MLP
import argparse


def parse_args():
    parser = argparse.ArgumentParser("TweetGage Params")
    a = parser.add_argument
    a('--LOAD_CSV', action='store_true')
    a('--EXTRACT_BERT', action='store_true')
    a('--USE_PCA', action='store_true')
    a('--USER_FEAT', action='store_true')
    a('--BERT_FEAT', action='store_true')
    a('--Model_Type', default='GCN', type=str)
    a('--MULTI_LABEL', action='store_true')
    return parser.parse_args()


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)


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
        learning_rate = 0.1
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
    if Model_type == 'XGBOOST':
        max_depth = 8
        learning_rate = 0.025
        subsample = 0.85
        colsample_bytree = 0.35
        eval_metric = 'logloss'
        objective = 'binary:logistic'
        tree_method = 'gpu_hist'
        seed = 1
        model = create_XGB(max_depth, learning_rate, subsample,
                           colsample_bytree, eval_metric, objective,
                           tree_method, seed)
        return model
    return hidden_units, num_classes, learning_rate, num_epochs, dropout_rate, batch_size, num_layers, num_heads, input, target, loss, optimizer, input_test, target_test, model


def main(LOAD_CSV=False, EXTRACT_BERT=True, USE_PCA=False, USER_FEAT=True, BERT_FEAT=True, Model_Type='GCN', MULTI_LABEL=True):
    reset_random_seeds()
    g = nx.read_gpickle('./network_tweets.pickle')
    print("POST:", len(g.nodes))
    print("ARCS:", len(g.edges))
    print("COMPONENTS:", nx.number_connected_components(g))
    if not LOAD_CSV:
        df = pd.read_csv("./first_week.csv", lineterminator="\n")
        if MULTI_LABEL:
            ths = [(0, 1), (1, 5), (5, 1e6)]
            df["class"] = df["engagement"].apply(lambda x: eng_mult_class(x, ths))
            num_classes = len(ths)
        else:
            df["class"] = df["engagement"].apply(lambda x: eng_class(x))
            num_classes = 2
        df = df.groupby('class').apply(sampling_k_elements).reset_index(drop=True)
        if EXTRACT_BERT:
            model = SentenceTransformer('efederici/sentence-bert-base')
            emb = model.encode(df["text"])
            if USE_PCA:
                pca = PCA(n_components=48)
                pca.fit(emb)
                emb = pca.transform(emb)
            df = pd.concat([df, pd.DataFrame(emb)], axis=1)
            del emb, model
            gc.collect()
        df = normalize(df)
    else:
        df = pd.read_csv("./first_week_posts_bert.csv")
        if USER_FEAT and not BERT_FEAT:
            df = df.iloc[:, 0:11]
        if not USER_FEAT and BERT_FEAT:
            df = df.iloc[:, 10:]
        if USE_PCA:
            pca = PCA(n_components=48)
            print('PCA 48 Components')
            pca.fit(df.drop(["class"], axis=1))
            emb = pca.transform(df.drop(["class"], axis=1))
            df = pd.concat([pd.DataFrame(emb), df[["class"]]], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(["class"], axis=1), df["class"], test_size=0.2,
                                                        random_state=42, stratify=df["class"])

    if not Model_Type == 'XGBOOST':
        hidden_units, num_classes, learning_rate, num_epochs, dropout_rate, batch_size, num_layers, \
        num_heads, input, target, loss, optimizer, input_test, target_test, model = select_params(Model_Type, X_train,
                                                                                                  y_train, X_test,
                                                                                                  y_test,
                                                                                                  df,
                                                                                                  g,
                                                                                                  num_classes=num_classes,
                                                                                                  num_epochs=300)
        run_experiment(model, input, target, learning_rate, loss, num_epochs, batch_size, optimizer)
        evaluate(model, input_test, target_test)
    else:
        model = select_params(Model_Type, X_train, y_train, X_test, y_test, df, g, num_classes=num_classes,
                              num_epochs=300)
        obj = run_experiment_XGB(model, X_train, y_train)
        evaluate_XGB(obj, X_test, y_test)


if __name__ == '__main__':
    args = vars(parse_args())
    main(*list(args.values))
