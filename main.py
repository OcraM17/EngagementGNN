import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer
import gc
from sklearn.model_selection import train_test_split
from Training import run_experiment
from Evaluation import evaluate
from utils import normalize, eng_class, select_params, sampling_k_elements


def main(LOAD_CSV=True, EXTRACT_BERT=False, PCA=False, Model_Type='MLP'):
    g = nx.read_gpickle('./fist_week.pickle')
    print("POST:", len(g.nodes))
    print("ARCS:", len(g.edges))
    print("COMPONENTS:", nx.number_connected_components(g))
    if not LOAD_CSV:
        df = pd.read_csv("./first_week.csv", lineterminator="\n")
        df["class"] = df["engagement"].apply(lambda x: eng_class(x))
        df = df.groupby('class').apply(sampling_k_elements).reset_index(drop=True)
        if EXTRACT_BERT:
            model = SentenceTransformer('efederici/sentence-bert-base')
            emb = model.encode(df["text"])
            if PCA:
                pca = PCA(n_components=48)
                pca.fit(emb)
                emb = pca.transform(emb)
            df = pd.concat([df, pd.DataFrame(emb)], axis=1)
            del emb, model
            gc.collect()
        df = normalize(df)
    else:
        df = pd.read_csv("./first_week_posts_bert.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["class"], axis=1), df["class"], test_size=0.2,
                                                        random_state=42, stratify=df["class"])
    hidden_units, num_classes, learning_rate, num_epochs, dropout_rate, batch_size, num_layers, \
    num_heads, input, target, loss, optimizer, input_test, target_test, model = select_params(Model_Type, X_train,
                                                                                              y_train, X_test, y_test,
                                                                                              g, num_classes=2,
                                                                                              num_epochs=300)
    run_experiment(model, input, target, learning_rate, loss, num_epochs, batch_size, optimizer)
    evaluate(model, input_test, target_test)


if __name__ == '__main__':
    main()
