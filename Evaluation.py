from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import numpy as np
from scipy.special import softmax
import xgboost as xgb


def evaluate(model, X_test, y_test):
    logits = model.predict(X_test)
    probs = softmax(logits, axis=1)
    classes = np.argmax(probs, axis=1)
    print_metrics(classes, y_test, probs[:,1])


def evaluate_XGB(obj, X_test, y_test):
    dtest = xgb.DMatrix(data=X_test)
    probs = obj.predict(dtest)
    classes = probs.copy()
    classes[classes > 0.5] = 1
    classes[classes <= 0.5] = 0
    print_metrics(classes, y_test, probs)


def print_metrics(classes, y_test, probs):
    print(classification_report(classes, y_test, labels=[0, 1]))
    prec, recall, thr = precision_recall_curve(y_test, probs, pos_label=1)
    prauc = auc(recall, prec)
    print(prauc)
    prauc = roc_auc_score(y_test, probs)
    print(prauc)
    print(confusion_matrix(classes, y_test))
