from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import numpy as np

def evaluate(model, X_test, y_test):
    logits = model.predict(X_test)
    probs = np.softmax(logits, axis=1)
    classes = np.argmax(probs, axis=1)
    print(classification_report(classes, y_test, labels=[0, 1]))
    prec, recall, thr = precision_recall_curve(y_test, probs[:, 1], pos_label=1)
    prauc = auc(recall, prec)
    print(prauc)
    prauc = roc_auc_score(y_test, probs[:, 1])
    print(prauc)
    print(confusion_matrix(classes, y_test))
