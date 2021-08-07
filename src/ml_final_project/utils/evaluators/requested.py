from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np


def pr_auc(y_true, y_pred):
    ''' if multiclass: pr_auc(y_true, y_pred)
        if binary class: pr_auc(y_true, y_pred[:,1])'''
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_bin.shape[1]
    print(n_classes)
    if n_classes > 2:
        auc_roc = roc_auc_score(y_true, y_pred,multi_class='ovr')
        # For each class
        precision_dict = dict()
        recall_dict = dict()
        auc_pr_curve_per_class = dict()
        for i in range(n_classes):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_true_bin[:, i],
                                                          y_pred[:, i])
            auc_pr_curve_per_class[i] = auc(recall_dict[i], precision_dict[i])
        auc_pr_curve = np.mean(list(auc_pr_curve_per_class.values()))
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        # Use AUC function to calculate the area under the curve of precision recall curve      
        auc_pr_curve = auc(recall, precision)
    return auc_pr_curve


def requested_evaluator(model, X_test, y_test):
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    res = {}
    if y_proba.shape[1] > 2: # multiclass version
        AUC = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
        PR_AUC = pr_auc(y_true, y_proba)
    else: # binary version
        AUC = roc_auc_score(y_true, y_proba[:,1])
        PR_AUC = pr_auc(y_true, y_proba[:,1])
    res['AUC'] = AUC
    res['PR_AUC'] = PR_AUC


    accuracy = accuracy_score(y_true, y_pred)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = np.divide(TP, (TP+FN), out=np.zeros_like(TP), where=(TP+FN)!=0).mean()
    # TPR = TP/(TP+FN)
    # TPR = np.nan_to_num(TPR).mean()
    res['TPR'] = TPR

    # Fall out or false positive rate
    FPR = np.divide(FP, (FP+TN), out=np.zeros_like(FP), where=(FP+TN)!=0).mean()
    # FPR = FP/(FP+TN)
    # FPR = np.nan_to_num(FPR).mean()
    res['FPR'] = FPR

    #  Precision orpositive predictive value
    # PPV = TP/(TP+FP)
    res['PPV'] = classification_report(y_true,y_pred, output_dict=True, zero_division=0)['macro avg']['precision']

    # Overall accuracy
    num, dem = (TP+TN), (TP+FP+FN+TN)
    ACC = np.divide(num, dem, out=np.zeros_like(num), where=(dem)!=0).mean()
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    # ACC = np.nan_to_num(ACC).mean()
    res['ACC'] = ACC
    return res