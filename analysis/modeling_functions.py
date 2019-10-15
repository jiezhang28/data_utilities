import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, KFold

def print_conf_matrix_scores(y, predictions):
    """
    Prints F1, Recall, and Precision scores based true classes against predicated classes

    Parameters
    ----------
    y           : List of true classes
    predictions : List of predicted classes
    """

    f1 = f1_score(y, predictions)
    recall = recall_score(y, predictions)
    precision = precision_score(y, predictions)

    print(f'F1: {f1}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')

def graph_roc_auc(y, score):
    """
    Prints AUC graph

    Parameters
    ----------
    y     : List of binary classes
    score : List of scores/probabilities for class 1
    """

    plt.figure()
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    lw=2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Area Under the ROC Curve')
    plt.legend(loc="lower right")

def graph_prec_recall(y, score, mark_thresh=0.5):
    """
    Prints precision-recall graph and place marker at specified threshold

    Parameters
    ----------
    y           : List of binary classes
    score       : List of scores/probabilities for class 1
    mark_thresh : Threshold value to place marker. DEFAULT: 0.5
    """

    plt.figure()
    average_precision = average_precision_score(y, score)
    precision, recall, thresholds = precision_recall_curve(y, score)

    thresh_ind = np.argmin(np.abs(thresholds - mark_thresh))

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.plot(recall[thresh_ind], precision[thresh_ind], '^', c='k', markersize=15)
    plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))

def _graph_prec_recall_threshold(precision, recall, thresholds):
    """
    Graphs Precision and Recall curves as function of threshold

    Parameters
    ----------
    precision  : Array of precision scores
    recall     : Array of recall scores
    thresholds : Array of decision thresholds
    """
    plt.figure()
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precision, color='orange', label="Precision")
    plt.plot(thresholds, recall, color='b', label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

def graph_prec_recall_threshold(y, score):
    """
    Gets precision, recall, and thresholds values and 
    plots precision-recall-thresholds graph

    Parameters
    ----------
    y     : True taget values
    score : Probability/score of predictions
    """
    precision, recall, thresholds = precision_recall_curve(y, score)
    _graph_prec_recall_threshold(precision[:-1], recall[:-1], thresholds)

def kfold_iter(X, y, splits=5, shuffle=False, stratified=True):
    """
    Creates kfold iterable

    Parameters
    ----------
    X          : feature data set
    y          : target value list
    splits     : number of splits to use. DEFAULT=5
    shuffle    : Shuffle flag to pass to KFold init. DEFAULT=False
    stratified : Flag to use StratifiedKFold instead of KFold. DEFAULT=True

    Returns
    -------
    Iterable that iterates over tuple X_train, X_test, y_train, y_test
    """
    kf = StratifiedKFold(n_splits=splits, shuffle=shuffle) if stratified else KFold(n_splits=splits, shuffle=shuffle)

    for train_ind, test_ind in kf.split(X, y):
        X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        yield X_train, X_test, y_train, y_test

def run_kfold_thresh(X, y, estimator, splits=5, thresh=0.5, stratified=True):
    """
    Runs Kfold on binary class estimator at specified probability threshold and returns model scores
    
    Parameters
    ----------
    X          : feature data set
    y          : target values list
    estimator  : estimator to use in kfold
    split      : number of splits to use. DEFAULT=5
    thresh     : decision threshold to use for predictions. DEFAULT=0.5
    stratified : Flag to use StratifiedKFold instead of KFold. DEFAULT=True

    Returns
    -------
    Dictionary containing lists for recall, precision, and auc.
    Can be sent into pd.DataFrame().
    """

    assert({0,1}==set(y)), "Target class is not binary 0/1"
    assert(hasattr(estimator, 'predict_proba')), "Estimator does not have a predict_proba function"

    scores = {
        'recall' : [],
        'precision' : [],
        'auc' : []
    }
    for X_train, X_test, y_train, y_test in kfold_iter(X, y, splits=splits, shuffle=False, stratified=stratified):
        estimator.fit(X_train, y_train)
        proba = estimator.predict_proba(X_test)[:,1]
        pred = [1 if p>thresh else 0 for p in proba]

        scores['recall'].append(recall_score(y_test, pred))
        scores['precision'].append(precision_score(y_test, pred))
        scores['auc'].append(roc_auc_score(y_test, proba))

    return scores

def run_kfold_curve(X, y, estimator, splits=5, stratified=True, shuffle=False):
    """
    Returns precision-recall curve data for each fold. Normalized threshold values between 0.01 and 0.99

    Parameters
    ----------
    X : feature data set
    y : target values list
    estimator : estimator to use for training and predicting
    splits : number of kfold splits to use. DEFAULT=5
    stratified : Flag to use StratifiedKFold instead of KFold. DEFAULT=True
    shuffle : Flag to shuffle data in KFold. DEFAULT=False

    Returns
    -------
    Array of dictionaries with keys: threshold, recall, precision
    """
    assert({0,1}==set(y)), "Target class is not binary 0/1"

    curves = []
    for X_train, X_test, y_train, y_test in kfold_iter(X, y, splits=splits, shuffle=shuffle, stratified=stratified):
        estimator.fit(X_train, y_train)

        if hasattr(estimator, 'predict_proba'):
            proba = estimator.predict_proba(X_test)[:,1]
        elif hasattr(estimator, 'decision_function'):
            score = estimator.decision_function(X_test)
            proba = np.exp(score) / (1 + np.exp(score))
        else:
            raise Exception('No predict_proba or decision_function')

        precision, recall, thresholds = precision_recall_curve(y_test, proba)

        scores = {
            'threshold' : [],
            'recall' : [],
            'precision' : []
        }
        for n in range(1,100):
            p = n/100
            ind = np.argmin(np.abs(thresholds - p))
            scores['threshold'].append(p)
            scores['recall'].append(recall[ind])
            scores['precision'].append(precision[ind])

        curves.append(scores)

    return curves

def graph_avg_prec_recall_curve(X, y, estimator, n=25):
    """
    Graph average precision, recall curves by running KFold n times (with shuffle turned on)
    and capturing the scoring output

    Parameters
    ----------
    X : feature data set
    y : target value list
    estimator : estimator to use in kfold
    n : number of times to run KFold

    Returns
    -------
    DataFrame containing all scores
    """
    df = pd.DataFrame()
    for i in range(n):
        data = run_kfold_curve(X, y, estimator, shuffle=True)
        df = pd.concat([df, pd.concat([pd.DataFrame(datum) for datum in data])])

    df = df.groupby('threshold').mean().reset_index()
    _graph_prec_recall_threshold(df['precision'], df['recall'], df['threshold'])
    return df



