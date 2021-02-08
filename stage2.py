import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import (
    plot_roc_curve, accuracy_score,
    confusion_matrix, plot_precision_recall_curve,
    f1_score, precision_score, recall_score, roc_auc_score
)
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline
import os


def get_data(save_file: bool = False) -> pd.DataFrame:
    d = pd.read_excel('data/2021MCMProblemC_DataSet.xlsx')
    tmp = pd.read_csv('data/textprocess.csv',index_col=0)
    ref_data = pd.read_csv('output/real_infer.csv', index_col=0)
    d['text_pro'] = tmp['text_pro']
    merged = ref_data.groupby(by='GlobalID').max()['1']
    tmp = merged.mean()
    d['image'] = tmp
    for i, p in zip(merged.index, merged.values):
        j = d[d.GlobalID == i].index[0]
        d.loc[j, 'image'] = pd
    from datetime import datetime
    d['Detection Date'] = d['Detection Date'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    d.sort_values(by='Detection Date', inplace=True, kind='mergesort', ignore_index=True)
    begin_time = d[d['Lab Status'] == 'Positive ID'].head(1)['Detection Date']
    d['time'] = d['Detection Date'] - begin_time
    d['time'] = d['time'].apply(lambda x: 1/x.days if x.days > 1 else 1)
    from math import exp
    d['lat'] = (d['Latitude']-49.149394).abs().apply(lambda x: exp(-x))
    d['lon'] = (d['Longitude']+123.943134).abs().apply(lambda x: exp(-x))
    d['label'] = d['Lab Status'].map(
        {'Unverified': -1, 'Negative ID': 0, 'Unprocessed': -1, 'Positive ID': 1})
    d.drop(axis=1, columns=['Detection Date', 'Notes', 'Submission Date',
                            'Lab Comments', 'Latitude', 'Longitude', 'Lab Status'], inplace=True)
    d.set_index('GlobalID', verify_integrity=True, inplace=True)
    if save_file:
        d.to_csv('data/dataset2.csv', index=True, index_label='GlobalID')
    return d


def plot_tsne(data: pd.DataFrame, filename: str, n_iter: int = 500) -> None:
    data.reset_index(inplace=True, drop=True)
    label = data['label']
    X = MinMaxScaler().fit_transform(data)
    X_embed = TSNE(n_components=3, n_iter=n_iter, init='pca', n_jobs=-1, random_state=97).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = {-1: 'grey', 0: 'blue', 1: 'red'}
    leg = {-1: 'Unverified or Unprocessed', 0: 'Negative', 1:'Positive'}
    alpha = {-1: 0.75, 0: 0.5, 1: 0.95}
    for i in [-1, 0, 1]:
        X_draw = X_embed[label[label == i].index.tolist()]
        ax.scatter(
            X_draw[:, 0], X_draw[:, 1], X_draw[:, 2],
            color=color[i],
            alpha=alpha[i],
            label=leg[i],
            s=30 if i == 1 else 18
        )
    plt.legend(loc="best")
    plt.savefig(filename, bbox_inches='tight')
    try:
        plt.show()
    except Exception as e:
        print(e.args)


def plot_confusion_matrix(cm, classes, filename, title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    try:
        plt.show()
    except Exception as e:
        print(e.args)


def plot_curve(model, X, y, filename, roc=True):
    ax = plt.gca()
    if roc:
        dis = plot_roc_curve(model, X, y, ax=ax)
    else:
        dis = plot_precision_recall_curve(model, X, y, ax=ax)
    dis.plot(ax=ax, alpha=0.8)
    plt.savefig(filename)
    try:
        plt.show()
    except Exception as e:
        print(e.args)


def over_sample(ds: pd.DataFrame, method: str) -> pd.DataFrame:
    assert method in ['borderline', 'adasyn', 'borderline2']
    X = ds[ds.label != -1]
    y = X['label']
    X.drop(axis=1, columns='label', inplace=True)
    if method == 'borderline':
        model = BorderlineSMOTE(n_jobs=-1, random_state=97, kind='borderline-1')
    elif method == 'borderline2':
        model = BorderlineSMOTE(n_jobs=-1, random_state=97, kind='borderline-2')
    else:
        model = ADASYN(n_jobs=-1, random_state=97)
    Xe, ye = model.fit_resample(X, y)
    Xe['label'] = ye
    return Xe


def LRGSCV(ds: pd.DataFrame):
    lr = LogisticRegression(max_iter=200, n_jobs=-1)
    ovs = ADASYN(n_jobs=-1)
    pipe = Pipeline([('ovs', ovs), ('lr', lr)])
    #C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    grid = [
        # {
        #     'ovs__sampling_strategy': [1],
        #     'lr__penalty': ['l1'],
        #     'lr__C': [1.0, 2.0],
        #     'lr__fit_intercept': [True],
        #     'lr__class_weight': ['balanced'],
        #     'lr__solver': ['liblinear'],
        # },
        {
            'ovs__sampling_strategy': [0.25, 0.5, 1],
            'lr__penalty': ['l1'],
            'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'lr__fit_intercept': [True, False],
            'lr__class_weight': ['balanced', None],
            'lr__solver': ['liblinear'],
        },
        {
            'ovs__sampling_strategy': [0.25, 0.5, 1],
            'lr__penalty': ['l2'],
            'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'lr__fit_intercept': [True, False],
            'lr__class_weight': ['balanced', None],
            'lr__solver': ['liblinear', 'lbfgs', 'sag']
        },
        {
            'ovs__sampling_strategy': [0.25, 0.5, 1],
            'lr__penalty': ['elasticnet'],
            'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'lr__fit_intercept': [True, False],
            'lr__class_weight': ['balanced', None],
            'lr__solver': ['saga'],
            'lr__l1_ratio': [0.2, 0.5, 0.8]
        }
    ]
    gsCV = GridSearchCV(
        estimator=pipe, cv=5, n_jobs=-1, param_grid=grid,
        scoring={
            'p': 'precision',
            'r': 'recall',
            'roc': 'roc_auc'
        }, 
        refit='r' , verbose=2
    )
    from time import strftime, localtime
    log_dir = 'output/' + strftime("%Y_%m_%d_%H_%M_%S", localtime())+'/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    Xtrain = ds[ds.label != -1]
    ytrain = Xtrain['label']
    Xtrain.drop(axis=1, columns='label', inplace=True)
    gsCV.fit(Xtrain, ytrain)
    from joblib import dump
    dump(gsCV, log_dir+'gsCV')
    file = open(log_dir + 'log.txt', 'w')
    file.write(gsCV.cv_results_.__str__())
    para = gsCV.best_params_
    file.write(para.__str__())
    file.flush()
    file.write('start refit')
    ovs.set_params(sampling_strategy=para['ovs__sampling_strategy'])
    lr = LogisticRegressionCV(
        Cs=[para['lr__C'], 0.002, 0.02, 0.05, 0.2, 0.3, 0.5, 0.75, 1.2, 1.5, 2, 5, 15, 50]
        + list(np.arange(0.8, 1.2, 0.02)),
        cv=5, n_jobs=-1, penalty=para['lr__penalty'],
        solver=para['lr__solver'], max_iter=200,
        class_weight=para['lr__class_weight']
    )
    if para['lr__penalty'] == 'elasticnet':
        lr.set_params(l1_ratio=para['lr__l1_ratio'])
    best_model = Pipeline([('ovs', ovs), ('lr', lr)])
    best_model.fit(Xtrain, ytrain)
    ypredict = best_model.predict(Xtrain)
    dump(best_model, log_dir+'best_model')
    file.write('refit done')
    file.write('Metrics on original Dataset: {}\n'.format({
        'acc': accuracy_score(ytrain, ypredict),
        'f1': f1_score(ytrain, ypredict),
        'precision': precision_score(ytrain, ypredict),
        'recall': recall_score(ytrain, ypredict),
        'roc_auc': roc_auc_score(ytrain, ypredict)
    }))
    cm = confusion_matrix(ytrain, ypredict)
    plot_confusion_matrix(
        cm, ['Negative', 'Positive'], log_dir + 'train_cm.png')
    file.write('\ntrain_cm:\n')
    file.write(cm.__str__())
    plot_curve(best_model, Xtrain, ytrain, log_dir + 'roc.png', True)
    plot_curve(best_model, Xtrain, ytrain, log_dir + 'pr.png', False)
    file.close()
    return gsCV


if __name__=='__main__':
    ds = pd.read_csv('data/dataset2.csv', index_col='GlobalID')
    ds = ds[ds.label>-1]
    LRGSCV(ds)
    # plot_tsne(ds, 'tsne.png')
    # plot_tsne(over_sample(ds, 'borderline'), 'borderline.png')
    # plot_tsne(over_sample(ds, 'borderline2'), 'borderline2.png')
    # plot_tsne(over_sample(ds, 'adasyn'), 'adasyn.png')
