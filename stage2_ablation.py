from stage2 import *


def LRGSCV_without_ADSYN(ds: pd.DataFrame):
    lr = LogisticRegression(max_iter=200, n_jobs=-1)
    pipe = Pipeline([('lr', lr)])
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
    log_dir = 'output/' + strftime("%Y_%m_%d_%H_%M_%S", localtime())+'-ablation/'
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
    lr = LogisticRegressionCV(
        Cs=[para['lr__C'], 0.002, 0.02, 0.05, 0.2, 0.3, 0.5, 0.75, 1.2, 1.5, 2, 5, 15, 50]
        + list(np.arange(0.8, 1.2, 0.02)),
        cv=5, n_jobs=-1, penalty=para['lr__penalty'],
        solver=para['lr__solver'], max_iter=200,
        class_weight=para['lr__class_weight']
    )
    if para['lr__penalty'] == 'elasticnet':
        lr.set_params(l1_ratio=para['lr__l1_ratio'])
    best_model = Pipeline([('lr', lr)])
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
    LRGSCV_without_ADSYN(ds)
    # plot_tsne(ds, 'tsne.png')
    # plot_tsne(over_sample(ds, 'borderline'), 'borderline.png')
    # plot_tsne(over_sample(ds, 'borderline2'), 'borderline2.png')
    # plot_tsne(over_sample(ds, 'adasyn'), 'adasyn.png')
