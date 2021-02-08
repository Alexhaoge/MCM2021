import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

def get_data():
    data = pd.read_csv('data/textprocess.csv')
    ref_data = pd.read_csv('output/infer.csv')
    smote_enn = SMOTE(random_state=0)
    smote_enn.fit_sample()


def LRGSCV():
    lr = LogisticRegression(C=10, penalty='l1', solver='liblinear')
    C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    tuned_parameters = dict(C=C_list)
    grid = GridSearchCV(lr, tuned_parameters, cv=5, scoring='neg_log_loss')
    print(grid.best_params_)