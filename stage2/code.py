data=pd.read_csv(r'/home/kesci/2021MCMProblemC_DataSet.csv')
smote_enn = SMOTE(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(X_train, y_train)

lr = LogisticRegression(C=10, penalty='l1', solver='liblinear')
C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
tuned_parameters = dict(C=C_list)
grid = GridSearchCV(lr, tuned_parameters, cv=5, scoring='neg_log_loss')
grid.fit(X_resampled, y_resampled)
print(grid.best_params_)

data=pd.read_csv(r'/home/kesci/2021MCMProblemC_DataSet.csv')
smote_enn = SMOTE(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(X_train, y_train)

lr = LogisticRegression(C=10, penalty='l1', solver='liblinear')
C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
tuned_parameters = dict(C=C_list)
grid = GridSearchCV(lr, tuned_parameters, cv=5, scoring='neg_log_loss')
grid.fit(X_resampled, y_resampled)
print(grid.best_params_)