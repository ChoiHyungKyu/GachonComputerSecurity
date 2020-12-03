#####################################
# Import Modules
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

#####################################
# Load Train dataset
df = pd.read_csv('data/subtrain_noscale.csv')
temp_df = pd.read_csv('data/test_benign.csv')
temp_df = temp_df.drop(columns=['Timestamp'])
temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
temp_df = temp_df.dropna(axis=0)
df = pd.concat([df, temp_df])

# Add Benign data
temp_df = pd.read_csv('data/train.csv')
temp_df = temp_df.drop(columns=['Timestamp'])
temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
temp_df = temp_df.dropna(axis=0)
temp_df = temp_df[temp_df['Label'] == 'Benign']
df = pd.concat([df, temp_df])
print(df.groupby('Label').count())

# Target value
X = df.drop(columns='Label')
Y = df['Label'].values

#####################################
# Load Test dataset
df_test = pd.read_csv('data/train.csv')
df_test = df_test.drop(columns=['Timestamp'])
df_test = df_test.replace([np.inf, -np.inf], np.nan)
df_test = df_test.dropna(axis=0)
print(df_test.groupby('Label').count())

# Target value
x_test = df_test.drop(columns='Label')
y_test = df_test['Label'].values

#####################################
# Convert Label to 1 or -1
for inx in range(len(Y)):
    if Y[inx] == 'Benign':
        Y[inx] = 1
    else:
        Y[inx] = -1
for inx in range(len(y_test)):
    if y_test[inx] == 'Benign':
        y_test[inx] = 1
    else:
        y_test[inx] = -1


#####################################
# Function: Getting result using model
def get_result(clf, title, x_train, y_train, x_test, y_test):
    clf.fit(x_train)
    y_train_pred = clf.predict(x_train)
    print('\n[{}]'.format(title))
    print('Train Score: {}'.format(accuracy_score(list(y_train), list(y_train_pred))))
    sn.heatmap(pd.crosstab(y_train, y_train_pred, rownames=['Actual'], colnames=['Predicted']), annot=True)
    plt.title('{}: Train'.format(title))
    plt.show()
    y_test_pred = clf.predict(x_test)
    print('Test Score: {}'.format(accuracy_score(list(y_test), list(y_test_pred))))
    sn.heatmap(pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted']), annot=True)
    plt.title('{}: Test'.format(title))
    plt.show()
    print(classification_report(list(y_test), list(y_test_pred)))
    # print('Outlier Score: {}')
    return y_train_pred, y_test_pred


# Construct Model and Run
clf_isf = IsolationForest(n_estimators=50, max_samples=50, contamination='auto', max_features=2, bootstrap=False)
clf_lof = LocalOutlierFactor(novelty=True)
clf_svm = OneClassSVM(nu=0.01)
isf_train_pred, isf_test_pred = get_result(clf_isf, 'IsolationForest', X, Y, x_test, y_test)
lof_train_pred, lof_test_pred = get_result(clf_lof, 'LocalOutlierFactor', X, Y, x_test, y_test)
svm_train_pred, svm_test_pred = get_result(clf_svm, 'OneClassSVM', X, Y, x_test, y_test)

#####################################
# Ensemble: Hard Voting
print('\n[Ensemble]')
ens_train_pred = []
for inx in range(len(isf_train_pred)):
    if (isf_train_pred[inx] + lof_train_pred[inx] + svm_train_pred[inx]) > 1:
        ens_train_pred.append(1)
    else:
        ens_train_pred.append(-1)
print('Train Score: {}'.format(accuracy_score(list(Y), list(ens_train_pred))))
ens_test_pred = []
for inx in range(len(isf_test_pred)):
    if (isf_test_pred[inx] + lof_test_pred[inx] + svm_test_pred[inx]) > 1:
        ens_test_pred.append(1)
    else:
        ens_test_pred.append(-1)
print('Test Score: {}'.format(accuracy_score(list(y_test), list(ens_test_pred))))
print(classification_report(list(y_test), list(ens_test_pred)))

#####################################
"""
# Find the Best Estimator(IsolationForest)
parameter = {'n_estimators': [50, 100, 200],
             'max_samples': [50, 100, 200, 'auto'],
             'contamination': [0.001, 0.01, 0.1, 'auto'],
             'max_features': [0.5, 1, 2],
             'bootstrap': [True, False]}
param_result_list = []
for ne in parameter['n_estimators']:
    for ms in parameter['max_samples']:
        for ct in parameter['contamination']:
            for mf in parameter['max_features']:
                for bs in parameter['bootstrap']:
                    clf = IsolationForest(n_estimators=ne, max_samples=ms, contamination=ct, max_features=mf, bootstrap=bs)
                    # Prediction
                    print('\n=============================')
                    print('<Parameter>')
                    print('n_estimators: {}, max_samples: {}, contamination: {}, max_features: {}, bootstrap: {}'.format(ne, ms, ct, mf, bs))
                    clf.fit(X)
                    y_pred = clf.predict(X)
                    print('\n<Train Result>')
                    # print('Actual: {}'.format(Y))
                    # print('Prediction: {}'.format(y_pred))
                    print('Train Score: {}'.format(accuracy_score(list(Y), list(y_pred))))
                    # print(classification_report(list(Y), list(y_pred)))
                    # sn.heatmap(pd.crosstab(Y, y_pred, rownames=['Actual'], colnames=['Predicted']), annot=True)
                    # plt.title('Training Result')
                    # plt.show()
                    # Evaluation
                    y_test_pred = clf.predict(x_test)
                    print('\n<Evaluation Result>')
                    # print('Actual: {}'.format(y_test))
                    # print('Prediction: {}'.format(y_test_pred))
                    print('Score: {}'.format(accuracy_score(list(y_test), list(y_test_pred))))
                    # print(classification_report(list(y_test), list(y_test_pred)))
                    # sn.heatmap(pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted']), annot=True)
                    # plt.title('Evaluation Result')
                    # plt.show()
                    param_result_list.append([ne, ms, ct, mf, bs, accuracy_score(list(y_test), list(y_test_pred))])
param_result = pd.DataFrame(data=param_result_list, columns=['n_estimators', 'max_samples', 'contamination', 'max_features', 'bootstrap', 'score'])
print(param_result)
param_result.to_excel('param_result.xlsx')
"""
#####################################
