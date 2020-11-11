from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

# Load dataset and preprocessing
df = pd.read_csv('data/train.csv')
df = df.drop(columns=['Timestamp'])
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(axis=0)

# Scaling data
rs = RobustScaler()
df[['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Var', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']] = rs.fit_transform(df[['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Var', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']])

# Target value
x = df.drop(columns='Label')
y = df['Label'].values

"""
# Find the best parameter -> {'eta': 0.35, 'gamma': 0, 'max_depth':7} is the best
CV = 10
param_xgb = {"eta": [0.20, 0.25, 0.30, 0.35, 0.40], "gamma": [0, 1, 2, 3, 4, 5], "max_depth": [3, 4, 5, 6, 7]}
clf_xgb = XGBClassifier()
grid_xgb = GridSearchCV(estimator=clf_xgb, param_grid=param_xgb, scoring='accuracy', n_jobs=4, cv=CV, verbose=10)
grid_xgb.fit(x, y)
print('Parameter:', grid_xgb.best_params_)
print('Score:', grid_xgb.best_score_)
"""

# Construct model with the best parameter
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf = XGBClassifier(eta=0.35, gamma=0, max_depth=7)
clf.fit(x_train, y_train)
print("Train score: {}".format(clf.score(x_test, y_test)))

# Load validation dataset and preprocessing
df_valid = pd.read_csv('data/test.csv')
df_valid = df_valid.drop(columns=['Timestamp'])
df_valid = df_valid.replace([np.inf, -np.inf], np.nan)
df_valid = df_valid.dropna(axis=0)

# Scaling validation data
rs = RobustScaler()
df_valid[['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Var', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']] = rs.fit_transform(df_valid[['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Var', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']])

# Target value
x_valid = df_valid.drop(columns='Label')
y_valid = df_valid['Label'].values

# Validation: for 'Botnet' label
print("Valid score: {}".format(clf.score(x_valid, y_valid)))
y_pred = clf.predict(x_valid)
sn.heatmap(pd.crosstab(y_valid, y_pred, rownames=['Actual'], colnames=['Predicted']), annot=True)
plt.title('Confusion Matrix(XGBoost)')
plt.show()