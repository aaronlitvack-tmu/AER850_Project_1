#Step 1: Data Processing
import pandas as pd
data = pd.read_csv("data/Project 1 Data.csv")
data = data.dropna().reset_index(drop=True)


#Step 2: Data Visualization
import matplotlib.pyplot as plt
import numpy as np

plt.plot(data['Step'], data['X'], label='X')
plt.plot(data['Step'], data['Y'], label='Y')
plt.plot(data['Step'], data['Z'], label='Z')
plt.xlabel('Step')
plt.ylabel('XYZ Coordinate')
plt.title('XYZ Coordiantes')
plt.legend(loc='best')
plt.show()

#Step 3: Correlation Analysis
import seaborn as sns

corr_matrix = data.corr()
sns.heatmap(np.abs(corr_matrix))

masked_corr_matrix = np.abs(corr_matrix) < 0.7
sns.heatmap(masked_corr_matrix)


#Step 4: Classification Model Development/Engineering
from sklearn.model_selection import  train_test_split, StratifiedShuffleSplit
data["step_categories"] = pd.cut(data["Z"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])
my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in my_splitter.split(data, data["step_categories"]):
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)
strat_data_train = strat_data_train.drop(columns=["step_categories"], axis = 1)
strat_data_test = strat_data_test.drop(columns=["step_categories"], axis = 1)

# y_train = strat_data_train['Step']
# X_train = strat_data_train.drop(columns=['Step'])
# y_test = strat_data_test['Step']
# X_test = strat_data_test.drop(columns=['Step'])

#print(np.abs(y_train.corr(X_train['X'])))

