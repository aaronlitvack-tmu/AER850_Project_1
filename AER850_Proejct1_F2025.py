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
from sklearn.model_selection import StratifiedShuffleSplit

#4.1 data split into training and testing blocks
data["step_categories"] = pd.cut(data["Y"],
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

y_train = strat_data_train['Step']
X_train = strat_data_train.drop(columns=['Step'])
y_test = strat_data_test['Step']
X_test = strat_data_test.drop(columns=['Step'])

#4.2 read coorlation values of training data
print(np.abs(y_train.corr(X_train['X'])))
print(np.abs(y_train.corr(X_train['Y'])))
print(np.abs(y_train.corr(X_train['Z'])))

#4.3 drop low corrlation variables from dataset
X_train = X_train.drop(columns=['Y'])
X_train = X_train.drop(columns=['Z'])
X_test = X_test.drop(columns=['Y'])
X_test = X_test.drop(columns=['Z'])

#4.4 Scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

pd.DataFrame(X_train).to_csv("UnscaledOriginalData.csv")
X_train = sc.transform(X_train)
pd.DataFrame(X_train).to_csv("NowScaledData.csv")

X_test = sc.transform(X_test)

#4.4 Developing linear regression model
from sklearn.linear_model import LinearRegression, LogisticRegression
mdl1 = LinearRegression()
mdl1.fit(X_train, y_train)

y_pred_train1 = mdl1.predict(X_train)
for i in range(5):
    print("Predictions:", y_pred_train1[i], "Actual values:", y_train[i])

#4.5 Developing logistic regression model
mdl2 = LogisticRegression()
mdl2.fit(X_train, y_train)

y_pred_train2 = mdl2.predict(X_train)
for i in range(5):
    print("Predictions:", y_pred_train2[i], "Actual values:", y_train[i])

#4.6 Developing random forest model
from sklearn.ensemble import RandomForestRegressor
mdl3 = RandomForestRegressor(n_estimators=50, random_state=42)
mdl3.fit(X_train, y_train)
y_pred_train3 = mdl3.predict(X_train)
for i in range(5):
    print("Predictions:", y_pred_train3[i], "Actual values:", y_train[i])

#4.6 Developing decision tree model
from sklearn import tree
mdl4 = tree.DecisionTreeClassifier()
mdl4.fit(X_train, y_train)
y_pred_train4 = mdl4.predict(X_train)
for i in range(5):
    print("Predictions:", y_pred_train4[i], "Actual values:", y_train[i])


#4.7 Pipeline
from sklearn.model_selection import cross_val_score
cv_scores_model1 = cross_val_score(mdl1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae1 = -cv_scores_model1.mean()
print("Model 1 Mean Absolute Error (CV):", round(cv_mae1, 2))


cv_scores_model2 = cross_val_score(mdl2, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae2 = -cv_scores_model2.mean()
print("Model 2 Mean Absolute Error (CV):", round(cv_mae2, 2))

cv_scores_model3 = cross_val_score(mdl3, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae3 = -cv_scores_model3.mean()
print("Model 3 Mean Absolute Error (CV):", round(cv_mae3, 2))

cv_scores_model4 = cross_val_score(mdl4, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae4 = -cv_scores_model4.mean()
print("Model 4 Mean Absolute Error (CV):", round(cv_mae4, 2))


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
pipeline1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())])
cv_scores1 = cross_val_score(pipeline1, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_mae1 = -cv_scores1.mean()
print("Model 1 CV MAE:", round(cv_mae1, 2))


pipeline1.fit(X_train, y_train)
y_pred_test1 = pipeline1.predict(X_test)
mae_test1 = mean_absolute_error(y_test, y_pred_test1)
print("Model 1 Test MAE:", round(mae_test1, 2))



pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())])
cv_scores2 = cross_val_score(pipeline2, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_mae2 = -cv_scores2.mean()
print("Model 2 CV MAE:", round(cv_mae2, 2))

pipeline2.fit(X_train, y_train)
y_pred_test2 = pipeline2.predict(X_test)
mae_test2 = mean_absolute_error(y_test, y_pred_test2)
print("Model 2 Test MAE:", round(mae_test2, 2))



pipeline3 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=10, random_state=42))])
cv_scores3 = cross_val_score(pipeline3, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_mae3 = -cv_scores3.mean()
print("Model 3 CV MAE:", round(cv_mae3, 2))

pipeline3.fit(X_train, y_train)
y_pred_test3 = pipeline3.predict(X_test)
mae_test3 = mean_absolute_error(y_test, y_pred_test3)
print("Model 3 Test MAE:", round(mae_test3, 2))



pipeline4 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', tree.DecisionTreeClassifier())])
cv_scores4 = cross_val_score(pipeline4, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_mae4 = -cv_scores4.mean()
print("Model 4 CV MAE:", round(cv_mae4, 2))

pipeline4.fit(X_train, y_train)
y_pred_test4 = pipeline4.predict(X_test)
mae_test4 = mean_absolute_error(y_test, y_pred_test4)
print("Model 4 Test MAE:", round(mae_test4, 2))

#4.8 cross validation for random forest
from sklearn.model_selection import GridSearchCV, KFold
param_grid = {
    'model__n_estimators': [10, 30, 50],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=pipeline3,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid.fit(X_train, y_train)

print("Best CV MAE:", -grid.best_score_)
print("Best params:", grid.best_params_)
y_pred = grid.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))


#4.9 cross validation for random forest using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

paramRND_grid = {
    'model__n_estimators': [10, 30, 50],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}
cvRND = KFold(n_splits=5, shuffle=True, random_state=42)
gridRND = RandomizedSearchCV(
    estimator=pipeline3,
    param_distributions=paramRND_grid,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
gridRND.fit(X_train, y_train)

print("Best CV MAE:", -gridRND.best_score_)
print("Best params:", gridRND.best_params_)
y_predRND = gridRND.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_predRND))
#Step 5: Model Performance Analysis


# #5.1 MAE for first model
# mae_train1 = mean_absolute_error(y_pred_train1, y_train)
# print("Model 1 training MAE is: ", round(mae_train1,2))

# #5.2 MAE for the second model
# mae_train2 = mean_absolute_error(y_pred_train2, y_train)
# print("Model 2 training MAE is: ", round(mae_train2,2))

# #5.3 MAE for the third model
# mae_train3 = mean_absolute_error(y_pred_train3, y_train)
# print("Model 3 training MAE is: ", round(mae_train3,2))

# #5.3 MAE for the fourth model
# mae_train4 = mean_absolute_error(y_pred_train4, y_train)
# print("Model 4 training MAE is: ", round(mae_train4,2))
