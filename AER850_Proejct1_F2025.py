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

data["step_categories"] = pd.cut(data["Step"],
                          bins=[0, 6, 10, np.inf],
                          labels=[1, 2, 3])
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


#4.3 Scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train.values)

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
mdl3 = RandomForestRegressor(n_estimators=10, random_state=42)
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

#4.8 cross validation for linear regression
from sklearn.model_selection import GridSearchCV, KFold
param_grid1 = {
    'model__copy_X': [True,False],
    'model__fit_intercept': [True,False],
    'model__n_jobs': [1,5,10,15,None], 
    'model__positive': [True,False]
}
cv1 = KFold(n_splits=5, shuffle=True, random_state=42)
grid1 = GridSearchCV(
    estimator=pipeline1,
    param_grid=param_grid1,
    scoring='neg_mean_absolute_error',
    cv=cv1,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid1.fit(X_train, y_train)

print("Best CV MAE:", -grid1.best_score_)
print("Best params:", grid1.best_params_)
y_pred1 = grid1.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred1))

#4.9 cross validation for logistic regression, ITERATION PROBLEMS, RUNNING GUMS UP CONSOLE
# param_grid2 = {
#     'model__penalty':['l1','l2','elasticnet','none'],
#     'model__C' : np.logspace(-4,4,20),
#     'model__solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
#     'model__max_iter'  : [100, 1000, 2500, 5000]
# }
# cv2 = KFold(n_splits=5, shuffle=True, random_state=42)
# grid2 = GridSearchCV(
#     estimator=pipeline2,
#     param_grid=param_grid2,
#     scoring='neg_mean_absolute_error',
#     cv=cv2,
#     n_jobs=-1,
#     refit=True,           
#     verbose=1,
#     return_train_score=True
# )
# grid2.fit(X_train, y_train)

# print("Best CV MAE:", -grid2.best_score_)
# print("Best params:", grid2.best_params_)
# y_pred2 = grid2.predict(X_test)
# print("Test MAE:", mean_absolute_error(y_test, y_pred2))

#4.10 cross validation for random forest
param_grid3 = {
    'model__n_estimators': [10, 30, 50],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}
cv3 = KFold(n_splits=5, shuffle=True, random_state=42)
grid3 = GridSearchCV(
    estimator=pipeline3,
    param_grid=param_grid3,
    scoring='neg_mean_absolute_error',
    cv=cv3,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid3.fit(X_train, y_train)

print("Best CV MAE:", -grid3.best_score_)
print("Best params:", grid3.best_params_)
y_pred3 = grid3.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred3))

#4.11 cross validation for decision tree
param_grid4 = {
    'model__n_estimators': [10, 30, 50],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}
cv4 = KFold(n_splits=5, shuffle=True, random_state=42)
grid4 = GridSearchCV(
    estimator=pipeline3,
    param_grid=param_grid3,
    scoring='neg_mean_absolute_error',
    cv=cv4,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid4.fit(X_train, y_train)

print("Best CV MAE:", -grid3.best_score_)
print("Best params:", grid3.best_params_)
y_pred4 = grid3.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred4))


#4.12 cross validation for random forest using RandomizedSearchCV
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
    cv=cvRND,
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

#5.1 Linear regression analysis
clf1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearRegression())])
clf1.fit(X_train, y_train)
print("Training accuracy:", clf1.score(X_train, y_train))
print("Test accuracy:", clf1.score(X_test, y_test))

# Evaluate the classifier using various metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#rounding prediction matrix (why do we do this?)
y_pred_clf1 = np.round(clf1.predict(X_test))
cm_clf1 = confusion_matrix(y_test, y_pred_clf1)
print("Confusion Matrix:")
print(cm_clf1)
precision_clf1 = precision_score(y_test, y_pred_clf1, average='micro')
recall_clf1 = recall_score(y_test, y_pred_clf1, average='micro')
f1_clf1 = f1_score(y_test, y_pred_clf1, average='micro')
print("Precision:", precision_clf1)
print("Recall:", recall_clf1)
print("F1 Score:", f1_clf1)


#5.2 logisitic regression analysis

clf2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
clf2.fit(X_train, y_train)
print("Training accuracy:", clf2.score(X_train, y_train))
print("Test accuracy:", clf2.score(X_test, y_test))


y_pred_clf2 = clf2.predict(X_test)
cm_clf2 = confusion_matrix(y_test, y_pred_clf2)
print("Confusion Matrix:")
print(cm_clf2)
precision_clf2 = precision_score(y_test, y_pred_clf2, average='micro')
recall_clf2 = recall_score(y_test, y_pred_clf2, average='micro')
f1_clf2 = f1_score(y_test, y_pred_clf2, average='micro')
print("Precision:", precision_clf2)
print("Recall:", recall_clf2)
print("F1 Score:", f1_clf2)


#5.3 random forest analysis

from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=10, random_state=42)
clf3.fit(X_train, y_train)
print("RF Training accuracy:", clf3.score(X_train, y_train))
print("RF Test accuracy:", clf3.score(X_test, y_test))

y_pred_clf3 = clf3.predict(X_test)
cm_clf3 = confusion_matrix(y_test, y_pred_clf3)
print("RF Confusion Matrix:")
print(cm_clf3)
precision_clf3 = precision_score(y_test, y_pred_clf3, average='micro')
recall_clf3 = recall_score(y_test, y_pred_clf3, average='micro')
f1_clf3 = f1_score(y_test, y_pred_clf3, average='micro')
print("RF Precision:", precision_clf3)
print("RF Recall:", recall_clf3)
print("RF F1 Score:", f1_clf3)

#5.4 decision tree analysis

from sklearn.tree import DecisionTreeClassifier
clf4 = DecisionTreeClassifier(max_depth=4, random_state=42)
clf4.fit(X_train, y_train)
print("DT Training accuracy:", clf4.score(X_train, y_train))
print("DT Test accuracy:", clf4.score(X_test, y_test))

y_pred_clf4 = clf4.predict(X_test)
cm_clf4 = confusion_matrix(y_test, y_pred_clf4)
print("DT Confusion Matrix:")
print(cm_clf4)
precision_clf4 = precision_score(y_test, y_pred_clf4, average='micro')
recall_clf4 = recall_score(y_test, y_pred_clf4, average='micro')
f1_clf4 = f1_score(y_test, y_pred_clf4, average='micro')
print("DT Precision:", precision_clf4)
print("DT Recall:", recall_clf4)
print("DT F1 Score:", f1_clf4)

#Step 6: Stacked Model Performance
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
estimators = [
    ('rf', mdl3),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]

clf5 = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(max_iter=1000, random_state=42)
)

clf5.fit(X_train, y_train)
print("SC Training accuracy:", clf5.score(X_train, y_train))
print("SC Test accuracy:", clf5.score(X_test, y_test))

y_pred_clf5 = clf5.predict(X_test)
cm_clf5 = confusion_matrix(y_test, y_pred_clf5)
print("SC Confusion Matrix:")
print(cm_clf5)
precision_clf5 = precision_score(y_test, y_pred_clf5, average='micro')
recall_clf5 = recall_score(y_test, y_pred_clf5, average='micro')
f1_clf5 = f1_score(y_test, y_pred_clf5, average='micro')
print("SC Precision:", precision_clf5)
print("SC Recall:", recall_clf5)
print("SC F1 Score:", f1_clf5)

#Step 7: Model Evaluation
import joblib

joblib.dump(clf5, "Project1_F2025.joblib")

loaded_clf = joblib.load("Project1_F2025.joblib")

testdata = np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])

testdata = sc.transform(testdata)
predictions = loaded_clf.predict(testdata)
print(predictions)