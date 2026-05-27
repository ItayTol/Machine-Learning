''' 318720604
    Itay Toledo '''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score as precision, recall_score as recall, f1_score as f1, roc_auc_score as auc, confusion_matrix as confusion
from sklearn.model_selection import GridSearchCV as grid, RandomizedSearchCV as randomized

# Filling the train set and the test set
folder = 'C:/Users/dror_toledo/Desktop/Y4S1/Machine Learning/Ex3/'
train = pd.read_csv(folder+'mnist_train.csv', header=None)
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, 0]

test = pd.read_csv(folder+'mnist_test.csv', header=None)
test_x = test.iloc[:, 1:]
test_y = test.iloc[:, 0]


# Logistic Regression
now=datetime.now()
logit_model = LogisticRegression()
logit_model.fit(train_x, train_y)
logit_pred=logit_model.predict(test_x)
print('Time spent in logistic regression is', datetime.now()-now)
# Metrics
logit_precision=precision(test_y, logit_pred, average='macro')
logit_recall=recall(test_y, logit_pred, average='macro')
logit_f1=f1(test_y, logit_pred, average='macro')
logit_auc=auc(test_y, logit_model.predict_proba(test_x), multi_class='ovr')
logit_confusion=confusion(test_y, logit_pred)
# Plot confusion matrix
sns.heatmap(logit_confusion, annot=True, annot_kws={"fontsize":8}, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label')
plt.title('Logistic regression - confusion matrix', fontsize=14)
plt.xlabel('Predicted label')
plt.savefig('logistic regression confusion matrix')


# Decision Tree
now=datetime.now()
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(train_x, train_y)
tree_y_pred=tree_model.predict(test_x)
print('Time spent in decision tree is', datetime.now()-now)
# Metrics
tree_precision=precision(test_y, tree_y_pred, average='macro')
tree_recall=recall(test_y, tree_y_pred, average='macro')
tree_f1=f1(test_y, tree_y_pred, average='macro')
tree_auc=auc(test_y, tree_model.predict_proba(test_x), multi_class='ovr')
tree_confusion=confusion(test_y, tree_y_pred)
# Plot confusion matrix
sns.heatmap(tree_confusion, annot=True, annot_kws={"fontsize":8}, fmt=".3f", linewidths=.5, square = True, cmap = 'plasma_r');
plt.ylabel('Actual label')
plt.title('Decision tree - confusion matrix', fontsize=14)
plt.xlabel('Predicted label')
plt.savefig('Decision tree confusion matrix')


# Random Forest
now=datetime.now()
forest_model = RandomForestClassifier()
forest_model.fit(train_x, train_y)
forest_y_pred=forest_model.predict(test_x)
print('Time spent in random forest', datetime.now()-now)
# Metrics
forest_precision=precision(test_y, forest_y_pred, average='macro')
forest_recall=recall(test_y, forest_y_pred, average='macro')
forest_f1=f1(test_y, forest_y_pred, average='macro')
forest_confusion=confusion(test_y, forest_y_pred)
forest_auc=auc(test_y, forest_model.predict_proba(test_x),multi_class='ovr')
# Plot confusion matrix
sns.heatmap(forest_confusion, annot=True, annot_kws={"fontsize":8}, fmt=".3f", linewidths=.5, square = True, cmap = 'cool');
plt.ylabel('Actual label')
plt.title('Random forest - confusion matrix', fontsize=14)
plt.xlabel('Predicted label')
plt.savefig('Random forest confusion matrix')


# Decision Tree - Grid Search CV
now=datetime.now()
parameters={'criterion': ['gini', 'entropy'],
            'max_depth': [10, 100, 500, 1000],
            'min_samples_leaf': [10, 100, 500, 1000]}
grid_search=grid(tree_model, parameters, cv=5, n_jobs=8)
grid_search.fit(train_x, train_y)
grid_best=grid_search.best_params_
grid_pred=grid_search.predict(test_x)
print('time spent in grid search is', datetime.now()-now)
# Metrics
grid_precision=precision(test_y, grid_pred, average='macro')
grid_recall=recall(test_y, grid_pred, average='macro')
grid_f1=f1(test_y, grid_pred, average='macro')
grid_auc=auc(test_y, grid_search.predict_proba(test_x), multi_class='ovr')
grid_confusion=confusion(test_y, grid_pred)
# Plot confusion matrix

plt.figure(figsize=(9,9))
sns.heatmap(grid_confusion, annot=True, annot_kws={"fontsize":8}, fmt=".3f", linewidths=.5, square = True, cmap = 'icefire_r');
plt.ylabel('Actual label')
plt.title('grid search of decision tree - confusion matrix', fontsize=14)
plt.xlabel('Predicted label')
plt.savefig('grid search of decision tree - confusion matrix')


# Random Forest - Randomized Search CV
now=datetime.now()
parameters={'n_estimators': [10, 100, 500, 1000],
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 100, 500, 1000],
            'min_samples_leaf': [10, 100, 500, 1000]}
randomized_search=randomized(forest_model, parameters, n_iter=10, cv=4, n_jobs=10)
randomized_search.fit(train_x, train_y)
rand_best=randomized_search.best_params_
rand_pred=randomized_search.predict(test_x)
print('Time spent in randomized search is', datetime.now()-now)
# Metrics
rand_precision=precision(test_y, rand_pred, average='macro')
rand_recall=recall(test_y, rand_pred, average='macro')
rand_f1=f1(test_y, rand_pred, average='macro')
rand_auc=auc(test_y, randomized_search.predict_proba(test_x), multi_class='ovr')
rand_confusion=confusion(test_y, rand_pred)
# Plot confusion matrix
sns.heatmap(rand_confusion, annot=True, annot_kws={"fontsize":8}, fmt=".3f", linewidths=.5, square = True, cmap = 'icefire');
plt.ylabel('Actual label')
plt.title('Randomized search of random forest - confusion matrix', fontsize=14)
plt.xlabel('Predicted label')
plt.savefig('Randomized search of random forest - confusion matrix')