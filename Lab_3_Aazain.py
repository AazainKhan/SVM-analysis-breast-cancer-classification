
# # Lab assignment #3: “Support Vector Machines”
# Aazain Ullah Khan - 301277063 - COMP 247
# <hr>


# ## Excercise 1


import pandas as pd
import numpy as np
data_aazain = pd.read_csv('breast_cancer.csv')


# initial investigation
data_aazain.head()


data_aazain.info()


# 1. Sample code number: id number
# 2. Clump Thickness: 1 - 10
# 3. Uniformity of Cell Size: 1 - 10
# 4. Uniformity of Cell Shape: 1 - 10
# 5. Marginal Adhesion: 1 - 10
# 6. Single Epithelial Cell Size: 1 - 10
# 7. Bare Nuclei: 1 - 10
# 8. Bland Chromatin: 1 - 10
# 9. Normal Nucleoli: 1 - 10
# 10. Mitoses: 1 - 10
# 11. Class: (2 for benign, 4 for malignant)


# missing values
data_aazain.isnull().sum()


data_aazain.describe()


# replace ? to NaN and convert to float in bare column
data_aazain['bare'] = data_aazain['bare'].replace('?', np.nan)
data_aazain['bare'] = data_aazain['bare'].astype(float)


data_aazain.info()


data_aazain.isnull().sum()


# fill missing values with median value of column
data_aazain = data_aazain.fillna(data_aazain.median())


data_aazain.describe()  # now shows the "bare" column statistics


# drop id column
data_aazain = data_aazain.drop('ID', axis=1)


# correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
correlation_matrix = data_aazain.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()


# 3d plot
from mpl_toolkits import mplot3d

# features
feature_1 = 'thickness'
feature_2 = 'size'
feature_3 = 'bare'
X_plot = data_aazain[[feature_1, feature_2, feature_3]]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=20)
ax.scatter3D(X_plot[feature_1], X_plot[feature_2], X_plot[feature_3],
             c=data_aazain['class'], cmap='rainbow')
ax.set_xlabel(feature_1)
ax.set_ylabel(feature_2)
ax.set_zlabel(feature_3)
plt.show()


# count plot for the target variable
sns.countplot(x='class', data=data_aazain)
plt.title('Distribution of Benign (2) and Malignant (4) Tumors')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

print('Benign Tumors:', data_aazain['class'].value_counts()[2])
print('Malignant Tumors:', data_aazain['class'].value_counts()[4])


# separate the features from the class
X = data_aazain.drop('class', axis=1)
y = data_aazain['class']


# split your data into train 80% and 20% test. random seed = 63

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=63)


# train svm classifier with linear kernel
from sklearn.svm import SVC

clf_linear_aazain = SVC(kernel='linear', C=0.1)
clf_linear_aazain.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix

# predictions
y_train_pred = clf_linear_aazain.predict(X_train)
y_test_pred = clf_linear_aazain.predict(X_test)

# accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix: \n", conf_matrix)



# repeat the process with rbf kernel and no C value

clf_rbf_aazain = SVC(kernel='rbf')
clf_rbf_aazain.fit(X_train, y_train)


# predictions
y_train_pred = clf_rbf_aazain.predict(X_train)
y_test_pred = clf_rbf_aazain.predict(X_test)

# accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix: \n", conf_matrix)


# repeat with poly kernel and C is not set to any value

clf_poly_aazain = SVC(kernel='poly')
clf_poly_aazain.fit(X_train, y_train)


# predictions
y_train_pred = clf_poly_aazain.predict(X_train)
y_test_pred = clf_poly_aazain.predict(X_test)

# accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix: \n", conf_matrix)


# repeat with sigmoid kernel and C is not set to any value

clf_sigmoid_aazain = SVC(kernel='sigmoid')
clf_sigmoid_aazain.fit(X_train, y_train)


# predictions
y_train_pred = clf_sigmoid_aazain.predict(X_train)
y_test_pred = clf_sigmoid_aazain.predict(X_test)

# accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix: \n", conf_matrix)


# ## Excercise 2


data_aazain_df2 = pd.read_csv('breast_cancer.csv')


# replace ? to nan in bare column
data_aazain_df2['bare'] = data_aazain_df2['bare'].replace('?', np.nan)


# drop id column
data_aazain_df2 = data_aazain_df2.drop('ID', axis=1)

# separate the features from the class
X = data_aazain_df2.drop('class', axis=1)
y = data_aazain_df2['class']


# split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=63)


# transform training data with imputer and scaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# fill missing values with median
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)


# combine transformers into a pipeline
from sklearn.pipeline import Pipeline

num_pipe_aazain = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# pipline with num_pipe_aazain and svm
pipe_svm_aazain = Pipeline([
    ('preprocessing', num_pipe_aazain),
    ('svm', SVC(random_state=63))
])


num_pipe_aazain


# define the grid search parameters
param_grid = [
    {'svm__kernel': ['linear', 'rbf', 'poly'],
     'svm__C': [0.01, 0.1, 1, 10, 100],
     'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
     'svm__degree': [2, 3]}
]

param_grid


# create a grid search object
from sklearn.model_selection import GridSearchCV

grid_search_aazain = GridSearchCV(
    pipe_svm_aazain, param_grid, scoring='accuracy', refit=True, verbose=3)

grid_search_aazain


# fit training data to the gird search object
grid_search_aazain.fit(X_train, y_train)


# print out the best parameters best estimator
print(f"Best Parameters: {grid_search_aazain.best_params_}")
print(f"Best Estimator: {grid_search_aazain.best_estimator_}")


best_model_aazain = grid_search_aazain.best_estimator_


# fit the best model to the training data
best_model_aazain.fit(X_train, y_train)

# predictions
y_train_pred = best_model_aazain.predict(X_train)
y_test_pred = best_model_aazain.predict(X_test)

# accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")


# save the model using joblib
import joblib

joblib.dump(best_model_aazain, 'best_model_aazain.pkl')

# save the pipeline using joblib
joblib.dump(pipe_svm_aazain, 'full_pipeline_aazain.pkl')


