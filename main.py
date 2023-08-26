import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn import ensemble



data = load_breast_cancer()

X,y = data.data,data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

import seaborn as sns
def scatter_plot(feature1, feature2):
    plt.scatter(X_train[:, feature1], X_train[:, feature2], c=y_train, cmap='viridis')
    plt.xlabel(f'Feature {feature1}')
    plt.ylabel(f'Feature {feature2}')
    plt.title(f'Scatter plot of Feature {feature1} vs Feature {feature2}')
    plt.colorbar(label='Class')
    plt.show()

# Visualize two features using scatter plot
scatter_plot(0, 1)  # Example features, you can choose any two features

# Visualize correlation matrix using a heatmap
correlation_matrix = np.corrcoef(X_train, rowvar=False)
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


tree_list = []

for i in range(100):
    tree = DecisionTreeClassifier(max_features='sqrt')
    subset_indices = np.random.choice(np.arange(len(X_train)), size=len(X_train) // 2, replace=False) 
    X_train_subset = X_train[subset_indices]
    y_train_subset = y_train[subset_indices]
    tree.fit(X_train_subset, y_train_subset)  # Fit the tree with the subset of training data
    tree_list.append(tree)

preds =[]


for i , tree in enumerate(tree_list):
    individual_preds = tree.predict(X_test)
    individual_accuracy = accuracy_score(y_test,individual_preds)
    print(f"tree {i +1 } accuracy: {individual_accuracy}")
    preds.append(individual_preds)

preds = np.array(preds)
mean_preds = np.round(np.mean(preds,axis = 0 ))

ensemble_accuracy = accuracy_score(y_test,mean_preds)

print(ensemble_accuracy)