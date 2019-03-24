# Build a forest and compute the feature importances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import system

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz



#dot_data = tree.export_graphviz(CLF, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris")


dataset = pd.read_csv('SalmonPredictionData3.csv')
X = dataset.iloc[:, 4:8].values
y = dataset.iloc[:, 0].values


forest = DecisionTreeRegressor(random_state=0)


forest.fit(X, y)
importances = forest.feature_importances_



#clf = tree.DecisionTreeRegressor()

#tree.export_graphviz(clf, out_file='tree.dot')      


# Print the feature ranking
print("Feature ranking:")
print(importances)

y = importances
N = len(y)
x = range(N)
width = 1/1.5
plt.bar(x, y, width, color="green")
labels =["Temperature", "Water discharge", "Precipitation", "Water Temperature" ]
plt.xticks(x,labels)
plt.ylabel('Relative Correlation')
plt.xlabel('Variables Inputted')




