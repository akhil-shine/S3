from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,classification_report

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x_train,y_train)

print(dt.predict(x_test))
v=dt.predict(x_test)
report=classification_report(y_test,v)
result=accuracy_score(y_test,v)

print("Accuracy::",result)
print("\nClassification Report::\n",report)

plt.figure(figsize=(15, 10))
plot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()

