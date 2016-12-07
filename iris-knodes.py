import sklearn.datasets as ds
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = ds.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)


# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

clf = KNeighborsClassifier()

clf.fit(x_train,y_train)

predictions = clf.predict(x_test)

print(accuracy_score(y_test,predictions))



