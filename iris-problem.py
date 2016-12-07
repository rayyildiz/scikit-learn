import numpy
from sklearn import datasets
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus

iris = datasets.load_iris()

test_idx = [0,50,100]

# Training
train_target = numpy.delete(iris.target,test_idx)
train_data = numpy.delete(iris.data, test_idx, axis=0)

# Testing
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]



clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)


print(test_target)
print(clf.predict(test_data))


# Viziluation


dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=True)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("./out/iris.pdf")
graph.write_png("./out/iris.png")

