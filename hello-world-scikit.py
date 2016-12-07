from sklearn import tree

# smooth = 0
# bumpy = 1
features = [[140,0], [130, 0], [150, 1], [170,1]]

# apple = 0
# orange = 1
labels = [0, 0, 1, 1]


classifier = tree.DecisionTreeClassifier()

clf = classifier.fit(features,labels)

print( classifier.predict([[140,0]]) )