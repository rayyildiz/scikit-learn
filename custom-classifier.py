from scipy.spatial import distance

def measurement(a,b):
    return distance.euclidean(a,b)


class MyClassifier():
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predictions = []

        for row in x_test:
            label = self.closet(row)
            predictions.append(label)
        return  predictions

    def closet(self,row):
        best_dist = measurement(row,x_train[0])
        best_idx = 0

        for i in range(len(self.x_train)):
            dist = measurement(row,self.x_train[i])
            if ( dist < best_dist):
                best_dist = dist
                best_idx = i

        return self.y_train[best_idx]

import sklearn.datasets as ds

iris = ds.load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)


clf = MyClassifier()

clf.fit(x_train,y_train)

predictions = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))



