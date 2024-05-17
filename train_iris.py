import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.feature_names)
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print(X_train.shape, y_train.shape)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.predict(X_test))
joblib.dump(model, 'model.joblib')

print(accuracy_score(y_test, model.predict(X_test)))