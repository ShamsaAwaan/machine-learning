from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load iris dataset
# iris = load_iris()
# X = iris.data
# y = (iris.target == 0).astype(int)  # Binary classification: Is setosa or not
# Load dataset
iris = load_iris()
x, y = iris.data, iris.target
# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Fit Logistic Regression
# clf = LogisticRegression()
# clf.fit(x_train, y_train)

model = KNeighborsClassifier(n_neighbors=8)
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
print (y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

