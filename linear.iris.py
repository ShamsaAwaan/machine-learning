from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("Iris.csv")
print(df.head())
print("shape\n",df.shape)
print(df.value_counts)
print(df.nunique())
print(df['Species'].unique())
x = df.drop(['Id','Species'], axis=1)
y = df['Species']


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)
print("only features\n", x)
print(x.shape)

print("\n Label \n",y)

X_train,X_test, y_train,y_test  = train_test_split(x,y)
print(X_train.shape)
print("train_feature")
# clf = LogisticRegression()
# clf.fit(x_train, y_train)

model = KNeighborsClassifier(n_neighbors=31)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print (y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
