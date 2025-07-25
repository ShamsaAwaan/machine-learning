from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
data ={'Experience':[1,2,3,4,5],'Salary':[30000,35000,50000,55000,60000]};
df = pd.DataFrame(data)
print(df.shape)
x= df[['Experience']]
y= df['Salary']
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2)
print(" training feature\n",x_train)
print("testing feature\n",x_test)
print("training label\n",y_train)
print("testing label\n",y_test)
print(x_train.shape)
print(y_train.shape)
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print ("Predictions:", predictions)
print ("MSE:",mean_squared_error(y_test,predictions))