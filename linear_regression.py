import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class ScratchLR:
    def __init__(self):
        self.m=None
        self.b=None

    def fit(self, X_train,y_train):
        num=0
        den=0
        for i in range(X_train.shape[0]):
            num= num + ((X_train[i]- X_train.mean()) * (y_train[i]- y_train.mean()))
            den= den + (X_train[i]- X_train.mean())**2
        self.m= num/den
        self.b= y_train.mean() - (self.m*X_train.mean())
        print("m=", self.m)
        print("b=", self.b)
            
    def predict(self, X_test):
        print(X_test)
        return self.m*X_test+self.b
    

df= pd.read_csv("placement.csv")

print(df.head())

X = df.iloc[:, 0].values 
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

lr = ScratchLR()
lr.fit(X_train, y_train)

# Test prediction
user_input= float(input("enter the cgpa:"))
print("predicted package:", lr.predict(user_input))
