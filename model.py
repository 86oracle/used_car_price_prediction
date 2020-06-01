import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 12].values

# Encoding some independent variable
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1]) 
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = OneHotEncoder(categorical_features = [1])
onehotencoder = OneHotEncoder(categorical_features = [2])
onehotencoder = OneHotEncoder(categorical_features = [3])
onehotencoder = OneHotEncoder(categorical_features = [4])
onehotencoder = OneHotEncoder(categorical_features = [5])
onehotencoder = OneHotEncoder(categorical_features = [6])

X = onehotencoder.fit_transform(X).toarray()


#Splitting data into Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting Multiple Linear Regression to Trainig set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train )      

pickle.dump(regressor, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl" , "rb"))
print(model.predict([[4, 300, 500]]))
        
