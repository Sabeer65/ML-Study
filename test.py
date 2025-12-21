import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

diabetes = pd.read_csv('diabetes.csv')

#count = diabetes['Outcome'].value_counts()
#print(count)

# diabetes.groupby('Outcome').mean() # Group dataset to check values

X = diabetes.drop(columns='Outcome', axis=1)
Y = diabetes['Outcome']

# Data standardization 

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=1)

