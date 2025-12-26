import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

flower_dataset = pd.read_csv('iris.csv')

# Splitting dataset
X = flower_dataset.drop(columns='species',axis=1)
Y = flower_dataset['species']

# train test split 

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

# model training 

model = LogisticRegression()

model.fit(X_train,Y_train)

# prediction 

training_pred = model.predict(X_train)
train_accuracy = accuracy_score(training_pred,Y_train)

test_pred = model.predict(X_test)
test_accuracy = accuracy_score(test_pred,Y_test)

print(f'Accuracy on training data: {train_accuracy * 100:.2f}%\n Accuracy on test data: {test_accuracy*100:.2f}%')