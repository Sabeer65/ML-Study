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


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

# Model training 
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy evaluation for training data 

X_train_prediction= classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_accuracy)

# Accuracy evaluation for training data 

X_test_prediction= classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accuracy)


# prediction system 
feature_names = diabetes.drop(columns='Outcome').columns
input_df = pd.DataFrame([[7,103,66,32,0,39.1,0.344,31]], columns=feature_names)

# 2. Transform and Predict
std_data = scaler.transform(input_df)
prediction = classifier.predict(std_data)

# 3. Output the result clearly
print(f"Prediction: {prediction[0]}")
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')