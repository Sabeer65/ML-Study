import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  

# Load dataset
sonar_data = pd.read_csv("sonar data.csv", header = None)

# Split into Data and Labels
X = sonar_data.drop(columns=60, axis = 1)
Y = sonar_data[60]

# Train and Test Data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)

# Model Training -> Logistic Regression 
model = LogisticRegression()

model.fit(X_train, Y_train) # Training model with training data

# Accuracy on training data 
X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)

print(f"Accuracy on training : ", train_accuracy)

# Accuracy on training data 
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)

print(f"Accuracy on test : ", test_accuracy)

# Making a Predictive System 

input_data = (0.0654,0.0649,0.0737,0.1132,0.2482,0.1257,0.1797,0.0989,0.2460,0.3422,0.2128,0.1377,0.4032,0.5684,0.2398,0.4331,0.5954,0.5772,0.8176,0.8835,0.5248,0.6373,0.8375,0.6699,0.7756,0.8750,0.8300,0.6896,0.3372,0.6405,0.7138,0.8202,0.6657,0.5254,0.2960,0.0704,0.0970,0.3941,0.6028,0.3521,0.3924,0.4808,0.4602,0.4164,0.5438,0.5649,0.3195,0.2484,0.1299,0.0825,0.0243,0.0210,0.0361,0.0239,0.0447,0.0394,0.0355,0.0440,0.0243,0.0098)

# Changing datatype to numpy array
input_data_num = np.asarray(input_data)

# Reshape np array (predicting for one instance)
input_reshaped = input_data_num.reshape(1, -1)

# Predicting for input data
prediction = model.predict(input_reshaped)

if (prediction[0] == 'R'):
    print("It's a rock")
elif (prediction[0] == 'M'):
    print("It's a Mine")
else:
    print("Error in prediction")