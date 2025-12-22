import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

house_data = pd.read_csv('housing.csv', sep=r'\s+', header=None) # Data has no separators so using regular expression to seperate between white spaces
house_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] # Adding column names.

"""print(house_data.head())
print(house_data.isnull().sum())
print(house_data.describe())

# correlation and heatmap plot
correlation = house_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues') 
plt.show()
"""

X = house_data.drop(['MEDV'], axis=1)
Y = house_data['MEDV']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training 

model = XGBRegressor()

model.fit(X_train, Y_train)


# Training data evaluation 

trainining_data_predicition = model.predict(X_train)

# R squared error 
score_1 = metrics.r2_score(Y_train, trainining_data_predicition)
# Mean Absolute error
score_2 = metrics.mean_absolute_error(Y_train, trainining_data_predicition)

print(f'R squared error : ', score_1, ', Mean Absolute error : ', score_2)

plt.scatter(Y_train, trainining_data_predicition)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Price vs Predicted Price on training data ')
plt.show()

# Test data evaluation 

test_data_predicition = model.predict(X_test)

# R squared error 
real_r2 = metrics.r2_score(Y_test, test_data_predicition)
# Mean Absolute error
real_mae = metrics.mean_absolute_error(Y_test, test_data_predicition)

print(f'R squared error : ', real_r2, ', Mean Absolute error : ', real_mae)

plt.scatter(Y_test, test_data_predicition)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Price vs Predicted Price on test data ')
plt.show()
