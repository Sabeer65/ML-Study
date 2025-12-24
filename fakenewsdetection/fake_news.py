import numpy as np 
import pandas as pd 
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# print(stopwords.words('english')) # Stopwords used 

stop_words = set(stopwords.words('english'))

# Function to check for and display nullwords
"""
def print_null_rows(df): 
      null_rows = news_data[df.isnull().any(axis=1)]
      if len(null_rows) > 0:
            print(f"Found {len(null_rows)} rows with null values:")
            print(null_rows)
      else:
            print("No null values found in the dataset!")

      return null_rows
"""

news_data = pd.read_csv('FakeNewsNet.csv')

# print_null_rows(news_data) # running null value function 

news_data = news_data.fillna('') # replacing null rows with empty string

# merging title and source domain 

news_data['content'] = news_data['source_domain']+' '+news_data['title']


# stemming (reducing word to root form)

port_stem = PorterStemmer()

def stemming(content):
    stemmed = re.sub('[^a-zA-Z]',' ', content).lower().split()
    stemmed = [port_stem.stem(word) for word in stemmed if not word in stop_words]    
    return ' '.join(stemmed)

news_data['content'] = news_data['content'].apply(stemming)

# print(news_data['content'])

# separating the data 

X = news_data['content'].values
Y = news_data['real'].values

# converting text data to numerical data

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# splitting data into training and test data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

# Training the model

model = LogisticRegression()
model.fit(X_train,Y_train)

# Accuracy score 

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction,Y_train)

X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction,Y_test)

print(f'Accuracy score of training : {training_accuracy}\n Accuracy score of test : {test_accuracy}')


# Testing 

X_input = X_test[0]

prediction = model.predict(X_input)

if prediction == 0:
    print('News is Fake')
else:
    print("New is real")

print(Y_test[0])