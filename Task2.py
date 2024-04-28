
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Read data
df_movies = pd.read_csv("C:/Users/HP/Documents/IMDbMoviesIndia.csv")
df_ratings = pd.read_csv("C:/Users/HP/Documents/IMDbMoviesIndia.csv")
df_users = pd.read_csv("C:/Users/HP/Documents/IMDbMoviesIndia.csv")

# Rename columns
df_movies.columns = ['MovieID', 'MovieName', 'Genre']
df_ratings.columns = ['UserID', 'MovieID', 'Ratings', 'TimeStamp']
df_users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

# Merge dataframes
user_ratings = pd.merge(df_ratings, df_users, on='UserID')
movie_ratings = df_movies.merge(user_ratings, on='MovieID')

# Visualizations and Data Analysis (Code not changed)

# Model training
features = movie_ratings[['MovieID', 'Age', 'Occupation']].values
labels = movie_ratings['Ratings'].values

# Splitting the dataset into training and testing data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.3)

# Logistic Regression
lr = LogisticRegression()
lr.fit(train, train_labels)
y_predict = lr.predict(test)
acc_model1 = accuracy_score(test_labels, y_predict)
score1 = round(lr.score(train, train_labels) * 100, 2)

Classification_Report = classification_report(test_labels, y_predict)

print('accuracy score of Logistic Regression model is :', acc_model1)
print('score of Logistic Regression model is :', score1)
print('classification report of Logistic Regression model is :', Classification_Report)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, train_labels)
y_pred_rf = rf.predict(test)
acc_model2 = accuracy_score(test_labels, y_pred_rf)
score2 = round(rf.score(train, train_labels) * 100, 2)

Classification_Report = classification_report(test_labels, y_pred_rf)

print('accuracy score of Random Forest model is :', acc_model2)
print('score of  Random Forest model is :', score2)
print('classification report of Random Forest model is :', Classification_Report)

# kNN classifier
knn = KNeighborsClassifier()
knn.fit(train, train_labels)
y_pred_knn = knn.predict(test)
acc_model3 = accuracy_score(test_labels, y_pred_knn)
score3 = round(knn.score(train, train_labels) * 100, 2)

Classification_Report = classification_report(test_labels, y_pred_knn)

print('accuracy score of KNN model is :', acc_model3)
print('score of  KNN model is :', score3)
print('classification report of KNN model is :', Classification_Report)

models = pd.DataFrame({'Model': ['Logistic Regression', 'Random Forest Classifier', 'K-Nearest Neighbor'],
                       'Score': [score1, score2, score3]})

models.sort_values(ascending=False, by='Score')

# Debugging Prediction
# Check Data Format
print("Train shape:", train.shape)
print("Labels shape:", labels.shape)

# Print the input data for prediction
input_data = [[1150, 45, 1]]  # Example input data for prediction
print("Input data for prediction:", input_data)

# Predict using Random Forest model
prediction = rf.predict(input_data)
print("Prediction:", prediction)