import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('C:/Users/HP/Documents/IRIS.csv')


print(df.head())
print(df.tail())
print(df.describe())
print(df.info())
print(df['species'].value_counts())
print(df.isnull().sum())

# Visualizing  the data
df['sepal_length'].hist()
plt.title('Sepal Length')
plt.show()




le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Splitting the data into training and testing sets
X = df.drop(columns=['species'])
Y = df['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# Training and evaluating models
models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test) * 100
    print(f'Accuracy of {name}: {accuracy:.2f}%')
