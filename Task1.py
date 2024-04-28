import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

test_df = pd.read_csv("C:/Users/HP/Documents/Titanic-Dataset.csv")
train_df = pd.read_csv("C:/Users/HP/Documents/Titanic-Dataset.csv")

train_df.info()
train_df.head(8)
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)

train_df.columns.values


# Defining the  data 
survived = 'survived'
not_survived = 'not survived'
women_survived = train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 1)]['Age'].dropna()
women_not_survived = train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 0)]['Age'].dropna()
men_survived = train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 1)]['Age'].dropna()
men_not_survived = train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 0)]['Age'].dropna()

# Creating the subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot for females
axes[0].hist([women_survived, women_not_survived], bins=18, label=[survived, not_survived], alpha=0.6, density=True)
axes[0].legend()
axes[0].set_title('Female')

# Plot for males
axes[1].hist([men_survived, men_not_survived], bins=18, label=[survived, not_survived], alpha=0.6, density=True)
axes[1].legend()
axes[1].set_title('Male')

plt.show()