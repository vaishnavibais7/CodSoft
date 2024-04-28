import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("C:/Users/HP/Downloads/advertising.csv")


print(df.head())
print(df.shape)
print(df.describe())


sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


df['TV'].plot.hist(bins=10, color="green")
plt.xlabel("TV")
plt.show()

df['Radio'].plot.hist(bins=10, color="orange")
plt.xlabel("Radio")
plt.show()

df['Newspaper'].plot.hist(bins=10, color="black")
plt.xlabel("Newspaper")
plt.show()


sns.heatmap(df.corr(), annot=True)
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df['Sales'], test_size=0.3, random_state=0)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
res = model.predict(X_test)
print(res)


print("Accuracy Score: ", model.score(X_test, y_test) * 100)


print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Plotting
plt.scatter(X_test, y_test)
plt.plot(X_test, model.predict(X_test), color='red')
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("Linear Regression Fit")
plt.show()
