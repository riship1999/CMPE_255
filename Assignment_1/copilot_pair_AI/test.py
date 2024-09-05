import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


"""
 1)Basic Data Exploration
 2)Handling Missing Values --> Check for missing values in the dataset and handle them appropriately.
 3) Model Building --> Build a machine learning model to predict the survival of passengers using logistic regression.
 4) Exploratory Data Analysis (EDA) --> Survival Rate by Gender, 
 5) Model Evaluation --> Evaluate the model using accuracy, precision, 
 recall, F1-score, confusion matrix.

"""
# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Perform basic data exploration
# Display the first few rows of the dataset
print(titanic_data.head())

# Check the shape of the dataset
print("Shape of the dataset:", titanic_data.shape)

# Check for missing values in the dataset
print("Missing values:\n", titanic_data.isnull().sum())

# Get summary statistics of the dataset
print("Summary statistics:\n", titanic_data.describe())

# Handle missing values
# Drop rows with missing values
titanic_data.dropna(inplace=True)

# Check for missing values again to confirm
print("Missing values after handling:", titanic_data.isnull().sum())

# Select the features and target variable
X = titanic_data[['Age', 'Sex', 'Fare']]
y = titanic_data['Survived']

# Convert categorical variables to numerical
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# Calculate confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_matrix)

import matplotlib.pyplot as plt

# Plot survival rate by gender
survival_by_gender = titanic_data.groupby('Sex')['Survived'].mean()
survival_by_gender.plot(kind='bar', color=['blue', 'pink'])
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()