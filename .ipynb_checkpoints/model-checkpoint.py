import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv("50 Startups.csv")

# Select independent (X) and dependent (y) variables
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]  # Features
y = df['Profit']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
pickle.dump(model, open("model.pkl", "wb"))

print("Linear Regression model has been trained and saved as model.pkl")
