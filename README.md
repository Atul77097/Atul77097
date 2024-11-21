# Step 1 : Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Step 2 : Import the dataset
default = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')

# Display the first few rows of the data
print(default.head())

# Display general info and statistics about the dataset
default.info()
default.describe()

# Checking the class distribution
print(default['Default'].value_counts())

# Check column names to confirm no issues
print(default.columns)

# Step 3: Prepare the data for training
# Separate the features (X) and target variable (y)
y = default['Default']
X = default.drop(['Default'], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Optional: Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warning occurs
model.fit(X_train_scaled, y_train)

# Get the intercept and coefficients of the model
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Step 5: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate the model's performance
# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification report (precision, recall, f1-score, support)
print("Classification Report:")
print(classification_report(y_test, y_pred))
