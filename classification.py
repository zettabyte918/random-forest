# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Generate example data
import numpy as np
from sklearn.tree import plot_tree
np.random.seed(42)
# Assuming the relationship: more hours -> higher chance of passing
hours_studied = np.random.randint(0, 10, 100)  # Random hours between 0 and 10
pass_or_fail = (hours_studied + np.random.randn(100) * 2) > 5  # Adding some randomness

# Create a DataFrame for clarity
import pandas as pd
data = pd.DataFrame({'Hours_Studied': hours_studied, 'Pass_or_Fail': pass_or_fail})

# Separate features (X) and target variable (y)
X = data[['Hours_Studied']]
y = data['Pass_or_Fail'].astype(int)  # Convert boolean to 0 or 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Data: {accuracy:.2f}")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf.estimators_[0], feature_names=['Hours_Studied'], class_names=['Fail', 'Pass'], filled=True, rounded=True)
plt.title("Example Decision Tree for Classification")
plt.savefig('output_plot_class_tree.png')

# Now, let's make predictions on new input data
new_hours_studied = np.array([2, 4, 8]).reshape(-1, 1)  # New hours studied data

# Suppress the warning about feature names
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure that the feature names match those used during training
new_predictions = clf.predict(new_hours_studied)
print(f"Predictions on New Data: {new_predictions}")

# Plotting the decision boundary
plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_train, y_train, label='Training Data', color='blue', marker='o')
plt.scatter(X_test, y_test, label='Testing Data', color='green', marker='o', alpha=0.5)

# Plot the decision boundary
x_values = np.linspace(0, 10, 100).reshape(-1, 1)
y_values = clf.predict(x_values)
plt.plot(x_values, y_values, label='Decision Boundary', color='red', linestyle='--')

plt.title('Random Forest Classifier - Decision Boundary')
plt.xlabel('Hours Studied')
plt.ylabel('Pass or Fail (0 or 1)')
plt.legend()
plt.savefig('output_plot_class.png')