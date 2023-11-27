# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
np.random.seed(42)

# Generate some example data
X = np.random.rand(100, 1) * 10  # House sizes (features)
y = 3 * X + np.random.randn(100, 1) * 2  # House prices (target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
regressor = LinearRegression()

# Train the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


# Now, let's make predictions on new input data
new_house_sizes = np.array([200, 500, 800]).reshape(-1, 1)  # New house sizes data

# Suppress the warning about feature names
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure that the feature names match those used during training
new_predictions = regressor.predict(new_house_sizes)
print(f"Predictions on New Data: {new_predictions}")

import matplotlib.pyplot as plt

# Plotting the linear regression results
plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_train, y_train, label='Training Data', color='blue', marker='o')
plt.scatter(X_test, y_test, label='Testing Data', color='green', marker='o', alpha=0.5)

# Plot the linear regression line
x_values = np.linspace(0, 10, 100).reshape(-1, 1)
y_values = regressor.predict(x_values)

plt.plot(x_values, y_values, label='Linear Regression Line', color='red')

plt.title('Linear Regression - House Prices')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.legend()
plt.savefig('output_plot_reg.png')