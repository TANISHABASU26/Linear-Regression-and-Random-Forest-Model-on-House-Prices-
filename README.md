# # Linear-Regression-and-Random-Forest-Model-on-House-Prices-

## Overview

This project focuses on predicting house prices based on various features of the houses. The dataset includes features such as location, size, number of rooms, age, median income, and proximity to the ocean. The project uses Linear Regression and Random Forest Regressor to build predictive models and evaluates their performance using several key business metrics.

## Dataset

The dataset used in this project includes the following columns:
- `longitude`: Longitude of the house location
- `latitude`: Latitude of the house location
- `housing_median_age`: Median age of the houses in the area
- `total_rooms`: Total number of rooms in the houses
- `total_bedrooms`: Total number of bedrooms in the houses
- `population`: Population in the area
- `households`: Number of households in the area
- `median_income`: Median income of the households
- `median_house_value`: Median house value (target variable)
- `ocean_proximity`: Proximity to the ocean (categorical variable)

## Preprocessing

1. Handle missing values by filling them with the median of the respective columns.
2. Convert the categorical variable `ocean_proximity` into dummy variables using one-hot encoding.
3. Concatenate the dummy variables with the original dataset and drop the original `ocean_proximity` column.

## Exploratory Data Analysis

### Correlation Matrix Heatmap
A heatmap is created to visualize the correlation between different features and the target variable, `median_house_value`.
 ![image](https://github.com/TANISHABASU26/Portfolio/assets/174117644/13d73fb4-fba2-42b6-a5df-27a44c43b0d6)


### Key Visualizations
1. **Median House Price with Respect to Median Income**: Shows the relationship between median income and median house prices.
![image](https://github.com/TANISHABASU26/Portfolio/assets/174117644/937ab861-bef4-4676-a349-38ee4ef2d4de)
2. **Value of House with Respect to Age**: Visualizes how house prices vary with the age of the houses.
![image](https://github.com/TANISHABASU26/Portfolio/assets/174117644/1d9a3165-cf94-4e6e-ac43-e936c29ab91a)

## Model Building and Evaluation

### Linear Regression
- Train a Linear Regression model on the training data.
- Evaluate the model using the following metrics:
  - Mean Absolute Error (MAE): $50,670.74
  - Mean Squared Error (MSE): $4,908,476,721.16
  - R-squared (R²): 0.63
  - Median Absolute Error: $38,377.26
  - Explained Variance Score: 0.63

### Random Forest Regressor
- Use GridSearchCV to find the best hyperparameters for the Random Forest Regressor.
- Train the model using the best parameters and evaluate it using the same metrics as above:
  - Mean Absolute Error (MAE): $31,447.06
  - Mean Squared Error (MSE): $2,378,261,530.83
  - R-squared (R²): 0.82
  - Median Absolute Error: $18,985.05
  - Explained Variance Score: 0.82

## Key Business Metrics Insights

### Linear Regression Model
- **Mean Absolute Error (MAE)**: Indicates the average absolute difference between the predicted and actual house prices.
- **Mean Squared Error (MSE)**: Indicates the average squared difference between the predicted and actual house prices, giving more weight to larger errors.
- **R-squared (R²)**: Shows how well the model explains the variance in the actual house prices. A value closer to 1 indicates a better fit.
- **Median Absolute Error**: Provides the median of the absolute differences between the predicted and actual prices, less sensitive to outliers compared to MAE.
- **Explained Variance Score**: Indicates the proportion of variance explained by the model, with values closer to 1 indicating a better model.

#### Based on these metrics:
- A Mean Absolute Error (MAE) of $50,670.74 suggests that, on average, the model's predictions are $50,670.74 away from the actual prices.
- A Mean Squared Error (MSE) of $4,908,476,721.16 indicates the average of the squared differences between the predicted and actual prices, which penalizes larger errors more.
- An R-squared (R²) value of 0.63 means that 62.54% of the variance in house prices is explained by the model.
- A Median Absolute Error of $38,377.26 shows that half of the errors are below $38,377.26, indicating the typical error magnitude.
- An Explained Variance Score of 0.63 suggests that 62.55% of the variance in the data is captured by the model.

### Random Forest Regressor Model
- **Mean Absolute Error (MAE)**: Indicates the average absolute difference between the predicted and actual house prices.
- **Mean Squared Error (MSE)**: Indicates the average squared difference between the predicted and actual house prices, giving more weight to larger errors.
- **R-squared (R²)**: Shows how well the model explains the variance in the actual house prices. A value closer to 1 indicates a better fit.
- **Median Absolute Error**: Provides the median of the absolute differences between the predicted and actual prices, less sensitive to outliers compared to MAE.
- **Explained Variance Score**: Indicates the proportion of variance explained by the model, with values closer to 1 indicating a better model.

#### Based on these metrics:
- A Mean Absolute Error (MAE) of $31,447.06 suggests that, on average, the model's predictions are $31,447.06 away from the actual prices.
- A Mean Squared Error (MSE) of $2,378,261,530.83 indicates the average of the squared differences between the predicted and actual prices, which penalizes larger errors more.
- An R-squared (R²) value of 0.82 means that 81.74% of the variance in house prices is explained by the model.
- A Median Absolute Error of $18,985.05 shows that half of the errors are below $18,985.05, indicating the typical error magnitude.
- An Explained Variance Score of 0.82 suggests that 81.76% of the variance in the data is captured by the model.

## Derivations and Recommendations

### Insights
- The correlation heatmap reveals that `median_income` has a strong positive correlation with `median_house_value`.
- House prices tend to be higher in areas closer to the ocean (`ocean_proximity`).

### Recommendations
- Focus on features with strong correlations to improve model performance.
- Consider additional data collection for features that might impact house prices, such as crime rates, school quality, and proximity to amenities.
- Regularly update the model with new data to maintain accuracy.

### Further Analysis
- Experiment with other machine learning algorithms such as Gradient Boosting, XGBoost, or neural networks.
- Perform feature engineering to create new features from existing ones.
- Conduct a time-series analysis if the dataset includes historical house prices.

## Code

Here is the complete code used in this project:

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv("C:\\Users\\tanis\\OneDrive\\Desktop\\OFFER LETTERS\\DATA SET USED FOR PROJECTS\\housing.csv")

# Handle missing values
df.fillna(df.median(), inplace=True)

# Convert categorical variable 'ocean_proximity' to dummy variables
df_dummies = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Concatenate dummy variables with original DataFrame and drop the original 'ocean_proximity' column
df_combined = pd.concat([df, df_dummies], axis=1)
df_combined.drop('ocean_proximity', axis=1, inplace=True)

# Define features and target variable
X = df_combined.drop(columns=['median_house_value'])
Y = df_combined['median_house_value']

# Plot median house price with respect to median income
plt.figure(figsize=(10, 6))
sns.lineplot(x='median_income', y='median_house_value', data=df_combined, marker='o')
plt.title('Median House Price with respect to Median Income')
plt.xlabel('Median Income ($)')
plt.ylabel('Median House Prices ($)')
plt.grid(True)
plt.show()

# Plot value of house with respect to age
plt.figure(figsize=(20, 10))
sns.barplot(x='housing_median_age', y='median_house_value', data=df)
plt.title('Value of house with respect to age')
plt.xlabel('House Age in Median')
plt.ylabel('Median House Prices ($)')
plt.grid(True)
plt.show()

# Calculate and display the correlation matrix
correlation_matrix = df_combined.corr()
print(correlation_matrix)

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(24, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the Linear Regression model
mae
