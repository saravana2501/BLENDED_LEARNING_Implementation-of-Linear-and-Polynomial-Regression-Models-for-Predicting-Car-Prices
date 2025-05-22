### BLENDED_LEARNING
## Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices
### DATE:17-04-2025
## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary libraries such as `pandas`, `sklearn`, `matplotlib`, and `numpy`.
2. Load the dataset using `pandas.read_csv()` and display the first few rows to understand its structure.
3. Select relevant features (`enginesize`, `horsepower`, `citympg`, `highwaympg`) as input variables (X) and `price` as the output variable (y).
4. Split the dataset into training and testing sets using `train_test_split()`.
5. Build a pipeline for Linear Regression with feature scaling (`StandardScaler`) and fit it on the training data.
6. Predict car prices on the test set using the linear model and evaluate performance using metrics like MSE and R².
7. Build a second pipeline for Polynomial Regression with degree 2, including scaling and the regression model.
8. Fit the polynomial model on the training data and evaluate it similarly.
9. Plot the actual vs predicted prices for both models to visually compare performance.


## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: SARAVANA KUMAR S
RegisterNumber:  212224220090
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
df=pd.read_csv('encoded_car_data.csv')
print(df.head())
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(x_train,y_train)
y_pred_linear=lr.predict(x_test)
poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(x_train,y_train)
y_pred_poly = poly_model.predict(x_test)
# Evaluate models
print("Linear Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_linear):.2f}")
print(f"R^2: {r2_score(y_test, y_pred_linear):.2f}")

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R^2: {r2_score(y_test, y_pred_poly):.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial (degree=2)', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
*/
```


## Output:
### PREVIEW OF THE DATASET:
![image](https://github.com/user-attachments/assets/60093b29-539f-4001-93c8-a99bce669c22)
### MODEL PERFORMANCE:
![image](https://github.com/user-attachments/assets/6402c164-9a6c-46d5-b7f4-adfc7043d0b8)
### COMPARISION OF ACTUAL PRICES AND PREDICTED PRICES:
![image](https://github.com/user-attachments/assets/b5278b75-e8f5-4f8b-9814-12fab3e2b016)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
