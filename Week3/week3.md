# 📚 Linear Models & Probability Distributions  

_A structured guide to understanding and implementing linear models and probability distributions using Python._

---

## 📖 Table of Contents  

- [� Linear Models \& Probability Distributions](#-linear-models--probability-distributions)
  - [📖 Table of Contents](#-table-of-contents)
  - [📌 Learning Objectives](#-learning-objectives)
  - [📈 Linear Models \& Regression](#-linear-models--regression)
    - [📉 Functional Form of a Linear Model](#-functional-form-of-a-linear-model)
    - [⚡ Loss Function for Linear Regression](#-loss-function-for-linear-regression)
    - [🛠 Fitting a Linear Model with Scikit-Learn](#-fitting-a-linear-model-with-scikit-learn)
- [Sample Data](#sample-data)
- [Fit Model](#fit-model)
- [Predictions](#predictions)

---

## 📌 Learning Objectives  

At the end of this guide, you should be able to:  

✔ **Understand** the functional form of a linear model and its loss function  
✔ **Use** `scikit-learn` to fit linear models and make predictions  
✔ **Visualize** model predictions against real data  
✔ **Describe** situations where discrete random variables are useful  
✔ **Create** histograms from discrete random variables  
✔ **Sample** from a given dataset and compute probabilities  
✔ **Explain and compute** entropy for a dataset  

---

## 📈 Linear Models & Regression  

### 📉 Functional Form of a Linear Model  

A **linear model** predicts an output \( y \) as a **linear combination** of input features:  

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

Where:  

- \( w_0 \) is the **intercept**  
- \( w_1, w_2, \dots, w_n \) are the **weights** (coefficients)  
- \( x_1, x_2, \dots, x_n \) are the **features**  

---

### ⚡ Loss Function for Linear Regression  

The **Mean Squared Error (MSE)** is the most common **loss function** used in linear regression:  

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where:  

- \( y_i \) is the **actual value**  
- \( \hat{y}_i \) is the **predicted value**  

---

### 🛠 Fitting a Linear Model with Scikit-Learn  

We can fit a **linear model** using `scikit-learn`:  

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit Model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)