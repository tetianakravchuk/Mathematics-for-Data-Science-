# DX 601 Week 1 Homework

## Introduction
This repository contains the Week 1 homework assignment for **DX 601**. The purpose of this assignment is to provide exposure to **basic Python programming** to support the mathematical concepts taught in this course.

If you are not taking **Module 2** consecutively with this module, you may find it helpful to review the following Jupyter notebook:

* [Python as a Calculator](https://github.com/bu-cds-omds/dx602-examples/blob/main/week01/video_python_as_a_calculator.ipynb) - This notebook from Module 2 contains several examples of using Python for basic mathematical computations.

## Instructions
You should replace every instance of `...` in the homework file with your own Python code to solve each problem.

Be sure to **run each code block** after editing it to ensure that your code executes correctly.

**Before submitting**, we strongly recommend you run all the code **from scratch** (`Runtime menu -> Restart and Run All`) to verify that your final version works without errors.

If your code raises an exception when run from scratch, it will interfere with the **auto-grader**, which may result in losing points. If you get stuck, ask for help on **YellowDig** or schedule an appointment with a **learning facilitator**.

---

## Submission Guidelines
To submit your homework, follow these steps:

1. **Save and commit** this notebook.
2. **Push your changes** to GitHub.
3. **Verify** that your changes are visible in GitHub.
4. **Delete your Codespace** to prevent using up your free quota.

The **auto-grading process** typically completes within a few minutes of pushing to GitHub but may take up to an hour. If you submit early, you can review the auto-grading results and make corrections before the deadline.

---

## Dataset Information
This repository includes a **sample dataset** in the file `colors.tsv`, which contains three numerical columns:

- **red**
- **green**
- **blue**

The dataset is used in **Problem 20**, where you are required to calculate the **average of the "red" column** using Python and Pandas.

---

## Example Usage
If you need to read `colors.tsv` into a **Pandas DataFrame**, use the following Python code:

```python
import pandas as pd

df = pd.read_csv("colors.tsv", sep="\t")  # Read tab-separated file
p20 = df["red"].mean()  # Compute the average of the "red" column
print("Average red value:", p20)
```

---

## Problems Overview
The problems in this homework cover:

- **Basic Python expressions** (addition, multiplication, exponentiation, etc.)
- **List operations and comprehensions**
- **Using built-in functions like `sum()` and `len()`**
- **Reading and processing datasets using Pandas**
- **Using NumPy for mathematical operations**

Each problem includes a **clear prompt** and a section labeled `# YOUR CHANGES HERE`, where you should enter your solution.


# DX 601 Week 2 Homework

## Introduction
This repository contains the Week 2 homework assignment for **DX 601**. In this assignment, you will practice calculating **statistics and sampling techniques** covered this week using Python.

If you are not familiar with Python and are not taking **Module 2** concurrently, we strongly recommend reviewing **[A Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/)**.
- **Chapter 5** covers basic expressions needed to implement mathematical formulas.
- **Chapter 9**, particularly the first two sections, covers writing functions in Python.

## Instructions
You should replace every instance of `...` in the homework file with your own **Python code** to solve each problem.

Be sure to **run each code block** after editing it to ensure that your code executes correctly.

**Before submitting**, we strongly recommend you run all the code **from scratch** (`Runtime menu -> Restart and Run All`) to verify that your final version works without errors.

If your code raises an exception when run from scratch, it will interfere with the **auto-grader**, which may result in losing points. If you get stuck, ask for help on **YellowDig** or schedule an appointment with a **learning facilitator**.

---

## Submission Guidelines
To submit your homework, follow these steps:

1. **Save and commit** this notebook.
2. **Push your changes** to GitHub.
3. **Verify** that your changes are visible in GitHub.
4. **Delete your Codespace** to prevent using up your free quota.

The **auto-grading process** typically completes within a few minutes of pushing to GitHub but may take up to an hour. If you submit early, you can review the auto-grading results and make corrections before the deadline.

---

## Shared Imports
This assignment requires using **Python libraries** such as:
```python
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

**Important:** Do not install or use additional modules. Installing extra modules may cause **autograder failures**, resulting in a **zero score** for the affected problems.

---

## Dataset Information
This repository includes a **sample dataset** in the file `colors.tsv`, which contains three numerical columns:

- **red**
- **green**
- **blue**

The dataset is used in **Problem 20**, where you are required to calculate the **average of the "red" column** using Python and Pandas.

---

## Example Usage
To read `colors.tsv` into a **Pandas DataFrame** and compute the average of the "red" column, use:

```python
import pandas as pd
import numpy as np

# Load the TSV file
q20 = pd.read_csv("colors.tsv", sep="\t")

# Compute the mean of the "red" column
p20 = np.mean(q20["red"])
print("Average red value:", p20)
```

---

## Problems Overview
The problems in this homework cover:

- **Basic statistics calculations** (mean, variance, percentiles, etc.)
- **List operations and comprehensions**
- **Using built-in functions like `sum()` and `len()`**
- **Reading and processing datasets using Pandas**
- **Using NumPy for statistical computations**
- **Plotting histograms using Matplotlib**

Each problem includes a **clear prompt** and a section labeled `# YOUR CHANGES HERE`, where you should enter your solution.

---

# DX 601 Week 3 Homework

# What Models Do and Managing Model Errors

## Introduction
This repository contains an introduction to **models**, their purposes, common errors, and methods for improvement. The focus is on **linear models**, but the concepts discussed apply to more advanced models as well. By understanding how models work, their limitations, and the ways to optimize them, you can build effective solutions to real-world problems.

---

## Learning Objectives
By the end of this module, you should be able to:

1. Describe the basic purpose of building models.
2. Explain the common limitations of models.
3. Understand what residuals and loss functions are.
4. Calculate and use common loss functions to evaluate models.

---

## What Do Models Do?

### Why Do We Build Models?
Models are built to:

1. **Make Predictions**:
   - Estimate future outcomes (e.g., stock prices, rainfall, or customer behavior).
2. **Understand Relationships**:
   - Identify and quantify relationships between variables (e.g., the impact of education on income).
3. **Automate Processes**:
   - Automate decision-making tasks like fraud detection or email classification.
4. **Support Risk Assessment**:
   - Quantify and manage risks in finance, insurance, and other fields.
5. **Optimize Outcomes**:
   - Find the best course of action for a specific goal, like maximizing profit or minimizing cost.

---

### Examples of Model Applications
| **Field**            | **Predictions**                                          |
|----------------------|---------------------------------------------------------|
| **Finance**          | Future stock prices, probability of price increase      |
| **Digital Marketing**| Revenue per click, optimal bid price                    |
| **Meteorology**      | Probability of rain, predicted range of snowfall        |
| **Agriculture**      | Crop yield per acre, market price trends                |
| **Insurance**        | Mortality rates, risk scores                            |
| **Retail**           | Unit sales under different pricing strategies           |

---

## Evaluating Model Performance

### How Do We Decide If Models Are Doing Their Job?
1. **Use Performance Metrics**:
   - **Regression Models**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared.
   - **Classification Models**: Accuracy, Precision, Recall, F1 Score, AUC-ROC.

2. **Check Residuals**:
   - Residuals are the differences between actual and predicted values.
   - A good model has residuals that are random and normally distributed.

3. **Cross-Validation**:
   - Split the dataset into training and testing sets or use k-fold cross-validation.
   - Ensure the model generalizes well to unseen data.

4. **Compare Against Baselines**:
   - Assess the model’s performance relative to simple approaches like mean prediction or random guessing.

5. **Interpret Results**:
   - Ensure predictions align with real-world expectations and make sense in the given context.

---

## Improving Models

### How Do We Improve Models?

1. **Improve Data Quality**:
   - Handle missing values, outliers, and inconsistencies.
   - Engineer better features from the raw data.

2. **Optimize Algorithms**:
   - Experiment with different models (e.g., linear regression, decision trees, neural networks).

3. **Tune Hyperparameters**:
   - Adjust parameters like learning rate, tree depth, or number of neurons to improve performance.

4. **Regularization**:
   - Use techniques like L1 (Lasso) or L2 (Ridge) regularization to prevent overfitting.

5. **Ensemble Methods**:
   - Combine multiple models (e.g., random forests, gradient boosting) for more robust predictions.

6. **Increase Data**:
   - Gather more relevant data to improve generalization.

7. **Use Advanced Techniques**:
   - Apply deep learning, transfer learning, or advanced feature selection methods.

---

## Summary

### Why Do We Build Models?
- To predict, classify, and understand data, automate processes, assess risks, and optimize outcomes.

### What Do Models Do?
- Perform tasks like prediction, classification, clustering, anomaly detection, and uncertainty quantification.

### How Do We Decide If They Are Doing Their Job?
- Use performance metrics, validate results, analyze residuals, and compare against baselines.

### How Do We Improve Them?
- Improve data quality, optimize algorithms, tune hyperparameters, and leverage advanced methods.

---

## Questions for Reflection
1. **What kinds of predictions help you make decisions every day?**
   - Examples: Weather forecasts, traffic conditions, stock trends, fitness app recommendations.

2. **If you could predict something new with high accuracy, what would it be?**
   - Examples: Future health risks, career growth opportunities, environmental changes.

3. **Can this prediction be achieved using historical data?**
   - Evaluate whether enough relevant data exists to build a reliable model.

---

## Next Steps
- Experiment with building and evaluating simple models (e.g., linear regression).
- Explore the impact of different loss functions and how they guide model optimization.
- Dive into more advanced modeling techniques like decision trees and neural networks.

Feel free to contribute examples or share insights as you explore these topics further. Happy modeling! 🚀


# What Models Do and Managing Model Errors

## Introduction
This repository contains an introduction to **models**, their purposes, common errors, and methods for improvement. The focus is on **linear models**, but the concepts discussed apply to more advanced models as well. By understanding how models work, their limitations, and the ways to optimize them, you can build effective solutions to real-world problems.

---

## Learning Objectives
By the end of this module, you should be able to:

1. Describe the basic purpose of building models.
2. Explain the common limitations of models.
3. Understand what residuals and loss functions are.
4. Calculate and use common loss functions to evaluate models.

---

## What Do Models Do?

### Why Do We Build Models?
Models are built to:

1. **Make Predictions**:
   - Estimate future outcomes (e.g., stock prices, rainfall, or customer behavior).
2. **Understand Relationships**:
   - Identify and quantify relationships between variables (e.g., the impact of education on income).
3. **Automate Processes**:
   - Automate decision-making tasks like fraud detection or email classification.
4. **Support Risk Assessment**:
   - Quantify and manage risks in finance, insurance, and other fields.
5. **Optimize Outcomes**:
   - Find the best course of action for a specific goal, like maximizing profit or minimizing cost.

---

### Examples of Model Applications
| **Field**            | **Predictions**                                          |
|----------------------|---------------------------------------------------------|
| **Finance**          | Future stock prices, probability of price increase      |
| **Digital Marketing**| Revenue per click, optimal bid price                    |
| **Meteorology**      | Probability of rain, predicted range of snowfall        |
| **Agriculture**      | Crop yield per acre, market price trends                |
| **Insurance**        | Mortality rates, risk scores                            |
| **Retail**           | Unit sales under different pricing strategies           |

---

## Evaluating Model Performance

### How Do We Decide If Models Are Doing Their Job?
1. **Use Performance Metrics**:
   - **Regression Models**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared.
   - **Classification Models**: Accuracy, Precision, Recall, F1 Score, AUC-ROC.

2. **Check Residuals**:
   - Residuals are the differences between actual and predicted values.
   - A good model has residuals that are random and normally distributed.

3. **Cross-Validation**:
   - Split the dataset into training and testing sets or use k-fold cross-validation.
   - Ensure the model generalizes well to unseen data.

4. **Compare Against Baselines**:
   - Assess the model’s performance relative to simple approaches like mean prediction or random guessing.

5. **Interpret Results**:
   - Ensure predictions align with real-world expectations and make sense in the given context.

---

## Improving Models

### How Do We Improve Models?

1. **Improve Data Quality**:
   - Handle missing values, outliers, and inconsistencies.
   - Engineer better features from the raw data.

2. **Optimize Algorithms**:
   - Experiment with different models (e.g., linear regression, decision trees, neural networks).

3. **Tune Hyperparameters**:
   - Adjust parameters like learning rate, tree depth, or number of neurons to improve performance.

4. **Regularization**:
   - Use techniques like L1 (Lasso) or L2 (Ridge) regularization to prevent overfitting.

5. **Ensemble Methods**:
   - Combine multiple models (e.g., random forests, gradient boosting) for more robust predictions.

6. **Increase Data**:
   - Gather more relevant data to improve generalization.

7. **Use Advanced Techniques**:
   - Apply deep learning, transfer learning, or advanced feature selection methods.

---

## Summary

### Why Do We Build Models?
- To predict, classify, and understand data, automate processes, assess risks, and optimize outcomes.

### What Do Models Do?
- Perform tasks like prediction, classification, clustering, anomaly detection, and uncertainty quantification.

### How Do We Decide If They Are Doing Their Job?
- Use performance metrics, validate results, analyze residuals, and compare against baselines.

### How Do We Improve Them?
- Improve data quality, optimize algorithms, tune hyperparameters, and leverage advanced methods.

---

## Questions for Reflection
1. **What kinds of predictions help you make decisions every day?**
   - Examples: Weather forecasts, traffic conditions, stock trends, fitness app recommendations.

2. **If you could predict something new with high accuracy, what would it be?**
   - Examples: Future health risks, career growth opportunities, environmental changes.

3. **Can this prediction be achieved using historical data?**
   - Evaluate whether enough relevant data exists to build a reliable model.

---

## Next Steps
- Experiment with building and evaluating simple models (e.g., linear regression).
- Explore the impact of different loss functions and how they guide model optimization.
- Dive into more advanced modeling techniques like decision trees and neural networks.

Feel free to contribute examples or share insights as you explore these topics further. Happy modeling! 🚀

