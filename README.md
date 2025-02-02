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


