# Introduction to Neural Networks, Mini-Lab 1 – Regression, Model Selection, and Gradient Descent

This repository contains the materials for the **first minilab** of the course *Introduction to Neural Networks (WS 2025)* at the University of Münster.  The goal is to explore **regression models on real data**, progressively moving from basic linear models to polynomial and gradient-based methods (as used in Online Learning).

---

## Learning Goals

This exercise should let you
- Work with **real-world housing data** (Immobilienscout24 dataset, 2018 to 2019)
- Implement and evaluate **linear regression models**
- Investigate the **importance of features** and **model selection**
- Apply **polynomial extensions** and explore **overfitting**
- Extend models using **gradient descent** and **transfer learning**

The detailed task descriptions can be found in  
📄 **`enn_minilab_regression.pdf`**

---

## Task Overview

| Task  | Topic                               | Core Idea                                                    |
| ----- | ----------------------------------- | ------------------------------------------------------------ |
| **1** | Data Analysis & Cleaning            | Inspect, clean, and prepare the data; create a baseline regression model |
| **2** | Feature Analysis                    | Rank features by predictive power; perform stepwise feature selection |
| **3** | Polynomial Modeling                 | Compare polynomial model degrees and visualize model selection |
| **4** | Transfer to New City                | Evaluate generalization and overfitting across datasets      |
| **5** | Temporal Transfer / Online Learning | Extend linear regression via gradient descent and incremental updates |

---

## Folder Structure

| **Folder / File**            | **Description**                                              |
| ---------------------------- | ------------------------------------------------------------ |
| data/                        | Contains the training, validation, and transfer datasets (CSV files) |
| notebooks/                   | Jupyter notebooks guiding each exercise task                 |
| src/                         | Source code for cleaning, feature analysis, models, and visualization |
| tests/                       | Test scripts verifying correctness of your implementations (start here to understand what you are expected to reach) |
| results/                     | Output directory for generated plots and model results       |
| enn_minilab_1_regression.pdf | PDF version of the exercise sheet (start here for detailed explanation) |
| README.md                    | This project overview file                                   |



## Running Tests

All functions can be validated via:

>  pytest -s tests/

Alternatively, tests can be run directly from within the corresponding Jupyter notebooks for interactive feedback.
