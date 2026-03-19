---
layout: assignment
permalink: Assignments/Regression
title: "Programming Assignment: Regression Models"

info:
  points: 100
  goals:
    - "Demonstrate the ability to select, justify, and clearly describe datasets appropriate for linear and logistic regression analyses."
    - "Implement linear regression and logistic regression from scratch (NumPy only) and validate correctness against closed-form or reference solutions."
    - "Apply scikit-learn implementations for both models, selecting appropriate preprocessing and hyperparameters."
    - "Evaluate model generalization using principled train/test protocols and appropriate metrics (RMSE, <span>\\(R^2\\)</span>, Accuracy, Precision, Recall, F1, ROC AUC)."
    - "Analyze the effects of L2 (Ridge) and L1 (Lasso) regularization, with attention to the bias–variance trade-off and coefficient behavior."
    - "Interpret residuals, ROC curves, and hyperparameter sweep plots to diagnose underfitting and overfitting and inform model improvement."
    - "Articulate theoretically grounded reflections connecting optimization dynamics, regularization, and generalization performance."
    - "Produce a reproducible workflow with clear documentation, fixed random seeds, and environment/version disclosure."
  concepts:
    - "Problem formulation: supervised learning, regression vs classification, and the design matrix with intercept augmentation."
    - "Optimization: normal equations, least squares, gradient descent (learning rate, convergence criteria), and convexity."
    - "Logistic regression: sigmoid function, cross-entropy loss, decision boundaries, and probabilistic interpretation of outputs."
    - "Regularization: Ridge (L2) and Lasso (L1), geometric intuition, coefficient shrinkage and sparsity, and the bias–variance trade-off."
    - "Model evaluation: train/test splits, stratified sampling, and regression metrics (RMSE, <span>\\(R^2\\)</span>) vs classification metrics (Accuracy, Precision, Recall, F1, ROC AUC)."
    - "Diagnostics and visualization: residual plots, ROC curves, calibration analysis, and decision region visualization."
    - "Hyperparameter tuning: systematic grids over λ (or C), learning rate schedules, iteration limits, and early-stopping heuristics."
    - "Data preprocessing: feature scaling and standardization, intercept handling, and basic feature engineering considerations."
    - "Reproducibility and software engineering practices: fixed random seeds, configuration capture, versioning, and structured, documented code."
  tasks:
    - "Select two datasets: one for regression (continuous target, ≥100 samples, ≥2 features) and one for classification (binary or multiclass). Provide a clear description for each dataset, including its source, features, target definition, summary statistics, and preprocessing steps."
    - "Implement linear regression from scratch using NumPy, either through the normal equation with matrix decomposition or via gradient descent, and validate correctness on a simple synthetic example."
    - "Reproduce the linear regression analysis using scikit-learn’s LinearRegression API and compare model coefficients, predictions, and generalization performance to the from-scratch results."
    - "Evaluate linear regression using RMSE and <span>\\(R^2\\)</span> metrics on both training and testing sets, and visualize residuals versus fitted values to identify systematic patterns."
    - "Perform a regularization study for Ridge and Lasso regression using λ ∈ {1e-3, 1e-2, 1e-1, 1, 10}, comparing train/test RMSE and plotting performance trends to interpret bias–variance behavior."
    - "Implement logistic regression from scratch using the sigmoid activation and binary cross-entropy loss, optimizing via gradient descent and verifying gradient correctness numerically."
    - "Apply scikit-learn’s LogisticRegression with standardized features, tuning the inverse regularization parameter C across a logarithmic grid, and report Accuracy, Precision, Recall, F1, and ROC AUC metrics."
    - "Plot ROC curves for binary classification and, if applicable, visualize decision regions for 2D feature spaces to illustrate model boundaries and separability."
    - "Analyze the effect of regularization (L2 penalty) and hyperparameters (learning rate, max iterations) on convergence, training stability, and overfitting versus underfitting trends."
    - "Write a reflective summary comparing train/test outcomes, relating λ and learning rate to generalization and convergence behavior, and integrating theoretical justifications for observed phenomena."
    - "Submit a complete, reproducible notebook containing clean code, markdown explanations, and plots, along with a two-page report summarizing datasets, modeling choices, evaluation results, and key insights."
  rubric:
    - weight: 10
      description: Data Selection and Description
      preemerging: Selects a dataset with minimal consideration for relevance or scale; provides limited description.
      beginning: Chooses an appropriate dataset with brief discussion of size and complexity.
      progressing: Selects a well-justified dataset with clear explanation of its properties, scope, and relevance to the task.
      proficient: Provides a thorough justification of data choice, discussing representativeness, complexity, and preprocessing considerations with precise documentation.
    - weight: 25
      description: From-Scratch Implementations
      preemerging: Provides partial or minimally functional code with limited explanation.
      beginning: Implements key algorithmic components correctly with basic verification of results.
      progressing: Produces complete, correct, and clear implementations that converge appropriately, with evidence of testing and debugging.
      proficient: Demonstrates robust, efficient, and well-documented implementations with convergence analysis, error checks, and design justification.
    - weight: 15
      description: Library Implementations
      preemerging: Uses libraries without clear rationale or explanation of parameters.
      beginning: Appropriately applies standard APIs for modeling and evaluation.
      progressing: Demonstrates proper use of libraries with suitable preprocessing and parameterization, validating expected behavior.
      proficient: Employs library APIs effectively and thoughtfully, explaining preprocessing, hyperparameter tuning, and implementation decisions with clarity.
    - weight: 20
      description: Evaluation and Plots
      preemerging: Provides limited or incorrect evaluation; few or unclear plots.
      beginning: Reports standard metrics and generates basic plots.
      progressing: Includes well-structured metrics, clear visualizations (e.g., residuals, ROC), and relevant parameter sweeps.
      proficient: Delivers comprehensive evaluations with insightful plots, interprets metrics rigorously, and discusses the implications of model behavior across hyperparameter variations.
    - weight: 15
      description: Regularization Analysis
      preemerging: Mentions regularization without concrete analysis or results.
      beginning: Demonstrates basic understanding of ridge and lasso effects on coefficients.
      progressing: Analyzes the impact of regularization strength (<span>\\(\lambda\\)</span>) with empirical comparisons between ridge and lasso.
      proficient: Provides a detailed quantitative and qualitative analysis of ridge and lasso behavior, discussing bias-variance trade-offs and theoretical implications.
    - weight: 15
      description: Reflection on Generalization
      preemerging: Offers minimal commentary on generalization or overfitting.
      beginning: Compares training and testing results with basic observations.
      progressing: Discusses overfitting and underfitting trends supported by data-driven reasoning.
      proficient: Provides a deep reflection on generalization, integrating quantitative evidence and theoretical rationale to explain observed performance differences.

tags:
  - regression
---

## Dataset Selection (for both regression and classification tasks)

For this assignment, you will work with **two datasets** — one for regression and one for classification. Each dataset should contain at least **100 samples** and at least **two input features**. For classification, a **binary** task is preferred, though a multiclass problem is acceptable. You may use publicly available datasets from reputable sources or generate your own synthetic data. Suggested sources include:  
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)  
- [OpenML](https://www.openml.org/)  
- [Kaggle Datasets](https://www.kaggle.com/datasets)  
- Built-in datasets from [`sklearn.datasets`](https://scikit-learn.org/stable/datasets/toy_dataset.html)  

Representative examples include:  
- **Regression:** [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg), [Wine Quality (Regression)](https://archive.ics.uci.edu/ml/datasets/wine+quality), or a small **synthetic dataset** you generate yourself (e.g., “study hours vs exam score”).  
- **Classification:** [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), [Pima Indians Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database), [Titanic Survival](https://www.kaggle.com/c/titanic/data), or [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) (restricted to two classes), or, again, a small **synthetic dataset** you generate yourself (e.g., “study hours vs receiving a passing grade”).

Do **not** select a dataset that we covered during class as a case-study; be creative!

After selecting your datasets, clearly describe their sources, the meaning of each feature, and the target variable. Construct your design matrix **`X`** (containing all predictor variables) and your target vector **`y`** (the output values or class labels). Ensure that these are clean NumPy arrays or pandas DataFrames suitable for model training.

---

## Part 1 — Linear Regression

In the **from-scratch implementation**, restrict yourself to a **linear relationship** between the input and output variables. That is, do not add squared or polynomial features and do not apply normalization or standardization. This will simplify your analysis and allow you to focus on the basic mathematics of linear regression. Construct the augmented design matrix

$$
\tilde{X} = [\mathbf{1}\;X]
$$ 

and compute the optimal parameter vector using  

$$
\boldsymbol{\theta}^* = (\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top \mathbf{y},
$$

or equivalently solve the least squares problem with `np.linalg.lstsq` or via QR/SVD decomposition. You may also choose to implement **gradient descent** for the same model, updating the parameters using  

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\frac{2}{n}\tilde{X}^\top(\tilde{X}\boldsymbol{\theta}-\mathbf{y}),
$$

where <span>\\(\alpha\\)</span> is the learning rate. Report your model’s **loss curve** (mean squared error versus iteration) for the gradient descent approach to show convergence behavior, and report what value of your **learning rate** <span>\\(\alpha\\)</span> gave you the best loss curve (be sure to try several!).

For the **library-based implementation**, use `sklearn.linear_model.LinearRegression`. If your dataset exhibits nonlinear trends, you should **extend** your feature matrix by including scaled polynomial terms such as `x**2`, `x**3`, or interaction features generated via `PolynomialFeatures` from `sklearn.preprocessing`. Include an option in your code to apply normalization using a **Scaler pipeline** (e.g., `StandardScaler` or `MinMaxScaler`) and to toggle normalization **on or off**. Experiment with both configurations and discuss how feature scaling affects convergence and generalization. Finally, add an option to fit your model using **Ridge** and/or **Lasso** regularization. Run your analysis using each configuration — Ridge alone, Lasso alone, both together, and neither — and compare their effects on model coefficients, training and testing RMSE, and overall fit quality.

Your linear regression section should include:  
- The fitted model parameters and predictions.  
- Train and test **RMSE** and **R²** (**coefficient of determination**) values within your training set and your testing set (be sure to split them into distinct sets, and don't train on the testing set!).  
- A plot of residuals versus fitted values, with commentary on any observed structure.  
- A plot of **train/test RMSE versus regularization strength** (<span>\\(\lambda\\)</span>) for Ridge and Lasso.  
- The **loss curve** from your gradient descent implementation, annotated with iteration count and convergence remarks.  

---

## Part 2 — Logistic Regression

In the **from-scratch implementation**, implement the logistic sigmoid function 

$$
\sigma(z) = 1 / (1 + e^{-z})
$$ 

and optimize the binary cross-entropy loss using gradient descent:

$$
J(\boldsymbol{\theta}) = -\frac{1}{n}\sum_i \left[y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i)\right],
\quad
\nabla J = \frac{1}{n}X^\top(\hat{\mathbf{p}} - \mathbf{y}),
$$

where <span>\\(\hat{\mathbf{p}}\\)</span> is the vector of predicted probabilities. Ensure that you monitor convergence by tracking the loss function over iterations.  

For the **library implementation**, use `sklearn.linear_model.LogisticRegression`. Always **standardize or normalize** your features for logistic regression, as this improves numerical stability and ensures fair comparison of coefficient magnitudes. Incorporate a `Pipeline` to streamline preprocessing, including feature scaling and optional polynomial feature generation for nonlinear relationships. Allow users to toggle normalization and polynomial augmentation on or off. As with linear regression, include options to apply **Lasso (L1)** and **Ridge (L2)** regularization separately or together, and document the impact of each configuration.  

Evaluate your logistic models using a comprehensive set of metrics: **RMSE**, **F1 score**, and **ROC AUC** both within your training set and testing set (again, be sure to split them into distinct sets, and don't train on the testing set!). Generate and label a **confusion matrix** and an **ROC curve** for the binary classification case, and if your features are two-dimensional, visualize **decision regions** to illustrate model boundaries. Report performance both with and without scaling to demonstrate the importance of normalization in gradient-based optimization.

---

## Deliverables

Submit one or two Jupyter notebooks (`.ipynb`), clearly labeled for the regression and classification tasks. Each notebook should include clean, well-commented code with narrative explanations in Markdown cells, describing your data choices, implementation steps, and evaluation results. Include plots of metrics versus hyperparameters (<span>\\(\lambda\\)</span>, <span>\\(\alpha\\)</span>) and figures showing loss curves, ROC curves, and residual analyses. Accompany your notebooks with a short, well-structured report (approximately two pages) summarizing your dataset rationale, modeling decisions, your findings on generalization, and the observed effects of regularization and normalization. Ensure reproducibility by fixing random seeds and listing software version information.
