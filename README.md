# ML-Study: From Mathematical Foundations to Advanced Models

Welcome to **ML-Study**, a Machine Learning portfolio that documents a **structured progression through data science**—starting from **low-level mathematical implementations** (e.g., Gradient Descent from scratch) to **high-level predictive modeling** using industry-standard libraries.

---

##  Repository Highlights

### Linear Regression (Built From Scratch)

📁 **Directory:** `Linear Regression (From Scratch)`

This project focuses on a **fundamental implementation of Linear Regression** without relying on high-level ML libraries.

**Key Concepts:**
- Manual implementation of **Gradient Descent**
- Calculation of **partial derivatives** to update:
  - Slope (**m**)
  - Intercept (**b**)
- Avoidance of libraries like `scikit-learn` for learning purposes

**Technical Challenges Addressed:**
- Managing the **Learning Rate (L)** to prevent gradient explosion (NaN errors)
- Optimizing the number of **Epochs** to ensure proper model convergence

---

### Supervised Learning Projects

These projects use **scikit-learn** and other established frameworks to solve **real-world classification and regression problems**:

- **Housing Price Prediction (XGBoost)**  
  Predictive modeling for real estate valuation using Gradient Boosting techniques.

- **Diabetes Prediction (SVM)**  
  Classification using Support Vector Machines to predict medical outcomes.

- **Fake News Detection (NLP)**  
  Natural Language Processing model to classify news as real or fake.

- **Sonar Dataset (Logistic Regression)**  
  Binary classification to distinguish between **rocks** and **mines**.

- **Iris Flower Classification**  
  A foundational multi-class classification project for species prediction.

---

## Technical Deep Dive: Gradient Descent Implementation

In the **from-scratch** projects, Gradient Descent was implemented to minimize the cost function:

\[
y = mx + b
\]

### Loss Function
- **Mean Squared Error (MSE)** was used to quantify prediction error.

### Gradient Calculation

\[
m_{gradient} = \frac{2}{n} \sum -x \cdot (y - (mx + b))
\]

\[
b_{gradient} = \frac{2}{n} \sum -(y - (mx + b))
\]

### Optimization Step

\[
m = m - (m_{gradient} \cdot L)
\]

\[
b = b - (b_{gradient} \cdot L)
\]

Where **L** is the learning rate.

---

## Technology Stack

- **Programming Language:** Python 3.x  
- **Data Manipulation:** Pandas, NumPy  
- **Data Visualization:** Matplotlib, Seaborn  
- **Machine Learning Libraries:** Scikit-learn, XGBoost  
- **NLP Tools:** Standard Python NLP toolkits  

---

## Project Roadmap & Future Objectives

- [x] Implement Linear Regression from scratch using Gradient Descent  
- [x] Build Binary Classification models (Sonar, Diabetes)  
- [x] Apply Gradient Boosting techniques (XGBoost)  
- [ ] Develop Neural Networks using only NumPy  
- [ ] Implement Feature Scaling (Normalization & Standardization) for custom algorithms  

---

## Navigation & Execution Instructions

- Each project directory is **self-contained**
- Ensure the required `.csv` dataset is present inside the project folder
- Run the corresponding `.py` script or open the Jupyter notebook
- Review plots and outputs to analyze **model performance and convergence**

---

📌 *This repository is intended to demonstrate both theoretical understanding and practical implementation of Machine Learning concepts.*
