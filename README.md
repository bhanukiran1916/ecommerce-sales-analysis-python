# 🚀 E-Commerce Customer Behavior & Sales Analysis

## 📌 Overview

This project focuses on analyzing e-commerce customer data using Data Science techniques. It covers the complete workflow from data preprocessing to machine learning model building.

The main objective is to understand customer purchasing behavior and predict spending patterns using statistical analysis and machine learning.

---

## 📊 Dataset

The dataset contains information about customers and their purchases, including:

* CustomerID
* Gender
* Age
* Category
* Quantity
* Price

A new feature **Total_Spending** is created using:

```
Total_Spending = Quantity × Price
```

---

## 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy

---

## 🧹 Data Preprocessing

* Handling missing values using forward fill
* Removing duplicate records
* Feature engineering (Total Spending calculation)

---

## 📈 Exploratory Data Analysis (EDA)

* Histogram & KDE plots for distribution analysis
* Scatter plots for relationship analysis
* Heatmap for correlation analysis
* Boxplot for outlier detection
* Bar charts for top product categories

---

## 📊 Statistical Analysis

* **Shapiro-Wilk Test** → Normality check
* **T-Test** → Compare spending between groups
* **A/B Testing** → High vs Low spending comparison

---

## 🤖 Machine Learning

### 🔹 Linear Regression

* Predicts Total Spending based on Quantity and Price
* Evaluated using R² Score

### 🔹 Classification (Decision Tree)

* Classifies customers into:

  * High Spending
  * Low Spending

---

## 📦 Outlier Detection

* Used IQR (Interquartile Range) method to remove extreme values

---

## 📌 Results & Insights

* Identified customer spending patterns
* Found relationships between price, quantity, and total spending
* Built predictive model with good accuracy
* Performed statistical tests to validate insights

---

## 💾 Output

* Cleaned dataset saved as: `final_ecommerce_data.csv`
* Visualizations for better understanding
* Machine learning predictions

---

## 🎯 Conclusion

This project demonstrates a complete Data Science pipeline and provides insights that can help businesses make data-driven decisions.

---

## 🔗 GitHub Repository

(Add your repo link here)

---

## 📸 Sample Visualizations

(Add screenshots of graphs here for better presentation)

---

## 🙌 Acknowledgment

This project was developed as part of learning Data Science and Machine Learning concepts.

