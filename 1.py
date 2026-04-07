 # ==============================
# 1. IMPORT LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, shapiro
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score

sns.set(style="darkgrid")

# ==============================
# 2. LOAD DATASET
# ==============================

df = pd.read_csv("Ecommerce_Customers_Data.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)

# ==============================
# 3. DATA CLEANING
# ==============================

print("\nMissing Values:\n", df.isnull().sum())

df.ffill(inplace=True)   # ✅ FIXED (no warning)
df.drop_duplicates(inplace=True)

# Feature Engineering
df['Total_Spending'] = df['Quantity'] * df['Price']

# ==============================
# 4. DESCRIPTIVE STATISTICS
# ==============================

print("\nSummary Statistics:\n", df.describe())

print("\nMean Spending:", np.mean(df['Total_Spending']))
print("Std Price:", np.std(df['Price']))

# ==============================
# 5. VISUALIZATION (EDA)
# ==============================

# Histogram
sns.histplot(df['Total_Spending'], kde=True, color='purple')
plt.title("Distribution of Total Spending")
plt.show()

# KDE
sns.kdeplot(df['Price'], fill=True, color='blue')
plt.title("Price Density")
plt.show()

# Pairplot
sns.pairplot(df[['Quantity','Price','Total_Spending']])
plt.show()

# Boxplot
sns.boxplot(data=df[['Quantity','Price','Total_Spending']], palette='Set2')
plt.title("Outlier Detection")
plt.show()

# Scatter
sns.scatterplot(x='Quantity', y='Total_Spending',
                hue='Gender', data=df, palette='viridis')
plt.title("Quantity vs Spending")
plt.show()

# Regression Plot
sns.regplot(x='Price', y='Total_Spending', data=df, color='red')
plt.title("Regression Plot")
plt.show()

# Heatmap
sns.heatmap(df[['Quantity','Price','Total_Spending']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Category Sales (✅ FIXED warning)
top_cat = df.groupby('Category')['Total_Spending'].sum().nlargest(10)

sns.barplot(x=top_cat.values, y=top_cat.index,
            hue=top_cat.index, palette='magma', legend=False)
plt.title("Top Categories")
plt.show()

# ==============================
# 6. OUTLIER REMOVAL
# ==============================

Q1 = df['Total_Spending'].quantile(0.25)
Q3 = df['Total_Spending'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Total_Spending'] >= Q1 - 1.5*IQR) &
        (df['Total_Spending'] <= Q3 + 1.5*IQR)]

# ==============================
# 7. CORRELATION
# ==============================

print("\nCorrelation:\n", df.corr(numeric_only=True))

# ==============================
# 8. STATISTICAL TESTING
# ==============================

# Shapiro Test
stat, p = shapiro(df['Total_Spending'].sample(500))
print("\nShapiro Test p-value:", p)

# T-Test (Male vs Female)
male = df[df['Gender'] == 'Male']['Total_Spending']
female = df[df['Gender'] == 'Female']['Total_Spending']

t_stat, p_val = ttest_ind(male, female)
print("T-Test p-value:", p_val)

# A/B Testing
high = df[df['Total_Spending'] > 500]['Price']
low = df[df['Total_Spending'] <= 500]['Price']

t_stat, p_val = ttest_ind(high, low)
print("A/B Testing p-value:", p_val)

# ==============================
# 9. LINEAR REGRESSION
# ==============================

X = df[['Quantity','Price']]
y = df['Total_Spending']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nR2 Score:", r2_score(y_test, y_pred))

# ==============================
# 10. CLASSIFICATION
# ==============================

df['Spending_Level'] = np.where(df['Total_Spending'] > 500,
                               'High', 'Low')

X_cls = df[['Quantity','Price']]
y_cls = df['Spending_Level']

clf = DecisionTreeClassifier()
clf.fit(X_cls, y_cls)

# ==============================
# 11. SAVE FILE
# ==============================

df.to_csv("final_ecommerce_data.csv", index=False)

print("\n✅ PROJECT COMPLETED SUCCESSFULLY")
