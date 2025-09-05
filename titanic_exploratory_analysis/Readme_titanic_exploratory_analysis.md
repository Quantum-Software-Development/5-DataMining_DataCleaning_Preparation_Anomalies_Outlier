 This guides through loading data, exploratory analysis, cleaning, outlier detection, normalization, modeling, and validation.

```python
# Cell 1: Import libraries and load the Titanic dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display data info and first rows to understand basic structure
print(df.info())
print(df.head())
```

*Explanation:* This cell imports necessary libraries and loads the Titanic dataset, showing initial information like columns, data types, and sample records.

***

```python
# Cell 2: Exploratory Data Analysis - summary statistics and missing data visualization
print(df.describe(include='all'))  # Statistics for numerical and categorical columns
print("Missing values per column:\n", df.isnull().sum())  # Count missing values

# Visualize missing values using heatmap for easy spotting
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

*Explanation:* Summarizes data distribution and reveals missing values per column. Visualization highlights columns needing cleaning.

***

```python
# Cell 3: Data cleaning - Imputation and column removal
df['Age'].fillna(df['Age'].median(), inplace=True)            # Fill missing Age with median
df['Embarked'].fillna(df['Embarked'].mode()[^0], inplace=True) # Fill missing Embarked with most frequent value
df.drop('Cabin', axis=1, inplace=True)                        # Drop Cabin because of many missing values

print("Missing values after cleaning:\n", df.isnull().sum())
```

*Explanation:* Handles missing values with imputation for moderate gaps and drops columns with too many missing records.

***

```python
# Cell 4: Feature selection and encoding categorical variables
# Drop unrelated features
df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# Convert categorical variables like Sex and Embarked to numeric with one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print(df.head())
```

*Explanation:* Removes irrelevant columns and converts categorical data to numeric form usable by machine learning algorithms.

***

```python
# Cell 5: Outlier detection with Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
num_features = ['Age', 'Fare', 'SibSp', 'Parch']

# Fit Isolation Forest and predict outliers (-1: outlier, 1: normal)
df['Outlier'] = iso_forest.fit_predict(df[num_features])

print(f"Number of outliers detected: {(df['Outlier'] == -1).sum()}")
```

*Explanation:* Applies ML algorithm to identify unusual records in key numeric features.

***

```python
# Cell 6: Visualize outliers for Age and Fare with boxplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['Outlier'], y=df['Age'])
plt.title('Outlier Distribution - Age')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Outlier'], y=df['Fare'])
plt.title('Outlier Distribution - Fare')

plt.show()
```

*Explanation:* Visual confirmation of outliers detected, showing their distribution relative to normal points.

***

```python
# Cell 7: Statistical outlier detection using z-score for Age and Fare
from scipy import stats

df['Age_zscore'] = np.abs(stats.zscore(df['Age']))
df['Fare_zscore'] = np.abs(stats.zscore(df['Fare']))

# Flag records with z-score > 3 as outliers
print(f"Age outliers by z-score: {(df['Age_zscore'] > 3).sum()}")
print(f"Fare outliers by z-score: {(df['Fare_zscore'] > 3).sum()}")
```

*Explanation:* Uses classic statistical approach to detect extreme values.

***

```python
# Cell 8: Normalize numerical data for modeling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print(df_scaled[['Age', 'Fare']].head())
```

*Explanation:* Standardizes data scale for better performance of many ML algorithms.

***

```python
# Cell 9: Model evaluation using cross-validation with Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X = df.drop(['Survived'], axis=1)
y = df['Survived']

rf = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf, X, y, cv=5)

print("5-fold cross-validation accuracies:", scores)
print("Mean CV accuracy:", scores.mean())
```

*Explanation:* Evaluates model using stratified folds to ensure robust performance estimation.

***

```python
# Cell 10: Clustering as anomaly detection using DBSCAN
from sklearn.cluster import DBSCAN

X_cluster = df_scaled[['Age', 'Fare']]
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster'] = dbscan.fit_predict(X_cluster)

# Cluster label -1 indicates noise/anomaly
print("Number of anomalies detected via DBSCAN:", (df['Cluster'] == -1).sum())
```

*Explanation:* Uses clustering to identify points that don’t fit any cluster, suggesting anomalies.

***

```python
# Cell 11: One-Class SVM anomaly detection (semi-supervised)
from sklearn.semi_supervised import OneClassSVM

ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
ocsvm.fit(X_cluster)
df['OCSVM_Anomaly'] = ocsvm.predict(X_cluster)

print("Anomalies detected with One-Class SVM:", (df['OCSVM_Anomaly'] == -1).sum())
```

*Explanation:* Alternative anomaly detection using semi-supervised learning.

***

```python
# Cell 12: Balancing dataset with SMOTE to handle class imbalance
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Original samples: {X.shape[^0]}, After SMOTE: {X_resampled.shape[^0]}")
```

*Explanation:* Synthetic sampling to create balanced classes for better supervised model training.

***

```python
# Cell 13: Supervised learning with Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(classification_report(y_test, y_pred))
```

*Explanation:* Builds a classification model and evaluates precision, recall, f1-score.

***

```python
# Cell 14: Lazy learning model K-Nearest Neighbors (k-NN)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print(classification_report(y_test, knn_pred))
```

*Explanation:* A non-parametric method that classifies new points based on majority class of neighbors.

***

```python
# Cell 15: Compare performance of two models
from sklearn.metrics import accuracy_score

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
```

*Explanation:* Compare final model accuracies to assist model selection.

***

```python
# Cell 16: Outlier detection using IQR (Interquartile Range) on Fare
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

fare_outliers_iqr = df[(df['Fare'] < Q1 - 1.5*IQR) | (df['Fare'] > Q3 + 1.5*IQR)]
print("Fare outliers detected by IQR method:")
print(fare_outliers_iqr[['Fare']])
```

*Explanation:* Classical method to flag outliers beyond 1.5 times IQR.

***

```python
# Cell 17: Remove outliers based on IQR
df_filtered = df[~((df['Fare'] < Q1 - 1.5*IQR) | (df['Fare'] > Q3 + 1.5*IQR))]
print(f"Data size: before={df.shape[^0]}, after removing outliers={df_filtered.shape[^0]}")
```

*Explanation:* Filters out extreme values before modeling.



