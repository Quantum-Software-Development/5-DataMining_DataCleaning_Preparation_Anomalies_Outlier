
<br>

**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]**


<br><br>

# 5- [Data Mining]() / Data Cleaning, Preparation and Detection of Anomalies (Outlier Detection)


<!-- ======================================= Start DEFAULT HEADER ===========================================  -->

<br><br>


[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP)  
[**School:**]() Faculty of Interdisciplinary Studies  
[**Program:**]() Humanistic AI and Data Science
[**Semester:**]() 2nd Semester 2025  
Professor:  [***Professor Doctor in Mathematics Daniel Rodrigues da Silva***](https://www.linkedin.com/in/daniel-rodrigues-048654a5/)

<br><br>

#### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)


<br><br>

<!--Confidentiality statement -->

#

<br><br><br>

> [!IMPORTANT]
> 
> ⚠️ Heads Up
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
> * The course emphasizes [**practical, hands-on experience**]() with real datasets to simulate professional consulting scenarios in the fields of **Data Analysis and Data Mining** for partner organizations and institutions affiliated with the university.
> * All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
> * Any content not authorized for public disclosure will remain [**confidential**]() and securely stored in [private repositories]().  
>


<br><br>

#

<!--END-->




<br><br><br><br>



<!-- PUC HEADER GIF
<p align="center">
  <img src="https://github.com/user-attachments/assets/0d6324da-9468-455e-b8d1-2cce8bb63b06" />
-->


<!-- video presentation -->


##### 🎶 Prelude Suite no.1 (J. S. Bach) - [Sound Design Remix]()

https://github.com/user-attachments/assets/4ccd316b-74a1-4bae-9bc7-1c705be80498

####  📺 For better resolution, watch the video on [YouTube.](https://youtu.be/_ytC6S4oDbM)


<br><br>


> [!TIP]
> 
>  This repository makes part of the  Data Mining, course from the undergraduate program Humanities, AI and Data Science at PUC-SP.
>
>  Access Data Mining [Main Reposi tory](https://github.com/Quantum-Software-Development/1-Main_DataMining_Repository)
> 

<!-- =======================================END DEFAULT HEADER ===========================================  -->

<br><br>


This repository addresses fundamental concepts and methodologies in Data Mining, with an emphasis on **data cleaning, preparation**, and the **identification of anomalies and outliers**. The material is grounded in a comprehensive reference document that integrates theoretical foundations with practical applications, including Python-based implementations for the treatment of heterogeneous and noisy datasets.

It constitutes a structured starting point for the systematic study and application of Data Mining techniques, particularly those related to data preprocessing, anomaly and outlier detection, and validation. The repository also provides contextualized examples and executable Python code to support empirical exploration and reproducibility.


<br><br>


## Table of Contents

- [Introduction](#introduction)
- [Dataset for Study](#dataset-for-study)
- [Pandas Functions for Data Exploration](#pandas-functions-for-data-exploration)
- [Key Concepts](#key-concepts)
  - Anomaly
  - Outlier
  - Anomaly Detection
  - Fraud Detection
- [Tips for Efficient and Effective Analysis](#tips-for-efficient-and-effective-analysis)
- [Statistical and Practical Significance](#statistical-and-practical-significance)
- [Characteristics and Understanding of Data](#characteristics-and-understanding-of-data)
- [Parsimony Principle in Model Selection](#parsimony-principle-in-model-selection)
- [Error Checking and Validation](#error-checking-and-validation)
- [Learning Paradigms](#learning-paradigms)
- [Applications](#applications)
- [Sentiment Analysis in Social Networks](#sentiment-analysis-in-social-networks)
- [Credit Card Fraud Detection](#credit-card-fraud-detection)
- [Non-Technical Losses in Electrical Energy](#non-technical-losses-in-electrical-energy)
- [Energy Load Segmentation](#energy-load-segmentation)
- [Steel Process Modeling](#steel-process-modeling)


<br><br>


## Introduction

The exponential growth of data generation necessitates intelligent techniques such as **Data Mining** to extract valuable knowledge from raw data. This process involves cleaning, preparing, mining, and validating data to enable effective decision-making.

<br><br>


## Dataset for Study

We use a publicly available, small, **dirty dataset** exemplifying missing values, duplicates, and inconsistencies to demonstrate concepts of data cleaning and anomaly detection.

Example dataset: [Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)  
This dataset contains missing values and requires preprocessing.


<br><br>


## Pandas Functions for Data Exploration

<br>

- `dataframe.describe()`  
  Displays statistical summary including count, mean, std, min, quartiles, and max.

<br>

- `dataframe.info()`  
  Shows information such as number of non-null entries and data types for each column.


<br>

Example usage:

<br>

```Python
import pandas as pd

df = pd.read_csv('titanic.csv')
print(df.describe())
print(df.info())
```

<br><br>



## Key Concepts

### Anomaly / Outlier
Anomalies or outliers are data points that deviate significantly from the majority and may indicate errors, rare events, or fraud.


<br>


### Anomaly Detection
Techniques to identify such unusual data points, including statistical, proximity-based, and machine learning methods.


<br>

### Fraud Detection
Identifying fraudulent transactions or activities that typically manifest as anomalies in data.

<br>

> [!TIP]
>
> [Access Code](https://github.com/Quantum-Software-Development/5-DataMining_DataCleaning_Preparation_Anomalies_Outlier/blob/f9ce063363d8b7139d6639efde5cc6c575dfaa47/Fraud%20Detection%20Concepts%20with%20Mini%20Data/code_Fraud%20Detection%20Concepts%20with%20Mini%20Data/Fraud_Detection_Concepts_with__MiniData.ipynb) - Fraud Detection Concepts with Mini Data
>



<br>

## Tips for Efficient and Effective Analysis

- **Significance of Mining**  
  - *Statistical significance*: Confidence in results, ensured by properly prepared datasets.  
  - *Practical significance*: Real-world applicability of insights.

<br>


- **Data Characteristics Influence Results**  
  The properties of the dataset affect analysis outcomes significantly.


<br>


- **Know Your Data**  
  Preliminary exploration and descriptive statistics help understand data distributions.

<br>


- **Parsimony Principle**  
  Choose models that balance complexity and interpretability.

<br>


- **Error Verification & Model Performance**  
  Check prediction errors, rule significance, and algorithm performance rigorously.


<br>

- **Validation of Results**  
  Compare multiple methods; assess generalization capacity; combine techniques; involve domain experts to validate findings.


<br><br>


## Formulas and Concepts

### Interquartile Range (IQR) rule for outliers:

<br>

$\Huge IQR = Q_3 - Q_1$

<br><br>

```latex
\Huge IQR = Q_3 - Q_1
```


<br><br>


$\Huge \text{Outlier if } x < Q_1 - 1.5 \times IQR \text{ or } x > Q_3 + 1.5 \times IQR$


<br><br>


```latex
\Huge \text{Outlier if } x < Q_1 - 1.5 \times IQR \text{ or } x > Q_3 + 1.5 \times IQR
```


<br><br>


### Z-Score for detecting outliers:

<br><br>


$\Huge Z = \frac{x - \mu}{\sigma}$


<br><br>

```latex
\Huge Z = \frac{x - \mu}{\sigma}
```


<br>

### [Where](): $\(x\)$ is a data point, $\(\mu\)$ mean, and $\(\sigma\)$ standard deviation.

<br><br>


## Learning Paradigms

<br>

| Paradigm               | Description                                                            | Example Algorithms                          |
|-----------------------|------------------------------------------------------------------------|---------------------------------------------|
| Supervised Learning    | Training with labeled data; learns mapping from inputs to outputs      | Decision Trees, Random Forest, SVM           |
| Unsupervised Learning  | Training with unlabeled data; discovers patterns or groups             | K-Means Clustering, DBSCAN, PCA              |
| Lazy Learning          | Defers generalization until a query is made                            | K-Nearest Neighbors (KNN)                     |


<br><br>

### Example: Decision Tree
A model is trained by partitioning data based on attribute splits optimizing a criterion like information gain.

<br>

### Example: K-Nearest Neighbors (KNN)
Classifies new data by looking at the 'k' closest known examples (lazy learning).


<br><br>


## Applications

Extensive use of data mining techniques in:

- Credit analysis and prediction
- Fraud detection
- Financial market prediction
- Customer relationship management
- Corporate bankruptcy prediction
- Energy sector
- Education, logistics, supply chain management
- Environment, social networks, ecommerce

<br><br>

## Sentiment Analysis in Social Networks

Classifying texts based on expressed sentiments (positive, negative, neutral) to measure public opinion, marketing effectiveness, and product feedback.


<br><br>

## Non-Technical Losses in Electrical Energy

- **Technical losses**: Intrinsic to electrical systems.
- **Commercial losses**: Errors, unmeasured consumption, fraud.

Data mining supports identifying irregularities and optimizing inspections.

<br><br>


## Energy Load Segmentation

Use clustering to segment typical daily electricity consumption patterns to improve demand prediction.


<br><br>


## Steel Process Modeling

Data mining to predict chemical composition and optimize production processes in steel industry.


<br><br>

## Credit Card Fraud Detection

Fraud categories:
- **Application Fraud**: Using fake personal info to obtain cards.
- **Behavioral Fraud**: Unauthorized use of genuine card user's data.

Fraud mitigation includes prevention (security measures) and detection (rapid identification of suspicious transactions).



<br><br>

## Python Example: [Data Cleaning and Anomaly Detection]()


<br><br>

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Load dataset

df = pd.read_csv('titanic.csv')

# Statistical overview

print(df.describe())
print(df.info())

# Handling missing values

df.fillna(df.median(numeric_only=True), inplace=True)  \# Impute missing numeric data

# Detecting outliers using Isolation Forest

iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(df.select_dtypes(include=[np.number]))
df['outlier'] = outliers

# Mark outliers (-1) and normal points (1)

print(df['outlier'].value_counts())
print(df[df['outlier'] == -1])

```

<br><br>


## Python Example: [Fraud Detection with Mini Data]()

<br>

###  [Cell 1]() - Below is the structured fraud detection code, cell-by-cell for including explanations about the dataset and additional techniques such as SMOTE for class balancing and Random Forest hyperparameter tuning.


<br><br>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load a smaller dataset (e.g., Iris dataset for binary classification - e.g., Versicolor vs Virginica)
# Carregar um conjunto de dados menor (por exemplo, conjunto de dados Iris para classificação binária - por exemplo, Versicolor vs Virginica)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# For binary classification, let's use only two classes (e.g., 1 and 2)
# Para classificação binária, vamos usar apenas duas classes (por exemplo, 1 e 2)
df_binary = df[df['target'].isin([1, 2])]
df_binary['target'] = df_binary['target'].replace({1: 0, 2: 1}) # Rename classes to 0 and 1
# Renomear classes para 0 e 1

# 2. Display the first few rows of the loaded DataFrame.
# Exibir as primeiras linhas do DataFrame carregado.
print("First 5 rows of the dataset:")
# Primeiras 5 linhas do conjunto de dados:
display(df_binary.head())

# 3. Display concise information about the DataFrame.
# Exibir informações concisas sobre o DataFrame.
print("\nDataset Info:")
# Informações do conjunto de dados:
df_binary.info()

# 4. Calculate and display the distribution of the target variable.
# Calcular e exibir a distribuição da variável alvo.
print("\nClass Distribution:")
# Distribuição de Classes:
display(df_binary['target'].value_counts())

# 5. Set up matplotlib for dark mode plotting.
# Configurar matplotlib para plotagem em modo escuro.
plt.style.use('dark_background')

# Set text color to white for better visibility in dark mode
# Definir a cor do texto para branco para melhor visibilidade no modo escuro
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = '#2b2b2b' # Dark background for figure
plt.rcParams['axes.facecolor'] = '#2b2b2b' # Dark background for axes

# 6. Define a turquoise color palette.
# Definir uma paleta de cores turquesa.
turquoise_palette = ['#40E0D0', '#48D1CC', '#00CED1', '#5F9EA0', '#008B8B']
```

<br><br>

###  [Cell 2]() - Exploratory Data Analysis (EDA) 

<br>

This code block carries out the initial steps of a data analysis workflow.
In essence, it prepares the dataset for further exploration and offers a first look at its main characteristics, laying the groundwork for more detailed analysis or modeling.


<br><br>


```python
import os

# Define the directory for saving plots
plot_dir = '/content/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# 1. Create histograms for each feature in df_binary with dual-language titles and labels.
# 1. Criar histogramas para cada característica em df_binary com títulos e rótulos em dois idiomas.
print("Feature Distributions (Histograms):")
# Distribuições das Características (Histogramas):
# Use only the first color from the palette for histograms
df_binary.hist(figsize=(12, 10), color=turquoise_palette[0], bins=15)
plt.suptitle('Feature Distributions / Distribuições das Características', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig(f'{plot_dir}/feature_histograms.png') # Save histogram plot

# 2. Generate box plots for each feature, comparing distributions across target classes with dual-language titles and labels.
# 2. Gerar box plots para cada característica, comparando as distribuições entre as classes alvo com títulos e rótulos em dois idiomas.
print("\nFeature Distributions by Target Class (Box Plots):")
# Distribuições das Características por Classe Alvo (Box Plots):
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(df_binary.columns[:-1]):
    # Removed palette argument from boxplot as it's not used with hue and causes a warning
    sns.boxplot(x='target', y=col, data=df_binary, ax=axes[i])
    axes[i].set_title(f'{col} Distribution by Target Class / Distribuição de {col} por Classe Alvo')
    axes[i].set_xlabel('Target Class / Classe Alvo')
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.show()
plt.savefig(f'{plot_dir}/feature_box_plots.png') # Save box plot

# 3. Create a pair plot of the features in df_binary, colored by the 'target' variable, with a dual-language title.
# 3. Criar um pair plot das características em df_binary, colorido pela variável 'target', com um título em dois idiomas.
print("\nPair Plot of Features by Target Class:")
# Pair Plot das Características por Classe Alvo:
# Use only the first two colors from the palette for the two classes
sns.pairplot(df_binary, hue='target', palette=turquoise_palette[:2], diag_kind='kde')
plt.suptitle('Pair Plot of Features by Target Class / Pair Plot das Características por Classe Alvo', y=1.02, fontsize=16)
plt.show()
plt.savefig(f'{plot_dir}/feature_pair_plot.png') # Save pair plot

# 4. Calculate and display the correlation matrix for the features in df_binary and visualize it with a heatmap and dual-language titles and labels.
# 4. Calcular e exibir a matriz de correlação para as características em df_binary e visualizá-la com um heatmap e títulos e rótulos em dois idiomas.
print("\nCorrelation Matrix:")
# Matriz de Correlação:
correlation_matrix_binary = df_binary.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_binary, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix / Matriz de Correlação', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
plt.savefig(f'{plot_dir}/correlation_matrix_heatmap.png') # Save heatmap plot
```


<br><br>



### 1- Feature Distributions (Histograms):

<br>

<img width="1358" height="1188" alt="Image" src="https://github.com/user-attachments/assets/c41a65c8-7725-4b9a-90ad-865c69aefadc" />

<br><br>

### 2- sepal length (cm) Distributions

<br>

<img width="1740" height="1160" alt="Image" src="https://github.com/user-attachments/assets/d34f548a-a84b-4200-91cf-9a0f8bb09bfb" />


<br><br>

### 3-Pair Plot of Features by Target Class

<br>

<img width="1230" height="1198" alt="Image" src="https://github.com/user-attachments/assets/26e95e05-6873-438c-8d76-1aa6ce277a7d" />

<br><

### 4- Correlation Matrix

<br>

<img width="1064" height="914" alt="Image" src="https://github.com/user-attachments/assets/0fac7a0a-37aa-4319-a2c6-6189ce46b46f" />


<br><br>




====================================🏄 Building ....===================================













































<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>



<!-- ========================== [Bibliographr ====================  -->

<br><br>


## [Bibliography]()

[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introduction to Data Mining: Basic Concepts, Algorithms, and Applications*. Saraiva.

[2](). **Ferreira, A. C. P. L. et al.** (2024). *Artificial Intelligence – A Machine Learning Approach*. 2nd Ed. LTC.

[3](). **Larson & Farber** (2015). *Applied Statistics*. Pearson.

<br><br>

      
<!-- ======================================= Bibliography Portugues ===========================================  -->

<!--

## [Bibliography]()


[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introdução à mineração de dados: conceitos básicos, algoritmos e aplicações*. Saraiva.

[2](). **Ferreira, A. C. P. L. et al.** (2024). *Inteligência Artificial - Uma Abordagem de Aprendizado de Máquina*. 2nd Ed. LTC.

[3](). **Larson & Farber** (2015). *Estatística Aplicada*. Pearson.


<br><br>
-->

<!-- ======================================= Start Footer ===========================================  -->


<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br><br>



#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />




<br><br><br>

<p align="center">  ────────────── 🔭⋆ ──────────────


<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>

<!--
<p align="center">  ────────────── ✦ ──────────────
-->



<!-- Programmers and artists are the only professionals whose hobby is their profession."

" I love people who are committed to transforming the world "

" I'm big fan of those who are making waves in the world! "

##### <p align="center">( Rafael Lain ) </p>   -->

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)










