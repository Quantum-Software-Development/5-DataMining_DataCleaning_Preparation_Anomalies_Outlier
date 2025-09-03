
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


==============================  Still shaping this repo 🏄 ==============================



































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










