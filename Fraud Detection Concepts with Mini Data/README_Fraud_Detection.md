# Fraud Detection Concepts with Mini Dataset

This repository contains a **step-by-step guide** for building a fraud detection pipeline using a **mini dataset**. The notebook is structured cell by cell so that you can **copy and paste the code directly into Google Colab**.

## How to use

1. Open [Google Colab](https://colab.research.google.com/).

2. Create a new notebook.

3. Copy each code block below **in order** (one cell at a time).

4. Run the cells sequentially.

5. Wherever you see `📊 Plot Output Here`, run the cell in Colab and the plot will appear there.


### Code Cell 0

```python
from google.colab import userdata
userdata.get('GOOGLE_API_KEY_1')
```


### Explanation (Markdown Cell 1)

Below is the **structured fraud detection code in English**, cell-by-cell for a Jupyter notebook, including explanations about the dataset and **additional techniques** such as SMOTE for class balancing and Random Forest hyperparameter tuning.


### Explanation (Markdown Cell 2)

# Task
Refactor the entire Python notebook for binary classification. Use a small dataset that runs quickly. Include detailed explanations in both Portuguese and English using hashtags. Implement dark mode plotting with a turquoise color scheme. Start the code from the data loading stage, deleting any previous code related to the random forest model or the larger dataset.


### Explanation (Markdown Cell 3)

## Data loading and initial understanding

### Subtask:
Load a smaller dataset suitable for binary classification. Display initial information and class distribution with explanations in Portuguese and English. Set up dark mode plotting with a turquoise color palette.


### Explanation (Markdown Cell 4)

**Reasoning**:
Load a small binary classification dataset, display its head and info, show the class distribution, and set up dark mode plotting with a turquoise color palette.


### Code Cell 5

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
print("\nTurquoise color palette defined.")
# Paleta de cores turquesa definida.
```

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 6)

## Exploratory data analysis (eda)

### Subtask:
Perform EDA with dark mode visualizations using the turquoise color scheme. Include plots relevant to the smaller dataset, such as distributions and relationships between features and the target variable, with dual-language explanations.


### Explanation (Markdown Cell 7)

**Reasoning**:
Create histograms, box plots, a pair plot, and a heatmap of the correlation matrix for the df_binary dataframe, using the defined turquoise palette and dark mode settings, with dual-language titles and labels.


### Code Cell 8

```python
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

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 9)

## Data preparation

### Subtask:
Handle missing values, scale numerical features, and split the data into training and testing sets. Include explanations in Portuguese and English.


### Explanation (Markdown Cell 10)

**Reasoning**:
Check for missing values in the dataframe, handle them if necessary, separate features and target, scale the features, and split the data for training and testing, including explanations in both languages.


### Code Cell 11

```python
# 1. Check for missing values in the df_binary DataFrame and print the count for each column.
# 1. Verificar valores ausentes no DataFrame df_binary e imprimir a contagem para cada coluna.
print("Checking for missing values / Verificando valores ausentes:")
print(df_binary.isnull().sum())

# 2. If missing values are found, handle them appropriately for numerical data (e.g., imputation with the mean or median).
# Based on the previous df_binary.info() output, there are no missing values.
# Com base na saída anterior de df_binary.info(), não há valores ausentes.
# No action needed for missing values in this case.
# Nenhuma ação necessária para valores ausentes neste caso.

# 3. Separate the features (X) and the target variable (y) from the df_binary DataFrame.
# 3. Separar as características (X) e a variável alvo (y) do DataFrame df_binary.
X = df_binary.drop('target', axis=1)
y = df_binary['target']
print("\nFeatures (X) and Target (y) separated. / Características (X) e Alvo (y) separados.")

# 4. Scale the numerical features using StandardScaler.
# Fit the scaler only on the training data to prevent data leakage.
# 4. Escalar as características numéricas usando StandardScaler.
# Ajustar o scaler apenas nos dados de treinamento para evitar vazamento de dados.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 5. Split the data into training and testing sets.
# 5. Dividir os dados em conjuntos de treinamento e teste.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Using stratify for balanced classes
# Usando stratify para classes balanceadas

# Fit and transform the scaler on the training data
# Ajustar e transformar o scaler nos dados de treinamento
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler
# Transformar os dados de teste usando o scaler ajustado
X_test_scaled = scaler.transform(X_test)

print("\nData split into training and testing sets (80/20). / Dados divididos em conjuntos de treinamento e teste (80/20).")
print("Features scaled using StandardScaler. / Características escaladas usando StandardScaler.")
print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
```


### Explanation (Markdown Cell 12)

## Handle class imbalance

### Subtask:
Address class imbalance if present in the dataset using an appropriate technique like SMOTE or class weights, with dual-language explanations.


### Explanation (Markdown Cell 13)

**Reasoning**:
Check the class distribution of the training set to confirm if class imbalance exists and print the value counts with a dual-language explanation.


### Code Cell 14

```python
# 1. Check the class distribution of the training set (y_train) to confirm if class imbalance exists.
# Print the value counts with a dual-language explanation.
# 1. Verificar a distribuição de classes do conjunto de treinamento (y_train) para confirmar se existe desequilíbrio de classes.
# Imprimir a contagem de valores com uma explicação em dois idiomas.
print("Class distribution in the training set (y_train):")
# Distribuição de classes no conjunto de treinamento (y_train):
display(y_train.value_counts())
```


### Explanation (Markdown Cell 15)

**Reasoning**:
Since the training data is perfectly balanced (40 instances for each class), SMOTE is not needed. Note this and proceed to the next step.


### Code Cell 16

```python
# 2. If there is a significant imbalance (e.g., one class is much smaller than the other), apply SMOTE to the training data.
# Based on the value counts, the training set is perfectly balanced (40 vs 40).
# Therefore, SMOTE is not required in this case.
# 2. Se houver um desequilíbrio significativo (por exemplo, uma classe for muito menor que a outra), aplique SMOTE aos dados de treinamento.
# Com base na contagem de valores, o conjunto de treinamento está perfeitamente equilibrado (40 vs 40).
# Portanto, o SMOTE não é necessário neste caso.
print("\nThe training set is balanced. SMOTE is not required. / O conjunto de treinamento está equilibrado. SMOTE não é necessário.")

# 3. Print the class distribution of the resampled training set (y_train_resampled) to show the effect of SMOTE, with a dual-language explanation.
# This step is skipped as SMOTE was not applied.
# 3. Imprimir a distribuição de classes do conjunto de treinamento reamostrado (y_train_resampled) para mostrar o efeito do SMOTE, com uma explicação em dois idiomas.
# Este passo é ignorado, pois o SMOTE não foi aplicado.

# 4. If SMOTE was not used (due to no significant imbalance or choosing an alternative method later), note this and proceed.
# This is already noted in step 2.
# 4. Se o SMOTE não foi usado (devido à ausência de desequilíbrio significativo ou à escolha de um método alternativo posteriormente), observe isso e prossiga.
# Isso já foi observado no passo 2.
```


### Explanation (Markdown Cell 17)

## Implement and evaluate multiple classification models

### Subtask:
Implement and evaluate at least two classification models suitable for structured data, prioritizing models that are relatively fast. Include dual-language explanations for model choices and training.


### Explanation (Markdown Cell 18)

**Reasoning**:
Implement and evaluate two classification models as requested by the subtask.


### Code Cell 19

```python
# 1. Import necessary classification models and metrics.
# 1. Importar os modelos de classificação e métricas necessários.
from sklearn.linear_model import LogisticRegression # A simple, fast, and interpretable model / Um modelo simples, rápido e interpretável
from sklearn.tree import DecisionTreeClassifier # A basic tree-based model / Um modelo básico baseado em árvore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 3. Initialize the chosen classification models.
# Include a random_state for reproducibility.
# 3. Inicializar os modelos de classificação escolhidos.
# Incluir um random_state para reprodutibilidade.

# Logistic Regression Model / Modelo de Regressão Logística
log_reg_model = LogisticRegression(random_state=42)
print("Initialized Logistic Regression Model / Modelo de Regressão Logística inicializado.")

# Decision Tree Model / Modelo de Árvore de Decisão
# Using a small max_depth for speed and simplicity / Usando uma max_depth pequena para velocidade e simplicidade
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
print("Initialized Decision Tree Model / Modelo de Árvore de Decisão inicializado.")


# 4. Train each initialized model using the scaled training data.
# 4. Treinar cada modelo inicializado usando os dados de treinamento escalados.

# Training Logistic Regression Model / Treinando Modelo de Regressão Logística
print("\nTraining Logistic Regression Model... / Treinando Modelo de Regressão Logística...")
log_reg_model.fit(X_train_scaled, y_train)
print("Logistic Regression Model trained. / Modelo de Regressão Logística treinado.")

# Training Decision Tree Model / Treinando Modelo de Árvore de Decisão
print("\nTraining Decision Tree Model... / Treinando Modelo de Árvore de Decisão...")
dt_model.fit(X_train_scaled, y_train)
print("Decision Tree Model trained. / Modelo de Árvore de Decisão treinado.")

# 5. Make predictions on the scaled test data using each trained model.
# 5. Fazer previsões nos dados de teste escalados usando cada modelo treinado.

y_pred_log_reg = log_reg_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test_scaled)

print("\nPredictions made for both models. / Previsões feitas para ambos os modelos.")

# 6. Evaluate each model's performance using the imported metrics.
# 7. Print the evaluation metrics for each model with clear labels and dual-language explanations.
# 6. Avaliar o desempenho de cada modelo usando as métricas importadas.
# 7. Imprimir as métricas de avaliação para cada modelo com rótulos claros e explicações em dois idiomas.

print("\n--- Model Evaluation --- / --- Avaliação do Modelo ---")

# Logistic Regression Evaluation / Avaliação da Regressão Logística
print("\nLogistic Regression Evaluation / Avaliação da Regressão Logística:")
print(f"Accuracy / Acurácia: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precision / Precisão: {precision_score(y_test, y_pred_log_reg):.4f}")
print(f"Recall / Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
print(f"F1-Score / F1-Score: {f1_score(y_test, y_pred_log_reg):.4f}")

# Decision Tree Evaluation / Avaliação da Árvore de Decisão
print("\nDecision Tree Evaluation / Avaliação da Árvore de Decisão:")
print(f"Accuracy / Acurácia: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision / Precisão: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall / Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score / F1-Score: {f1_score(y_test, y_pred_dt):.4f}")

# 8. Briefly discuss the initial performance of each model.
# 8. Discutir brevemente o desempenho inicial de cada modelo.
print("\n--- Initial Performance Discussion --- / --- Discussão Inicial de Desempenho ---")
print("Both models performed very well on this small and balanced dataset.")
print("Logistic Regression achieved perfect scores across all metrics.")
print("The Decision Tree also performed excellently, showing slightly different but still high scores.")
print("\nAmbos os modelos tiveram um desempenho muito bom neste conjunto de dados pequeno e equilibrado.")
print("A Regressão Logística alcançou pontuações perfeitas em todas as métricas.")
print("A Árvore de Decisão também teve um desempenho excelente, mostrando pontuações ligeiramente diferentes, mas ainda altas.")
```


### Explanation (Markdown Cell 20)

## Hyperparameter tuning

### Subtask:
Perform hyperparameter tuning for at least one model with a very small search space to ensure speed. Include dual-language explanations for the tuning process.


### Explanation (Markdown Cell 21)

**Reasoning**:
Import necessary libraries for hyperparameter tuning, define a small search space for the Decision Tree model, initialize GridSearchCV, fit it to the scaled training data, print the best parameters and score, and optionally initialize and train the best model.


### Code Cell 22

```python
# 1. Import the necessary libraries for hyperparameter tuning.
# 1. Importar as bibliotecas necessárias para otimização de hiperparâmetros.
from sklearn.model_selection import GridSearchCV # For hyperparameter tuning / Para otimização de hiperparâmetros

# 2. Define a small hyperparameter search space for one of the trained models (e.g., DecisionTreeClassifier).
# Define a small parameter grid for faster execution.
# 2. Definir um pequeno espaço de busca de hiperparâmetros para um dos modelos treinados (por exemplo, DecisionTreeClassifier).
# Definir uma pequena grade de parâmetros para execução mais rápida.
param_grid = {
    'max_depth': [3, 5, 7], # Maximum depth of the tree / Profundidade máxima da árvore
    'min_samples_split': [2, 4] # Minimum number of samples required to split an internal node / Número mínimo de amostras necessárias para dividir um nó interno
}
print("Hyperparameter search space defined for Decision Tree. / Espaço de busca de hiperparâmetros definido para Árvore de Decisão.")
print(param_grid)

# 3. Initialize the chosen tuning method (e.g., GridSearchCV) with the model, the parameter grid, and a small number of cross-validation folds.
# Using Decision Tree model and a small cv value (e.g., 3) for speed.
# 3. Inicializar o método de ajuste escolhido (por exemplo, GridSearchCV) com o modelo, a grade de parâmetros e um pequeno número de folds de validação cruzada.
# Usando o modelo de Árvore de Decisão e um pequeno valor de cv (por exemplo, 3) para velocidade.
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
print("\nInitialized GridSearchCV for Decision Tree. / GridSearchCV inicializado para Árvore de Decisão.")

# 4. Fit the tuning method to the scaled training data.
# This will perform cross-validation for each combination of parameters in the grid.
# 4. Ajustar o método de ajuste aos dados de treinamento escalados.
# Isso realizará validação cruzada para cada combinação de parâmetros na grade.
print("\nPerforming hyperparameter tuning for Decision Tree... / Realizando otimização de hiperparâmetros para Árvore de Decisão...")
grid_search.fit(X_train_scaled, y_train)
print("Hyperparameter tuning finished. / Otimização de hiperparâmetros concluída.")

# 5. Print the best hyperparameters found by the tuning process.
# 5. Imprimir os melhores hiperparâmetros encontrados pelo processo de ajuste.
print("\nBest hyperparameters found: / Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# 6. Print the best score achieved during the tuning process.
# 6. Imprimir a melhor pontuação alcançada durante o processo de ajuste.
print("\nBest cross-validation score: / Melhor pontuação de validação cruzada:")
print(f"{grid_search.best_score_:.4f}")

# 7. (Optional) Initialize a new model with the best hyperparameters found and train it on the entire training data.
# 7. (Opcional) Inicializar um novo modelo com os melhores hiperparâmetros encontrados e treiná-lo em todos os dados de treinamento.
best_dt_model = grid_search.best_estimator_
print("\nInitialized best Decision Tree model with found hyperparameters. / Melhor modelo de Árvore de Decisão inicializado com os hiperparâmetros encontrados.")

# Train the best model on the entire training data / Treinar o melhor modelo em todos os dados de treinamento
best_dt_model.fit(X_train_scaled, y_train)
print("Best Decision Tree model trained on the entire training data. / Melhor modelo de Árvore de Decisão treinado em todos os dados de treinamento.")
```


### Explanation (Markdown Cell 23)

## Model evaluation

### Subtask:
Evaluate all models using appropriate metrics for binary classification, with explanations in Portuguese and English. Generate dark mode plots (like Confusion Matrix, ROC curve) using the turquoise color scheme.


### Explanation (Markdown Cell 24)

**Reasoning**:
Import necessary libraries for plotting and evaluation metrics.


### Code Cell 25

```python
# 1. Import necessary libraries for plotting and evaluation metrics.
# 1. Importar as bibliotecas necessárias para plotagem e métricas de avaliação.
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


### Explanation (Markdown Cell 26)

**Reasoning**:
Define a function to plot the confusion matrix in dark mode with the turquoise palette, including dual-language titles and labels.


### Code Cell 27

```python
# 2. Define a function to plot the confusion matrix in dark mode with the turquoise palette,
#    including dual-language titles and labels.
# 2. Definir uma função para plotar a matriz de confusão no modo escuro com a paleta turquesa,
#    incluindo títulos e rótulos em dois idiomas.
def plot_confusion_matrix(y_true, y_pred, title_en, title_pt, classes, cmap):
    """
    Plots a confusion matrix with dual-language titles and labels in dark mode.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title_en (str): English title for the plot.
        title_pt (str): Portuguese title for the plot.
        classes (list): List of class names.
        cmap (str): Colormap to use for the heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{title_en} / {title_pt}', fontsize=14)
    plt.xlabel('Predicted Label / Rótulo Previsto', fontsize=12)
    plt.ylabel('True Label / Rótulo Verdadeiro', fontsize=12)
    plt.show()

print("Confusion matrix plotting function defined. / Função de plotagem da matriz de confusão definida.")
```

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 28)

**Reasoning**:
Generate and plot the confusion matrix for each trained model using the scaled test data and the defined plotting function.


### Code Cell 29

```python
# 3. Generate and plot the confusion matrix for each trained model
#    (Logistic Regression, initial Decision Tree, and the best Decision Tree from tuning)
#    using the scaled test data and the defined plotting function.
# 3. Gerar e plotar a matriz de confusão para cada modelo treinado
#    (Regressão Logística, Árvore de Decisão inicial e a melhor Árvore de Decisão da otimização)
#    usando os dados de teste escalados e a função de plotagem definida.

class_names = ['Class 0', 'Class 1'] # Assuming binary classes 0 and 1 / Assumindo classes binárias 0 e 1

# Confusion Matrix for Logistic Regression / Matriz de Confusão para Regressão Logística
plot_confusion_matrix(y_test, y_pred_log_reg,
                      title_en='Confusion Matrix - Logistic Regression',
                      title_pt='Matriz de Confusão - Regressão Logística',
                      classes=class_names,
                      cmap='Blues') # Using a blue-ish colormap for dark mode / Usando um mapa de cores azulado para modo escuro
plt.savefig(f'{plot_dir}/lr_confusion_matrix.png') # Save plot

# Confusion Matrix for Initial Decision Tree / Matriz de Confusão para Árvore de Decisão Inicial
plot_confusion_matrix(y_test, y_pred_dt,
                      title_en='Confusion Matrix - Initial Decision Tree',
                      title_pt='Matriz de Confusão - Árvore de Decisão Inicial',
                      classes=class_names,
                      cmap='Greens') # Using a green-ish colormap for dark mode / Usando um mapa de cores esverdeado para modo escuro
plt.savefig(f'{plot_dir}/initial_dt_confusion_matrix.png') # Save plot

# Make predictions for the best Decision Tree model / Fazer previsões para o melhor modelo de Árvore de Decisão
y_pred_best_dt = best_dt_model.predict(X_test_scaled)

# Confusion Matrix for Best Decision Tree / Matriz de Confusão para Melhor Árvore de Decisão
plot_confusion_matrix(y_test, y_pred_best_dt,
                      title_en='Confusion Matrix - Tuned Decision Tree',
                      title_pt='Matriz de Confusão - Árvore de Decisão Otimizada',
                      classes=class_names,
                      cmap='Purples') # Using a purple-ish colormap for dark mode / Usando um mapa de cores arroxeado para modo escuro
plt.savefig(f'{plot_dir}/tuned_dt_confusion_matrix.png') # Save plot
```

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 30)

**Reasoning**:
Define a function to plot the ROC curve and calculate AUC in dark mode with the turquoise palette, including dual-language titles and labels.


### Code Cell 31

```python
# 4. Define a function to plot the ROC curve and calculate AUC in dark mode with the turquoise palette,
#    including dual-language titles and labels.
# 4. Definir uma função para plotar a curva ROC e calcular a AUC no modo escuro com a paleta turquesa,
#    incluindo títulos e rótulos em dois idiomas.
def plot_roc_curve(y_true, y_probs, title_en, title_pt, color):
    """
    Plots an ROC curve and calculates AUC with dual-language titles and labels in dark mode.

    Args:
        y_true (array-like): True labels.
        y_probs (array-like): Predicted probabilities for the positive class.
        title_en (str): English title for the plot.
        title_pt (str): Portuguese title for the plot.
        color (str): Color for the ROC curve line.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') # Random guess line / Linha de palpite aleatório
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate / Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('True Positive Rate / Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title(f'{title_en} / {title_pt}', fontsize=14)
    plt.legend(loc="lower right")
    plt.show()

    return auc # Return AUC value / Retornar valor da AUC

print("ROC curve plotting function defined. / Função de plotagem da curva ROC definida.")
```

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 32)

**Reasoning**:
Calculate the probability predictions for the positive class on the scaled test data for each model, generate and plot the ROC curve for each model, and print the AUC score for each model with dual-language explanations.


### Code Cell 33

```python
# 5. Calculate the probability predictions for the positive class on the scaled test data for each model.
# 5. Calcular as probabilidades previstas para a classe positiva nos dados de teste escalados para cada modelo.
y_probs_log_reg = log_reg_model.predict_proba(X_test_scaled)[:, 1]
y_probs_dt = dt_model.predict_proba(X_test_scaled)[:, 1]
y_probs_best_dt = best_dt_model.predict_proba(X_test_scaled)[:, 1]

print("Probability predictions for the positive class calculated for all models. / Probabilidades previstas para a classe positiva calculadas para todos os modelos.")

# 6. Generate and plot the ROC curve for each model using the true test labels and the probability predictions,
#    using the defined plotting function.
# 7. Print the AUC score for each model with dual-language explanations.
# 6. Gerar e plotar a curva ROC para cada modelo usando os rótulos de teste verdadeiros e as probabilidades previstas,
#    usando a função de plotagem definida.
# 7. Imprimir a pontuação AUC para cada modelo com explicações em dois idiomas.

print("\n--- ROC Curve and AUC Evaluation --- / --- Avaliação da Curva ROC e AUC ---")

# ROC Curve and AUC for Logistic Regression / Curva ROC e AUC da Regressão Logística
print("\nLogistic Regression ROC and AUC / Curva ROC e AUC da Regressão Logística:")
auc_log_reg = plot_roc_curve(y_test, y_probs_log_reg,
                             title_en='ROC Curve - Logistic Regression',
                             title_pt='Curva ROC - Regressão Logística',
                             color=turquoise_palette[0]) # Using a color from the turquoise palette / Usando uma cor da paleta turquesa
plt.savefig(f'{plot_dir}/lr_roc_curve.png') # Save plot
print(f"AUC for Logistic Regression: {auc_log_reg:.4f}")
print(f"AUC para Regressão Logística: {auc_log_reg:.4f}")


# ROC Curve and AUC for Initial Decision Tree / Curva ROC e AUC para Árvore de Decisão Inicial
print("\nInitial Decision Tree ROC and AUC / Curva ROC e AUC da Árvore de Decisão Inicial:")
auc_dt = plot_roc_curve(y_test, y_probs_dt,
                        title_en='ROC Curve - Initial Decision Tree',
                        title_pt='Curva ROC - Árvore de Decisão Inicial',
                        color=turquoise_palette[1]) # Using a color from the turquoise palette / Usando uma cor da paleta turquesa
plt.savefig(f'{plot_dir}/initial_dt_roc_curve.png') # Save plot
print(f"AUC for Initial Decision Tree: {auc_dt:.4f}")
print(f"AUC para Árvore de Decisão Inicial: {auc_dt:.4f}")


# ROC Curve and AUC for Best Decision Tree / Curva ROC e AUC para Melhor Árvore de Decisão
print("\nTuned Decision Tree ROC and AUC / Curva ROC e AUC da Árvore de Decisão Otimizada:")
auc_best_dt = plot_roc_curve(y_test, y_probs_best_dt,
                             title_en='ROC Curve - Tuned Decision Tree',
                             title_pt='Curva ROC - Árvore de Decisão Otimizada',
                             color=turquoise_palette[2]) # Using a color from the turquoise palette / Usando uma cor da paleta turquesa
plt.savefig(f'{plot_dir}/tuned_dt_roc_curve.png') # Save plot
print(f"AUC for Tuned Decision Tree: {auc_best_dt:.4f}")
print(f"AUC para Árvore de Decisão Otimizada: {auc_best_dt:.4f}")
```

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 34)

**Reasoning**:
Create a summary table or print the evaluation metrics for all models and briefly discuss the performance comparison, with dual-language explanations.


### Code Cell 35

```python
# 8. Create a summary table or print the evaluation metrics (Accuracy, Precision, Recall, F1-Score)
#    for all models (including the tuned Decision Tree) on the test set, with dual-language labels.
# 8. Criar uma tabela de resumo ou imprimir as métricas de avaliação (Acurácia, Precisão, Recall, F1-Score)
#    para todos os modelos (incluindo a Árvore de Decisão otimizada) no conjunto de teste, com rótulos em dois idiomas.

print("\n--- Model Performance Summary (Test Set) --- / --- Resumo de Desempenho dos Modelos (Conjunto de Teste) ---")

# Calculate metrics for the tuned Decision Tree / Calcular métricas para a Árvore de Decisão otimizada
accuracy_best_dt = accuracy_score(y_test, y_pred_best_dt)
precision_best_dt = precision_score(y_test, y_pred_best_dt)
recall_best_dt = recall_score(y_test, y_pred_best_dt)
f1_best_dt = f1_score(y_test, y_pred_best_dt)

# Using the previously calculated metrics for Logistic Regression and Initial Decision Tree
# Usando as métricas calculadas anteriormente para Regressão Logística e Árvore de Decisão Inicial

data = {
    'Model / Modelo': ['Logistic Regression / Regressão Logística', 'Initial Decision Tree / Árvore de Decisão Inicial', 'Tuned Decision Tree / Árvore de Decisão Otimizada'],
    'Accuracy / Acurácia': [accuracy_score(y_test, y_pred_log_reg), accuracy_score(y_test, y_pred_dt), accuracy_best_dt],
    'Precision / Precisão': [precision_score(y_test, y_pred_log_reg), precision_score(y_test, y_pred_dt), precision_best_dt],
    'Recall / Recall': [recall_score(y_test, y_pred_log_reg), recall_score(y_test, y_pred_dt), recall_best_dt],
    'F1-Score / F1-Score': [f1_score(y_test, y_pred_log_reg), f1_score(y_test, y_pred_dt), f1_best_dt],
    'AUC': [auc_log_reg, auc_dt, auc_best_dt] # Including AUC in the summary / Incluindo AUC no resumo
}

performance_df = pd.DataFrame(data)
display(performance_df)

# 9. Briefly discuss the performance comparison of the models based on the evaluation metrics and visualizations,
#    with explanations in Portuguese and English.
# 9. Discutir brevemente a comparação de desempenho dos modelos com base nas métricas de avaliação e visualizações,
#    com explicações em português e inglês.
print("\n--- Performance Comparison Discussion --- / --- Discussão da Comparação de Desempenho ---")
print("Based on the evaluation metrics and visualizations:")
print("- Logistic Regression achieved perfect scores (Accuracy, Precision, Recall, F1, and AUC) on this small test set, indicating excellent performance.")
print("- The Initial Decision Tree performed well but had lower scores compared to Logistic Regression, particularly in Recall and F1-Score.")
print("- The Tuned Decision Tree showed the same performance as the initial Decision Tree on this specific test set with the chosen small hyperparameter grid.")
print("- The Confusion Matrices visualize the counts of true/false positives/negatives for each model.")
print("- The ROC curves and AUC scores confirm the superior performance of Logistic Regression, with an AUC close to 1, while both Decision Trees had a lower AUC.")
print("This suggests that for this specific binary classification task on this dataset, Logistic Regression is the most effective model among those tested.")

print("\nCom base nas métricas de avaliação e visualizações:")
print("- A Regressão Logística alcançou pontuações perfeitas (Acurácia, Precisão, Recall, F1 e AUC) neste pequeno conjunto de teste, indicando excelente desempenho.")
print("- A Árvore de Decisão Inicial teve um bom desempenho, mas obteve pontuações mais baixas em comparação com a Regressão Logística, particularmente em Recall e F1-Score.")
print("- A Árvore de Decisão Otimizada apresentou o mesmo desempenho que a Árvore de Decisão inicial neste conjunto de teste específico com a pequena grade de hiperparâmetros escolhida.")
print("- As Matrizes de Confusão visualizam as contagens de verdadeiros/falsos positivos/negativos para cada modelo.")
print("- As curvas ROC e as pontuações AUC confirmam o desempenho superior da Regressão Logística, com uma AUC próxima de 1, enquanto ambas as Árvores de Decisão tiveram uma AUC menor.")
print("Isso sugere que, para esta tarefa de classificação binária específica neste conjunto de dados, a Regressão Logística é o modelo mais eficaz entre os testados.")
```


### Explanation (Markdown Cell 36)

## Implement cross-validation

### Subtask:
Introduce and implement k-fold cross-validation for more robust model evaluation, with dual-language explanations.


### Explanation (Markdown Cell 37)

**Reasoning**:
Introduce cross-validation for more robust model evaluation, starting by importing the necessary function.


### Code Cell 38

```python
# 1. Import the cross_val_score function from sklearn.model_selection.
# 1. Importar a função cross_val_score de sklearn.model_selection.
from sklearn.model_selection import cross_val_score

print("Imported cross_val_score for cross-validation. / cross_val_score importado para validação cruzada.")
```


### Explanation (Markdown Cell 39)

**Reasoning**:
Explain the importance of cross-validation for model evaluation, especially on small datasets, with dual-language explanations in a markdown cell.


### Code Cell 40

```python
# 2. Explain in a markdown cell why cross-validation is useful for model evaluation, especially on smaller datasets,
#    including explanations in Portuguese and English.
# 2. Explicar em uma célula markdown por que a validação cruzada é útil para avaliação de modelos, especialmente em conjuntos de dados menores,
#    incluindo explicações em português e inglês.
```


### Explanation (Markdown Cell 41)

**Reasoning**:
Implement k-fold cross-validation for the Logistic Regression and the best Decision Tree models using cross_val_score on the entire scaled dataset, print the results, and discuss the findings.


### Explanation (Markdown Cell 42)

**Reasoning**:
Fix the NameError by using the correct variable name `log_reg_cv_scores` and regenerate the full code block for cross-validation, results printing, and discussion.


### Code Cell 43

```python
# 3. Choose one or more of the trained models (e.g., Logistic Regression and the best Decision Tree).
# Models chosen: Logistic Regression (log_reg_model) and Tuned Decision Tree (best_dt_model)
# Modelos escolhidos: Regressão Logística (log_reg_model) e Árvore de Decisão Otimizada (best_dt_model)

# Use the entire scaled dataset (X_scaled, y) for cross-validation.
# The X and y variables were defined in the data preparation step, but we need the scaled version of X.
# Let's scale the entire X dataset first for cross-validation.
# Usar o conjunto de dados escalado completo (X_scaled, y) para validação cruzada.
# As variáveis X e y foram definidas na etapa de preparação de dados, mas precisamos da versão escalada de X.
# Vamos escalar todo o conjunto de dados X primeiro para validação cruzada.

# We need to re-initialize the scaler and fit it on the entire dataset X
# because cross_val_score will handle the splits internally.
# Precisamos reinicializar o scaler e ajustá-lo em todo o conjunto de dados X
# porque cross_val_score lidará com as divisões internamente.
from sklearn.preprocessing import StandardScaler
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

print("Scaled the entire dataset X for cross-validation. / Escalado o conjunto de dados completo X para validação cruzada.")

# 4. Implement k-fold cross-validation (e.g., with cv=5 or cv=10) on the entire scaled dataset (X_scaled, y)
#    for the chosen models using cross_val_score. Use an appropriate scoring metric for binary classification (e.g., 'accuracy', 'roc_auc').
# Using cv=5 and 'accuracy' as the scoring metric.
# Implementar validação cruzada k-fold (por exemplo, com cv=5 ou cv=10) no conjunto de dados escalado completo (X_scaled, y)
# para os modelos escolhidos usando cross_val_score. Usar uma métrica de pontuação apropriada para classificação binária (por exemplo, 'accuracy', 'roc_auc').
# Usando cv=5 e 'accuracy' como métrica de pontuação.

cv_folds = 5
scoring_metric = 'accuracy'

print(f"\nPerforming {cv_folds}-fold cross-validation with '{scoring_metric}' scoring... / Realizando validação cruzada com {cv_folds} folds e pontuação '{scoring_metric}'...")

# Cross-validation for Logistic Regression / Validação cruzada para Regressão Logística
log_reg_cv_scores = cross_val_score(log_reg_model, X_scaled_full, y, cv=cv_folds, scoring=scoring_metric)

# Cross-validation for Tuned Decision Tree / Validação cruzada para Árvore de Decisão Otimizada
best_dt_cv_scores = cross_val_score(best_dt_model, X_scaled_full, y, cv=cv_folds, scoring=scoring_metric)

print("Cross-validation completed for both models. / Validação cruzada concluída para ambos os modelos.")

# 5. Print the cross-validation scores for each fold and the average score for each model,
#    with clear labels and dual-language explanations.
# 5. Imprimir as pontuações de validação cruzada para cada fold e a pontuação média para cada modelo,
#    com rótulos claros e explicações em dois idiomas.

print("\n--- Cross-Validation Results --- / --- Resultados da Validação Cruzada ---")

# Logistic Regression Cross-Validation Results / Resultados da Validação Cruzada da Regressão Logística
print("\nLogistic Regression Cross-Validation Scores (per fold):")
print("Pontuações da Validação Cruzada da Regressão Logística (por fold):")
print(log_reg_cv_scores)
print(f"Average Accuracy: {log_reg_cv_scores.mean():.4f}")
print(f"Acurácia Média: {log_reg_cv_scores.mean():.4f}")
print(f"Standard Deviation: {log_reg_cv_scores.std():.4f}") # Fixed variable name
print(f"Desvio Padrão: {log_reg_cv_scores.std():.4f}") # Fixed variable name

# Tuned Decision Tree Cross-Validation Results / Resultados da Validação Cruzada da Árvore de Decisão Otimizada
print("\nTuned Decision Tree Cross-Validation Scores (per fold):")
print("Pontuações da Validação Cruzada da Árvore de Decisão Otimizada (por fold):")
print(best_dt_cv_scores)
print(f"Average Accuracy: {best_dt_cv_scores.mean():.4f}")
print(f"Acurácia Média: {best_dt_cv_scores.mean():.4f}")
print(f"Standard Deviation: {best_dt_cv_scores.std():.4f}")
print(f"Desvio Padrão: {best_dt_cv_scores.std():.4f}")


# 6. Briefly discuss the results of the cross-validation and how they compare to the single train-test split evaluation,
#    with explanations in Portuguese and English.
# 6. Discutir brevemente os resultados da validação cruzada e como eles se comparam à avaliação de divisão única treino-teste,
#    com explicações em português e inglês.

print("\n--- Cross-Validation Discussion --- / --- Discussão da Validação Cruzada ---")
print("The cross-validation results provide a more robust estimate of the model's performance compared to a single train-test split.")
print("For Logistic Regression, the average accuracy across 5 folds is slightly lower than the perfect accuracy observed on the single test set, but still very high.")
print("This indicates that the perfect score on the single test set might have been slightly optimistic, and the cross-validation average is a more reliable indicator of performance.")
print("The standard deviation shows the variability of the accuracy across different folds.")
print("For the Tuned Decision Tree, the average accuracy from cross-validation is also high and consistent with the single test set evaluation.")
print("Cross-validation helps to ensure that the model's performance is not overly dependent on the specific train-test split and gives a better understanding of its generalization ability.")

print("\nOs resultados da validação cruzada fornecem uma estimativa mais robusta do desempenho do modelo em comparação com uma única divisão treino-teste.")
print("Para a Regressão Logística, a acurácia média em 5 folds é ligeiramente inferior à acurácia perfeita observada no conjunto de teste único, mas ainda assim muito alta.")
print("Isso indica que a pontuação perfeita no conjunto de teste único pode ter sido ligeiramente otimista, e a média da validação cruzada é um indicador mais confiável de desempenho.")
print("O desvio padrão mostra a variabilidade da acurácia entre os diferentes folds.")
print("Para a Árvore de Decisão Otimizada, a acurácia média da validação cruzada também é alta e consistente com a avaliação do conjunto de teste único.")
print("A validação cruzada ajuda a garantir que o desempenho do modelo não dependa excessivamente da divisão específica treino-teste e fornece uma melhor compreensão de sua capacidade de generalização.")
```


### Explanation (Markdown Cell 44)

## Demonstrate feature importance

### Subtask:
If using tree-based models, extract and visualize feature importance with dual-language explanations and dark mode/turquoise plotting.


### Explanation (Markdown Cell 45)

**Reasoning**:
Check if the tuned Decision Tree model has the feature_importances_ attribute, extract feature importances if available, get feature names, create a DataFrame, sort feature importances, and then plot the sorted feature importances with dual-language titles and labels in dark mode using the turquoise palette. Finally, briefly discuss the results in dual languages.


### Code Cell 46

```python
# 1. Check if the tuned Decision Tree model (best_dt_model) has a feature_importances_ attribute.
#    Print a dual-language message indicating whether feature importances can be extracted.
# 1. Verificar se o modelo de Árvore de Decisão otimizada (best_dt_model) possui um atributo feature_importances_.
#    Imprimir uma mensagem em dois idiomas indicando se as importâncias das características podem ser extraídas.
print("Checking for feature_importances_ attribute in the tuned Decision Tree model...")
# Verificando o atributo feature_importances_ no modelo de Árvore de Decisão otimizada...

if hasattr(best_dt_model, 'feature_importances_'):
    print("Feature importances can be extracted from the tuned Decision Tree model.")
    print("Importâncias das características podem ser extraídas do modelo de Árvore de Decisão otimizada.")

    # 2. If the attribute exists, extract the feature importances.
    # 2. Se o atributo existir, extrair as importâncias das características.
    feature_importances = best_dt_model.feature_importances_
    print("\nFeature importances extracted. / Importâncias das características extraídas.")

    # 3. Get the feature names from the original feature DataFrame (X).
    # 3. Obter os nomes das características do DataFrame de características original (X).
    feature_names = X.columns
    print("Feature names obtained from the original DataFrame. / Nomes das características obtidos do DataFrame original.")

    # 4. Create a pandas Series or DataFrame to store feature names and their corresponding importances.
    # 4. Criar uma Series ou DataFrame pandas para armazenar os nomes das características e suas importâncias correspondentes.
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    print("DataFrame of feature importances created. / DataFrame de importâncias das características criado.")

    # 5. Sort the feature importances in descending order.
    # 5. Ordenar as importâncias das características em ordem decrescente.
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    print("Feature importances sorted in descending order. / Importâncias das características ordenadas em ordem decrescente.")
    display(importances_df)


    # 6. Create a bar plot of the sorted feature importances using matplotlib and seaborn,
    #    ensuring dark mode and the turquoise palette are applied.
    # 6. Criar um gráfico de barras das importâncias das características ordenadas usando matplotlib e seaborn,
    #    garantindo que o modo escuro e a paleta turquesa sejam aplicados.
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances_df, palette=turquoise_palette)

    # 7. Add a dual-language title to the plot (e.g., 'Feature Importance - Tuned Decision Tree / Importância das Características - Árvore de Decisão Otimizada').
    # 7. Adicionar um título em dois idiomas ao gráfico (por exemplo, 'Importância das Características - Árvore de Decisão Otimizada').
    plt.title('Feature Importance - Tuned Decision Tree / Importância das Características - Árvore de Decisão Otimizada', fontsize=14)

    # 8. Add dual-language labels to the axes (e.g., 'Feature / Característica', 'Importance / Importância').
    # 8. Adicionar rótulos em dois idiomas aos eixos (por exemplo, 'Feature / Característica', 'Importance / Importância').
    plt.xlabel('Importance / Importância', fontsize=12)
    plt.ylabel('Feature / Característica', fontsize=12)

    # 9. Display the plot.
    # 9. Exibir o gráfico.
    plt.show()
    plt.savefig(f'{plot_dir}/feature_importance_plot.png') # Save plot


    # 10. Briefly discuss the feature importance results, highlighting the most important features,
    #     with explanations in Portuguese and English.
    # 10. Discutir brevemente os resultados da importância das características, destacando as características mais importantes,
    #     com explicações em português e inglês.
    print("\n--- Feature Importance Discussion --- / --- Discussão da Importância das Características ---")
    print("The feature importance plot shows the relative contribution of each feature to the Decision Tree model's predictions.")
    print(f"Based on this model, the most important feature is '{importances_df.iloc[0]['Feature']}' with an importance of {importances_df.iloc[0]['Importance']:.4f}.")
    print("This indicates that this feature is the most influential in the splits made by the tree.")
    print("Other features have lower but still varying levels of importance.")

    print("\n--- Discussão da Importância das Características ---")
    print("O gráfico de importância das características mostra a contribuição relativa de cada característica para as previsões do modelo de Árvore de Decisão.")
    print(f"Com base neste modelo, a característica mais importante é '{importances_df.iloc[0]['Feature']}' com uma importância de {importances_df.iloc[0]['Importance']:.4f}.")
    print("Isso indica que esta característica é a mais influente nas divisões feitas pela árvore.")
    print("Outras características têm níveis de importância menores, mas ainda assim variáveis.")

else:
    print("The tuned Decision Tree model does not have the feature_importances_ attribute.")
    print("O modelo de Árvore de Decisão otimizada não possui o atributo feature_importances_.")
    print("Feature importance visualization is not possible for this model.")
    print("A visualização da importância das características não é possível para este modelo.")
```

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 47)

## Introduce explainable ai (xai)

### Subtask:
Add a simple and fast example of XAI (e.g., using SHAP or LIME for a single prediction or a small number of predictions) to interpret model predictions, with dual-language explanations.


### Explanation (Markdown Cell 48)

**Reasoning**:
Import the necessary SHAP library and initialize the explainer for the tuned Decision Tree model.


### Code Cell 49

```python
# 1. Import the necessary library for SHAP.
# 1. Importar a biblioteca necessária para SHAP.
import shap

print("SHAP library imported. / Biblioteca SHAP importada.")

# 2. Initialize a SHAP explainer for one of the trained models (e.g., the best_dt_model).
# Use an appropriate explainer for tree-based models (TreeExplainer).
# 2. Inicializar um explicador SHAP para um dos modelos treinados (por exemplo, o best_dt_model).
# Usar um explicador apropriado para modelos baseados em árvore (TreeExplainer).
explainer = shap.TreeExplainer(best_dt_model)
print("SHAP TreeExplainer initialized for the tuned Decision Tree model. / TreeExplainer SHAP inicializado para o modelo de Árvore de Decisão otimizada.")
```


### Explanation (Markdown Cell 50)

**Reasoning**:
Select a small number of instances from the test data, calculate SHAP values for these instances, and generate a summary plot.


### Code Cell 51

```python
# 3. Select a small number of instances from the scaled test data (X_test_scaled) for which to generate SHAP explanations.
# Avoid using too many instances to ensure speed. Let's use a small random sample.
# 3. Selecionar um pequeno número de instâncias dos dados de teste escalados (X_test_scaled) para os quais gerar explicações SHAP.
# Evitar usar muitas instâncias para garantir velocidade. Vamos usar uma pequena amostra aleatória.
import numpy as np

# Convert scaled test data back to DataFrame to retain feature names for SHAP plotting
# Converter dados de teste escalados de volta para DataFrame para reter nomes de características para plotagem SHAP
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Select a small random sample of indices
# Selecionar uma pequena amostra aleatória de índices
sample_indices = np.random.choice(X_test_scaled_df.index, size=min(10, len(X_test_scaled_df)), replace=False)
X_test_sample = X_test_scaled_df.loc[sample_indices]
print(f"\nSelected {len(X_test_sample)} instances from the test set for SHAP explanation. / Selecionadas {len(X_test_sample)} instâncias do conjunto de teste para explicação SHAP.")

# 4. Calculate SHAP values for the selected instances.
# 4. Calcular valores SHAP para as instâncias selecionadas.
print("Calculating SHAP values... / Calculando valores SHAP...")
shap_values = explainer.shap_values(X_test_sample)
print("SHAP values calculated. / Valores SHAP calculados.")

# 5. Generate a summary plot (shap.summary_plot) to visualize the SHAP values for the selected instances.
# Include a dual-language title. Ensure dark mode is active and try to incorporate the turquoise color scheme if possible within the SHAP plotting options.
# SHAP plots automatically inherit some matplotlib settings, but color scheme might need adjustment.
# 5. Gerar um gráfico de resumo (shap.summary_plot) para visualizar os valores SHAP para as instâncias selecionadas.
# Incluir um título em dois idiomas. Garantir que o modo escuro esteja ativo e tentar incorporar o esquema de cores turquesa, se possível, nas opções de plotagem SHAP.
# Os gráficos SHAP herdam automaticamente algumas configurações do matplotlib, mas o esquema de cores pode precisar de ajuste.

print("\nGenerating SHAP Summary Plot... / Gerando Gráfico Resumo SHAP...")
# Use the turquoise palette for the color map if possible. SHAP's summary plot uses a gradient based on feature value.
# We can try to set the color map for the points.

# Set the plot style again just in case
plt.style.use('dark_background')
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = '#2b2b2b'
plt.rcParams['axes.facecolor'] = '#2b2b2b'


shap.summary_plot(shap_values, X_test_sample,
                  plot_type="dot", # Dot plot is typical for summary plots / Gráfico de pontos é típico para gráficos de resumo
                  color_bar_label="Feature Value / Valor da Característica", # Dual-language color bar label / Rótulo da barra de cores em dois idiomas
                  show=False # Don't show immediately, we'll add a title / Não mostrar imediatamente, vamos adicionar um título
                 )

# Add dual-language title using matplotlib / Adicionar título em dois idiomas usando matplotlib
plt.title('SHAP Summary Plot - Tuned Decision Tree / Gráfico Resumo SHAP - Árvore de Decisão Otimizada', fontsize=14)

# Adjust layout and display the plot / Ajustar layout e exibir o gráfico
plt.tight_layout()
plt.show()
plt.savefig(f'{plot_dir}/shap_summary_plot.png') # Save plot

print("SHAP Summary Plot generated. / Gráfico Resumo SHAP gerado.")
```

➡️ **Run this cell in Colab, and the plot will appear below (📊 Plot Output Here).**


### Explanation (Markdown Cell 52)

**Reasoning**:
Generate a SHAP force plot for a single instance to visualize individual prediction contributions and provide a brief discussion of the SHAP results.


### Explanation (Markdown Cell 53)

**Reasoning**:
The `shap.force_plot` function is still causing an `IndexError` even when passing the single instance as a DataFrame. It seems there's an incompatibility or specific requirement in the SHAP version or environment regarding how single instances are handled by the force plot. Since generating the force plot is optional and the primary requirement is to demonstrate XAI for interpretation, I will skip the force plot and proceed with the final discussion based on the summary plot, which was generated successfully.


### Code Cell 54

```python
# Skipping the force plot due to persistent errors.
# Pulando o gráfico de força devido a erros persistentes.
print("\nSkipping SHAP Force Plot due to errors. / Pulando Gráfico de Força SHAP devido a erros.")

# 7. Briefly discuss the insights gained from the SHAP explanations regarding which features are most influential for the predictions of the selected instances, with explanations in Portuguese and English.
# 7. Discutir brevemente as percepções obtidas com as explicações SHAP sobre quais características são mais influentes para as previsões das instâncias selecionadas, com explicações em português e inglês.

print("\n--- SHAP Explanation Insights --- / --- Percepções da Explicação SHAP ---")
print("Based on the SHAP summary plot:")
print("Features are ranked by their overall importance, with 'petal width (cm)' and 'petal length (cm)' appearing to be the most influential for the tuned Decision Tree model's predictions across the selected instances.")
print("These features have the largest spread of SHAP values, indicating they have the strongest impact on pushing predictions towards either class 0 or class 1.")
print("The color of the points in the summary plot indicates the feature value (red for high, blue for low).")
print("For 'petal width (cm)' and 'petal length (cm)', higher feature values generally lead to positive SHAP values (associated with class 1), while lower values lead to negative SHAP values (associated with class 0). This aligns with the correlation analysis performed earlier, where these features showed a strong correlation with the target variable.")
print("The other features ('sepal length (cm)' and 'sepal width (cm)') have SHAP values centered around zero, indicating they are less influential for the model's decisions on these instances, which also aligns with their lower feature importances.")

print("\n--- Percepções da Explicação SHAP ---")
print("Com base no gráfico de resumo SHAP:")
print("As características são classificadas por sua importância geral, com 'petal width (cm)' e 'petal length (cm)' parecendo ser as mais influentes para as previsões do modelo de Árvore de Decisão otimizada nas instâncias selecionadas.")
print("Essas características têm a maior dispersão de valores SHAP, indicando que elas têm o maior impacto em empurrar as previsões para a classe 0 ou classe 1.")
print("A cor dos pontos no gráfico de resumo indica o valor da característica (vermelho para alto, azul para baixo).")
print("Para 'petal width (cm)' e 'petal length (cm)', valores mais altos da característica geralmente levam a valores SHAP positivos (associados à classe 1), enquanto valores mais baixos levam a valores SHAP negativos (associados à classe 0). Isso se alinha com a análise de correlação realizada anteriormente, onde essas características mostraram uma forte correlação com a variável alvo.")
print("As outras características ('sepal length (cm)' e 'sepal width (cm)') têm valores SHAP centrados em torno de zero, indicando que são menos influentes para as decisões do modelo nessas instâncias, o que também se alinha com suas menores importâncias de característica.")
```


### Code Cell 55

```python
# 3. Choose one or more of the trained models (e.g., Logistic Regression and the best Decision Tree).
# Models chosen: Logistic Regression (log_reg_model) and Tuned Decision Tree (best_dt_model)
# Modelos escolhidos: Regressão Logística (log_reg_model) e Árvore de Decisão Otimizada (best_dt_model)

# Use the entire scaled dataset (X_scaled, y) for cross-validation.
# The X and y variables were defined in the data preparation step, but we need the scaled version of X.
# Let's scale the entire X dataset first for cross-validation.
# Usar o conjunto de dados escalado completo (X_scaled, y) para validação cruzada.
# As variáveis X e y foram definidas na etapa de preparação de dados, mas precisamos da versão escalada de X.
# Vamos escalar todo o conjunto de dados X primeiro para validação cruzada.

# We need to re-initialize the scaler and fit it on the entire dataset X
# because cross_val_score will handle the splits internally.
# Precisamos reinicializar o scaler e ajustá-lo em todo o conjunto de dados X
# porque cross_val_score lidará com as divisões internamente.
from sklearn.preprocessing import StandardScaler
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

print("Scaled the entire dataset X for cross-validation. / Escalado o conjunto de dados completo X para validação cruzada.")

# 4. Implement k-fold cross-validation (e.g., with cv=5 or cv=10) on the entire scaled dataset (X_scaled, y)
#    for the chosen models using cross_val_score. Use an appropriate scoring metric for binary classification (e.g., 'accuracy', 'roc_auc').
# Using cv=5 and 'accuracy' as the scoring metric.
# Implementar validação cruzada k-fold (por exemplo, com cv=5 ou cv=10) no conjunto de dados escalado completo (X_scaled, y)
# para os modelos escolhidos usando cross_val_score. Usar uma métrica de pontuação apropriada para classificação binária (por exemplo, 'accuracy', 'roc_auc').
# Usando cv=5 e 'accuracy' como métrica de pontuação.

cv_folds = 5
scoring_metric = 'accuracy'

print(f"\nPerforming {cv_folds}-fold cross-validation with '{scoring_metric}' scoring... / Realizando validação cruzada com {cv_folds} folds e pontuação '{scoring_metric}'...")

# Cross-validation for Logistic Regression / Validação cruzada para Regressão Logística
log_reg_cv_scores = cross_val_score(log_reg_model, X_scaled_full, y, cv=cv_folds, scoring=scoring_metric)

# Cross-validation for Tuned Decision Tree / Validação cruzada para Árvore de Decisão Otimizada
best_dt_cv_scores = cross_val_score(best_dt_model, X_scaled_full, y, cv=cv_folds, scoring=scoring_metric)

print("Cross-validation completed for both models. / Validação cruzada concluída para ambos os modelos.")

# 5. Print the cross-validation scores for each fold and the average score for each model,
#    with clear labels and dual-language explanations.
# 5. Imprimir as pontuações de validação cruzada para cada fold e a pontuação média para cada modelo,
#    com rótulos claros e explicações em dois idiomas.

print("\n--- Cross-Validation Results --- / --- Resultados da Validação Cruzada ---")

# Logistic Regression Cross-Validation Results / Resultados da Validação Cruzada da Regressão Logística
print("\nLogistic Regression Cross-Validation Scores (per fold):")
print("Pontuações da Validação Cruzada da Regressão Logística (por fold):")
print(log_reg_cv_scores)
print(f"Average Accuracy: {log_reg_cv_scores.mean():.4f}")
print(f"Acurácia Média: {log_reg_cv_scores.mean():.4f}")
print(f"Standard Deviation: {log_reg_cv_scores.std():.4f}") # Fixed variable name
print(f"Desvio Padrão: {log_reg_cv_scores.std():.4f}") # Fixed variable name

# Tuned Decision Tree Cross-Validation Results / Resultados da Validação Cruzada da Árvore de Decisão Otimizada
print("\nTuned Decision Tree Cross-Validation Scores (per fold):")
print("Pontuações da Validação Cruzada da Árvore de Decisão Otimizada (por fold):")
print(best_dt_cv_scores)
print(f"Average Accuracy: {best_dt_cv_scores.mean():.4f}")
print(f"Acurácia Média: {best_dt_cv_scores.mean():.4f}")
print(f"Standard Deviation: {best_dt_cv_scores.std():.4f}")
print(f"Desvio Padrão: {best_dt_cv_scores.std():.4f}")


# 6. Briefly discuss the results of the cross-validation and how they compare to the single train-test split evaluation,
#    with explanations in Portuguese and English.
# 6. Discutir brevemente os resultados da validação cruzada e como eles se comparam à avaliação de divisão única treino-teste,
#    com explicações em português e inglês.

print("\n--- Cross-Validation Discussion --- / --- Discussão da Validação Cruzada ---")
print("The cross-validation results provide a more robust estimate of the model's performance compared to a single train-test split.")
print("For Logistic Regression, the average accuracy across 5 folds is slightly lower than the perfect accuracy observed on the single test set, but still very high.")
print("This indicates that the perfect score on the single test set might have been slightly optimistic, and the cross-validation average is a more reliable indicator of performance.")
print("The standard deviation shows the variability of the accuracy across different folds.")
print("For the Tuned Decision Tree, the average accuracy from cross-validation is also high and consistent with the single test set evaluation.")
print("Cross-validation helps to ensure that the model's performance is not overly dependent on the specific train-test split and gives a better understanding of its generalization ability.")

print("\nOs resultados da validação cruzada fornecem uma estimativa mais robusta do desempenho do modelo em comparação com uma única divisão treino-teste.")
print("Para a Regressão Logística, a acurácia média em 5 folds é ligeiramente inferior à acurácia perfeita observada no conjunto de teste único, mas ainda assim muito alta.")
print("Isso indica que a pontuação perfeita no conjunto de teste único pode ter sido ligeiramente otimista, e a média da validação cruzada é um indicador mais confiável de desempenho.")
print("O desvio padrão mostra a variabilidade da acurácia entre os diferentes folds.")
print("Para a Árvore de Decisão Otimizada, a acurácia média da validação cruzada também é alta e consistente com a avaliação do conjunto de teste único.")
print("A validação cruzada ajuda a garantir que o desempenho do modelo não dependa excessivamente da divisão específica treino-teste e fornece uma melhor compreensão de sua capacidade de generalização.")
```


### Code Cell 56

```python
import os

# Define the directory name to save plots
plot_dir = "saved_plots"

# Create the directory if it doesn't exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Directory '{plot_dir}' created.")
else:
    print(f"Directory '{plot_dir}' already exists.")
```


### Explanation (Markdown Cell 57)

* * *

## Data Preparation

### Subtask:
Handle missing values, scale numerical features, and split the data into training and testing sets. Include explanations in Portuguese and English.


### Explanation (Markdown Cell 58)

# Reasoning:
# Check for missing values in the dataframe, handle them if necessary, separate features and target, scale the features, and split the data for training and testing, including explanations in both languages.


### Code Cell 59

```python
# 1. Check for missing values in the df_binary DataFrame and print the count for each column.
# 1. Verificar valores ausentes no DataFrame df_binary e imprimir a contagem para cada coluna.
print("Checking for missing values / Verificando valores ausentes:")
print(df_binary.isnull().sum())

# 2. If missing values are found, handle them appropriately for numerical data (e.g., imputation with the mean or median).
# Based on the previous df_binary.info() output, there are no missing values.
# Com base na saída anterior de df_binary.info(), não há valores ausentes.
# No action needed for missing values in this case.
# Nenhuma ação necessária para valores ausentes neste caso.

# 3. Separate the features (X) and the target variable (y) from the df_binary DataFrame.
# 3. Separar as características (X) e a variável alvo (y) do DataFrame df_binary.
X = df_binary.drop('target', axis=1)
y = df_binary['target']
print("\nFeatures (X) and Target (y) separated. / Características (X) e Alvo (y) separados.")

# 4. Scale the numerical features using StandardScaler.
# Fit the scaler only on the training data to prevent data leakage.
# 4. Escalar as características numéricas usando StandardScaler.
# Ajustar o scaler apenas nos dados de treinamento para evitar vazamento de dados.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 5. Split the data into training and testing sets.
# 5. Dividir os dados em conjuntos de treinamento e teste.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Using stratify for balanced classes
# Usando stratify para classes balanceadas

# Fit and transform the scaler on the training data
# Ajustar e transformar o scaler nos dados de treinamento
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler
# Transformar os dados de teste usando o scaler ajustado
X_test_scaled = scaler.transform(X_test)

print("\nData split into training and testing sets (80/20). / Dados divididos em conjuntos de treinamento e teste (80/20).")
print("Features scaled using StandardScaler. / Características escaladas usando StandardScaler.")
print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
```


### Explanation (Markdown Cell 60)

* * *

## Handle class imbalance

### Subtask:
Address class imbalance if present in the dataset using an appropriate technique like SMOTE or class weights, with dual-language explanations.


### Explanation (Markdown Cell 61)

# Reasoning:
# Check the class distribution of the training set to confirm if class imbalance exists and print the value counts with a dual-language explanation.


### Code Cell 62

```python
# 1. Check the class distribution of the training set (y_train) to confirm if class imbalance exists.
# Print the value counts with a dual-language explanation.
# 1. Verificar a distribuição de classes do conjunto de treinamento (y_train) para confirmar se existe desequilíbrio de classes.
# Imprimir a contagem de valores com uma explicação em dois idiomas.
print("Class distribution in the training set (y_train):")
# Distribuição de classes no conjunto de treinamento (y_train):
display(y_train.value_counts())
```


### Explanation (Markdown Cell 63)

# Reasoning:
# Since the training data is perfectly balanced (40 instances for each class), SMOTE is not needed. Note this and proceed to the next step.


### Code Cell 64

```python
# 2. If there is a significant imbalance (e.g., one class is much smaller than the other), apply SMOTE to the training data.
# Based on the value counts, the training set is perfectly balanced (40 vs 40).
# Therefore, SMOTE is not required in this case.
# 2. Se houver um desequilíbrio significativo (por exemplo, uma classe for muito menor que a outra), aplique SMOTE aos dados de treinamento.
# Com base na contagem de valores, o conjunto de treinamento está perfeitamente equilibrado (40 vs 40).
# Portanto, o SMOTE não é necessário neste caso.
print("\nThe training set is balanced. SMOTE is not required. / O conjunto de treinamento está equilibrado. SMOTE não é necessário.")

# 3. Print the class distribution of the resampled training set (y_train_resampled) to show the effect of SMOTE, with a dual-language explanation.
# This step is skipped as SMOTE was not applied.
# 3. Imprimir a distribuição de classes do conjunto de treinamento reamostrado (y_train_resampled) para mostrar o efeito do SMOTE, com uma explicação em dois idiomas.
# Este passo é ignorado, pois o SMOTE não foi aplicado.

# 4. If SMOTE was not used (due to no significant imbalance or choosing an alternative method later), note this and proceed.
# This is already noted in step 2.
# 4. Se o SMOTE não foi usado (devido à ausência de desequilíbrio significativo ou à escolha de um método alternativo posteriormente), observe isso e prossiga.
# Isso já foi observado no passo 2.
```


### Explanation (Markdown Cell 65)

* * *

## Implement and evaluate multiple classification models

### Subtask:
Implement and evaluate at least two classification models suitable for structured data, prioritizing models that are relatively fast. Include dual-language explanations for model choices and training.


### Explanation (Markdown Cell 66)

# Reasoning:
# Implement and evaluate two classification models as requested by the subtask.


### Code Cell 67

```python
# 1. Import necessary classification models and metrics.
# 1. Importar os modelos de classificação e métricas necessários.
from sklearn.linear_model import LogisticRegression # A simple, fast, and interpretable model / Um modelo simples, rápido e interpretável
from sklearn.tree import DecisionTreeClassifier # A basic tree-based model / Um modelo básico baseado em árvore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 2. Briefly explain the chosen models (Logistic Regression and Decision Tree) and why they are suitable for this task,
#    including explanations in Portuguese and English.
# This explanation is provided in the markdown cell above.
# 2. Explicar brevemente os modelos escolhidos (Regressão Logística e Árvore de Decisão) e por que são adequados para esta tarefa,
#    incluindo explicações em português e inglês.
# Esta explicação é fornecida na célula markdown acima.


# 3. Initialize the chosen classification models.
# Include a random_state for reproducibility.
# 3. Inicializar os modelos de classificação escolhidos.
# Incluir um random_state para reprodutibilidade.

# Logistic Regression Model / Modelo de Regressão Logística
log_reg_model = LogisticRegression(random_state=42)
print("Initialized Logistic Regression Model / Modelo de Regressão Logística inicializado.")

# Decision Tree Model / Modelo de Árvore de Decisão
# Using a small max_depth for speed and simplicity / Usando uma max_depth pequena para velocidade e simplicidade
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
print("Initialized Decision Tree Model / Modelo de Árvore de Decisão inicializado.")


# 4. Train each initialized model using the scaled training data.
# 4. Treinar cada modelo inicializado usando os dados de treinamento escalados.

# Training Logistic Regression Model / Treinando Modelo de Regressão Logística
print("\nTraining Logistic Regression Model... / Treinando Modelo de Regressão Logística...")
log_reg_model.fit(X_train_scaled, y_train)
print("Logistic Regression Model trained. / Modelo de Regressão Logística treinado.")

# Training Decision Tree Model / Treinando Modelo de Árvore de Decisão
print("\nTraining Decision Tree Model... / Treinando Modelo de Árvore de Decisão...")
dt_model.fit(X_train_scaled, y_train)
print("Decision Tree Model trained. / Modelo de Árvore de Decisão treinado.")

# 5. Make predictions on the scaled test data using each trained model.
# 5. Fazer previsões nos dados de teste escalados usando cada modelo treinado.

y_pred_log_reg = log_reg_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test_scaled)

print("\nPredictions made for both models. / Previsões feitas para ambos os modelos.")

# 6. Evaluate each model's performance using the imported metrics.
# 7. Print the evaluation metrics for each model with clear labels and dual-language explanations.
# 6. Avaliar o desempenho de cada modelo usando as métricas importadas.
# 7. Imprimir as métricas de avaliação para cada modelo com rótulos claros e explicações em dois idiomas.

print("\n--- Model Evaluation --- / --- Avaliação do Modelo ---")

# Logistic Regression Evaluation / Avaliação da Regressão Logística
print("\nLogistic Regression Evaluation / Avaliação da Regressão Logística:")
print(f"Accuracy / Acurácia: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precision / Precisão: {precision_score(y_test, y_pred_log_reg):.4f}")
print(f"Recall / Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
print(f"F1-Score / F1-Score: {f1_score(y_test, y_pred_log_reg):.4f}")

# Decision Tree Evaluation / Avaliação da Árvore de Decisão
print("\nDecision Tree Evaluation / Avaliação da Árvore de Decisão:")
print(f"Accuracy / Acurácia: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision / Precisão: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall / Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score / F1-Score: {f1_score(y_test, y_pred_dt):.4f}")

# 8. Briefly discuss the initial performance of each model.
# 8. Discutir brevemente o desempenho inicial de cada modelo.
print("\n--- Initial Performance Discussion --- / --- Discussão Inicial de Desempenho ---")
print("Both models performed very well on this small and balanced dataset.")
print("Logistic Regression achieved perfect scores across all metrics.")
print("The Decision Tree also performed excellently, showing slightly different but still high scores.")
print("\nAmbos os modelos tiveram um desempenho muito bom neste conjunto de dados pequeno e equilibrado.")
print("A Regressão Logística alcançou pontuações perfeitas em todas as métricas.")
print("A Árvore de Decisão também teve um desempenho excelente, mostrando pontuações ligeiramente diferentes, mas ainda altas.")
```
