# Previsão de Diagnóstico de Câncer de Mama com Redes Neurais

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)

## 📖 Sobre o Projeto

Este projeto consiste na construção e otimização de um modelo de Machine Learning para prever o diagnóstico de câncer de mama, classificando tumores como **Maligno (M)** ou **Benigno (B)**. O desenvolvimento foi focado no uso de uma **Rede Neural Artificial (RNA)**, com uma exploração detalhada do processo de otimização de hiperparâmetros para maximizar a precisão do modelo.

O objetivo principal não era apenas criar um classificador funcional, mas também documentar a importância da **Análise Exploratória de Dados (EDA)** e da busca sistemática por hiperparâmetros para construir um modelo robusto e confiável.

## 📊 Dataset

O conjunto de dados utilizado é o **Breast Cancer Wisconsin (Diagnostic) Data Set**, um dos datasets mais conhecidos para problemas de classificação binária. Ele contém 569 amostras com 30 características numéricas que foram computadas a partir de imagens digitalizadas de aspirados por agulha fina (FNA) de massas mamárias.

As features incluem raio, textura, perímetro, área, suavidade, entre outras, calculadas para cada núcleo celular presente na imagem.

## 🛠️ Metodologia

O fluxo de trabalho foi estruturado nas seguintes etapas:

### 1. Análise Exploratória de Dados (EDA)
A primeira fase consistiu em uma análise detalhada para entender a distribuição dos dados e a relação entre as características e o diagnóstico.
- **Visualização da Distribuição:** Foram gerados **histogramas** e **boxplots** para cada uma das 30 features, comparando a distribuição entre os diagnósticos Maligno e Benigno.
- **Matriz de Correlação:** Uma matriz de correlação foi visualizada como um heatmap para identificar características altamente correlacionadas.

### 2. Pré-processamento dos Dados
- **Label Encoding:** A variável alvo (diagnóstico) foi convertida de categórica ('M'/'B') para numérica (0/1).
- **Escalonamento (Scaling):** Todas as features numéricas foram escalonadas usando o `StandardScaler` do Scikit-learn para normalizar a escala dos dados e garantir que todas as características contribuíssem de forma equilibrada para o treinamento do modelo.

### 3. Modelagem e Otimização da Rede Neural

O núcleo do projeto foi a construção e otimização de uma Rede Neural com TensorFlow/Keras.

#### a) Busca Manual de Hiperparâmetros
Foi realizada uma busca manual para entender o impacto de diferentes configurações na performance do modelo. Os seguintes hiperparâmetros foram testados:
- **Funções de Ativação:** `relu`, `tanh`, `sigmoid`, `softplus`, etc.
- **Arquitetura da Rede (neurônios nas camadas ocultas):** `[32, 16]`, `[16, 8]`, `[8, 4]`
- **Taxa de Aprendizagem (Learning Rate):** `0.01`, `0.003`, `0.001`, `0.0001`

#### b) Otimização Automatizada com GridSearchCV
Para encontrar a melhor combinação de hiperparâmetros de forma sistemática, foi utilizado o `GridSearchCV` do Scikit-learn em conjunto com o `KerasClassifier`.
- **Validação Cruzada (K-Fold):** A estabilidade do modelo foi avaliada testando diferentes números de `folds` (k de 2 a 10).
- **Modelo Final:** O `GridSearchCV` identificou a melhor combinação de `batch_size`, `epochs` e `optimizer` para o modelo final.

### 4. Modelo de Baseline
Para fins de comparação, um modelo de **Regressão Logística** foi treinado e avaliado, servindo como um baseline para medir o ganho de performance obtido com a Rede Neural.

## 📈 Resultados

O modelo final de Rede Neural, após a otimização, demonstrou uma performance excelente, superando o modelo de baseline. A avaliação foi feita com base no **Relatório de Classificação** (`classification_report`) e na **Matriz de Confusão**.

O modelo final alcançou alta precisão, recall e F1-score para ambas as classes, indicando um classificador bem balanceado e com poucos erros.

## 🚀 Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)
    cd nome-do-repositorio
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```

4.  **Execute o Jupyter Notebook:**
    ```bash
    jupyter notebook PrevDeCancer.ipynb
    ```
