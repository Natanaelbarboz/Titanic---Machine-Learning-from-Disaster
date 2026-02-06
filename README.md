# 🚢 Projeto Titanic – Machine Learning

## 📌 Visão Geral

Este projeto tem como objetivo prever a **sobrevivência de passageiros do Titanic** utilizando técnicas de **Machine Learning**, com foco em **classificação supervisionada**. O dataset utilizado é o clássico **Titanic Dataset** do Kaggle https://kaggle.com/competitions/titanic.

O projeto abrange desde a **análise exploratória dos dados**, **engenharia de atributos**, **tratamento de valores ausentes**, até a **modelagem, avaliação e salvamento do modelo** para reutilização futura.

---

## 🎯 Objetivo

Construir um modelo preditivo capaz de estimar se um passageiro sobreviveu ou não ao naufrágio, com base em características como:

* Classe social
* Sexo
* Idade
* Possui familia no navia
* Porto de embarque

---

## 🧠 Tecnologias e Bibliotecas Utilizadas

* **Python 3.10+**
* **Pandas** – Manipulação de dados
* **NumPy** – Operações numéricas
* **Matplotlib / Seaborn** – Visualização de dados
* **Scikit-learn** – Modelagem e avaliação
* **Joblib** – Salvamento e reutilização de modelos

---

## 🔧 Pré-processamento de Dados

As principais etapas de pré-processamento incluem:

* Tratamento de valores ausentes:

  * **Idade**: imputação pela mediana
  * **Embarked**: imputação pelo valor mais frequente

---

## 🧩 Engenharia de Atributos

Foram criadas novas features para melhorar o desempenho do modelo:

* **Family**: Indica se o passageiro viajava sozinho ou com família
* **Cabin_filled**: Indica se possuia cabine definida.
* **Title**: Título extraído do nome (Mr, Mrs, Rare, etc.)

---

## 🤖 Modelagem

O modelo principal utilizado foi o **RandomForestClassifier**, com ajustes de hiperparâmetros como:

* `n_estimators`
* `max_features`
* `min_samples_leaf`
* `random_state`

A avaliação foi feita utilizando:

* Acurácia
* ROC AUC
* Matriz de confusão

---

## 📈 Resultados

O modelo alcançou aproximadamente:

* **Acurácia**: ~77%
* **ROC AUC**: ~0.77

Esses resultados mostram um bom equilíbrio entre viés e variância para o problema proposto.
Valor obtido na submissão no Kaggle: 0.77751

---

## 💾 Salvamento e Reutilização do Modelo

O modelo final foi salvo utilizando `joblib`, permitindo reutilização futura sem necessidade de novo treinamento.

```python
import joblib
joblib.dump(model, 'Scripts/random_forest_model.pkl')
```

---

## 🚀 Como Executar o Projeto

1. Clone o repositório:

```bash
git clone https://github.com/Natanaelbarboz/Titanic---Machine-Learning-from-Disaster.git
```

2. Instale as dependências:

```bash 
pip install -r requirements.txt
```

3. Execute o script principal ou notebook:

```bash
python Scripts/RandomForestClassifierModel_v2.py
```

---

## 📌 Próximos Passos

* Ajuste fino de hiperparâmetros com GridSearchCV
* Testar outros algoritmos (XGBoost, Gradient Boosting)

---

## 👤 Autor

**Natanael Barboza**
Projeto desenvolvido para estudos em **Ciência de Dados e Machine Learning**.

---

## 📄 Fonte

Base de dados do site Kaggle: https://kaggle.com/competitions/titanic
