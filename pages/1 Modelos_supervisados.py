# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:49:06 2024

@author: jperezr
"""


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from itertools import combinations
import time

class ModelAnalysis:
    def __init__(self, input_df, model_type=None, target_column='Class', seed=None):
        self.input_df = input_df
        self.target_column = target_column
        self.X = self.input_df.drop(columns=[self.target_column])  # Características
        self.y = self.input_df[self.target_column]  # Etiquetas
        self.feature_names = self.X.columns  # Nombres de las características
        self.seed = seed

        self.models = {
            'SVM': SVC(),
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(max_iter=1000)
        }

        self.model_type = model_type if model_type else 'SVM'

    def run_all_combinations(self, train_test_split_ratio=0.7):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=1-train_test_split_ratio, random_state=self.seed)

        results = []

        for n in range(1, len(self.feature_names) + 1):
            for comb in combinations(self.feature_names, n):
                for model_name, model in self.models.items():
                    start_time = time.time()
                    X_train_comb = X_train[list(comb)]
                    X_test_comb = X_test[list(comb)]

                    model.fit(X_train_comb, y_train)
                    predictions = model.predict(X_test_comb)

                    elapsed_time = time.time() - start_time

                    accuracy = accuracy_score(y_test, predictions)
                    conf_matrix = confusion_matrix(y_test, predictions)

                    if conf_matrix.shape == (2, 2):
                        tn, fp, fn, tp = conf_matrix.ravel()
                    else:
                        tn, fp, fn, tp = [np.nan]*4

                    report = classification_report(y_test, predictions, output_dict=True)

                    results.append({
                        'Model': model_name,
                        'Combination': comb,
                        'Accuracy': accuracy,
                        'True Positive (TP)': tp,
                        'False Positive (FP)': fp,
                        'True Negative (TN)': tn,
                        'False Negative (FN)': fn,
                        'Precision': report['weighted avg']['precision'],
                        'Recall': report['weighted avg']['recall'],
                        'F1-Score': report['weighted avg']['f1-score'],
                        'Time': elapsed_time,
                        'Confusion Matrix': f"[{conf_matrix.tolist()}]"  # Encerrar la matriz de confusión entre corchetes
                   })

        results_df = pd.DataFrame(results)

        # Save results
        self.save_results_to_csv(results_df)

        return results_df

    def save_results_to_csv(self, results_df):
        results_df.to_csv("all_combinations_results.csv", index=False)

def highlight_best_models(df):
    def calculate_score(row):
        metrics = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
        return np.mean(metrics)

    df['Score'] = df.apply(calculate_score, axis=1)

    def highlight(row):
        num_classes = len(row['Combination'])
        color_map = {
            1: 'background-color: yellow',
            2: 'background-color: red',
            3: 'background-color: green',
            4: 'background-color: blue'
        }

        score = row['Score']
        best_score = df['Score'].max()

        if score == best_score:
            return [color_map.get(num_classes, '')] * len(row)  # Apply color to all columns of the best row
        return [''] * len(row)

    styled_df = df.style.apply(highlight, axis=1)

    return styled_df

# Streamlit code for visualization
st.sidebar.title("Ayuda")
st.sidebar.write("""
Este código realiza un análisis de modelos de clasificación utilizando diferentes combinaciones de atributos. A continuación, se describen los pasos y funcionalidades principales:

1. **Base de Datos:** Utiliza el conjunto de datos Iris.
2. **Modelos Utilizados:** Soporte Vectorial (SVM), Árbol de Decisión, Bosques Aleatorios y Regresión Logística.
3. **Combinaciones por Atributos:** Prueba todas las combinaciones posibles de atributos del conjunto de datos.
4. **Tiempo de Ejecución:** Calcula el tiempo que cada modelo tarda en resolver cada combinación de atributos.

Autor: Javier Horacio Pérez Ricárdez
""")

st.title('Análisis de modelos para todas las combinaciones')

# Load data (modify as needed)
input_df = pd.read_csv("iris.csv")
st.write("### Iris Dataset", input_df)
target_column = 'Class'

# Initialize Model Analysis
analysis = ModelAnalysis(input_df=input_df, target_column=target_column, seed=1271673)
results_df = analysis.run_all_combinations()

# Highlight the best models and apply colors
styled_df = highlight_best_models(results_df)

# Display the results with Streamlit
st.write("### Model Performance Results", styled_df)
