# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:28:06 2024

@author: jperezr
"""


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from itertools import combinations
import time

class ModelAnalysis:
    def __init__(self, input_df, target_column=None, seed=None):
        self.input_df = input_df
        self.target_column = target_column
        # Asegúrate de eliminar la columna de etiquetas ('Class') si está presente
        if target_column:
            self.X = self.input_df.drop(columns=[self.target_column])  # Solo características
        else:
            self.X = self.input_df  # Sin cambiar si no hay target_column
        self.feature_names = self.X.columns  # Nombres de las características
        self.seed = seed

        self.models = {
            'KMeans': KMeans(),
            'MeanShift': MeanShift(),
            'HierarchicalClustering': AgglomerativeClustering()
        }

    def run_all_combinations(self, train_test_split_ratio=0.7):
        results = []

        # Evaluar todas las combinaciones posibles de características
        for n in range(1, len(self.feature_names) + 1):
            for comb in combinations(self.feature_names, n):
                for model_name, model in self.models.items():
                    start_time = time.time()
                    X_comb = self.X[list(comb)]

                    # Ajustar y predecir el modelo
                    model.fit(X_comb)
                    if hasattr(model, 'labels_'):  # Solo para modelos que generan etiquetas
                        labels = model.labels_
                    else:
                        labels = np.nan  # En caso de que no genere etiquetas

                    # Métricas de evaluación
                    if len(set(labels)) > 1:  # Si hay más de un cluster
                        silhouette_avg = silhouette_score(X_comb, labels)
                    else:
                        silhouette_avg = np.nan

                    elapsed_time = time.time() - start_time

                    results.append({
                        'Model': model_name,
                        'Combination': comb,
                        'Silhouette Score': silhouette_avg,
                        'Time': elapsed_time,
                        'Labels': labels.tolist()  # Guardamos las etiquetas generadas
                    })

        results_df = pd.DataFrame(results)

        # Guardar resultados en un archivo CSV
        self.save_results_to_csv(results_df)

        return results_df

    def save_results_to_csv(self, results_df):
        results_df.to_csv("unsupervised_combinations_results.csv", index=False)

def highlight_best_models(df):
    def calculate_score(row):
        return row['Silhouette Score'] if not np.isnan(row['Silhouette Score']) else -1

    df['Score'] = df.apply(calculate_score, axis=1)

    def highlight(row):
        score = row['Score']
        best_score = df['Score'].max()

        if score == best_score:
            return ['background-color: yellow'] * len(row)  # Aplicamos color amarillo a las mejores filas
        return [''] * len(row)

    styled_df = df.style.apply(highlight, axis=1)

    return styled_df

# Streamlit code for visualization
st.sidebar.title("Ayuda")
st.sidebar.write("""
Este código realiza un análisis de modelos de **clustering** utilizando diferentes combinaciones de atributos. A continuación, se describen los pasos y funcionalidades principales:

1. **Base de Datos:** Utiliza el conjunto de datos Iris.
2. **Modelos Utilizados:** KMeans, MeanShift y Hierarchical Clustering.
3. **Combinaciones por Atributos:** Prueba todas las combinaciones posibles de atributos del conjunto de datos.
4. **Evaluación:** Utiliza la **Silhouette Score** para medir la calidad de los clusters generados.

Autor: Javier Horacio Pérez Ricárdez
""")

st.title('Análisis de Modelos de Clustering para Todas las Combinaciones')

# Load data (modificar según sea necesario)
input_df = pd.read_csv("iris.csv")
st.write("### Iris Dataset", input_df)

# Inicializar el análisis de modelos
analysis = ModelAnalysis(input_df=input_df, target_column='Class', seed=1271673)
results_df = analysis.run_all_combinations()

# Resaltar los mejores modelos y aplicar colores
styled_df = highlight_best_models(results_df)

# Mostrar los resultados con Streamlit
st.write("### Resultados del Rendimiento del Modelo", styled_df)