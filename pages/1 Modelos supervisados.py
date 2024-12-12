# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:23:49 2024

@author: jahop
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


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
                    report = classification_report(y_test, predictions, output_dict=True)

                    results.append({
                        'Model': model_name,
                        'Combination': comb,
                        'Accuracy': accuracy,
                        'Time': elapsed_time,
                        'Precision': report['weighted avg']['precision'],
                        'Recall': report['weighted avg']['recall'],
                        'F1-Score': report['weighted avg']['f1-score'],
                    })

        results_df = pd.DataFrame(results)

        # Save results
        self.save_results_to_csv(results_df)

        return results_df

    def save_results_to_csv(self, results_df):
        results_df.to_csv("all_combinations_results.csv", index=False)

    def best_models_by_num_attributes(self, results_df):
        best_models = {}

        # Filtrar por número de atributos (1, 2, 3, 4)
        for num_attributes in [1, 2, 3, 4]:
            subset = results_df[results_df['Combination'].apply(lambda x: len(x) == num_attributes)]
            best_model_row = subset.loc[subset['Accuracy'].idxmax()]  # Obtener el mejor modelo basado en la precisión
            best_models[num_attributes] = best_model_row

        best_models_df = pd.DataFrame.from_dict(best_models, orient='index')
        return best_models_df


def highlight_best_models(df):
    return df.sort_values(by='Accuracy', ascending=False)


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

# Highlight the best models and apply colors (now without styling)
results_df = highlight_best_models(results_df)

# Display the results without colors
st.write("### Resultados de rendimiento del modelo", results_df)

# Get the best models for 1, 2, 3, and 4 attributes
best_models_df = analysis.best_models_by_num_attributes(results_df)

# Display the best models by attributes
st.write("### Mejores modelos por número de atributos", best_models_df)

# Obtener los 4 mejores modelos por precisión
top_4_models = best_models_df[['Model', 'Combination', 'Accuracy', 'Time']].head(4)

# Crear gráfico interactivo con Plotly
fig = go.Figure()

# Añadir barras para Accuracy
fig.add_trace(go.Bar(
    x=top_4_models['Model'],
    y=top_4_models['Accuracy'],
    name='Accuracy',
    marker_color='skyblue',
    text=top_4_models['Combination'].apply(lambda x: f'Combinación: {x}'),
    hovertemplate='<b>Modelo:</b> %{x}<br><b>Accuracy:</b> %{y:.2f}<br><b>Combinación:</b> %{text}<extra></extra>',
    width=0.3  # Establecer el ancho de las barras
))

# Añadir barras para Time
fig.add_trace(go.Bar(
    x=top_4_models['Model'],
    y=top_4_models['Time'],
    name='Time',
    marker_color='red',
    text=top_4_models['Combination'].apply(lambda x: f'Combinación: {x}'),
    hovertemplate='<b>Modelo:</b> %{x}<br><b>Tiempo:</b> %{y:.2f} segundos<br><b>Combinación:</b> %{text}<extra></extra>',
    yaxis="y2",
    width=0.3  # Establecer el ancho de las barras
))

# Configuración de ejes y diseño
fig.update_layout(
    title='Mejores Modelos: Accuracy y Tiempo',
    barmode='group',
    xaxis=dict(title='Modelo'),
    yaxis=dict(title='Accuracy', side='left'),
    yaxis2=dict(title='Tiempo (segundos)', side='right', overlaying='y'),
    hovermode='closest',
    bargap=0.1  # Reducir el espacio entre las barras para hacer las barras más estrechas
)

# Mostrar gráfico interactivo
st.plotly_chart(fig)