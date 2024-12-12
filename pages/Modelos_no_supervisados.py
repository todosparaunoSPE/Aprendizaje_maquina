# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:09:03 2024

@author: jahop
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score, adjusted_rand_score
from itertools import combinations
import time
import plotly.express as px


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

class UnsupervisedModelAnalysis:
    def __init__(self, input_df, model_type=None, seed=None):
        # Limpia los nombres de las columnas
        input_df.columns = input_df.columns.str.strip()
        
        self.input_df = input_df
        self.X = self.input_df.drop(columns=['Class'])  # Asegúrate de que la columna 'Class' esté presente
        self.feature_names = self.X.columns  # Nombres de las características
        self.seed = seed

        # Modelos de clustering
        self.models = {
            'KMeans': KMeans(),
            'MeanShift': MeanShift(),
            'DBSCAN': DBSCAN(),
            'Hierarchical': AgglomerativeClustering(),
            'BIRCH': Birch(threshold=0.1)  # Ajustamos el threshold para BIRCH
        }

        self.model_type = model_type if model_type else 'KMeans'

    def run_all_combinations(self, train_test_split_ratio=0.7):
        results = []

        for n in range(1, len(self.feature_names) + 1):
            for comb in combinations(self.feature_names, n):
                for model_name, model in self.models.items():
                    start_time = time.time()
                    X_comb = self.X[list(comb)]  # Selecciona las columnas de la combinación

                    # Ajuste y predicción
                    model.fit(X_comb)
                    labels = model.labels_ if hasattr(model, 'labels_') else None  # Etiquetas generadas por el modelo

                    # Evaluación de la calidad del clustering, evitando métricas vacías
                    if labels is not None:
                        # Si el modelo genera un solo clúster o solo ruido (DBSCAN), los scores no serán válidos
                        if len(set(labels)) > 1:  # Al menos dos clústeres
                            sil_score = silhouette_score(X_comb, labels)
                            ari_score = adjusted_rand_score(self.input_df['Class'], labels)  # Usando 'Class' como referencia
                        else:
                            sil_score = np.nan
                            ari_score = np.nan
                    else:
                        sil_score = np.nan
                        ari_score = np.nan

                    elapsed_time = time.time() - start_time

                    results.append({
                        'Model': model_name,
                        'Combination': str(comb),  # Almacenar la combinación como una cadena de texto
                        'Silhouette Score': sil_score,
                        'Adjusted Rand Index (ARI)': ari_score,
                        'Time': elapsed_time,
                        'Num Attributes': len(comb)
                    })

        results_df = pd.DataFrame(results)

        # Guardar resultados
        self.save_results_to_csv(results_df)

        return results_df

    def save_results_to_csv(self, results_df):
        results_df.to_csv("unsupervised_combinations_results.csv", index=False)

    def select_best_model_per_combination(self, results_df):
        # Seleccionar el mejor modelo para cada número de atributos (1, 2, 3, 4)
        best_models = []

        for num_attributes in range(1, 5):  # De 1 a 4 atributos
            # Filtramos las combinaciones con el número específico de atributos
            df_subset = results_df[results_df['Num Attributes'] == num_attributes]

            # Calculamos el score combinando Silhouette y ARI, y ordenamos por el tiempo
            df_subset['Score'] = df_subset.apply(lambda row: np.nanmean([row['Silhouette Score'], row['Adjusted Rand Index (ARI)']]) if not np.isnan(row['Silhouette Score']) and not np.isnan(row['Adjusted Rand Index (ARI)']) else np.nan, axis=1)

            # El modelo con el mejor score y menor tiempo
            best_row = df_subset.loc[df_subset['Score'].idxmax()]
            best_models.append(best_row)

        best_models_df = pd.DataFrame(best_models)

        return best_models_df


# Streamlit code for visualization
st.sidebar.title("Ayuda")
st.sidebar.write("""
Este código realiza un análisis de modelos de clustering utilizando diferentes combinaciones de atributos. A continuación, se describen los pasos y funcionalidades principales:

1. **Base de Datos:** Utiliza el conjunto de datos Iris.
2. **Modelos Utilizados:** K-Means, MeanShift, DBSCAN, Clustering Jerárquico y BIRCH.
3. **Combinaciones por Atributos:** Prueba todas las combinaciones posibles de atributos del conjunto de datos.
4. **Métricas de Evaluación:** Usa el **Silhouette Score** y el **Adjusted Rand Index** (ARI).
5. **Tiempo de Ejecución:** Calcula el tiempo que cada modelo tarda en resolver cada combinación de atributos.

Autor: Javier Horacio Pérez Ricárdez
""")

st.title('Análisis de modelos no supervisados para todas las combinaciones')

# Cargar datos (modificar según sea necesario)
input_df = pd.read_csv("iris.csv")
st.write("### Iris Dataset", input_df)

# Inicializar Análisis de Modelos No Supervisados
analysis = UnsupervisedModelAnalysis(input_df=input_df, seed=1271673)
results_df = analysis.run_all_combinations()

# Selección del mejor modelo por combinación de atributos
best_models_df = analysis.select_best_model_per_combination(results_df)

# Mostrar los resultados completos
st.write("### Resultados de rendimiento del modelo (todos los modelos evaluados)", results_df)

# Mostrar resultados de los mejores modelos
st.write("### Los mejores modelos para cada combinación de atributos", best_models_df)

# Filtrar los datos para el gráfico con solo los mejores modelos
filtered_df = best_models_df[['Model', 'Silhouette Score', 'Time', 'Combination']]

# Crear gráfico de barras dinámico con Plotly para los mejores modelos
fig = px.bar(filtered_df, 
             x='Model', 
             y=['Silhouette Score', 'Time'], 
             color='Model',
             text='Combination',  # Agregar combinación en las etiquetas
             title="Comparación de Modelos por Silhouette Score y Tiempo",
             labels={'Silhouette Score': 'Silhouette Score', 'Time': 'Tiempo (segundos)'},
             height=500)

# Mostrar gráfico
st.plotly_chart(fig)