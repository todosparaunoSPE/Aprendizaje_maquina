# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:36:52 2024

@author: jperezr
"""

import pandas as pd
import streamlit as st

# Cargar el conjunto de datos (en este caso, Iris)
#input_df = pd.read_csv("iris.csv")

# Título de la aplicación en Streamlit
st.title("Modelos Supervisados y No Supervisados")

# Descripción de la aplicación
#st.sidebar.title("Ayuda")
#st.sidebar.write("""
#Este aplicativo permite realizar análisis con modelos supervisados y no supervisados utilizando el conjunto de datos Iris.

### Modelos Supervisados:
#- **SVM**
#- **Árbol de Decisión**
#- **Random Forest**
#- **Regresión Logística**

### Modelos No Supervisados:
#- **KMeans**
#- **MeanShift**
#- **DBSCAN**
#- **BIRCH**

#Para cada modelo, se explorarán combinaciones de atributos para analizar los resultados.

**Autor**: Javier Horacio Pérez Ricárdez
""")

# Mostrar el conjunto de datos
st.write("### Iris Dataset", input_df)
