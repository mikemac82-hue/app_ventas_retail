import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

# 1. CONFIGURACIÓN DE PÁGINA STREAMLIT
st.set_page_config(page_title="RetailMax Analytics", layout="wide")

# 2. CONFIGURACIÓN CENTRALIZADA
CONFIG = {
    'data_path': 'tema1_ventas_retail.csv',
    'target_column': 'ventas_semanales',
    'date_column': 'fecha',
    'store_column': 'tienda_id'
}

# 3. FUNCIONES DE APOYO (Optimizadas para Web)
@st.cache_data
def asegurar_y_cargar_datos():
    """Genera datos si no existen y los carga con caché para velocidad"""
    if not os.path.exists(CONFIG['data_path']):
        np.random.seed(42)
        fechas = pd.date_range(start="2023-01-01", periods=104, freq="W")
        data = []
        for t_id in range(1, 6):
            base = np.random.normal(15000, 2000)
            for f in fechas:
                promo = np.random.choice([0, 1], p=[0.7, 0.3])
                temp = np.random.normal(20, 8)
                ventas = base + (promo * 3000) - (abs(temp-20)*50) + np.random.normal(0, 500)
                data.append({
                    'tienda_id': t_id, 'fecha': f, 'ventas_semanales': max(0, ventas),
                    'promocion_activa': promo, 'inventario_inicial': np.random.randint(30000, 70000),
                    'temperatura_promedio': round(temp, 1)
                })
        df = pd.DataFrame(data)
        df.to_csv(CONFIG['data_path'], index=False)
    
    df = pd.read_csv(CONFIG['data_path'])
    df[CONFIG['date_column']] = pd.to_datetime(df[CONFIG['date_column']])
    return df

def preprocesar(df):
    df = df.copy()
    df['año'] = df[CONFIG['date_column']].dt.year
    df['mes'] = df[CONFIG['date_column']].dt.month
    df['semana'] = df[CONFIG['date_column']].dt.isocalendar().week.astype(int)
    features = [CONFIG['store_column'], 'año', 'mes', 'semana', 'promocion_activa', 'inventario_inicial', 'temperatura_promedio']
    return df[features], df[CONFIG['target_column']]

# 4. INTERFAZ Y EJECUCIÓN
st.title("🚀 RetailMax - Dashboard de Predicción de Ventas")

df = asegurar_y_cargar_datos()

# Sidebar para parámetros
st.sidebar.header("Parámetros del Modelo")
n_est = st.sidebar.slider("Número de Árboles (Estimators)", 10, 200, 100)
max_d = st.sidebar.slider("Profundidad Máxima", 5, 30, 10)

if st.button("Ejecutar Entrenamiento y Análisis"):
    with st.spinner('Entrenando modelo...'):
        # Proceso
        X, y = preprocesar(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelo = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.success("✅ ¡Pipeline completado!")
        
        # Visualización de Métricas
        col1, col2 = st.columns(2)
        col1.metric("MAE (Error Medio)", f"${mae:.2f}")
        col2.metric("R² (Ajuste)", f"{r2:.3f}")
        
        # Gráficos
        st.subheader("Análisis Visual")
        c_alt1, c_alt2 = st.columns(2)
        
        with c_alt1:
            fig_scatter = px.scatter(x=y_test, y=y_pred, trendline="ols",
                                     title="Ventas Reales vs Predichas",
                                     labels={'x': 'Real', 'y': 'Predicción'})
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with c_alt2:
            df_imp = pd.DataFrame({'Feature': X.columns, 'Importancia': modelo.feature_importances_}).sort_values('Importancia')
            fig_bar = px.bar(df_imp, x='Importancia', y='Feature', orientation='h', title="Importancia de Variables")
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("Presiona el botón para iniciar el análisis con los datos históricos.")
    st.write("Vista previa de los datos actuales:")
    st.dataframe(df.head())
