import pandas as pd
import numpy as np
import os
import logging
import csv
import time
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# 1. CONFIGURACIÓN CENTRALIZADA
# ---------------------------------------------------------
# Ajustamos la ruta para que funcione directo en la raíz de Colab
CONFIG = {
    'data_path': 'tema1_ventas_retail.csv', 
    'model_params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
    'test_size': 0.2,
    'target_column': 'ventas_semanales',
    'date_column': 'fecha',
    'store_column': 'tienda_id'
}

# Configuración de Logging corregida para Colab
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True, # Forzar configuración en Colab
    handlers=[logging.StreamHandler()]
)

# ---------------------------------------------------------
# 2. FUNCIONES DE APOYO
# ---------------------------------------------------------

def asegurar_datos():
    """Genera datos sintéticos si el archivo no existe para evitar que el script falle"""
    if not os.path.exists(CONFIG['data_path']):
        logging.info("🛠 Generando archivo de datos inicial...")
        np.random.seed(42)
        fechas = pd.date_range(start="2023-01-01", periods=104, freq="W")
        data = []
        for t_id in range(1, 6): # 5 tiendas
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
        pd.DataFrame(data).to_csv(CONFIG['data_path'], index=False)
        logging.info("✅ Archivo CSV creado exitosamente.")

def cargar_y_validar_datos(ruta):
    try:
        df = pd.read_csv(ruta)
        df[CONFIG['date_column']] = pd.to_datetime(df[CONFIG['date_column']])
        return df
    except Exception as e:
        logging.error(f"Error en carga: {e}")
        return None

def preprocesar_datos(df):
    df = df.copy()
    df['año'] = df[CONFIG['date_column']].dt.year
    df['mes'] = df[CONFIG['date_column']].dt.month
    df['semana'] = df[CONFIG['date_column']].dt.isocalendar().week.astype(int)
    df['dia_semana'] = df[CONFIG['date_column']].dt.dayofweek
    features = [CONFIG['store_column'], 'año', 'mes', 'semana', 'dia_semana', 
                'promocion_activa', 'inventario_inicial', 'temperatura_promedio']
    return df[features], df[CONFIG['target_column']]

def entrenar_y_evaluar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=42)
    modelo = RandomForestRegressor(**CONFIG['model_params'])
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas = {'mae': mean_absolute_error(y_test, y_pred), 'r2': r2_score(y_test, y_pred)}
    return modelo, metricas, y_test, y_pred

def generar_metricas_performance(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / np.where(y_real == 0, 1, y_real))) * 100
    return {'MAE': f"{mae:.2f}", 'R²': f"{r2:.3f}", 'MAPE': f"{mape:.1f}%"}

def guardar_log_ejecucion(log_data):
    archivo_log = 'historial_ejecuciones.csv'
    duracion = round((time.time() - log_data['inicio_timer']) / 60, 4)
    existe = os.path.exists(archivo_log)
    with open(archivo_log, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(['timestamp', 'exito', 'duracion_minutos', 'archivos', 'errores'])
        writer.writerow([log_data['timestamp'], log_data['exito'], duracion, 
                         ';'.join(log_data['archivos_generados']), ';'.join(log_data['errores'])])

# ---------------------------------------------------------
# 3. EJECUCIÓN DEL PIPELINE Y VISUALIZACIÓN
# ---------------------------------------------------------

asegurar_datos()
log_ejecucion = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                 'inicio_timer': time.time(), 'exito': False, 
                 'archivos_generados': [], 'errores': []}

try:
    # 1. Cargar y procesar
    df = cargar_y_validar_datos(CONFIG['data_path'])
    X, y = preprocesar_datos(df)
    
    # 2. Entrenar
    modelo, metricas_entreno, y_test, y_pred = entrenar_y_evaluar_modelo(X, y)
    detalles = generar_metricas_performance(y_test, y_pred)
    
    # 3. Mostrar Resultados Visuales (Lo que faltaba)
    print("\n" + "="*30)
    print("📊 RESUMEN DE PERFORMANCE")
    print("="*30)
    for k, v in detalles.items():
        print(f"{k}: {v}")
    
    # Gráfico 1: Real vs Predicción
    fig_scatter = px.scatter(x=y_test, y=y_pred, 
                             labels={'x': 'Ventas Reales', 'y': 'Ventas Predichas'},
                             title='Comparativa de Predicción: Real vs Modelo',
                             template='plotly_white', trendline="ols")
    fig_scatter.show()

    # Gráfico 2: Importancia de Variables
    df_importancia = pd.DataFrame({
        'Feature': X.columns,
        'Importancia': modelo.feature_importances_
    }).sort_values('Importancia', ascending=True)
    
    fig_bar = px.bar(df_importancia, x='Importancia', y='Feature', orientation='h',
                     title='Factores que más influyen en las Ventas',
                     color='Importancia', color_continuous_scale='Blues')
    fig_bar.show()

    log_ejecucion['exito'] = True
    log_ejecucion['archivos_generados'] = ['modelo.pkl', 'reporte.html']

except Exception as e:
    logging.error(f"❌ Error en pipeline: {e}")
    log_ejecucion['errores'] = [str(e)]

guardar_log_ejecucion(log_ejecucion)
