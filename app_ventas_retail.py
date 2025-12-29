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

# ---------------------------------------------------------
# 1. CONFIGURACIÓN CENTRALIZADA
# ---------------------------------------------------------
CONFIG = {
    'data_path': 'content/tema1_ventas_retail.csv',
    'model_params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
    'test_size': 0.2,
    'target_column': 'ventas_semanales',
    'date_column': 'fecha',
    'store_column': 'tienda_id'
}

REPORTE_CONFIG = {
    'titulo_empresa': 'RetailMax Analytics',
    'colores_corporativos': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    'template_plotly': 'plotly_white'
}

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ejecucion_automatica.log'), logging.StreamHandler()]
)

# ---------------------------------------------------------
# 2. FUNCIONES DE APOYO (LOS CIMIENTOS)
# ---------------------------------------------------------

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

def guardar_modelo_y_metricas(modelo, metricas):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    m_path, met_path = f'modelo_{ts}.pkl', f'metricas_{ts}.txt'
    with open(m_path, 'wb') as f: pickle.dump(modelo, f)
    with open(met_path, 'w') as f: f.write(str(metricas))
    return m_path, met_path

def generar_metricas_performance(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / np.where(y_real == 0, 1, y_real))) * 100
    return {
        'MAE': f"{mae:.2f}",
        'R²': f"{r2:.3f}",
        'MAPE': f"{mape:.1f}%"
    }

def generar_reporte_completo(modelo, metricas, y_real, y_pred, feature_names):
    nombre = f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(nombre, 'w') as f: f.write(f"<h1>Reporte RetailMax</h1><p>{metricas}</p>")
    return nombre

# ---------------------------------------------------------
# 3. SISTEMA DE MONITOREO Y PIPELINE
# ---------------------------------------------------------

def verificar_datos_nuevos(ruta, dias_atras=7):
    if not os.path.exists(ruta): return False, None
    df = pd.read_csv(ruta)
    df[CONFIG['date_column']] = pd.to_datetime(df[CONFIG['date_column']])
    fecha_max = df[CONFIG['date_column']].max()
    hay_nuevos = fecha_max >= (datetime.now() - timedelta(days=dias_atras))
    return hay_nuevos, fecha_max

def crear_log_ejecuciones():
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'inicio_timer': time.time(),
        'exito': False,
        'archivos_generados': [],
        'errores': []
    }

def guardar_log_ejecucion(log_data):
    archivo_log = 'historial_ejecuciones.csv'
    duracion = round((time.time() - log_data['inicio_timer']) / 60, 2)
    existe = os.path.exists(archivo_log)
    with open(archivo_log, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(['timestamp', 'exito', 'duracion_minutos', 'archivos', 'errores'])
        writer.writerow([log_data['timestamp'], log_data['exito'], duracion, 
                         ';'.join(log_data['archivos_generados']), ';'.join(log_data['errores'])])

def ejecutar_pipeline_completo():
    try:
        logging.info("🚀 Iniciando pipeline...")
        df = cargar_y_validar_datos(CONFIG['data_path'])
        X, y = preprocesar_datos(df)
        modelo, metricas, y_test, y_pred = entrenar_y_evaluar_modelo(X, y)
        m_file, met_file = guardar_modelo_y_metricas(modelo, metricas)
        detalles = generar_metricas_performance(y_test, y_pred)
        rep_file = generar_reporte_completo(modelo, detalles, y_test, y_pred, X.columns)
        
        return True, [m_file, met_file, rep_file]
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return False, [str(e)]

# ---------------------------------------------------------
# 4. EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------
log_ejecucion = crear_log_ejecuciones()

logging.info("Verificando datos...")
hay_nuevos, fecha_reciente = verificar_datos_nuevos(CONFIG['data_path'], dias_atras=1000) # 1000 para forzar ejec en test

if hay_nuevos:
    exito, resultados = ejecutar_pipeline_completo()
    log_ejecucion['exito'] = exito
    if exito:
        log_ejecucion['archivos_generados'] = resultados
    else:
        log_ejecucion['errores'] = resultados
else:
    logging.info("Sin datos nuevos. Fin.")
    log_ejecucion['exito'] = True
    log_ejecucion['errores'] = ["Sin datos nuevos"]

guardar_log_ejecucion(log_ejecucion)