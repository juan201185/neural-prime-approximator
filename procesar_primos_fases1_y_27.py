import os
import math
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.initializers import HeNormal 
from sklearn.metrics import mean_squared_error
import joblib 
import tensorflow as tf # Asegurarse de importar tensorflow completo

# --- IMPRESIÓN DE INICIO PARA CONFIRMAR QUE EL SCRIPT EMPEZA ---
print("--- Script 'procesar_primos_fases1_y_2.py' iniciado ---")

# --- Configuración global del SNI (Constantes) ---
K_SNI = 1.258104526
e_K = math.exp(K_SNI) 
# ----------------------------------------------------

# Directorio donde se descargaron los archivos .txt de primos
directorio_primos = "primos_descargados"

# --- RUTAS ESTÁTICAS PARA GUARDAR/CARGAR MODELO Y SCALER ---
MODEL_SAVE_PATH = 'modelo_fi_ideal_sn_universal.keras'
SCALER_SAVE_PATH = 'scaler_features.pkl'

# --- FUNCIONES DE LA FASE 1: PREPARACIÓN DE DATOS ---

def consolidar_primos(num_millones):
    """
    Lee los archivos primosX.txt de cada millón y los consolida en un único dataset en memoria.
    """
    dataset_primos_consolidado = []
    X_global = 0 

    print(f"\n--- Iniciando consolidación de {num_millones} millón(es) de primos ---")

    for i in range(1, num_millones + 1):
        file_name = f"primes{i}.txt"
        file_path = os.path.join(directorio_primos, file_name)
        
        if not os.path.exists(file_path):
            print(f"  Advertencia: Archivo '{file_path}' no encontrado. Saltando.")
            continue

        print(f"  Procesando archivo: '{file_path}'...")
        
        try: 
            with open(file_path, 'r') as f:
                for line in f:
                    line_stripped = line.strip() 
                    if not line_stripped: 
                        continue 
                    
                    numeros_en_linea_str = line_stripped.split() 
                    
                    for num_str in numeros_en_linea_str:
                        if num_str:
                            try: 
                                P_X = int(num_str) 
                                dataset_primos_consolidado.append({"X": X_global, "P_X": P_X})
                                X_global += 1
                            except ValueError: 
                                print(f"    Saltando línea/segmento no numérico: '{num_str}' en '{file_name}'.")
                                continue 
            print(f"    '{file_path}' procesado. Total de primos consolidados hasta ahora: {X_global}.")
        
        except Exception as e: 
            print(f"    Ocurrió un error al leer '{file_path}': {e}. Deteniendo consolidación.")
            break 
                
    print(f"\n--- Consolidación completada. Total de primos en el dataset: {len(dataset_primos_consolidado)} ---")
    return dataset_primos_consolidado

def calcular_f_ideal(dataset_primos):
    """
    Calcula F_ideal para cada primo en el dataset consolidado. Válida para X >= 1.
    """
    dataset_f_ideal = []
    print("\n--- Iniciando cálculo de F_ideal para el dataset consolidado ---")

    primos_procesados = 0
    
    for entrada in dataset_primos:
        X = entrada["X"]
        P_X = entrada["P_X"]

        if X == 0:
            print(f"  Omitiendo X=0 (P(0)={P_X}) ya que X es el denominador en la fórmula de F_ideal.")
            continue 
        
        try: 
            f_ideal_val = X * e_K / P_X 
            
            dataset_f_ideal.append({
                "X": X, 
                "P_X": P_X, 
                "F_ideal": f_ideal_val
            })
            primos_procesados += 1
            
            if primos_procesados % 1000000 == 0: 
                print(f"  {primos_procesados} primos procesados para F_ideal...")

        except ValueError:
            print(f"  Advertencia: No se pudo calcular F_ideal para X={X}, P(X)={P_X}. Saltando.")
            continue
        except ZeroDivisionError: 
            print(f"  Advertencia: División por cero al calcular F_ideal para P(X)={P_X}. Saltando.")
            continue
        except Exception as e: 
            print(f"  Error inesperado al calcular F_ideal para X={X}, P(X)={P_X}: {e}. Deteniendo.")
            break
            
    print(f"\n--- Cálculo de F_ideal completado. Total de puntos con F_ideal: {len(dataset_f_ideal)} ---")
    return dataset_f_ideal

# --- VARIABLE GLOBAL PARA EL SCALER ---
scaler_features = None 

def generar_features(dataset_f_ideal):
    """
    Genera un vector de características (features) a partir de la posición X Y el valor P(X).
    INCLUYE ESCALADO DE CARACTERÍSTICAS para estabilidad de la RN.
    """
    global scaler_features 
    
    print("\n--- Iniciando Ingeniería de Características para la RN ---")

    raw_features_list = []
    y_targets_list = [] 
    
    primos_procesados = 0
    
    for entrada in dataset_f_ideal:
        X = entrada["X"]
        P_X = entrada["P_X"] 
        F_ideal = entrada["F_ideal"]
        
        # --- CÁLCULO DE TODAS LAS CARACTERÍSTICAS (Versión con todas las Features) ---
        feature_X = float(X)
        feature_ln_X = math.log(X) 
        feature_X_squared = float(X**2)
        feature_sqrt_X = float(math.sqrt(X)) 
        feature_one_over_X = 1.0/X 
        feature_nd_X = float(int(math.log10(X)) + 1) 
        feature_X_cubed = float(X**3) 

        feature_P_X = float(P_X) 
        feature_ln_P_X = math.log(P_X) 
        
        feature_log10_X = math.log10(X) if X > 0 else 0.0 
        feature_log10_PX = math.log10(P_X) if P_X > 0 else 0.0 
        feature_pow10_nd_X = 10**(int(math.log10(X))) if X > 0 else 1.0 
        feature_pow10_nd_PX = 10**(int(math.log10(P_X))) if P_X > 0 else 1.0
        
        features_vector = [
            feature_X, 
            feature_ln_X,         
            feature_X_squared, 
            feature_sqrt_X,
            feature_one_over_X, 
            feature_nd_X,
            feature_X_cubed,
            feature_P_X,         
            feature_ln_P_X,       
            feature_log10_X,      
            feature_log10_PX,     
            feature_pow10_nd_X,   
            feature_pow10_nd_PX   
        ]
        
        raw_features_list.append(features_vector)
        y_targets_list.append(F_ideal) 

        primos_procesados += 1
        if primos_procesados % 1000000 == 0:
            print(f"  {primos_procesados} puntos procesados para características (con todas las features)...")
        
    X_features_np = np.array(raw_features_list, dtype=np.float32) 
    y_targets_np = np.array(y_targets_list, dtype=np.float32) 

    # --- ¡PASO CRUCIAL: ESCALADO DE CARACTERÍSTICAS! ---
    global scaler_features 
    scaler_features = StandardScaler() 
    X_features_scaled = scaler_features.fit_transform(X_features_np) 
    
    dataset_con_features = []
    for i in range(len(X_features_scaled)):
        dataset_con_features.append({
            "features": X_features_scaled[i].tolist(), 
            "target_f_ideal": y_targets_np[i].item() 
        })
    
    print(f"\n--- Generación de características y ESCALADO completados para {len(dataset_con_features)} puntos ---")
    return dataset_con_features

# --- FUNCIÓN DE ENTRENAMIENTO DE LA RN ---
def entrenar_modelo_rn(dataset_entrenamiento):
    print("\n--- Iniciando Entrenamiento y Optimización de la Red Neuronal ---")
    print("  Objetivo: La red debe entender el PATRÓN de F_ideal y replicarlo con alta precisión.")

    X_features = np.array([d["features"] for d in dataset_entrenamiento], dtype=np.float32)
    y_targets = np.array([d["target_f_ideal"] for d in dataset_entrenamiento], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_targets, test_size=0.01, random_state=42
    ) 

    print(f"Dataset dividido: Entrenamiento={len(X_train)} puntos, Prueba={len(X_test)} puntos.")

    num_features = X_train.shape[1] 
    
    # --- ARQUITECTURA DE RN PARA ALTA PRECISIÓN ---
    model = Sequential([
        Dense(units=128, input_dim=num_features, activation='relu', kernel_initializer='he_normal'), 
        BatchNormalization(), 
        Dense(units=64, activation='relu', kernel_initializer='he_normal'),                             
        BatchNormalization(), 
        Dense(units=32, activation='relu', kernel_initializer='he_normal'),                              
        BatchNormalization(), 
        Dense(units=16, activation='relu', kernel_initializer='he_normal'),                              
        BatchNormalization(), 
        Dense(units=1, activation='linear')                              
    ])

    # --- Optimizador Adam con Tasa de Aprendizaje AUMENTADA y Clipping ---
    optimizer_adam = Adam(learning_rate=0.001, clipvalue=1.0) 
    model.compile(optimizer=optimizer_adam, loss='mse')

    print("Modelo de RN configurado. Iniciando entrenamiento (esto tomará un tiempo considerable)...")
    print("El progreso del entrenamiento se mostrará época por época.")
    
    history = model.fit(
        X_train, y_train,
        epochs=1000,                
        batch_size=32,             
        verbose=1,                 
        validation_split=0.1,      
        callbacks=[]               
    )
    
    print("\nEntrenamiento de la Red Neuronal completado.")

    loss_test = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Evaluación del Modelo en el Conjunto de Prueba Final ---")
    print(f"MSE (Error Cuadrático Medio) en datos no vistos: {loss_test:.6f}")
    
    if loss_test < 0.3: 
        print("¡Éxito! La red ha replicado el patrón de F_ideal con alta fidelidad (MSE < 0.3).")
    else:
        print("Advertencia: El MSE es más alto de lo esperado. La red podría necesitar más épocas o ajustes de arquitectura.")

    return model

# --- LLAMADA PRINCIPAL AL FLUJO DE TRABAJO DEL PROYECTO ---
num_millones_a_procesar = 1 
num_primos_entrenamiento_actual = 999999 

print(f"Iniciando el proceso completo de la Fase 1 para {num_primos_entrenamiento_actual} primos (prueba a escala de 1 millón)...")

# 1. Consolidar los primos (se consolida 1 millón)
mi_dataset_primos_consolidado = consolidar_primos(num_millones_a_procesar)

# --- Verificar si ya tenemos el modelo y scaler REALES entrenados y guardados ---
modelo_rn_entrenado = None
scaler_features = None
try:
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
        # Necesitamos la clase Adam para que Keras pueda reconstruir el optimizador.
        # Keras maneja 'mse' por string, pero el error indicó que buscaba una función.
        # Al usar .keras, Keras debería ser más inteligente.
        
        # Para ser muy explícitos si fuera necesario, se define custom_objects
        # custom_objects_for_load = {
        #    'Adam': lambda *args, **kwargs: tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0, *args, **kwargs)
        # }
        # Si el modelo se guardó con string 'mse' como loss, no se necesita pasarlo a custom_objects si es .keras
        
        # La forma más simple y robusta para .keras es intentar cargar sin custom_objects si es posible.
        # Si falla, entonces sí se añaden.
        
        # Como el error fue 'mse' not found, y no Adam, probamos sin custom_objects para mse.
        # Pero para Adam, sí es bueno pasarlo si se customizó el LR.
        # La solución más limpia es pasar custom_objects con Adam, y Keras debería resolver 'mse'.
        
        custom_objects_for_load = {
             'Adam': tf.keras.optimizers.Adam # Pasa la CLASE Adam
             # 'mse': tf.keras.losses.MeanSquaredError() # Si 'mse' sigue siendo un problema
        }
        
        # Solo cargamos si los archivos existen
        modelo_rn_entrenado = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects=custom_objects_for_load)
        scaler_features = joblib.load(SCALER_SAVE_PATH)
        print("\n--- Modelo y Scaler REALES PRE-ENTRENADOS cargados exitosamente. ¡No se requiere reentrenamiento! ---")
    else:
        print("\n--- Modelo o Scaler no encontrados. Procediendo con el entrenamiento completo. ---")
except Exception as e:
    print(f"\nAdvertencia: Error al intentar cargar modelo/scaler existente: {e}")
    print("Esto es esperado si el modelo/scaler no existe o hubo un problema de serialización/deserialización.")
    print("Procediendo con el entrenamiento completo para generar/regenerar el modelo.")

# --- Solo entrenar si el modelo no fue cargado exitosamente ---
if modelo_rn_entrenado is None:
    mi_dataset_primos_para_entrenamiento_actual = []
    if mi_dataset_primos_consolidado:
        if len(mi_dataset_primos_consolidado) >= num_primos_entrenamiento_actual:
            mi_dataset_primos_para_entrenamiento_actual = mi_dataset_primos_consolidado[:num_primos_entrenamiento_actual]
        else:
            print(f"Advertencia: El dataset consolidado ({len(mi_dataset_primos_consolidado)} primos) es menor que los primos solicitados para entrenamiento ({num_primos_entrenamiento_actual}). Usando todos los consolidados.")
            mi_dataset_primos_para_entrenamiento_actual = mi_dataset_primos_consolidado
    else:
        print("No se pudo consolidar ningún primo. Verifique la descarga y los archivos.")

    mi_dataset_f_ideal_consolidado = []
    if mi_dataset_primos_para_entrenamiento_actual:
        mi_dataset_f_ideal_consolidado = calcular_f_ideal(mi_dataset_primos_para_entrenamiento_actual)
    else:
        print("No hay datos de primos para calcular F_ideal. Verifique el paso anterior.")

    dataset_final_entrenamiento = []
    if mi_dataset_f_ideal_consolidado:
        dataset_final_entrenamiento = generar_features(mi_dataset_f_ideal_consolidado)
        print(f"\nDataset Final de Entrenamiento (features escaladas) listo para {len(dataset_final_entrenamiento)} puntos.")
    else:
        print("No hay datos de F_ideal para generar características. Verifique el paso anterior.")

    if dataset_final_entrenamiento:
        modelo_rn_entrenado = entrenar_modelo_rn(dataset_final_entrenamiento)
    else:
        print("\n--- Fallo en la preparación de datos para el entrenamiento de la RN ---")

# --- Sección de Guardado (si el entrenamiento fue exitoso o se cargó) ---
if modelo_rn_entrenado:
    print("\n--- Fase 1: Preparación de Datos y Entrenamiento de la RN COMPLETADA (Éxito) ---")
    print("Modelo de RN entrenado disponible en la variable 'modelo_rn_entrenado'.")
    try:
        # Guardar el modelo Keras en formato .keras (más robusto)
        modelo_rn_entrenado.save(MODEL_SAVE_PATH)
        print(f"Modelo de RN guardado como '{MODEL_SAVE_PATH}'.")
    except Exception as e:
        print(f"No se pudo guardar el modelo: {e}")
    
    try:
        # Guardar el scaler de scikit-learn
        joblib.dump(scaler_features, SCALER_SAVE_PATH)
        print(f"Scaler guardado como '{SCALER_SAVE_PATH}'.")
    except Exception as e:
        print(f"No se pudo guardar el scaler: {e}")

else: # Si modelo_rn_entrenado sigue siendo None (no se entrenó ni se cargó)
    print("\n--- Fase 1: Fallo en la preparación de datos o entrenamiento de la RN ---")