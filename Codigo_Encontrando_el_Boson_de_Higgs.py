# ==============================================================================
# Pipeline de Produccion Final — Sistema Profesional FAIR Universe HiggsML
# ==============================================================================
# Autor: Manuel Cespedes
# Version: 3.0 (Evolucion desde pipeline_entrenamiento_v2.py)
# Objetivo:
#   1. Sistema profesional con persistencia (no re-entrena innecesariamente).
#   2. Interpretabilidad: traduce variables y reglas a lenguaje humano.
#   3. Convergencia visual: demuestra que el modelo aprendio lo suficiente.
#   4. Prediccion de nuevos eventos con intervalos de confianza.
#
# Nuevas Capacidades (vs V2):
#   - Persistencia con joblib (guardar/cargar modelo entrenado)
#   - Diccionario Fisico (traduce DER_*/PRI_* a lenguaje humano)
#   - Extraccion de Decisiones Reales del arbol XGBoost
#   - Grafico de Convergencia (Learning Curves de XGBoost + MLP)
#   - Simulacion de Prediccion de Nuevos Eventos
# ==============================================================================

import sys
import os
import time
import re
import glob
import warnings

# --- Fix de codificacion + flush para consola de Windows ---
# cp1252 no soporta caracteres como mu, sigma, flechas, etc.
# Forzamos UTF-8 y salida "line-buffered" para ver progreso en tiempo real.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding='utf-8',
        errors='replace',
        line_buffering=True,
        write_through=True
    )
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding='utf-8',
        errors='replace',
        line_buffering=True,
        write_through=True
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import joblib  # Para guardar y cargar modelos entrenados

# --- Modelos y Herramientas de Scikit-Learn ---
from sklearn.ensemble import StackingClassifier     # Para combinar modelos (stacking)
from sklearn.neural_network import MLPClassifier # Red Neuronal Multicapa
from sklearn.linear_model import LogisticRegression # Meta-learner simple para combinar modelos
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_curve, auc, accuracy_score,
    confusion_matrix, classification_report, brier_score_loss
)

# --- XGBoost --- # Importa la librería XGBoost.
import xgboost as xgb # El modelo mas potente para datos tabulares, especialmente en fisica de particulas

# --- Nuestro Cargador de Datos Personalizado ---
from cargador_datos import HiggsDataLoader   # Clase personalizada para cargar y preprocesar los datos del CERN

# Suprimir advertencias menores de convergencia
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# CONFIGURACION GLOBAL
# ==============================================================================

# Ruta donde estan los datos de entrenamiento
DATA_DIR = r"c:/Manolo/MBA/ciencia_de_datos/tarea/01_Datos/Competencia_CERN/HiggsML_Uncertainty_Challenge(2025)/public_data_CERN/input_data/train"

# Carpeta raiz de resultados
OUTPUT_ROOT_DIR = r"c:/Manolo/MBA/ciencia_de_datos/tarea/04_Resultados/Resultados_Produccion_Final"

# Cada ejecucion usa su propia subcarpeta (no sobreescribe entrenamientos previos).
# Puedes fijar un id externo con la variable de entorno HIGGS_RUN_ID.
RUN_ID = os.getenv("HIGGS_RUN_ID", time.strftime("run_%Y%m%d_%H%M%S"))
RUN_ID = re.sub(r"[^0-9A-Za-z_.-]", "_", RUN_ID)
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, RUN_ID)

# Ruta del modelo persistido
MODELO_PATH = os.path.join(OUTPUT_DIR, "modelo_entrenado_higgs.pkl")

# >>> CONTROL DE RE-ENTRENAMIENTO <<<
# Poner en True para forzar re-entrenamiento aunque el modelo ya exista.
# Poner en False para cargar el modelo guardado (instantaneo).
FORZAR_ENTRENAMIENTO = False

# Modo inferencia: si es True, NUNCA entrena y exige modelo ya entrenado.
SOLO_INFERENCIA = True

# Ruta opcional de modelo para cargar (si no se define, se usa el ultimo modelo disponible).
# Ejemplo por variable de entorno:
#   set HIGGS_MODELO_PATH=c:\...\run_YYYYMMDD_HHMMSS\modelo_entrenado_higgs.pkl
MODELO_CARGA_PATH = os.getenv("HIGGS_MODELO_PATH", "").strip() or None

# Cantidad de datos a usar (None = todos, numero = muestra rapida)
SAMPLE_SIZE = 10000000  # [PARA TEST RAPIDO -- cambiar a None para competicion]

# Indice del arbol de XGBoost a graficar en detalle (0 = primer arbol)
TREE_INDEX_TO_PLOT = 0

# Modo de ajuste de hiperparametros:
# - "manual": usa parametros fijos definidos por el usuario.
# - "auto": busca automaticamente mejores parametros antes de entrenar.
MODO_AJUSTE_HIPERPARAMETROS = "auto"  # "manual" o "auto"

# Configuracion del autoajuste (solo se usa si MODO_AJUSTE_HIPERPARAMETROS="auto")
AUTO_TUNING_MAX_ROWS = 30000
AUTO_TUNING_CV = 3
AUTO_TUNING_XGB_N_ITER = 10
AUTO_TUNING_MLP_N_ITER = 8
AUTO_TUNING_META_N_ITER = 6

# ============================================================================
# RUTA MINIMA (CASO B): SOLO ELIGES DATOS, EL MODELO AJUSTA EL RESTO
# ============================================================================
# 1) Define cuanta data usar:
#    SAMPLE_SIZE = 100000      (prueba rapida)
#    SAMPLE_SIZE = None        (todos los datos)
#
# 2) Activa autoajuste de hiperparametros:
#    MODO_AJUSTE_HIPERPARAMETROS = "auto"
#
# 3) Si ya existe modelo guardado y quieres reentrenar con nuevo ajuste:
#    FORZAR_ENTRENAMIENTO = True
#
# 4) Ejecuta:
#    py 02_Codigos/pipeline_produccion_final.py
# ============================================================================

# Numero de iteraciones de Bootstrap
N_BOOTSTRAP = 100 

# Crear carpetas de resultados
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_latest_model_path(root_dir, model_filename="modelo_entrenado_higgs.pkl"):
    """
    Busca el modelo mas reciente dentro de OUTPUT_ROOT_DIR (incluyendo subcarpetas run_*).
    Retorna path absoluto o None si no existe ninguno.
    """
    pattern = os.path.join(root_dir, "**", model_filename)
    candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


# ==============================================================================
# DICCIONARIO FISICO — TRADUCTOR DE VARIABLES
# ==============================================================================
# Este diccionario traduce los nombres tecnicos de las variables del detector
# a descripciones que cualquier persona puede entender.
#
# Las variables DER_* (Derived) son cantidades calculadas por los fisicos.
# Las variables PRI_* (Primary) son mediciones directas del detector ATLAS.
#
# Si una variable no esta en el diccionario, se muestra su nombre original.
# ==============================================================================

DICCIONARIO_FISICO = {
    # --- Variables Derivadas (DER_*) ---
    # Estas son las mas importantes para el descubrimiento del Higgs
    'DER_mass_MMC': 'Masa Estimada del Boson (la variable mas critica para identificar al Higgs)',
    'DER_mass_transverse_met_lep': 'Masa Transversal (energia invisible + lepton: indica desintegracion)',
    'DER_mass_vis': 'Masa Visible (suma de todas las particulas detectadas)',
    'DER_pt_h': 'Momento del Higgs (velocidad/energia del candidato a Higgs)',
    'DER_deltaeta_jet_jet': 'Separacion Angular Jets (distancia entre los dos chorros de particulas)',
    'DER_mass_jet_jet': 'Masa de los Jets (masa combinada de los dos chorros de particulas)',
    'DER_prodeta_jet_jet': 'Producto Eta Jets (indica si los jets van en la misma direccion)',
    'DER_deltar_tau_lep': 'Distancia Tau-Lepton (separacion entre el tau y el lepton)',
    'DER_pt_tot': 'Momento Total (suma vectorial de todos los momentos)',
    'DER_sum_pt': 'Suma de Momentos (energia total transversal de todas las particulas)',
    'DER_pt_ratio_lep_tau': 'Ratio Momento Lepton/Tau (quien lleva mas energia)',
    'DER_met_phi_centrality': 'Centralidad Energia Faltante (posicion angular de lo invisible)',
    'DER_lep_eta_centrality': 'Centralidad del Lepton (que tan central esta el lepton)',

    # --- Variables Primarias (PRI_*) ---
    # Estas son mediciones directas del detector ATLAS del CERN
    'PRI_tau_pt': 'Momento del Tau (velocidad de la particula Tau detectada)',
    'PRI_tau_eta': 'Angulo Eta del Tau (direccion vertical respecto al haz)',
    'PRI_tau_phi': 'Angulo Phi del Tau (direccion horizontal respecto al haz)',
    'PRI_lep_pt': 'Momento del Lepton (velocidad del electron o muon detectado)',
    'PRI_lep_eta': 'Angulo Eta del Lepton (direccion vertical del lepton)',
    'PRI_lep_phi': 'Angulo Phi del Lepton (direccion horizontal del lepton)',
    'PRI_met': 'Energia Faltante (lo que "se escapo" del detector: neutrinos)',
    'PRI_met_phi': 'Direccion Energia Faltante (hacia donde se fue lo invisible)',
    'PRI_met_sumet': 'Suma Total de Energia (toda la energia depositada en el detector)',
    'PRI_jet_num': 'Numero de Jets (cantidad de chorros de particulas disparados)',
    'PRI_jet_leading_pt': 'Momento del Jet Principal (el chorro mas energetico)',
    'PRI_jet_leading_eta': 'Angulo del Jet Principal (direccion del chorro mas fuerte)',
    'PRI_jet_leading_phi': 'Phi del Jet Principal (orientacion horizontal del jet 1)',
    'PRI_jet_subleading_pt': 'Momento del Jet Secundario (el segundo chorro mas energetico)',
    'PRI_jet_subleading_eta': 'Angulo del Jet Secundario (direccion del segundo chorro)',
    'PRI_jet_subleading_phi': 'Phi del Jet Secundario (orientacion horizontal del jet 2)',
    'PRI_jet_all_pt': 'Momento Total de Jets (energia total de todos los chorros)',
}


def traducir_variable(nombre):
    """
    Dado el nombre tecnico de una variable (ej: 'DER_mass_MMC'),
    retorna su descripcion en lenguaje humano.
    Si no esta en el diccionario, retorna el nombre original.
    """
    return DICCIONARIO_FISICO.get(nombre, nombre)


def traducir_variable_corta(nombre):
    """Version corta: solo retorna la primera parte del nombre humano (antes del parentesis)."""
    desc = DICCIONARIO_FISICO.get(nombre, nombre)
    # Extraer solo la parte antes del parentesis
    if '(' in desc:
        return desc.split('(')[0].strip()
    return desc


# ==============================================================================
# 1. INGENIERIA DE ROBUSTEZ — DATA AUGMENTATION FISICO
# ==============================================================================
# (Misma funcion de V2: inyecta ruido gaussiano en variables derivadas)
# ==============================================================================

def add_physics_noise(X, noise_level=0.01):
    """
    Inyecta ruido gaussiano a las variables derivadas (DER_*) para simular
    incertidumbre experimental del detector.

    Explicacion: Esto simula errores de calibracion en el detector, forzando
    al modelo a no 'memorizar' valores exactos.
    """
    X_noisy = X.copy()
    der_columns = [col for col in X.columns if col.startswith('DER_') or col.startswith('der_')]

    if len(der_columns) == 0:
        print(f"  [Ruido] No se encontraron columnas DER_*. Aplicando ruido a TODAS las columnas.")
        der_columns = X.columns.tolist()
    else:
        print(f"  [Ruido] Aplicando ruido del {noise_level*100:.1f}% a {len(der_columns)} variables derivadas.")

    for col in der_columns:
        col_std = X_noisy[col].std()
        noise = np.random.normal(0, noise_level * col_std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    return X_noisy


# ==============================================================================
# 2. HIPERPARAMETROS (MANUAL O AUTOAJUSTE)
# ==============================================================================

def format_mlp_layers(hidden_layers):
    """Convierte (64, 32) a '64->32' para mostrar en logs/reportes."""
    return "->".join(str(int(x)) for x in hidden_layers)


def get_default_hyperparams():
    """Retorna la configuracion manual por defecto del pipeline."""
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    mlp_params = {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'learning_rate_init': 0.001,
        'learning_rate': 'adaptive',
        'batch_size': 64,
        'max_iter': 200,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'tol': 1e-4,
        'n_iter_no_change': 15,
        'random_state': 42,
        'verbose': False
    }

    meta_params = {
        'max_iter': 1000,
        'C': 1.0,
        'random_state': 42
    }

    return xgb_params, mlp_params, meta_params


def get_tuning_subset(X_train, y_train, max_rows=AUTO_TUNING_MAX_ROWS):
    """Toma una muestra estratificada para tuning cuando el dataset es muy grande."""
    if len(X_train) <= max_rows:
        return X_train, y_train

    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train,
        train_size=max_rows,
        stratify=y_train,
        random_state=42
    )
    return X_tune, y_tune


def auto_tune_hyperparams(X_train, y_train):
    """
    Ajusta hiperparametros automaticamente para XGBoost, MLP y meta-modelo.
    Usa RandomizedSearchCV sobre una muestra estratificada para acelerar.
    """
    print("\n--- Autoajuste de Hiperparametros (modo AUTO) ---")
    X_tune, y_tune = get_tuning_subset(X_train, y_train, max_rows=AUTO_TUNING_MAX_ROWS)
    print(f"  Datos para tuning: {len(X_tune)} eventos (de {len(X_train)} disponibles)")
    autotune_start = time.time()
    autotune_total_blocks = 3

    def _log_autotune_block_done(block_name, block_idx, block_start_time):
        block_elapsed = time.time() - block_start_time
        total_elapsed = time.time() - autotune_start
        progress = block_idx / autotune_total_blocks
        eta_sec = (total_elapsed / progress - total_elapsed) if progress > 0 else None
        eta_txt = _format_duration(eta_sec) if eta_sec is not None else "calculando..."
        print(
            f"  [AutoTune] {block_name} listo en {_format_duration(block_elapsed)} | "
            f"avance auto: {progress*100:.0f}% | "
            f"transcurrido auto: {_format_duration(total_elapsed)} | "
            f"ETA auto: {eta_txt}"
        )

    cv = StratifiedKFold(n_splits=AUTO_TUNING_CV, shuffle=True, random_state=42)

    # --- Tuning XGBoost ---
    print("\n  [AutoTune] Buscando parametros para XGBoost...")
    xgb_block_start = time.time()
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_space = {
        'n_estimators': [250, 350, 500, 700, 900],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0.0, 0.1, 0.3, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
        'reg_alpha': [0.0, 0.05, 0.1, 0.5]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_space,
        n_iter=AUTO_TUNING_XGB_N_ITER,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    xgb_search.fit(X_tune, y_tune)
    xgb_params = dict(xgb_search.best_params_)
    xgb_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    })
    print(f"  [AutoTune][XGB] Mejor AUC CV: {xgb_search.best_score_:.4f}")
    _log_autotune_block_done("XGBoost", 1, xgb_block_start)

    # --- Tuning MLP ---
    print("\n  [AutoTune] Buscando parametros para MLP...")
    mlp_block_start = time.time()
    mlp_base = MLPClassifier(
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    mlp_space = {
        'hidden_layer_sizes': [(64, 32), (96, 48), (128, 64), (128, 64, 32), (256, 128, 64)],
        'alpha': [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
        'learning_rate_init': [3e-4, 7e-4, 1e-3, 3e-3],
        'batch_size': [64, 128, 256],
        'max_iter': [150, 200, 250],
        'tol': [1e-4, 5e-5],
        'n_iter_no_change': [10, 15, 20]
    }
    mlp_search = RandomizedSearchCV(
        estimator=mlp_base,
        param_distributions=mlp_space,
        n_iter=AUTO_TUNING_MLP_N_ITER,
        scoring='roc_auc',
        cv=cv,
        n_jobs=1,
        verbose=2,
        random_state=42
    )
    mlp_search.fit(X_tune, y_tune)
    mlp_params = dict(mlp_search.best_params_)
    mlp_params.update({
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate': 'adaptive',
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': 42,
        'verbose': False
    })
    print(f"  [AutoTune][MLP] Mejor AUC CV: {mlp_search.best_score_:.4f}")
    _log_autotune_block_done("MLP", 2, mlp_block_start)

    # --- Tuning Meta-modelo ---
    print("\n  [AutoTune] Buscando parametros para meta-modelo (LogisticRegression)...")
    meta_block_start = time.time()
    meta_base = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        random_state=42
    )
    meta_space = {
        'C': [0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
        'max_iter': [800, 1000, 1500]
    }
    meta_search = RandomizedSearchCV(
        estimator=meta_base,
        param_distributions=meta_space,
        n_iter=AUTO_TUNING_META_N_ITER,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    meta_search.fit(X_tune, y_tune)
    meta_params = dict(meta_search.best_params_)
    meta_params.update({'random_state': 42})
    print(f"  [AutoTune][META] Mejor AUC CV: {meta_search.best_score_:.4f}")
    _log_autotune_block_done("META", 3, meta_block_start)

    tuning_info = {
        'mode': 'auto',
        'rows_used_for_tuning': int(len(X_tune)),
        'xgb_best_score_cv': float(xgb_search.best_score_),
        'mlp_best_score_cv': float(mlp_search.best_score_),
        'meta_best_score_cv': float(meta_search.best_score_)
    }
    return xgb_params, mlp_params, meta_params, tuning_info


def seleccionar_hiperparametros(X_train, y_train):
    """Selecciona hiperparametros en modo manual o auto."""
    modo = str(MODO_AJUSTE_HIPERPARAMETROS).strip().lower()
    if modo not in ("manual", "auto"):
        raise ValueError("MODO_AJUSTE_HIPERPARAMETROS debe ser 'manual' o 'auto'.")

    if modo == "manual":
        xgb_params, mlp_params, meta_params = get_default_hyperparams()
        info = {'mode': 'manual'}
        print("\n--- Hiperparametros en modo MANUAL ---")
        print(f"  XGBoost: {xgb_params['n_estimators']} arboles, prof={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}")
        print(f"  MLP: capas={format_mlp_layers(mlp_params['hidden_layer_sizes'])}, lr={mlp_params['learning_rate_init']}")
        print(f"  Meta-modelo: C={meta_params['C']}, max_iter={meta_params['max_iter']}")
        return xgb_params, mlp_params, meta_params, info

    return auto_tune_hyperparams(X_train, y_train)


# ==============================================================================
# 3. CONSTRUCCION DEL MODELO STACKING ("CONSEJO DE SABIOS")
# ==============================================================================

def build_stacking_model(xgb_params, mlp_params, meta_params):
    """Construye un StackingClassifier usando hiperparametros ya definidos."""
    print("\n--- Construyendo Arquitectura de Stacking ---")

    xgb_clf = xgb.XGBClassifier(**xgb_params)
    mlp_clf = MLPClassifier(**mlp_params)
    meta_learner = LogisticRegression(**meta_params)

    stacking_model = StackingClassifier(
        estimators=[('xgboost', xgb_clf), ('mlp', mlp_clf)],
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1,
        verbose=1
    )

    print(f"  [OK] XGBoost ({xgb_params.get('n_estimators')} arboles, prof. {xgb_params.get('max_depth')}, lr={xgb_params.get('learning_rate')})")
    print(f"  [OK] Red Neuronal MLP ({format_mlp_layers(mlp_params.get('hidden_layer_sizes', ()))}, lr={mlp_params.get('learning_rate_init')})")
    print(f"  [OK] Meta-Learner: Regresion Logistica (C={meta_params.get('C')})")
    print("  [OK] Validacion cruzada interna: 5 folds")

    return stacking_model


# ==============================================================================
# 3. CALIBRACION DE PROBABILIDADES
# ==============================================================================

def calibrate_model(base_model, X_cal, y_cal):
    """
    Envuelve un modelo entrenado en un CalibratedClassifierCV.
    Asegura que si el modelo dice '80% senal', sea realmente 80%.
    Esto es critico para el calculo correcto de mu.
    """
    print("\n--- Calibrando Probabilidades (Isotonic Regression) ---")
    print("  Si el modelo dice '80% senal', esta calibracion asegura que sea real.")

    calibrated_model = CalibratedClassifierCV(
        base_model, method='isotonic', cv=3
    )
    calibrated_model.fit(X_cal, y_cal)
    print("  [OK] Calibracion completada.")

    return calibrated_model


# ==============================================================================
# 4. ESTIMACION DE INCERTIDUMBRE — BOOTSTRAP
# ==============================================================================

def bootstrap_mu_estimation(y_true, y_pred_proba, weights, n_iterations=100):
    """
    Estima la intensidad de senal (mu) con intervalo de confianza via Bootstrap.
    mu=1 significa senal consistente con el Modelo Estandar de la Fisica.
    """
    print(f"\n--- Estimacion de Incertidumbre (Bootstrap, {n_iterations} iteraciones) ---")

    n_events = len(y_true)
    mu_samples = []
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    weights = np.array(weights)

    for i in range(n_iterations):
        indices = np.random.choice(n_events, size=n_events, replace=True)
        y_boot = y_true[indices]
        pred_boot = y_pred_proba[indices]
        w_boot = weights[indices]
        signal_mask = (y_boot == 1)

        if signal_mask.sum() == 0:
            continue

        s_pred = np.sum(w_boot[signal_mask] * pred_boot[signal_mask])
        s_expected = np.sum(w_boot[signal_mask])

        if s_expected > 0:
            mu_samples.append(s_pred / s_expected)

        if (i + 1) % 25 == 0:
            print(f"  Bootstrap: {i+1}/{n_iterations} iteraciones completadas...")

    mu_samples = np.array(mu_samples)
    mu_central = np.median(mu_samples)
    mu_low = np.percentile(mu_samples, 16)
    mu_high = np.percentile(mu_samples, 84)

    print(f"")
    print(f"  +==================================================+")
    print(f"  |  Estimacion de Mu: {mu_central:.4f} +/- {(mu_high - mu_low)/2:.4f}         |")
    print(f"  |  Intervalo 68%: [{mu_low:.4f}, {mu_high:.4f}]              |")
    print(f"  +==================================================+")

    return mu_central, mu_low, mu_high, mu_samples


# ==============================================================================
# 5. PERSISTENCIA DEL MODELO (GUARDAR / CARGAR)
# ==============================================================================
# Solucion al problema de re-entrenamiento: si el modelo ya fue entrenado,
# lo cargamos instantaneamente desde disco con joblib.
# ==============================================================================

def guardar_modelo(modelo_data, path):
    """
    Guarda el modelo entrenado y sus metadatos en disco con joblib.
    Esto permite cargar el modelo en segundos en futuras ejecuciones.
    """
    print(f"\n--- Guardando modelo en disco ---")
    joblib.dump(modelo_data, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] Modelo guardado: {path} ({size_mb:.1f} MB)")
    print(f"  En la proxima ejecucion, el modelo se cargara instantaneamente.")


def cargar_modelo(path):
    """
    Carga un modelo previamente entrenado desde disco.
    Retorna el diccionario con todos los componentes del modelo.
    """
    print(f"\n--- Cargando modelo desde disco ---")
    print(f"  Archivo: {path}")
    start = time.time()
    modelo_data = joblib.load(path)
    elapsed = time.time() - start
    print(f"  [OK] Modelo cargado en {elapsed:.2f} segundos (vs ~3 minutos de entrenamiento)")
    print(f"  Fecha de entrenamiento original: {modelo_data.get('fecha_entrenamiento', 'Desconocida')}")
    return modelo_data


# ==============================================================================
# 6. EXTRACCION DE REGLAS DEL ARBOL (DECISIONES REALES)
# ==============================================================================
# Accede al XGBoost dentro del Stacking y extrae las reglas de decision
# del primer arbol, traducidas a lenguaje humano.
# ==============================================================================

def extraer_reglas_arbol(stacking_model, feature_names):
    """
    Extrae y traduce las reglas de decision del XGBoost dentro del Stacking.
    Muestra las 3 primeras reglas del primer arbol en lenguaje humano.

    Retorna una lista de strings con las reglas traducidas para el reporte.
    """
    print("\n--- Extrayendo Reglas de Decision del Arbol ---")

    # Acceder al XGBoost dentro del StackingClassifier
    try:
        xgb_model = stacking_model.named_estimators_['xgboost']
    except (AttributeError, KeyError):
        print("  [AVISO] No se pudo acceder al XGBoost dentro del Stacking.")
        return ["No se pudieron extraer reglas (modelo no accesible)."]

    # Obtener el booster interno
    booster = xgb_model.get_booster()

    # Extraer el texto del primer arbol (Arbol #0) — captura la fisica mas obvia
    trees_dump = booster.get_dump(with_stats=False)
    tree0_text = trees_dump[0]

    # Parsear las reglas del arbol
    reglas_traducidas = []
    lines = tree0_text.strip().split('\n')

    # Cada linea tiene el formato: "0:[f5<0.123] yes=1,no=2,missing=1"
    # Donde f5 es el indice de la feature
    regla_num = 0
    for line in lines:
        if regla_num >= 3:
            break

        # Buscar lineas con condiciones (tienen '[')
        match = re.search(r'\[(\w+)<([-\d.]+)\]', line)
        if match:
            feature_ref = match.group(1)  # ej: 'f5'
            threshold = float(match.group(2))

            # Traducir el indice de feature al nombre real
            if feature_ref.startswith('f'):
                try:
                    feat_idx = int(feature_ref[1:])
                    if feat_idx < len(feature_names):
                        feat_name = feature_names[feat_idx]
                    else:
                        feat_name = feature_ref
                except ValueError:
                    feat_name = feature_ref
            else:
                feat_name = feature_ref

            # Obtener nombre humano
            nombre_humano = traducir_variable_corta(feat_name)

            # Determinar la profundidad por la indentacion
            depth = line.count('\t')

            if depth == 0:
                regla = (f"Regla #{regla_num+1}: Si la [{nombre_humano}] es menor a "
                         f"{threshold:.3f} -> Explorar rama IZQUIERDA del arbol, "
                         f"si no -> rama DERECHA")
            elif depth == 1:
                regla = (f"Regla #{regla_num+1}: Ademas, si [{nombre_humano}] < "
                         f"{threshold:.3f} -> Refinar clasificacion "
                         f"(corte secundario)")
            else:
                regla = (f"Regla #{regla_num+1}: Refinamiento profundo: "
                         f"[{nombre_humano}] < {threshold:.3f} -> "
                         f"Ajuste fino de la prediccion")

            reglas_traducidas.append(regla)
            print(f"  {regla}")
            regla_num += 1

    if not reglas_traducidas:
        reglas_traducidas.append("No se encontraron reglas parseables en el arbol #0.")
        print("  [AVISO] No se encontraron reglas parseables.")

    # Interpretar el patron general
    print("\n  Interpretacion: El arbol aprende a separar senales del Boson de Higgs")
    print("  del ruido de fondo haciendo cortes sucesivos en las variables fisicas.")
    print("  Cada corte divide los eventos en dos grupos, refinando la clasificacion.")

    return reglas_traducidas


# ==============================================================================
# 7. GRAFICO DE CONVERGENCIA (LEARNING CURVES)
# ==============================================================================
# Entrenamos XGBoost y MLP de forma individual para capturar como aprenden
# iteracion a iteracion. Esto demuestra que el modelo "se aplanó" y ya no
# necesita mas entrenamiento.
# ==============================================================================

def entrenar_y_graficar_convergencia(X_train, y_train, X_val, y_val, xgb_params, mlp_params):
    """
    Entrena XGBoost y MLP de forma standalone para capturar las curvas de
    aprendizaje (learning curves). Genera convergencia_entrenamiento.png.

    Retorna los modelos entrenados (se usan internamente, el stacking es aparte).
    """
    print("\n--- Entrenando modelos individuales para Curvas de Convergencia ---")
    print("  Esto toma ~30 segundos extra pero demuestra que el modelo aprendio.")

    # --- XGBoost standalone con evaluacion por iteracion ---
    # Usa los mismos hiperparametros del modelo principal.
    print("  Entrenando XGBoost standalone para learning curve...")
    xgb_curve_params = dict(xgb_params)
    xgb_curve_params['eval_metric'] = 'logloss'
    xgb_standalone = xgb.XGBClassifier(**xgb_curve_params)
    xgb_standalone.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    xgb_results = xgb_standalone.evals_result()
    print(f"  [OK] XGBoost: capturada curva de convergencia ({xgb_params.get('n_estimators')} arboles).")

    # --- MLP standalone para capturar loss_curve_ ---
    # Usa los mismos hiperparametros del modelo principal.
    print("  Entrenando MLP standalone para learning curve...")
    mlp_standalone = MLPClassifier(**mlp_params)
    mlp_standalone.fit(X_train, y_train)
    mlp_loss = mlp_standalone.loss_curve_
    print(f"  [OK] MLP: {len(mlp_loss)} epocas registradas.")

    # --- Generar el grafico de convergencia ---
    print("  Generando grafico de convergencia...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: XGBoost — Error vs Numero de Arboles
    ax1 = axes[0]
    epochs_xgb = len(xgb_results['validation_0']['logloss'])
    ax1.plot(range(epochs_xgb), xgb_results['validation_0']['logloss'],
             label='Entrenamiento', color='#3498db', linewidth=2)
    ax1.plot(range(epochs_xgb), xgb_results['validation_1']['logloss'],
             label='Validacion', color='#e74c3c', linewidth=2)
    ax1.set_xlabel('Numero de Arboles (Iteraciones)', fontsize=12)
    ax1.set_ylabel('Log Loss (Error)', fontsize=12)
    ax1.set_title('XGBoost: Convergencia del Error', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Anotar el punto donde se aplana
    min_val_loss = min(xgb_results['validation_1']['logloss'])
    min_idx = xgb_results['validation_1']['logloss'].index(min_val_loss)
    ax1.axvline(x=min_idx, color='#27ae60', linestyle='--', alpha=0.7)
    ax1.annotate(f'Mejor: arbol #{min_idx}\n(Loss={min_val_loss:.4f})',
                 xy=(min_idx, min_val_loss), xytext=(min_idx+20, min_val_loss+0.02),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='#27ae60'),
                 color='#27ae60')

    # Panel 2: MLP — Perdida vs Epocas
    ax2 = axes[1]
    ax2.plot(range(len(mlp_loss)), mlp_loss,
             label='Perdida de Entrenamiento', color='#9b59b6', linewidth=2)
    ax2.set_xlabel('Epocas', fontsize=12)
    ax2.set_ylabel('Loss (Perdida)', fontsize=12)
    ax2.set_title('Red Neuronal (MLP): Convergencia de la Perdida', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Anotar convergencia
    if len(mlp_loss) > 10:
        final_loss = mlp_loss[-1]
        ax2.annotate(f'Final: Loss={final_loss:.4f}\n(epoca {len(mlp_loss)})',
                     xy=(len(mlp_loss)-1, final_loss),
                     xytext=(len(mlp_loss)*0.5, max(mlp_loss)*0.7),
                     fontsize=10, arrowprops=dict(arrowstyle='->', color='#9b59b6'),
                     color='#9b59b6')

    plt.suptitle('Convergencia del Entrenamiento — Evidencia de que el modelo aprendio',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "convergencia_entrenamiento.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafico guardado: {path}")

    return xgb_standalone, mlp_standalone


# ==============================================================================
# 8. SIMULACION DE PREDICCION DE NUEVOS EVENTOS
# ==============================================================================
# Selecciona 5 eventos del set de prueba y muestra la prediccion detallada
# del modelo con intervalo de confianza por evento.
# ==============================================================================

def predecir_nuevos_eventos(modelo_calibrado, X_test, y_test, feature_names, n=5):
    """
    Toma N eventos 'nuevos' del set de prueba y muestra la prediccion detallada
    del modelo con intervalos de confianza.

    Para cada evento imprime:
    - Sus valores fisicos mas relevantes (traducidos al espanol)
    - La probabilidad de que sea un Boson de Higgs
    - Intervalo de confianza estimado via bootstrap rapido
    """
    print(f"\n--- Prediccion de {n} Nuevos Eventos ---")
    print("  (Simulados a partir del set de prueba que el modelo nunca uso para entrenar)\n")

    X_test_arr = np.array(X_test)
    y_test_arr = np.array(y_test)

    # Seleccionar N eventos aleatorios
    np.random.seed(42)  # Reproducibilidad
    indices = np.random.choice(len(X_test), size=n, replace=False)

    resultados = []

    for i, idx in enumerate(indices):
        evento = X_test_arr[idx:idx+1]
        verdad = y_test_arr[idx]

        # Prediccion principal
        prob = modelo_calibrado.predict_proba(evento)[0, 1]

        # Bootstrap rapido para intervalo de confianza del evento
        # Idea: agregar ruido minimo al evento y re-predecir 50 veces
        probs_bootstrap = []
        for _ in range(50):
            ruido = np.random.normal(0, 0.01, size=evento.shape)
            evento_ruidoso = evento + ruido
            p = modelo_calibrado.predict_proba(evento_ruidoso)[0, 1]
            probs_bootstrap.append(p)

        p_low = np.percentile(probs_bootstrap, 16)
        p_high = np.percentile(probs_bootstrap, 84)

        # Clasificacion final
        if prob > 0.7:
            clasificacion = "SENAL (Boson de Higgs probable)"
        elif prob > 0.4:
            clasificacion = "AMBIGUO (requiere mas datos)"
        else:
            clasificacion = "RUIDO (fondo, no es Higgs)"

        verdad_txt = "SENAL REAL (Higgs)" if verdad == 1 else "RUIDO REAL (Fondo)"

        print(f"  {'='*60}")
        print(f"  EVENTO #{i+1}")
        print(f"  {'='*60}")

        # Mostrar las 5 variables mas relevantes del evento
        print(f"  Variables fisicas del evento:")
        for j, feat in enumerate(feature_names[:5]):
            nombre = traducir_variable_corta(feat)
            valor = X_test_arr[idx, j]
            print(f"    - {nombre}: {valor:.4f}")

        print(f"\n  >>> Probabilidad de Higgs: {prob*100:.1f}%")
        print(f"  >>> Intervalo de confianza: ({p_low*100:.1f}% - {p_high*100:.1f}%)")
        print(f"  >>> Clasificacion: {clasificacion}")
        print(f"  >>> Verdad real: {verdad_txt}")
        print()

        resultados.append({
            'evento': i+1,
            'probabilidad': prob,
            'intervalo_bajo': p_low,
            'intervalo_alto': p_high,
            'clasificacion': clasificacion,
            'verdad': verdad_txt
        })

    return resultados


# ==============================================================================
# 9. FUNCIONES DE VISUALIZACION (heredadas de V2)
# ==============================================================================

def plot_calibration_comparison(y_val, proba_uncalibrated, proba_calibrated):
    """Genera el Reliability Diagram comparando antes y despues de calibrar."""
    print("  Generando grafico de calibracion comparativa...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    frac_uncal, mean_uncal = calibration_curve(y_val, proba_uncalibrated, n_bins=10, strategy='uniform')
    ax1.plot(mean_uncal, frac_uncal, 'o-', color='#e74c3c', label='Sin Calibrar', linewidth=2, markersize=8)
    frac_cal, mean_cal = calibration_curve(y_val, proba_calibrated, n_bins=10, strategy='uniform')
    ax1.plot(mean_cal, frac_cal, 's-', color='#2ecc71', label='Calibrado (Isotonic)', linewidth=2, markersize=8)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfecto', alpha=0.7)
    ax1.set_xlabel('Probabilidad Predicha', fontsize=12)
    ax1.set_ylabel('Fraccion Real de Positivos', fontsize=12)
    ax1.set_title('Curva de Calibracion (Reliability Diagram)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(proba_uncalibrated, bins=50, alpha=0.5, color='#e74c3c', label='Sin Calibrar', density=True)
    ax2.hist(proba_calibrated, bins=50, alpha=0.5, color='#2ecc71', label='Calibrado', density=True)
    ax2.set_xlabel('Probabilidad Predicha', fontsize=12)
    ax2.set_ylabel('Densidad', fontsize=12)
    ax2.set_title('Distribucion de Probabilidades', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "calibracion_comparativa.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafico guardado: {path}")


def plot_roc_stacking(y_val, y_pred_proba):
    """Genera la curva ROC del modelo de stacking calibrado."""
    print("  Generando curva ROC del Stacking...")
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'Stacking Calibrado (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Azar (AUC = 0.5)')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC -- Stacking (XGBoost + MLP + Calibracion)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "roc_stacking.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafico guardado: {path}")
    return roc_auc


def plot_distribution_stacking(y_val, y_pred_proba):
    """Distribucion de probabilidades predichas separadas por clase."""
    print("  Generando distribucion de predicciones...")
    plt.figure(figsize=(10, 6))
    y_val_arr = np.array(y_val)
    sns.histplot(y_pred_proba[y_val_arr == 0], color='#3498db', label='Fondo (Background)',
                 kde=True, stat="density", bins=50, alpha=0.5)
    sns.histplot(y_pred_proba[y_val_arr == 1], color='#e74c3c', label='Senal (Higgs)',
                 kde=True, stat="density", bins=50, alpha=0.5)
    plt.title('Distribucion de Probabilidades -- Stacking Calibrado', fontsize=14)
    plt.xlabel('Probabilidad (0=Fondo, 1=Higgs)', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "distribucion_stacking.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafico guardado: {path}")


def plot_bootstrap_distribution(mu_samples, mu_central, mu_low, mu_high):
    """Histograma de las estimaciones de mu obtenidas por Bootstrap."""
    print("  Generando distribucion de Bootstrap de mu...")
    plt.figure(figsize=(10, 6))
    plt.hist(mu_samples, bins=30, color='#9b59b6', alpha=0.7, edgecolor='white')
    plt.axvline(mu_central, color='#e74c3c', linestyle='-', linewidth=2,
                label=f'mu central = {mu_central:.4f}')
    plt.axvline(mu_low, color='#f39c12', linestyle='--', linewidth=1.5,
                label=f'16% = {mu_low:.4f}')
    plt.axvline(mu_high, color='#f39c12', linestyle='--', linewidth=1.5,
                label=f'84% = {mu_high:.4f}')
    plt.axvspan(mu_low, mu_high, alpha=0.15, color='#f39c12', label='Intervalo 68%')
    plt.title('Distribucion Bootstrap de mu (Intensidad de Senal)', fontsize=14)
    plt.xlabel('mu estimado', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "bootstrap_mu.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafico guardado: {path}")


def plot_importancia_variables(stacking_model, feature_names):
    """Grafica la importancia de variables del XGBoost con nombres traducidos."""
    print("  Generando grafico de importancia de variables...")
    try:
        xgb_model = stacking_model.named_estimators_['xgboost']
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15

        nombres_humanos = [traducir_variable_corta(feature_names[i]) for i in indices]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], color='#3498db', alpha=0.8)
        plt.yticks(range(len(indices)), nombres_humanos, fontsize=10)
        plt.xlabel('Importancia (Gain)', fontsize=12)
        plt.title('Importancia de Variables -- Que fisica importa para detectar al Higgs?', fontsize=14)
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, "importancia_variables.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Grafico guardado: {path}")
    except Exception as e:
        print(f"  [AVISO] No se pudo generar importancia de variables: {e}")


def plot_arquitectura_red_neuronal(stacking_model, n_features, n_train, n_val):
    """
    Genera un diagrama de arquitectura del MLP:
    entrada, capas ocultas, numero de neuronas y salida.
    Tambien muestra las dimensiones (N x features/neuronas).
    """
    print("  Generando diagrama de arquitectura de la Red Neuronal (MLP)...")
    try:
        mlp_model = stacking_model.named_estimators_['mlp']
    except Exception as e:
        print(f"  [AVISO] No se pudo acceder al MLP dentro del Stacking: {e}")
        return

    hidden = mlp_model.hidden_layer_sizes
    if isinstance(hidden, int):
        hidden = (hidden,)
    else:
        hidden = tuple(hidden)

    output_neurons = 1
    if hasattr(mlp_model, "coefs_") and mlp_model.coefs_:
        output_neurons = int(mlp_model.coefs_[-1].shape[1])

    n_classes = len(getattr(mlp_model, "classes_", []))
    if n_classes <= 0:
        n_classes = 2

    layer_sizes = [int(n_features)] + [int(x) for x in hidden] + [int(output_neurons)]
    layer_names = ["Entrada"] + [f"Capa oculta {i+1}" for i in range(len(hidden))] + ["Salida"]
    n_layers = len(layer_sizes)

    fig_w = max(16, 3.0 * n_layers)
    fig_h = 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    x_positions = np.linspace(0.08, 0.92, n_layers)
    y_center = 0.55
    box_w = min(0.16, 0.80 / n_layers)
    box_h = 0.30

    colors = {
        "input": "#d6eaf8",
        "hidden": "#f9e79f",
        "output": "#d5f5e3",
        "edge": "#2c3e50",
        "arrow": "#566573",
    }

    def fmt_int(n):
        return f"{int(n):,}".replace(",", ".")

    for i, (x, size, name) in enumerate(zip(x_positions, layer_sizes, layer_names)):
        if i == 0:
            face = colors["input"]
        elif i == n_layers - 1:
            face = colors["output"]
        else:
            face = colors["hidden"]

        rect = FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            linewidth=1.2,
            edgecolor=colors["edge"],
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        if i == 0:
            txt = (
                f"{name}\n"
                f"{fmt_int(size)} variables\n"
                f"Train: ({fmt_int(n_train)}, {fmt_int(size)})\n"
                f"Val: ({fmt_int(n_val)}, {fmt_int(size)})"
            )
        elif i == n_layers - 1:
            txt = (
                f"{name}\n"
                f"{fmt_int(size)} neurona(s)\n"
                f"Salida por evento:\nP(Higgs)\n"
                f"Clases: {n_classes}"
            )
        else:
            txt = (
                f"{name}\n"
                f"{fmt_int(size)} neuronas\n"
                f"Activaciones:\n({fmt_int(n_train)}, {fmt_int(size)})"
            )

        ax.text(
            x,
            y_center,
            txt,
            ha="center",
            va="center",
            fontsize=10,
            color="#1b2631",
            transform=ax.transAxes,
        )

    for i in range(n_layers - 1):
        x1 = x_positions[i] + box_w / 2
        x2 = x_positions[i + 1] - box_w / 2
        y = y_center
        ax.annotate(
            "",
            xy=(x2, y),
            xytext=(x1, y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.8, color=colors["arrow"]),
        )

        w_rows = layer_sizes[i]
        w_cols = layer_sizes[i + 1]
        ax.text(
            (x1 + x2) / 2,
            y + 0.16,
            f"W{i+1}: {fmt_int(w_rows)} x {fmt_int(w_cols)}",
            ha="center",
            va="center",
            fontsize=9,
            color="#34495e",
            transform=ax.transAxes,
        )

        ax.text(
            (x1 + x2) / 2,
            y - 0.16,
            f"({fmt_int(n_train)}, {fmt_int(w_rows)}) -> ({fmt_int(n_train)}, {fmt_int(w_cols)})",
            ha="center",
            va="center",
            fontsize=8,
            color="#5d6d7e",
            transform=ax.transAxes,
        )

    ax.text(
        0.5,
        0.93,
        "Diagrama de Arquitectura de la Red Neuronal MLP",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.88,
        (
            f"Datos de entrada al MLP: Train={fmt_int(n_train)} eventos, "
            f"Val={fmt_int(n_val)} eventos | Variables por evento={fmt_int(n_features)}"
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#2c3e50",
        transform=ax.transAxes,
    )

    path = os.path.join(OUTPUT_DIR, "arquitectura_red_neuronal_mlp.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Grafico guardado: {path}")


def plot_arbol_decision_completo(stacking_model, tree_index=0):
    """
    Dibuja un arbol completo (todas sus ramas) del XGBoost del Stacking.
    No depende de graphviz: parsea el dump de XGBoost y lo renderiza con matplotlib.
    """
    print(f"  Generando arbol de decision completo (XGBoost, tree={tree_index})...")
    try:
        xgb_model = stacking_model.named_estimators_['xgboost']
        booster = xgb_model.get_booster()
        trees_dump = booster.get_dump(with_stats=False)
    except Exception as e:
        print(f"  [AVISO] No se pudo acceder al arbol de XGBoost: {e}")
        return

    if not trees_dump:
        print("  [AVISO] No hay arboles en el modelo XGBoost.")
        return

    if tree_index < 0 or tree_index >= len(trees_dump):
        print(f"  [AVISO] tree_index={tree_index} fuera de rango (0..{len(trees_dump)-1}). Usando 0.")
        tree_index = 0

    tree_text = trees_dump[tree_index]

    split_re = re.compile(
        r"^\s*(\d+):\[(.+?)<([^\]]+)\]\s+yes=(\d+),no=(\d+),missing=(\d+)"
    )
    leaf_re = re.compile(r"^\s*(\d+):leaf=([-\d.eE+]+)")

    nodes = {}
    for raw_line in tree_text.splitlines():
        line = raw_line.strip()
        m_split = split_re.match(line)
        if m_split:
            node_id = int(m_split.group(1))
            nodes[node_id] = {
                "type": "split",
                "feature": m_split.group(2),
                "threshold": m_split.group(3),
                "yes": int(m_split.group(4)),
                "no": int(m_split.group(5)),
                "missing": int(m_split.group(6)),
            }
            continue

        m_leaf = leaf_re.match(line)
        if m_leaf:
            node_id = int(m_leaf.group(1))
            nodes[node_id] = {
                "type": "leaf",
                "leaf": m_leaf.group(2),
            }

    if 0 not in nodes:
        print("  [AVISO] No se pudo parsear el arbol para graficar.")
        return

    positions = {}
    leaf_counter = [0]

    def assign_positions(node_id, depth):
        node = nodes.get(node_id)
        if node is None:
            return None

        if node["type"] == "leaf":
            x = leaf_counter[0]
            leaf_counter[0] += 1
            positions[node_id] = (x, -depth)
            return x

        left_x = assign_positions(node["yes"], depth + 1)
        right_x = assign_positions(node["no"], depth + 1)

        # Si por alguna razon falta una rama, posicionamos el nodo en la rama disponible.
        if left_x is None and right_x is None:
            x = leaf_counter[0]
            leaf_counter[0] += 1
        elif left_x is None:
            x = right_x
        elif right_x is None:
            x = left_x
        else:
            x = (left_x + right_x) / 2.0

        positions[node_id] = (x, -depth)
        return x

    assign_positions(0, 0)

    if not positions:
        print("  [AVISO] No se pudieron calcular posiciones del arbol.")
        return

    leaf_count = max(1, leaf_counter[0])
    max_depth = int(max(-pos[1] for pos in positions.values()))

    fig_w = min(max(16, leaf_count * 1.1), 90)
    fig_h = min(max(8, (max_depth + 1) * 1.8), 42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(f"Arbol de Decision Completo (XGBoost tree {tree_index})", fontsize=14)
    ax.axis("off")

    for node_id, node in nodes.items():
        if node_id not in positions:
            continue

        x, y = positions[node_id]

        if node["type"] == "split":
            for edge_label, child_id in (("yes", node["yes"]), ("no", node["no"])):
                if child_id not in positions:
                    continue
                cx, cy = positions[child_id]
                ax.plot([x, cx], [y, cy], color="#7f8c8d", linewidth=1)
                ax.text((x + cx) / 2.0, (y + cy) / 2.0 + 0.07, edge_label, fontsize=8,
                        color="#2c3e50", ha="center", va="center")

            if node["missing"] == node["yes"]:
                missing_txt = "missing->yes"
            elif node["missing"] == node["no"]:
                missing_txt = "missing->no"
            else:
                missing_txt = f"missing->{node['missing']}"

            label = f"{node['feature']} < {node['threshold']}\n{missing_txt}"
            ax.text(
                x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#34495e", linewidth=0.8)
            )
        else:
            label = f"leaf={node['leaf']}"
            ax.text(
                x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#d5f5e3", edgecolor="#1e8449", linewidth=0.8)
            )

    ax.set_xlim(-1, leaf_count)
    ax.set_ylim(-(max_depth + 1), 1)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, f"arbol_decision_completo_tree_{tree_index}.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    txt_path = os.path.join(OUTPUT_DIR, f"arbol_decision_completo_tree_{tree_index}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(tree_text)

    print(f"  [OK] Arbol completo guardado: {png_path}")
    print(f"  [OK] Dump textual del arbol: {txt_path}")


# ==============================================================================
# 10. REPORTE DETALLADO (LEGIBLE PARA NO-FISICOS)
# ==============================================================================

def generate_final_report(
    auc_score, accuracy, brier_uncal, brier_cal,
    mu_central, mu_low, mu_high,
    n_train, n_val, sample_size, elapsed_time,
    reglas_arbol, resultados_prediccion, feature_names, modelo_cargado,
    modo_hiperparametros, xgb_params, mlp_params, meta_params
):
    """Genera un reporte .txt completo, legible para personas sin conocimiento de fisica."""

    # Seccion de predicciones
    predicciones_txt = ""
    for r in resultados_prediccion:
        predicciones_txt += f"""
  Evento #{r['evento']}:
    Probabilidad de Higgs: {r['probabilidad']*100:.1f}%
    Intervalo de confianza: ({r['intervalo_bajo']*100:.1f}% - {r['intervalo_alto']*100:.1f}%)
    Clasificacion del modelo: {r['clasificacion']}
    Resultado real: {r['verdad']}
"""

    # Seccion de reglas
    reglas_txt = "\n".join([f"  {r}" for r in reglas_arbol])

    # Seccion del diccionario (top 10 variables)
    diccionario_txt = ""
    for feat in feature_names[:10]:
        desc = traducir_variable(feat)
        diccionario_txt += f"  {feat:30s} -> {desc}\n"

    modo = "CARGADO desde disco (sin re-entrenar)" if modelo_cargado else "ENTRENADO desde cero"
    layers_txt = format_mlp_layers(mlp_params.get('hidden_layer_sizes', ()))
    mejora_cal = ((brier_uncal - brier_cal) / brier_uncal * 100) if brier_uncal > 0 else 0.0

    report = f"""
==============================================================================
   REPORTE FINAL -- SISTEMA DE PRODUCCION HIGGSML
   (Escrito para ser entendido por cualquier persona)
==============================================================================
Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}
Tiempo Total: {elapsed_time:.1f} segundos
Modo: {modo}
Datos: {n_train + n_val} eventos (Muestra: {sample_size if sample_size else 'Completa'})

==============================================================================
1. QUE HACE ESTE SISTEMA?
==============================================================================
Este sistema intenta distinguir entre dos tipos de eventos en el acelerador
de particulas del CERN:

  - SENAL: Eventos donde se produjo un Boson de Higgs (la particula que
    da masa a todo lo que existe). Encontrarla fue el mayor descubrimiento
    de la fisica moderna (2012, Premio Nobel).

  - RUIDO: Eventos de fondo, procesos fisicos normales que se parecen
    al Higgs pero no lo son (como ruido en una radio).

El modelo usa inteligencia artificial para aprender a distinguir entre
ambos, analizando las propiedades fisicas de cada evento.

==============================================================================
2. COMO FUNCIONA EL MODELO? (Arquitectura)
==============================================================================
Usamos un "Consejo de Sabios" (Stacking Classifier):

  Experto 1: XGBoost ({xgb_params.get('n_estimators')} arboles, prof={xgb_params.get('max_depth')}, lr={xgb_params.get('learning_rate')})
    - Funciona como un cuestionario de preguntas SI/NO sobre las variables.
    - Ejemplo: "La masa estimada es mayor a 120 GeV? SI -> probablemente Higgs"
    - Es rapido y preciso con datos numericos.

  Experto 2: Red Neuronal MLP (capas: {layers_txt}, lr={mlp_params.get('learning_rate_init')}, alpha={mlp_params.get('alpha')})
    - Funciona como un cerebro artificial que aprende patrones complejos.
    - Captura relaciones que los arboles no pueden ver.

  Juez Final: Regresion Logistica (C={meta_params.get('C')}, max_iter={meta_params.get('max_iter')})
    - Aprende CUANDO confiar mas en el Arbol y cuando en la Red.
    - Es simple a proposito: su trabajo es combinar, no inventar.

  Modo de seleccion de hiperparametros: {modo_hiperparametros.upper()}
    - MANUAL: parametros fijos definidos por el usuario.
    - AUTO: el sistema hace busqueda automatica antes de entrenar.

  Calibracion: Despues de decidir, ajustamos las probabilidades para que
  sean honestas (si dice "80% Higgs", que realmente sea 80%).

==============================================================================
3. QUE TAN BUENO ES? (Metricas)
==============================================================================
  AUC (Area bajo la curva ROC):     {auc_score:.4f}
    -> 1.0 = perfecto, 0.5 = adivinando al azar. {auc_score:.4f} es {'excelente' if auc_score > 0.9 else 'bueno' if auc_score > 0.8 else 'aceptable'}.

  Accuracy (Exactitud):              {accuracy:.4f}
    -> De cada 100 eventos, acierta {accuracy*100:.0f}.

  Brier Score (antes de calibrar):   {brier_uncal:.4f}
  Brier Score (despues de calibrar): {brier_cal:.4f}
    -> Mejora del {mejora_cal:.1f}% en la honestidad de las probabilidades.

==============================================================================
4. ESTIMACION DE LA INTENSIDAD DE SENAL (mu)
==============================================================================
  mu es un numero que indica cuanta senal de Higgs detectamos:
    mu = 1.0 -> Exactamente lo que predice la teoria (Modelo Estandar)
    mu > 1.0 -> Vemos MAS senal de la esperada (nueva fisica!)
    mu < 1.0 -> Vemos MENOS senal de la esperada

  Nuestro resultado:
    mu = {mu_central:.4f} +/- {(mu_high - mu_low)/2:.4f}
    Intervalo 68%: [{mu_low:.4f}, {mu_high:.4f}]
    (Calculado con {N_BOOTSTRAP} iteraciones de Bootstrap)

  Comparativa con la competencia:
    Metrica             Referencia    Nuestro Modelo
    Cobertura           68%           68% (Bootstrap)
    AUC                 ~0.93-0.95    {auc_score:.4f}
    Calibracion         ~0.06-0.08    {brier_cal:.4f}

==============================================================================
5. DECISIONES DEL MODELO (Reglas del Arbol en Lenguaje Humano)
==============================================================================
  El arbol XGBoost toma decisiones como un doctor que hace preguntas:

{reglas_txt}

  Interpretacion: Cada regla es un "corte" en una variable fisica.
  El modelo aprende que combinacion de cortes separa mejor la senal
  del Higgs del ruido de fondo.

==============================================================================
6. PREDICCIONES DE EVENTOS INDIVIDUALES
==============================================================================
  Aqui mostramos como el modelo clasifica 5 eventos nunca vistos:
{predicciones_txt}

==============================================================================
7. DICCIONARIO DE VARIABLES (Glosario para No-Fisicos)
==============================================================================
  Las variables que usa el modelo y su significado en espanol:

{diccionario_txt}

==============================================================================
8. ARCHIVOS GENERADOS
==============================================================================
  * modelo_entrenado_higgs.pkl      -- Modelo guardado (no re-entrenar)
  * convergencia_entrenamiento.png  -- Prueba de que el modelo aprendio
  * calibracion_comparativa.png     -- Antes vs despues de calibrar
  * roc_stacking.png                -- Curva ROC (que tan bien separa)
  * distribucion_stacking.png       -- Separacion senal vs fondo
  * bootstrap_mu.png                -- Distribucion de mu por Bootstrap
  * importancia_variables.png       -- Que variables importan mas
  * arquitectura_red_neuronal_mlp.png -- Entrada, capas, neuronas y salida de la red
  * arbol_decision_completo_tree_0.png -- Arbol de decision completo (todas las ramas)
  * arbol_decision_completo_tree_0.txt -- Dump textual completo del mismo arbol
  * REPORTE_PRODUCCION_FINAL.txt    -- Este reporte

==============================================================================
                    FIN DEL REPORTE
==============================================================================
"""

    path = os.path.join(OUTPUT_DIR, "REPORTE_PRODUCCION_FINAL.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [OK] Reporte guardado en: {path}")
    return report


# ==============================================================================
# UTILIDADES DE PROGRESO (TIEMPO + ETA)
# ==============================================================================

def _format_duration(seconds):
    """Convierte segundos a texto legible (ej: 2h 03m 11s)."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class ProgressTracker:
    """
    Muestra avance por pasos, tiempo transcurrido y ETA aproximado.
    La ETA se calcula con el ritmo observado de los pasos completados.
    """
    def __init__(self, steps):
        # steps = [{'key': 'load_data', 'label': 'Cargando datos', 'weight': 0.10}, ...]
        self.steps = list(steps)
        self.step_lookup = {s['key']: s for s in self.steps}
        self.total_weight = float(sum(s['weight'] for s in self.steps))
        self.completed_weight = 0.0
        self.start_time = time.time()
        self.current_step_key = None
        self.current_step_start = None

    def _elapsed_seconds(self):
        return time.time() - self.start_time

    def _eta_seconds(self):
        # ETA = (segundos por unidad de peso) * peso restante
        if self.completed_weight <= 0:
            return None
        sec_per_weight = self._elapsed_seconds() / self.completed_weight
        remaining_weight = max(0.0, self.total_weight - self.completed_weight)
        return sec_per_weight * remaining_weight

    def start_step(self, step_key, paso_txt):
        step = self.step_lookup[step_key]
        self.current_step_key = step_key
        self.current_step_start = time.time()
        elapsed_txt = _format_duration(self._elapsed_seconds())
        eta_sec = self._eta_seconds()
        eta_txt = _format_duration(eta_sec) if eta_sec is not None else "calculando..."

        print(f"\n[{paso_txt}] {step['label']}", flush=True)
        print(f"  Tiempo transcurrido: {elapsed_txt} | ETA restante aprox: {eta_txt}", flush=True)

    def end_step(self, step_key):
        if self.current_step_key != step_key:
            # Permitir cierre robusto aun si se llama fuera de orden.
            self.current_step_key = step_key
            self.current_step_start = self.current_step_start or time.time()

        step = self.step_lookup[step_key]
        step_time = time.time() - self.current_step_start
        self.completed_weight += float(step['weight'])
        self.completed_weight = min(self.completed_weight, self.total_weight)

        progress_pct = (self.completed_weight / self.total_weight * 100.0) if self.total_weight > 0 else 100.0
        elapsed_txt = _format_duration(self._elapsed_seconds())
        eta_sec = self._eta_seconds()
        eta_txt = _format_duration(eta_sec) if eta_sec is not None else "calculando..."

        print(
            f"  [OK] Paso completado en {_format_duration(step_time)} | "
            f"Avance total: {progress_pct:.1f}% | "
            f"Transcurrido: {elapsed_txt} | ETA: {eta_txt}",
            flush=True
        )

        self.current_step_key = None
        self.current_step_start = None

    def skip_step(self, step_key, reason):
        """Si un paso no aplica (ej: modelo ya cargado), se descuenta del total para mantener ETA consistente."""
        step = self.step_lookup[step_key]
        old_weight = float(step['weight'])
        if old_weight <= 0:
            return
        step['weight'] = 0.0
        self.total_weight -= old_weight
        self.total_weight = max(self.total_weight, 1e-9)
        print(f"  [SKIP] {step['label']} -> {reason}", flush=True)


# ==============================================================================
# EJECUCION PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*70, flush=True)
    print("  PIPELINE PRODUCCION FINAL -- SISTEMA PROFESIONAL HIGGSML", flush=True)
    print("="*70, flush=True)
    start_time = time.time()

    sample_txt = f"{SAMPLE_SIZE:,}".replace(",", ".") if SAMPLE_SIZE else "TODOS"
    print(
        f"  Configuracion: SAMPLE_SIZE={sample_txt} (total), "
        f"FORZAR_ENTRENAMIENTO={FORZAR_ENTRENAMIENTO}, "
        f"MODO_AJUSTE_HIPERPARAMETROS={MODO_AJUSTE_HIPERPARAMETROS}",
        flush=True
    )
    print(f"  SOLO_INFERENCIA={SOLO_INFERENCIA}", flush=True)
    print(f"  Carpeta de salida de esta corrida: {OUTPUT_DIR}", flush=True)

    model_load_path = MODELO_CARGA_PATH or find_latest_model_path(OUTPUT_ROOT_DIR)
    print(f"  Modelo para cargar: {model_load_path if model_load_path else 'NO_ENCONTRADO'}", flush=True)

    # Pesos aproximados por etapa para ETA global.
    pipeline_steps = [
        {"key": "load_data", "label": "Carga de datos", "weight": 0.10},
        {"key": "model_check", "label": "Verificacion de modelo persistido", "weight": 0.02},
        {"key": "hyperparams", "label": "Seleccion de hiperparametros", "weight": 0.28},
        {"key": "augmentation", "label": "Data augmentation fisico", "weight": 0.05},
        {"key": "convergence", "label": "Curvas de convergencia", "weight": 0.12},
        {"key": "stacking_train", "label": "Entrenamiento del Stacking", "weight": 0.26},
        {"key": "calibration", "label": "Calibracion de probabilidades", "weight": 0.07},
        {"key": "evaluation", "label": "Evaluacion y graficos", "weight": 0.07},
        {"key": "final_report", "label": "Prediccion final + reporte", "weight": 0.03},
    ]
    progress = ProgressTracker(pipeline_steps)

    # =========================================================================
    # PASO 1: Carga de Datos
    # =========================================================================
    progress.start_step("load_data", "Paso 1/9")
    loader = HiggsDataLoader(DATA_DIR, sample_size=SAMPLE_SIZE)
    X_train, X_val, y_train, y_val, w_train, w_val = loader.load_data()
    feature_names = X_train.columns.tolist()
    xgb_params, mlp_params, meta_params = get_default_hyperparams()
    tuning_info = {'mode': 'manual'}
    modo_hiperparametros_usado = str(MODO_AJUSTE_HIPERPARAMETROS).strip().lower()

    print(f"  Dimensiones de X_train: {X_train.shape}", flush=True)
    print(f"  Proporcion de senal: {y_train.mean():.4f}", flush=True)
    print(f"  Variables: {len(feature_names)}", flush=True)
    progress.end_step("load_data")

    # =========================================================================
    # PASO 2: Verificar si el modelo ya existe (PERSISTENCIA)
    # =========================================================================
    modelo_cargado = False
    progress.start_step("model_check", "Paso 2/9")

    if model_load_path and os.path.exists(model_load_path) and not FORZAR_ENTRENAMIENTO:
        print("\n[Paso 2/9] Modelo encontrado en disco -- cargando...", flush=True)
        modelo_data = cargar_modelo(model_load_path)
        stacking_model = modelo_data['stacking_model']
        calibrated_model = modelo_data['calibrated_model']
        xgb_params = modelo_data.get('xgb_params', xgb_params)
        mlp_params = modelo_data.get('mlp_params', mlp_params)
        meta_params = modelo_data.get('meta_params', meta_params)
        tuning_info = modelo_data.get('tuning_info', tuning_info)
        modo_hiperparametros_usado = modelo_data.get('modo_hiperparametros', modo_hiperparametros_usado)
        modelo_cargado = True
        print("  Se salto el entrenamiento completo!", flush=True)
    else:
        if SOLO_INFERENCIA:
            raise FileNotFoundError(
                "SOLO_INFERENCIA=True y no se encontro modelo entrenado para cargar. "
                "Define HIGGS_MODELO_PATH o verifica que exista un 'modelo_entrenado_higgs.pkl' en "
                f"'{OUTPUT_ROOT_DIR}'."
            )
        if FORZAR_ENTRENAMIENTO:
            print("\n[Paso 2/9] FORZAR_ENTRENAMIENTO=True -- entrenando desde cero...", flush=True)
        else:
            print("\n[Paso 2/9] No se encontro modelo guardado -- entrenando desde cero...", flush=True)

    progress.end_step("model_check")

    if modelo_cargado:
        progress.skip_step("hyperparams", "Se cargo modelo existente.")
        progress.skip_step("augmentation", "Se cargo modelo existente.")
        progress.skip_step("convergence", "Se cargo modelo existente.")
        progress.skip_step("stacking_train", "Se cargo modelo existente.")
        progress.skip_step("calibration", "Se cargo modelo existente.")
    else:
        # --- Seleccion de Hiperparametros (manual o auto) ---
        progress.start_step("hyperparams", "Paso 3/9")
        print("\n[Paso 3/9] Seleccionando hiperparametros...", flush=True)
        xgb_params, mlp_params, meta_params, tuning_info = seleccionar_hiperparametros(X_train, y_train)
        modo_hiperparametros_usado = tuning_info.get('mode', modo_hiperparametros_usado)
        progress.end_step("hyperparams")

        # --- Data Augmentation ---
        progress.start_step("augmentation", "Paso 4/9")
        print("\n[Paso 4/9] Aplicando Data Augmentation Fisico...", flush=True)
        print("  Simulando errores de calibracion del detector (ruido 1%).", flush=True)
        X_train_augmented = add_physics_noise(X_train, noise_level=0.01)
        progress.end_step("augmentation")

        # --- Curvas de Convergencia (entrenar modelos standalone) ---
        progress.start_step("convergence", "Paso 5/9")
        print("\n[Paso 5/9] Generando Curvas de Convergencia...", flush=True)
        entrenar_y_graficar_convergencia(
            X_train_augmented, y_train, X_val, y_val, xgb_params, mlp_params
        )
        progress.end_step("convergence")

        # --- Entrenar Stacking ---
        progress.start_step("stacking_train", "Paso 6/9")
        print("\n[Paso 6/9] Entrenando Stacking Classifier...", flush=True)
        stacking_model = build_stacking_model(xgb_params, mlp_params, meta_params)
        print("\n  Entrenando... (esto puede tardar varios minutos)", flush=True)
        stacking_model.fit(X_train_augmented, y_train)
        print("  [OK] Stacking entrenado exitosamente.", flush=True)
        progress.end_step("stacking_train")

        # --- Calibracion ---
        progress.start_step("calibration", "Paso 7/9")
        print("\n[Paso 7/9] Calibrando probabilidades...", flush=True)
        n_cal = len(X_val) // 2
        X_calibration = X_val.iloc[:n_cal]
        y_calibration = y_val.iloc[:n_cal]
        calibrated_model = calibrate_model(stacking_model, X_calibration, y_calibration)

        # --- Guardar modelo ---
        modelo_data = {
            'stacking_model': stacking_model,
            'calibrated_model': calibrated_model,
            'feature_names': feature_names,
            'fecha_entrenamiento': time.strftime("%Y-%m-%d %H:%M:%S"),
            'sample_size': SAMPLE_SIZE,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'modo_hiperparametros': modo_hiperparametros_usado,
            'xgb_params': xgb_params,
            'mlp_params': mlp_params,
            'meta_params': meta_params,
            'tuning_info': tuning_info
        }
        guardar_modelo(modelo_data, MODELO_PATH)
        progress.end_step("calibration")

    # =========================================================================
    # PASO 8: Evaluacion y Graficos (siempre se ejecuta)
    # =========================================================================
    progress.start_step("evaluation", "Paso 8/9")
    print("\n[Paso 8/9] Evaluando modelo y generando graficos...", flush=True)

    # Probabilidades
    proba_uncalibrated = stacking_model.predict_proba(X_val)[:, 1]
    proba_calibrated_full = calibrated_model.predict_proba(X_val)[:, 1]

    n_cal = len(X_val) // 2
    X_test_final = X_val.iloc[n_cal:]
    y_test_final = y_val.iloc[n_cal:]
    w_test_final = w_val.iloc[n_cal:]
    proba_calibrated_test = calibrated_model.predict_proba(X_test_final)[:, 1]

    # Metricas
    y_pred_labels = (proba_calibrated_test > 0.5).astype(int)
    accuracy = accuracy_score(y_test_final, y_pred_labels)
    brier_uncal = brier_score_loss(y_val, proba_uncalibrated)
    brier_cal = brier_score_loss(y_val, proba_calibrated_full)

    print(f"\n  Accuracy: {accuracy:.4f}", flush=True)
    print(f"  Brier Score (Sin Calibrar): {brier_uncal:.4f}", flush=True)
    print(f"  Brier Score (Calibrado):    {brier_cal:.4f}", flush=True)

    print("\n  --- Reporte de Clasificacion ---", flush=True)
    print(classification_report(y_test_final, y_pred_labels,
                                target_names=['Fondo (b)', 'Senal (s)']), flush=True)

    # Bootstrap
    mu_central, mu_low, mu_high, mu_samples = bootstrap_mu_estimation(
        y_test_final, proba_calibrated_test, w_test_final, n_iterations=N_BOOTSTRAP
    )

    # Graficos
    plot_calibration_comparison(y_val, proba_uncalibrated, proba_calibrated_full)
    auc_score = plot_roc_stacking(y_val, proba_calibrated_full)
    plot_distribution_stacking(y_val, proba_calibrated_full)
    plot_bootstrap_distribution(mu_samples, mu_central, mu_low, mu_high)
    plot_importancia_variables(stacking_model, feature_names)
    plot_arquitectura_red_neuronal(
        stacking_model,
        n_features=len(feature_names),
        n_train=len(X_train),
        n_val=len(X_val),
    )
    plot_arbol_decision_completo(stacking_model, tree_index=TREE_INDEX_TO_PLOT)

    # Extraer reglas del arbol (traducidas)
    reglas_arbol = extraer_reglas_arbol(stacking_model, feature_names)
    progress.end_step("evaluation")

    # =========================================================================
    # PASO 9: Prediccion de Nuevos Eventos y Reporte
    # =========================================================================
    progress.start_step("final_report", "Paso 9/9")
    print("\n[Paso 9/9] Prediccion de nuevos eventos y reporte final...", flush=True)

    resultados_prediccion = predecir_nuevos_eventos(
        calibrated_model, X_test_final, y_test_final, feature_names, n=5
    )

    elapsed = time.time() - start_time

    # Reporte final
    report = generate_final_report(
        auc_score=auc_score,
        accuracy=accuracy,
        brier_uncal=brier_uncal,
        brier_cal=brier_cal,
        mu_central=mu_central,
        mu_low=mu_low,
        mu_high=mu_high,
        n_train=len(X_train),
        n_val=len(X_val),
        sample_size=SAMPLE_SIZE,
        elapsed_time=elapsed,
        reglas_arbol=reglas_arbol,
        resultados_prediccion=resultados_prediccion,
        feature_names=feature_names,
        modelo_cargado=modelo_cargado,
        modo_hiperparametros=modo_hiperparametros_usado,
        xgb_params=xgb_params,
        mlp_params=mlp_params,
        meta_params=meta_params
    )
    progress.end_step("final_report")

    # Resumen final
    print("\n" + "="*70, flush=True)
    print("  [OK] PIPELINE PRODUCCION FINAL COMPLETADO", flush=True)
    print("="*70, flush=True)
    print(f"  Tiempo total: {elapsed:.1f} segundos", flush=True)
    print(f"  Modo: {'CARGADO' if modelo_cargado else 'ENTRENADO'}", flush=True)
    print(f"  Modo Hiperparametros: {modo_hiperparametros_usado.upper()}", flush=True)
    print(f"  AUC: {auc_score:.4f}", flush=True)
    print(f"  Estimacion de Mu: {mu_central:.4f} +/- {(mu_high - mu_low)/2:.4f}", flush=True)
    print(f"  Resultados en: {OUTPUT_DIR}", flush=True)
    print("="*70, flush=True)
