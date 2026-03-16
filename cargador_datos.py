import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HiggsDataLoader:
    def __init__(self, data_dir, sample_size=None, random_state=42):
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Carga características (Parquet), etiquetas (txt) y pesos (txt).
        Devuelve: X_train, X_val, y_train, y_val, w_train, w_val, scaler
        """
        print(f"--- [Data Loader] Loading data from: {self.data_dir} ---")
        
        # Rutas
        features_path = os.path.join(self.data_dir, "data", "data.parquet")
        labels_path = os.path.join(self.data_dir, "labels", "data.labels")
        weights_path = os.path.join(self.data_dir, "weights", "data.weights")
        
        if self.sample_size:
            # OPTIMIZACION: Si se muestrea, leer un poco más de lo necesario (p. ej. 2x) y luego muestrear, 
            # para evitar cargar 20 GB de datos en RAM.
            limit = self.sample_size * 5  # Cargar 5x para asegurar buena mezcla al barajar
            print(f"[Optimization] Loading partial data (limit={limit}) to save RAM...")
            
            # Parquet: leer las primeras N filas
            # Podemos usar read_parquet con slicing si se soporta, o leer con pyarrow y cortar
            try:
                # pandas read_parquet no tiene 'nrows', pero podemos leer el archivo como tabla pyarrow primero
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(features_path)
                
                # Calcular cuántos grupos de filas se necesitan
                num_rows = pf.metadata.num_rows
                num_groups = pf.num_row_groups
                rows_per_group = num_rows / num_groups
                groups_needed = int(np.ceil(limit / rows_per_group))
                groups_needed = max(1, min(groups_needed, num_groups)) # Al menos 1, como máximo todos
                
                print(f"[Parquet] Reading {groups_needed} row groups (out of {num_groups}) to get approx {int(groups_needed*rows_per_group)} rows...")
                
                # Leer los grupos de filas seleccionados
                X = pf.read_row_groups(range(0, groups_needed)).to_pandas()
                
                if len(X) > limit:
                     X = X.iloc[:limit]
            except Exception as e:
                print(f"Parquet slice failed: {e}. Reading full (might be slow)...")
                X = pd.read_parquet(features_path, engine='pyarrow')

            # CSV: leer con nrows
            y = pd.read_csv(labels_path, header=None, names=['label'], nrows=len(X))
            w = pd.read_csv(weights_path, header=None, names=['weight'], nrows=len(X))
        else:
             # Cargar datos completos
            X = pd.read_parquet(features_path, engine='pyarrow')
            y = pd.read_csv(labels_path, header=None, names=['label'])
            w = pd.read_csv(weights_path, header=None, names=['weight'])
        
        # Verificar alineación
        if len(X) != len(y) or len(X) != len(w):
            raise ValueError(f"Size mismatch! X: {len(X)}, y: {len(y)}, w: {len(w)}")
            
        print(f"Total events loaded: {len(X)}")
        
        # 3. Muestreo (si se solicita)
        if self.sample_size and len(X) > self.sample_size:
            print(f"Sampling {self.sample_size} random events...")
            # Usamos sample para mezclar y limitar el tamaño al mismo tiempo
            indices = X.sample(n=self.sample_size, random_state=self.random_state).index
            X = X.loc[indices]
            y = y.loc[indices]
            w = w.loc[indices]
        else:
            # Si no se muestrea, asumimos que aún podría ser necesario mezclar por la estructura del desafío
            # Pero sklearn train_test_split mezcla por defecto, así que está bien.
            pass

        # 4. Dividir datos (entrenamiento/validación)
        # Estratificar no es estrictamente necesario en datasets grandes, pero es buena práctica
        print("Splitting data into Train (80%) and Validation (20%)...")
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, w, test_size=0.2, random_state=self.random_state, shuffle=True
        )
        
        # 5. Normalización (ajustar en entrenamiento, aplicar en validación)
        # Crucial para redes neuronales
        print("Normalizing features (StandardScaler)...")
        # Asegurar que solo escalemos variables numéricas (excluir IDs si existen, aunque parquet suele estar limpio)
        # Usamos todas las columnas por ahora, ya que parecen ser variables físicas
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convertir de vuelta a tabla de datos para XGBoost (usa nombres de columnas)
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_val = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)
        
        print("Data Loading Complete.")
        return X_train, X_val, y_train['label'], y_val['label'], w_train['weight'], w_val['weight']



