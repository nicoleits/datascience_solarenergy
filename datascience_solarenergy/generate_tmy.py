import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Importar os

def clean_data(df):
    """Limpia columnas específicas del DataFrame y devuelve un log de las acciones."""
    print("Iniciando limpieza de datos...")
    cleaning_log = [] # Lista para guardar el resumen
    cleaning_log.append("Data Cleaning Summary:")
    cleaning_log.append("=====================")
    
    # Columnas a limpiar y acciones
    cols_to_clean = {
        'GHI': {'mark_upper_as_nan': 1250, 'interpolate': True, 'clip_lower': 0}, 
        'DNI': {'interpolate': True, 'clip_lower': 0, 'clip_upper': None}, 
        'DHI': {'interpolate': True, 'clip_lower': 0, 'clip_upper': None}, 
        'Tdry': {'interpolate': True, 'clip_lower': None, 'clip_upper': None},
        'Wspd': {'interpolate': True, 'clip_lower': 0, 'clip_upper': None} 
    }

    for col, actions in cols_to_clean.items():
        if col in df.columns:
            log_entry = [f"\nColumn: {col}"]
            print(f"Limpiando {col}...")
            initial_nan = df[col].isnull().sum()
            log_entry.append(f"  Initial NaNs: {initial_nan}")
            marked_as_nan = 0

            # 1. Marcar outliers superiores como NaN
            upper_nan_limit = actions.get('mark_upper_as_nan')
            if upper_nan_limit is not None:
                outlier_mask = df[col] > upper_nan_limit
                marked_as_nan = outlier_mask.sum()
                if marked_as_nan > 0:
                    df.loc[outlier_mask, col] = np.nan
                    print(f"  Marcados {marked_as_nan} valores > {upper_nan_limit} como NaN.")
                    log_entry.append(f"  Outliers > {upper_nan_limit} marked as NaN: {marked_as_nan}")

            # 2. Interpolar NaNs
            if actions.get('interpolate'):
                df[col] = df[col].interpolate(method='linear')
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(method='bfill', inplace=True)
                final_nan = df[col].isnull().sum()
                print(f"  NaNs interpolados/rellenados (originales: {initial_nan}, marcados: {marked_as_nan}, final: {final_nan}).")
                log_entry.append(f"  Interpolated/Filled NaNs (Original: {initial_nan}, Marked: {marked_as_nan}, Final: {final_nan})")
            
            # 3. Clip inferior
            lower_limit = actions.get('clip_lower')
            if lower_limit is not None:
                negative_mask = df[col] < lower_limit
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    df[col] = df[col].clip(lower=lower_limit)
                    print(f"  Corregidos {negative_count} valores < {lower_limit} en {col}.")
                    log_entry.append(f"  Values < {lower_limit} clipped: {negative_count}")
            
            cleaning_log.extend(log_entry)

    print("Limpieza de datos completada.")
    cleaning_log.append("\nLimpieza de datos completada.")
    return df, cleaning_log # Devolver DataFrame y log

def calculate_fs_stat(series1, series2, num_bins=100):
    """Calcula la estadística Finkelstein-Schafer entre dos series."""
    # Calcula los CDFs empíricos
    counts1, bin_edges = np.histogram(series1, bins=num_bins)
    cdf1 = np.cumsum(counts1) / series1.size
    counts2, _ = np.histogram(series2, bins=bin_edges) # Usar los mismos bins
    cdf2 = np.cumsum(counts2) / series2.size
    # Calcula la diferencia absoluta media (FS stat)
    fs_stat = np.sum(np.abs(cdf1 - cdf2)) / num_bins
    return fs_stat

def generate_tmy(df):
    """Genera el DataFrame TMY a partir de los datos horarios limpios."""
    print("\nIniciando generación de TMY...")
    
    # Variables para la selección TMY y sus pesos
    tmy_vars = {
        'GHI': 0.5,
        'DNI': 0.2, # Ponderación menor si hay más incertidumbre/NaNs originales
        'DHI': 0.1, # Ponderación menor si hay más incertidumbre/NaNs originales
        'Tdry': 0.2
    }
    
    available_years = df.index.year.unique()
    print(f"Años disponibles en los datos: {available_years.tolist()}")
    if len(available_years) < 3:
         print("Advertencia: Se recomienda tener al menos 3 años de datos para un TMY robusto.")

    monthly_stats = {}
    long_term_stats = {}

    # --- 1. Calcular estadísticas mensuales y a largo plazo --- 
    print("Calculando estadísticas mensuales y a largo plazo...")
    for month in range(1, 13):
        long_term_monthly_data = {}
        for var in tmy_vars:
            long_term_monthly_data[var] = pd.Series(dtype=float)

        monthly_stats[month] = {}
        
        for year in available_years:
            # Seleccionar datos para el mes/año específico
            mask = (df.index.month == month) & (df.index.year == year)
            month_year_data = df[mask]
            
            if not month_year_data.empty:
                monthly_stats[month][year] = {}
                for var in tmy_vars:
                    series = month_year_data[var]
                    monthly_stats[month][year][var] = {
                        'mean': series.mean(),
                        'data': series # Guardamos la serie para el cálculo FS
                    }
                    # Acumular datos para promedio a largo plazo
                    long_term_monthly_data[var] = pd.concat([long_term_monthly_data[var], series])

        # Calcular estadísticas a largo plazo para el mes
        long_term_stats[month] = {}
        for var in tmy_vars:
            series_lt = long_term_monthly_data[var]
            long_term_stats[month][var] = {
                'mean': series_lt.mean(),
                'data': series_lt
            }

    # --- 2. Calcular diferencias ponderadas y seleccionar meses --- 
    print("Calculando diferencias y seleccionando meses típicos...")
    selected_months = {}
    for month in range(1, 13):
        best_year = None
        min_weighted_diff = float('inf')
        
        if month not in monthly_stats: # Si no hay datos para este mes
            print(f"Advertencia: No hay datos para el mes {month}, no se puede seleccionar mes típico.")
            continue

        for year in monthly_stats[month]:
            total_weighted_diff = 0
            for var, weight in tmy_vars.items():
                # Diferencia de medias (normalizada por la media a largo plazo para evitar sesgos de escala)
                mean_diff = abs(monthly_stats[month][year][var]['mean'] - long_term_stats[month][var]['mean'])
                if long_term_stats[month][var]['mean'] != 0:
                     mean_diff /= abs(long_term_stats[month][var]['mean'])
                
                # Diferencia de distribución (FS Stat)
                fs_diff = calculate_fs_stat(monthly_stats[month][year][var]['data'], long_term_stats[month][var]['data'])

                # Ponderar (aquí simplemente sumamos, se puede ajustar)
                # Damos más peso a la diferencia de distribución (FS)
                total_weighted_diff += weight * (0.3 * mean_diff + 0.7 * fs_diff) 

            # Actualizar si es el mejor año hasta ahora para este mes
            if total_weighted_diff < min_weighted_diff:
                min_weighted_diff = total_weighted_diff
                best_year = year
        
        if best_year is not None:
            selected_months[month] = best_year
            print(f"  Mes {month}: Año seleccionado = {best_year}")
        else:
            print(f"Advertencia: No se pudo seleccionar un año típico para el mes {month}.")

    # --- 3. Construir el DataFrame TMY --- 
    print("\nConstruyendo el DataFrame TMY final...")
    tmy_df_list = []
    if not selected_months:
        print("Error: No se seleccionaron meses típicos. No se puede generar el TMY.")
        return None, None
        
    for month, year in selected_months.items():
        mask = (df.index.month == month) & (df.index.year == year)
        month_data = df[mask]
        # Ajustar el año a un año "típico" (opcional, aquí usamos el año original)
        # Podríamos crear un nuevo índice de tiempo para un año genérico si fuera necesario
        tmy_df_list.append(month_data)
    
    if not tmy_df_list:
        print("Error: No se pudieron recolectar datos para los meses seleccionados.")
        return None, None
        
    tmy_final_df = pd.concat(tmy_df_list).sort_index()
    print(f"TMY DataFrame construido con {len(tmy_final_df)} registros.")

    # --- 4. Ajustar el índice a un año genérico --- 
    print("Ajustando el índice de tiempo a un año genérico...")
    placeholder_year = 2000 # Puedes elegir otro año si prefieres
    try:
        # Crear nuevo índice reemplazando solo el año
        new_index = tmy_final_df.index.map(lambda ts: ts.replace(year=placeholder_year))
        tmy_final_df.index = new_index
        
        # Manejo de año bisiesto si es necesario
        # Si el año placeholder NO es bisiesto, eliminar 29 Feb
        if placeholder_year % 4 != 0 or (placeholder_year % 100 == 0 and placeholder_year % 400 != 0):
             feb_29_mask = (tmy_final_df.index.month == 2) & (tmy_final_df.index.day == 29)
             if feb_29_mask.any():
                 print(f"Advertencia: Año placeholder {placeholder_year} no es bisiesto. Se eliminarán {feb_29_mask.sum()} registros del 29 de Feb.")
                 tmy_final_df = tmy_final_df[~feb_29_mask]
        # (Podríamos añadir lógica para duplicar un día si el placeholder ES bisiesto y el original no, pero es menos común)

        # Verificar longitud final
        if len(tmy_final_df) != 8760:
             print(f"Advertencia: El TMY final tiene {len(tmy_final_df)} horas, no las 8760 esperadas después del ajuste de año.")
        
        print(f"Índice ajustado al año genérico {placeholder_year}.")
        
        # Actualizar las columnas Year, Month, Day, Hour para reflejar el nuevo índice
        tmy_final_df['Year'] = tmy_final_df.index.year
        tmy_final_df['Month'] = tmy_final_df.index.month
        tmy_final_df['Day'] = tmy_final_df.index.day
        tmy_final_df['Hour'] = tmy_final_df.index.hour
        # Minute si existe y fue usado
        if 'Minute' in tmy_final_df.columns:
             tmy_final_df['Minute'] = tmy_final_df.index.minute
            
    except Exception as e:
        print(f"Error al ajustar el índice al año genérico: {e}")
        # Continuar sin el ajuste si falla

    # --- 5. Asegurar orden cronológico final --- 
    print("Asegurando orden cronológico final del índice...")
    tmy_final_df.sort_index(inplace=True)

    # Devolver DataFrame TMY, meses seleccionados
    return tmy_final_df, selected_months

# --- Script Principal ---
# Definir carpeta de salida con ruta absoluta
output_dir = '/home/cparrado/datasciencesolar/datascience_solarenergy/datascience_solarenergy/' 
os.makedirs(output_dir, exist_ok=True) # Crear carpeta si no existe

# Cargar el dataset
# Usar ruta absoluta también para la carga, aunque ya funcionaba
file_path = '/home/cparrado/datasciencesolar/datascience_solarenergy/datascience_solarenergy/antofagasta_dirty.csv'
print(f"Cargando datos desde: {file_path}")
try:
    df = pd.read_csv(file_path)
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print(f"Error: Archivo no encontrado en {file_path}")
    exit() # Salir si el archivo no se encuentra

# Limpiar los datos y capturar el log
df_cleaned, cleaning_summary_log = clean_data(df.copy()) 

# Crear Timestamp y establecer como índice
print("Creando índice de tiempo (Timestamp)...")
try:
    # Asegurarse de que las columnas de fecha/hora sean numéricas (a veces pueden leerse como objetos)
    date_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    for col in date_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Eliminar filas donde la conversión falló (si alguna)
    rows_before = len(df_cleaned)
    df_cleaned.dropna(subset=date_cols, inplace=True)
    if len(df_cleaned) < rows_before:
        print(f"Advertencia: Se eliminaron {rows_before - len(df_cleaned)} filas con valores no numéricos en columnas de fecha/hora.")

    # Convertir a enteros para asegurar compatibilidad con to_datetime
    for col in ['Year', 'Month', 'Day', 'Hour', 'Minute']:
         df_cleaned[col] = df_cleaned[col].astype(int)

    # Crear el timestamp (ignorar 'Minute' si siempre es 0, como parece ser el caso)
    if (df_cleaned['Minute'] == 0).all():
        df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned[['Year', 'Month', 'Day', 'Hour']])
    else: # Si 'Minute' varía, incluirlo
        df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        
    df_cleaned.set_index('Timestamp', inplace=True)
    df_cleaned.sort_index(inplace=True) # Asegurar orden cronológico
    print("Índice de tiempo creado y establecido.")
except Exception as e:
    print(f"Error al crear el índice de tiempo: {e}")
    print("Asegúrate de que las columnas 'Year', 'Month', 'Day', 'Hour', 'Minute' existan y sean numéricas.")
    exit()

# Mostrar información básica del DataFrame limpio
print("\nInformación del DataFrame limpio:")
print(df_cleaned.info())
print("\nPrimeras filas del DataFrame limpio:")
print(df_cleaned.head())
print("\nÚltimas filas del DataFrame limpio:")
print(df_cleaned.tail())

# --- Generar TMY ---
# generate_tmy ahora solo devuelve df y selected_years
tmy_data, selected_years = generate_tmy(df_cleaned)

# --- Guardar TMY ---
if tmy_data is not None:
    output_file_csv = os.path.join(output_dir, 'antofagasta_tmy.csv') # Usar os.path.join
    # output_file_csv = f"{output_dir}/antofagasta_tmy.csv" # Alternativa con f-string
    try:
        tmy_data.to_csv(output_file_csv)
        print(f"\nArchivo TMY guardado exitosamente en: {output_file_csv}")
    except Exception as e:
        print(f"\nError al guardar el archivo TMY: {e}")
else:
    print("\nNo se generó el archivo TMY debido a errores previos.")

# --- Verificación Opcional del Orden del Índice --- 
if tmy_data is not None:
    print(f"\nVerificando orden del Timestamp en {output_file_csv}...")
    try:
        df_check = pd.read_csv(output_file_csv, index_col='Timestamp', parse_dates=True)
        if df_check.index.is_monotonic_increasing:
            print("Verificación exitosa: El índice Timestamp está ordenado cronológicamente.")
        else:
            print("Error de verificación: El índice Timestamp NO está ordenado cronológicamente.")
            # Opcional: Mostrar dónde no está ordenado
            # diffs = df_check.index.to_series().diff().dt.total_seconds()
            # print(diffs[diffs < 0])
    except Exception as e:
        print(f"Error durante la verificación del archivo guardado: {e}")

# --- Generación de Gráficos del TMY Final --- 
if tmy_data is not None:
    print("\nGenerando gráficos descriptivos del TMY final...")
    
    # 1. Gráfico Serie Anual GHI (TMY)
    output_file_png_series = os.path.join(output_dir, 'tmy_annual_series.png')
    print(f"  Generando gráfico de serie anual GHI (TMY) -> {output_file_png_series}...")
    try:
        plt.figure(figsize=(15, 6))
        plt.plot(tmy_data.index, tmy_data['GHI'])
        plt.title(f'Serie Anual de GHI (TMY - Año {tmy_data.index.year.unique()[0]})')
        plt.xlabel('Fecha')
        plt.ylabel('GHI (W/m^2)')
        plt.grid(True)
        plt.savefig(output_file_png_series)
        plt.close()
        print(f"    Gráfico guardado como {output_file_png_series}")
    except Exception as e:
        print(f"    Error al generar gráfico de serie anual: {e}")

    # 2. Histograma GHI (TMY)
    output_file_png_hist = os.path.join(output_dir, 'tmy_ghi_histogram.png')
    print(f"  Generando histograma GHI (TMY) -> {output_file_png_hist}...")
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(tmy_data['GHI'], kde=True, bins=50)
        plt.title(f'Histograma de GHI (TMY - Año {tmy_data.index.year.unique()[0]})')
        plt.xlabel('GHI (W/m^2)')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig(output_file_png_hist)
        plt.close()
        print(f"    Gráfico guardado como {output_file_png_hist}")
    except Exception as e:
        print(f"    Error al generar histograma: {e}")

    # 3. Matriz de Correlación (TMY)
    output_file_png_corr = os.path.join(output_dir, 'tmy_correlation_matrix.png')
    print(f"  Generando matriz de correlación (TMY) -> {output_file_png_corr}...")
    try:
        numeric_cols_tmy = tmy_data.select_dtypes(include=np.number)
        correlation_matrix_tmy = numeric_cols_tmy.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix_tmy, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'Matriz de Correlación (TMY - Año {tmy_data.index.year.unique()[0]})')
        plt.tight_layout()
        plt.savefig(output_file_png_corr)
        plt.close()
        print(f"    Gráfico guardado como {output_file_png_corr}")
    except Exception as e:
        print(f"    Error al generar matriz de correlación: {e}")

    # 4. Histograma GHI (Positivos) (TMY)
    output_file_png_hist_pos = os.path.join(output_dir, 'tmy_ghi_histogram_positive.png')
    print(f"  Generando histograma GHI > 0 (TMY) -> {output_file_png_hist_pos}...")
    try:
        ghi_positive = tmy_data[tmy_data['GHI'] > 0]['GHI']
        if not ghi_positive.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(ghi_positive, kde=True, bins=50)
            plt.title(f'Histograma de GHI > 0 (TMY - Año {tmy_data.index.year.unique()[0]})')
            plt.xlabel('GHI (W/m^2)')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            plt.savefig(output_file_png_hist_pos)
            plt.close()
            print(f"    Gráfico guardado como {output_file_png_hist_pos}")
        else:
            print("    No se encontraron valores de GHI > 0 para generar el histograma.")
    except Exception as e:
        print(f"    Error al generar histograma de GHI positivo: {e}")
        
# --- Guardar Informe Consolidado (Limpieza + Estadísticas TMY) --- 
report_file_path = os.path.join(output_dir, 'tmy_generation_report.txt')
print(f"\nGuardando informe consolidado en: {report_file_path}...")
try:
    with open(report_file_path, 'w') as f:
        f.write("=================================================\n")
        f.write(" TMY Generation Report for Antofagasta Data \n")
        f.write("=================================================\n")

        # 1. Resumen de Limpieza
        if 'cleaning_summary_log' in locals() and cleaning_summary_log:
            f.write("\n\n--- Data Cleaning Summary ---\n")
            for line in cleaning_summary_log:
                f.write(line + "\n")
        else:
            f.write("\n\n--- Data Cleaning Summary ---\n")
            f.write("No cleaning summary log available.\n")

        # 2. Selección de Meses Típicos
        if 'selected_years' in locals() and selected_years:
            f.write("\n\n--- Selected Source Year for Each Month ---\n")
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                           7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            for month_num in sorted(selected_years.keys()):
                f.write(f"  {month_names.get(month_num, month_num)}: {selected_years[month_num]}\n")
        else:
             f.write("\n\n--- Selected Source Year for Each Month ---\n")
             f.write("Month selection data not available (TMY generation might have failed).\n")

        # 3. Estadísticas Descriptivas del TMY Final
        if tmy_data is not None:
            f.write("\n\n--- Descriptive Statistics of Final TMY Data ---\n")
            desc_stats = tmy_data.describe()
            f.write(desc_stats.to_string()) 
            f.write("\n")
        else:
             f.write("\n\n--- Descriptive Statistics of Final TMY Data ---\n")
             f.write("TMY data not available for statistics (TMY generation might have failed).\n")

    print("  Informe consolidado guardado exitosamente.")
except Exception as e:
    print(f"  Error al guardar el informe consolidado: {e}")

# --- Fin del Script --- 
# (Se eliminó el print final redundante) 