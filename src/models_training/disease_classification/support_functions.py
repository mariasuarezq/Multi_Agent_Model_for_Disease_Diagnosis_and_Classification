import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import math
import seaborn as sns

def create_target_variable_boxplots(dataframe, target_col):
    # Identifica le colonne continue e la variabile categorica
    continuous_cols = [col for col in dataframe.columns if col != target_col and pd.api.types.is_numeric_dtype(dataframe[col])]

    # Numero di variabili continue
    total_plots = len(continuous_cols)

    # Calcola il layout della griglia dei subplots
    cols = 3  # Numero di colonne nei subplot
    rows = math.ceil(total_plots / cols)  # Numero di righe necessarie

    # Crea la figura e gli assi
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
    axes = axes.flatten()  # Per iterare facilmente sugli assi

    # Crea i boxplot
    for i, col in enumerate(continuous_cols):
        sns.boxplot(data=dataframe, x=target_col, y=col, ax=axes[i])
        axes[i].set_title(f"Boxplot of {col} by {target_col}")
        axes[i].set_xlabel(target_col)
        axes[i].set_ylabel(col)

    # Rimuove i subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Mostra il grafico
    plt.show()

def atipicosAmissing(varaux):
    """
    Esta función identifica valores atípicos en una serie de datos y los reemplaza por NaN.
    
    Datos de entrada:
    - varaux: Serie de datos en la que se buscarán valores atípicos.
    
    Datos de salida:
    - Una nueva serie de datos con valores atípicos reemplazados por NaN.
    - El número de valores atípicos identificados.
    """
    
    # Verifica si la distribución de los datos es simétrica o asimétrica
    if abs(varaux.skew()) < 1:
        # Si es simétrica, calcula los valores atípicos basados en la desviación estándar
        criterio1 = abs((varaux - varaux.mean()) / varaux.std()) > 3
    else:
        # Si es asimétrica, calcula la Desviación Absoluta de la Mediana (MAD) y los valores atípicos
        mad = sm.robust.mad(varaux, axis=0)
        criterio1 = abs((varaux - varaux.median()) / mad) > 8
    
    # Calcula los cuartiles 1 (Q1) y 3 (Q3) para determinar el rango intercuartílico (H)
    qnt = varaux.quantile([0.25, 0.75]).dropna()
    Q1 = qnt.iloc[0]
    Q3 = qnt.iloc[1]
    H = 3 * (Q3 - Q1)
    
    # Identifica valores atípicos que están fuera del rango intercuartílico
    criterio2 = (varaux < (Q1 - H)) | (varaux > (Q3 + H))
    
    # Crea una copia de la serie original y reemplaza los valores atípicos por NaN
    var = varaux.copy()
    var[criterio1 & criterio2] = np.nan
    
    # Retorna la serie con valores atípicos reemplazados y el número de valores atípicos identificados
    return [var, sum(criterio1 & criterio2)]

def graficoVcramer(matriz, target):
    """
    Genera un gráfico de barras horizontales que muestra el coeficiente V de Cramer entre cada columna de matriz y la variable target.

    Datos de entrada:
    - matriz: DataFrame con las variables a comparar.
    - target: Serie de la variable objetivo (categórica).

    Datos de salida:
    - Gráfico de barras horizontales que muestra el coeficiente V de Cramer para cada variable.
    """

    # Calcula el coeficiente V de Cramer para cada columna de matriz y target
    salidaVcramer = {x: Vcramer(matriz[x], target) for x in matriz.columns}

    # Ordena los resultados en orden descendente por el coeficiente V de Cramer
    sorted_data = dict(sorted(salidaVcramer.items(), key=lambda item: item[1], reverse=True))

    # Crea el gráfico de barras horizontales
    plt.figure(figsize=(6, 10))
    plt.barh(list(sorted_data.keys()), list(sorted_data.values()), color='skyblue')
    plt.xlabel('V de Cramer')
    plt.title(f'V de Cramer entre las variables input y la variable objetivo {target.name}')
    plt.show()

    return sorted_data

def Vcramer(v, target):
    """
    Calcula el coeficiente V de Cramer entre dos variables. Si alguna de ellas es continua, la discretiza.

    Datos de entrada:
    - v: Serie de datos categóricos o cuantitativos.
    - target: Serie de datos categóricos o cuantitativos.

    Datos de salida:
    - Coeficiente V de Cramer que mide la asociación entre las dos variables.
    """

    if v.dtype == 'float64' or v.dtype == 'int64':
        # Si v es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(v.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        v = pd.cut(v, bins=p)
        v = v.fillna(v.min())

    if target.dtype == 'float64' or target.dtype == 'int64':
        # Si target es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(target.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        target = pd.cut(target, bins=p)
        target = target.fillna(target.min())

    # Calcula una tabla de contingencia entre v y target
    tabla_cruzada = pd.crosstab(v, target)

    # Calcula el chi-cuadrado y el coeficiente V de Cramer
    chi2 = chi2_contingency(tabla_cruzada)[0]
    n = tabla_cruzada.sum().sum()
    v_cramer = np.sqrt(chi2 / (n * (min(tabla_cruzada.shape) - 1)))

    return v_cramer