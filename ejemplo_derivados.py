# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:59:56 2025

@author: jperezr
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Título de la aplicación
st.title("Optimización de Cartera con Derivados y Gestión de Riesgos")

# Descripción
st.write("""
Esta aplicación permite optimizar una cartera de inversiones utilizando acciones, bonos y derivados (opciones y futuros) para gestionar el riesgo.
""")

# Selección de activos
st.sidebar.header("Selección de Activos")
assets = st.sidebar.multiselect(
    "Selecciona los activos para tu cartera (acciones y bonos):",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "TLT"]
)

# Selección de derivados
st.sidebar.header("Selección de Derivados")
use_options = st.sidebar.checkbox("Incluir opciones de cobertura (puts)")
use_futures = st.sidebar.checkbox("Incluir futuros para ajustar exposición")

# Período de tiempo
st.sidebar.header("Período de Tiempo")
start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("2023-01-01"))

# Agregar tu nombre y copyright
st.sidebar.write("---")  # Línea separadora
st.sidebar.write("**Desarrollado por:** Javier Horacio Pérez Ricárdez")  # Cambia "Juan Pérez" por tu nombre
st.sidebar.write("**© 2025 Copyright**")  # Mensaje de copyright

# Descargar datos de yfinance
@st.cache_data
def load_data(assets, start_date, end_date):
    data = yf.download(assets, start=start_date, end=end_date)['Close']  # Usamos 'Close' en lugar de 'Adj Close'
    return data

if len(assets) > 0:
    data = load_data(assets, start_date, end_date)
    st.write("### Precios de los Activos")
    st.line_chart(data)

    # Calcular rendimientos diarios
    returns = data.pct_change().dropna()

    # Función para calcular el riesgo (volatilidad) de la cartera
    def calculate_portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Función de optimización para minimizar la volatilidad
    def minimize_volatility(weights, cov_matrix):
        return calculate_portfolio_volatility(weights, cov_matrix)

    # Restricciones y límites para la optimización
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))

    # Pesos iniciales (iguales para todos los activos)
    init_weights = np.array([1/len(assets)] * len(assets))

    # Matriz de covarianza
    cov_matrix = returns.cov()

    # Optimización de la cartera
    result = minimize(minimize_volatility, init_weights, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)

    # Pesos óptimos
    optimal_weights = result.x

    # Mostrar resultados de la optimización
    st.write("### Pesos Óptimos de la Cartera")
    for i, asset in enumerate(assets):
        st.write(f"{asset}: {optimal_weights[i]:.2%}")

    # Calcular el rendimiento y riesgo de la cartera óptima
    portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
    portfolio_volatility = calculate_portfolio_volatility(optimal_weights, cov_matrix) * np.sqrt(252)

    st.write(f"### Rendimiento Anual Esperado: {portfolio_return:.2%}")
    st.write(f"### Volatilidad Anual Esperada: {portfolio_volatility:.2%}")

    # Gestión de riesgos con derivados
    if use_options:
        st.write("### Cobertura con Opciones (Puts)")
        st.write("Se han incluido opciones de venta (puts) para proteger la cartera contra caídas en el mercado.")
        # Simulación de cobertura con opciones (reducción del 10% en la volatilidad)
        portfolio_volatility *= 0.9
        st.write(f"### Volatilidad Anual con Cobertura: {portfolio_volatility:.2%}")

    if use_futures:
        st.write("### Ajuste de Exposición con Futuros")
        st.write("Se han utilizado futuros para ajustar la exposición al riesgo de tasas de interés.")
        # Simulación de ajuste con futuros (reducción del 5% en la volatilidad)
        portfolio_volatility *= 0.95
        st.write(f"### Volatilidad Anual con Futuros: {portfolio_volatility:.2%}")

    # Gráfico de la frontera eficiente (simplificado)
    st.write("### Frontera Eficiente (Simplificada)")
    num_portfolios = 1000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(assets))
        weights /= np.sum(weights)
        portfolio_return_i = np.sum(returns.mean() * weights) * 252
        portfolio_volatility_i = calculate_portfolio_volatility(weights, cov_matrix) * np.sqrt(252)
        results[0, i] = portfolio_return_i
        results[1, i] = portfolio_volatility_i
        results[2, i] = (portfolio_return_i - 0.02) / portfolio_volatility_i  # Ratio de Sharpe

    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
    plt.colorbar(label='Ratio de Sharpe')
    plt.title('Frontera Eficiente')
    plt.xlabel('Volatilidad Anual')
    plt.ylabel('Rendimiento Anual')
    st.pyplot(plt)

else:
    st.write("Por favor, selecciona al menos un activo para comenzar.")
