"""
Dashboard de Actividad Económica Regional
TP Ingeniería del Software - UGR (Fase 6: Despliegue, metodología CRISP-DM)

"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Configuración
st.set_page_config(page_title="Actividad Económica Regional", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).parent / "data" / "raw"
ARCHIVO_DESEST = DATA_DIR / "serie-desestacionalizada-actividad-economica-region.csv"
URL_REPO = "https://github.com/FacuM87/TP-Ing-de-Software"

PAISES = ["Argentina", "Brasil", "EstadosUnidos", "Mexico", "Paraguay"]
PREDICTORAS = ["Brasil", "EstadosUnidos", "Mexico", "Paraguay"]
NOMBRES = {"Argentina": "Argentina", "Brasil": "Brasil", "EstadosUnidos": "EE.UU.",
           "Mexico": "México", "Paraguay": "Paraguay"}
UNIDADES = {"Argentina": "EMAE", "Brasil": "IBC-Br", "EstadosUnidos": "PIB, miles de M USD",
            "Mexico": "IGAE", "Paraguay": "IMAEP"}
COLORES = {"Argentina": "#1f77b4", "Brasil": "#2ca02c", "EstadosUnidos": "#d62728",
           "Mexico": "#9467bd", "Paraguay": "#ff7f0e"}

PARAMS_RF = {"n_estimators": 300, "max_depth": 4, "random_state": 42}
UMBRAL_ALTO, UMBRAL_BAJO = 70, 30 
CLASE_ALTA = "Alta Actividad"



@st.cache_data
def cargar_datos():
    df = pd.read_csv(ARCHIVO_DESEST)
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d-%m-%Y")
    df = df.sort_values("Fecha").reset_index(drop=True)[["Fecha"] + PAISES]
    mediana = df["Argentina"].median()
    df["Nivel_Actividad"] = np.where(df["Argentina"] >= mediana, CLASE_ALTA, "Baja Actividad")
    return df, mediana



@st.cache_resource
def entrenar_modelo(df):
    X, y = df[PREDICTORAS], df["Nivel_Actividad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    modelo = RandomForestClassifier(**PARAMS_RF).fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    corte = df["Fecha"] < "2019-01-01"
    modelo_temp = clone(modelo).fit(X[corte], y[corte])

    metricas = {
        "Exactitud": accuracy_score(y_test, y_pred),
        "Precisión (Alta)": precision_score(y_test, y_pred, pos_label=CLASE_ALTA),
        "Recall (Alta)": recall_score(y_test, y_pred, pos_label=CLASE_ALTA),
        "F1-Score (Alta)": f1_score(y_test, y_pred, pos_label=CLASE_ALTA),
        "Exactitud temporal (2019-2023)": accuracy_score(y[~corte], modelo_temp.predict(X[~corte])),
    }
    importancias = pd.Series(modelo.feature_importances_, index=PREDICTORAS).sort_values()
    return modelo, metricas, importancias


def probabilidad_alta(modelo, valores):
    escenario = pd.DataFrame([valores])[PREDICTORAS]
    return modelo.predict_proba(escenario)[0][list(modelo.classes_).index(CLASE_ALTA)] * 100


def semaforo(prob):
    if prob >= UMBRAL_ALTO:
        return ("🟢 ALTA ACTIVIDAD", "#28a745", "#e8f5e9",
                "Escenario favorable: evaluar aumento de turnos, anticipar compras de insumos "
                "y asegurar cupos logísticos, previa validación de los analistas.")
    if prob > UMBRAL_BAJO:
        return ("🟡 ZONA DE INCERTIDUMBRE", "#e0a800", "#fff8e1",
                "Señal no concluyente: mantener niveles actuales de producción y stock "
                "y monitorear los próximos datos publicados.")
    return ("🔴 BAJA ACTIVIDAD", "#dc3545", "#ffebee",
            "Escenario desfavorable: moderar la acumulación de stock, controlar costos fijos "
            "y orientar esfuerzos comerciales a los mercados más estables.")


def tarjeta(prob, titulo):
    estado, color, fondo, recomendacion = semaforo(prob)
    st.markdown(f"""
    <div style="background-color:{fondo}; border-left:6px solid {color}; padding:16px 20px;
                border-radius:6px; color:#222;">
        <p style="font-size:14px; margin:0; color:#444;">{titulo}</p>
        <h3 style="color:{color}; margin:6px 0 10px 0;">{estado}</h3>
        <p style="font-size:16px; margin:0 0 8px 0;"><b>Probabilidad de Alta Actividad en Argentina:</b> {prob:.1f}%</p>
        <p style="font-size:14px; margin:0;"><b>Recomendación:</b> {recomendacion}</p>
    </div>""", unsafe_allow_html=True)



# Carga de datoss
try:
    df, mediana = cargar_datos()
except Exception as e:
    st.error(f"No se pudieron cargar los datos desde `{ARCHIVO_DESEST}`. Detalle: {e}")
    st.stop()

modelo, metricas, importancias = entrenar_modelo(df)
ultimo = df.iloc[-1]
anterior = df.iloc[-2]
ultimo_mes = ultimo["Fecha"].strftime("%m/%Y")
valores_iniciales = {p: round(float(ultimo[p]), 1) for p in PREDICTORAS}


st.title("📊 Actividad Económica Regional")
st.caption(f"AgroSur Exportaciones S.A. · Series desestacionalizadas 01/2004 – {ultimo_mes} · "
           "Fuente: datos.produccion.gob.ar")

vista_panel, vista_explorador, vista_simulador = st.tabs(
    ["Panel Ejecutivo", "Explorador Histórico", "Simulador de Escenarios"])

# Panel
with vista_panel:
    st.subheader(f"Situación al último mes disponible ({ultimo_mes})")

    columnas = st.columns(len(PAISES))
    for col, pais in zip(columnas, PAISES):
        variacion = (ultimo[pais] / anterior[pais] - 1) * 100
        valor = f"{ultimo[pais]:,.0f}" if pais == "EstadosUnidos" else f"{ultimo[pais]:.1f}"
        col.metric(f"{NOMBRES[pais]} ({UNIDADES[pais]})", valor, f"{variacion:+.1f}% vs mes anterior")

    st.markdown("#### Régimen estimado para Argentina")
    tarjeta(probabilidad_alta(modelo, valores_iniciales), f"Estimación del modelo para {ultimo_mes}")

    st.markdown("#### Argentina: últimos 36 meses")
    recientes = df.tail(36)
    fig = px.line(recientes, x="Fecha", y="Argentina", labels={"Argentina": "Índice EMAE", "Fecha": ""})
    fig.update_traces(line_color=COLORES["Argentina"])
    fig.add_hline(y=mediana, line_dash="dash", line_color="gray",
                  annotation_text=f"Mediana histórica ({mediana:.1f}): umbral Alta / Baja")
    fig.update_layout(height=350, margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")

# Explorador Histórico 
with vista_explorador:
    st.subheader("Evolución comparada y co-movimiento")

    c1, c2 = st.columns([1, 2])
    seleccion = c1.multiselect("Países", PAISES, default=PAISES, format_func=lambda p: NOMBRES[p])
    fecha_min, fecha_max = df["Fecha"].min().date(), df["Fecha"].max().date()
    desde, hasta = c2.slider("Período", min_value=fecha_min, max_value=fecha_max,
                             value=(fecha_min, fecha_max), format="MM/YYYY")

    if not seleccion:
        st.info("Seleccioná al menos un país.")
    else:
        periodo = df[(df["Fecha"].dt.date >= desde) & (df["Fecha"].dt.date <= hasta)]
        normalizado = df[["Fecha"] + seleccion].copy()
        for pais in seleccion:
            normalizado[pais] = df[pais] / df[pais].iloc[0] * 100
        normalizado = normalizado[normalizado["Fecha"].isin(periodo["Fecha"])]

        largo = normalizado.melt(id_vars="Fecha", var_name="País", value_name="Índice")
        largo["País"] = largo["País"].map(NOMBRES)
        fig = px.line(largo, x="Fecha", y="Índice", color="País",
                      color_discrete_map={NOMBRES[p]: COLORES[p] for p in PAISES},
                      labels={"Índice": "Índice normalizado (Ene 2004 = 100)", "Fecha": ""})
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
        fig.update_layout(height=420, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")

        if len(seleccion) >= 2:
            st.markdown(f"#### Matriz de correlación ({desde.strftime('%m/%Y')} – {hasta.strftime('%m/%Y')})")
            corr = periodo[seleccion].corr().rename(index=NOMBRES, columns=NOMBRES)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", zmin=-1, zmax=1,
                            aspect="auto")
            fig.update_layout(height=420, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Seleccioná al menos dos países para ver la matriz de correlación.")

# Simulador de Escenarios
with vista_simulador:
    st.subheader("¿En qué régimen estaría Argentina si sus socios presentan estos niveles?")

    def restablecer():
        for pais in PREDICTORAS:
            st.session_state[f"slider_{pais}"] = valores_iniciales[pais]

    c1, c2 = st.columns([1, 1])
    with c1:
        valores = {}
        for pais in PREDICTORAS:
            clave = f"slider_{pais}"
            if clave not in st.session_state:  
                st.session_state[clave] = valores_iniciales[pais]
            paso = 50.0 if pais == "EstadosUnidos" else 0.5
            valores[pais] = st.slider(f"{NOMBRES[pais]} ({UNIDADES[pais]})",
                                      min_value=round(float(df[pais].min()), 1),
                                      max_value=round(float(df[pais].max()), 1),
                                      step=paso, key=clave)
        st.button(f"Volver a {ultimo_mes}", on_click=restablecer)

    with c2:
        titulo = (f"Último mes disponible ({ultimo_mes})" if valores == valores_iniciales
                  else "Escenario simulado")
        tarjeta(probabilidad_alta(modelo, valores), titulo)
        st.caption("Los controles se limitan al rango histórico de cada país. Umbrales del semáforo: "
                   f"verde ≥ {UMBRAL_ALTO}%, amarillo entre {UMBRAL_BAJO}% y {UMBRAL_ALTO}%, rojo ≤ {UMBRAL_BAJO}%.")

    with st.expander("Sobre el modelo y su alcance"):
        m1, m2 = st.columns([1, 1])
        with m1:
            st.markdown("**Random Forest** (300 árboles, profundidad máxima 4). Métricas sobre el conjunto de prueba:")
            st.dataframe(pd.DataFrame({"Métrica": list(metricas)[:4],
                                       "Valor": [f"{v*100:.2f}%" for v in list(metricas.values())[:4]]}),
                         hide_index=True, width="stretch")
        with m2:
            fig = px.bar(x=importancias.values * 100, y=[NOMBRES[p] for p in importancias.index],
                         orientation="h", labels={"x": "Importancia (%)", "y": ""},
                         title="Importancia de cada variable")
            fig.update_traces(marker_color=[COLORES[p] for p in importancias.index])
            fig.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        st.warning(
            f"**Alcance:** al entrenar con 2004-2018 y predecir 2019-2023, la exactitud cae a "
            f"{metricas['Exactitud temporal (2019-2023)']*100:.1f}%. El modelo sirve para analizar escenarios "
            "dentro de rangos históricos, no para pronosticar períodos futuros. Las alertas deben "
            "validarse con los analistas antes de comprometer producción o compras.")

st.divider()
st.caption(f"TP Ingeniería del Software · UGR · Código y datos: [{URL_REPO.replace('https://', '')}]({URL_REPO})")
