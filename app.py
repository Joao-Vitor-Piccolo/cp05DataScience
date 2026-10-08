"""Interface Streamlit do Checkpoint 05 — mesmo pipeline do notebook."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_PATH = Path(__file__).parent / "models" / "wine_quality_pipeline.joblib"

LIMITES = {
    "fixed acidity": (4.0, 16.0, 7.4, 0.1),
    "volatile acidity": (0.10, 1.60, 0.70, 0.01),
    "citric acid": (0.00, 1.00, 0.00, 0.01),
    "residual sugar": (0.8, 15.5, 1.9, 0.1),
    "chlorides": (0.010, 0.620, 0.076, 0.001),
    "free sulfur dioxide": (1.0, 72.0, 11.0, 1.0),
    "total sulfur dioxide": (6.0, 289.0, 34.0, 1.0),
    "density": (0.990, 1.004, 0.9978, 0.0001),
    "pH": (2.70, 4.01, 3.51, 0.01),
    "sulphates": (0.33, 2.00, 0.56, 0.01),
    "alcohol": (8.4, 14.9, 9.4, 0.1),
}

ROTULOS = {
    "fixed acidity": "Acidez fixa (g/L)",
    "volatile acidity": "Acidez volátil (g/L)",
    "citric acid": "Ácido cítrico (g/L)",
    "residual sugar": "Açúcar residual (g/L)",
    "chlorides": "Cloretos (g/L)",
    "free sulfur dioxide": "SO₂ livre (mg/L)",
    "total sulfur dioxide": "SO₂ total (mg/L)",
    "density": "Densidade (g/cm³)",
    "pH": "pH",
    "sulphates": "Sulfatos (g/L)",
    "alcohol": "Álcool (% vol)",
}


@st.cache_resource
def carregar_artefato():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute o notebook até o Exercício 7 "
            "para gerar models/wine_quality_pipeline.joblib."
        )
    return joblib.load(ARTIFACT_PATH)


st.set_page_config(page_title="Wine Quality — Checkpoint 05", layout="wide")
st.title("Previsão da nota de vinho tinto")
st.caption(
    "Mesmo pipeline avaliado no notebook do Checkpoint 05 "
    "(Random Forest / XGBoost / LightGBM)."
)

artefato = carregar_artefato()
pipeline = artefato["pipeline"]
feature_cols = artefato["feature_cols"]
caso = artefato.get("caso_paridade", {})

col_info, col_form = st.columns([1, 2])

with col_info:
    st.subheader("Modelo em produção")
    st.write(f"**Algoritmo:** {artefato.get('modelo', '—')}")
    st.write(f"**Estratégia:** {artefato.get('estrategia', '—')}")
    metricas = artefato.get("metricas_teste", {})
    if metricas:
        st.metric("RMSE no teste", f"{metricas.get('RMSE', float('nan')):.3f}")
        st.metric("MAE no teste", f"{metricas.get('MAE', float('nan')):.3f}")
        st.metric("R² no teste", f"{metricas.get('R2', float('nan')):.3f}")

with col_form:
    st.subheader("Perfil químico")
    valores = {}
    c1, c2 = st.columns(2)
    for i, feat in enumerate(feature_cols):
        lo, hi, default, step = LIMITES[feat]
        destino = c1 if i % 2 == 0 else c2
        valores[feat] = destino.slider(
            ROTULOS.get(feat, feat),
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=float(step),
        )

    if st.button("Prever nota", type="primary"):
        entrada = pd.DataFrame([valores], columns=feature_cols)
        y_hat = float(pipeline.predict(entrada)[0])
        st.success(f"Nota prevista  ŷ = **{y_hat:.3f}**")
        st.caption("Escala sensorial usual de 0 a 10 (nesta base, tipicamente 3 a 8).")

st.divider()
st.subheader("Teste de paridade notebook ↔ interface")
if caso:
    st.write(
        "Reproduza os valores abaixo e confira se ŷ da interface coincide "
        "com o ŷ do notebook."
    )
    entrada_ref = pd.DataFrame([caso["inputs"]], columns=feature_cols)
    y_hat_app = float(pipeline.predict(entrada_ref)[0])
    st.dataframe(entrada_ref.T.rename(columns={0: "valor"}), use_container_width=True)
    st.write(f"y observado no notebook: **{caso['y']:.3f}**")
    st.write(f"ŷ no notebook: **{caso['y_hat']:.6f}**")
    st.write(f"ŷ recalculado nesta interface: **{y_hat_app:.6f}**")
    if abs(y_hat_app - caso["y_hat"]) < 1e-6:
        st.success("Paridade confirmada: as duas previsões coincidem.")
    else:
        st.error("Divergência entre notebook e interface — investigar o artefato.")
else:
    st.info("Caso de paridade ainda não gravado no artefato.")
