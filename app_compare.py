# -*- coding: utf-8 -*-
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ADA_DIR = Path("models_ada")
IXE_DIR = Path("models_ixe")
BIME_DIR = Path("models_bime")

def load_meta(models_dir):
    with open(models_dir / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(models_dir, endpoint):
    meta = load_meta(models_dir)
    info = meta["models"][endpoint]
    path = info["path"] if isinstance(info, dict) else info
    return joblib.load(Path(path.replace("\\", "/")))

def predict(model, X):
    X = X.reindex(columns=model.feature_names_in_)
    return float(model.predict_proba(X)[0, 1])

st.set_page_config(page_title="Comparador ADA vs IXE vs BIME", layout="wide")
st.title("Comparador predictivo: Adalimumab vs Ixekizumab vs Bimekizumab")
st.caption("Predicción individual de PASI75 y PASI90 a semana 16 para un mismo paciente.")

# ---------------- INPUTS DEL PACIENTE ----------------
c1, c2, c3 = st.columns(3)

with c1:
    pasi = st.number_input("PASI basal", 0.0, 80.0, 20.0, step=0.5)
    edad = st.number_input("Edad (años)", 18, 100, 45)

with c2:
    imc = st.number_input("IMC", 15.0, 60.0, 27.0)
    sexo = st.selectbox("Sexo", ["Varón", "Mujer"])

with c3:
    artritis_txt = st.selectbox("Artritis psoriásica", ["No", "Sí"])
    nprev = st.number_input("Nº tratamientos/biológicos previos", 0, 20, 0)

if st.button("Comparar tratamientos"):
    # ---------- ADA ----------
    X_ada = pd.DataFrame([{
        "Sexo": sexo,
        "EDAD": int(edad),
        "IMC": float(imc),
        "PASI INICIAL ADA": float(pasi),
        "ARTRITIS": 1 if artritis_txt == "Sí" else 0,
        "N tratamientos previos": int(nprev),
    }])

    m_ada_75 = load_model(ADA_DIR, "PASI75_w16")
    m_ada_90 = load_model(ADA_DIR, "PASI90_w16")
    p_ada_75 = predict(m_ada_75, X_ada)
    p_ada_90 = predict(m_ada_90, X_ada)

    # ---------- IXE ----------
    X_ixe = pd.DataFrame([{
        "Sexo": sexo,
        "edad": int(edad),
        "IMC": float(imc),
        "ARTRITIS PSORIASICA": 1 if artritis_txt == "Sí" else 0,
        "años con psoriasis": 15,  # valor neutro si no se pregunta
        "N biológicos previos": int(nprev),
        "PASI INICIAL IXE": float(pasi),
    }])

    m_ixe_75 = load_model(IXE_DIR, "PASI75_w16")
    m_ixe_90 = load_model(IXE_DIR, "PASI90_w16")
    p_ixe_75 = predict(m_ixe_75, X_ixe)
    p_ixe_90 = predict(m_ixe_90, X_ixe)

    # ---------- BIME ----------
    X_bime = pd.DataFrame([{
        "Sexo": sexo,
        "Edad (autocálculo)": float(edad),
        "IMC (autocálculo)": float(imc),
        "Artritis": 1 if artritis_txt == "Sí" else 0,
        "N biológicos previos": int(nprev),
        "PASI INICIO TTO": float(pasi),
    }])

    m_bime_75 = load_model(BIME_DIR, "PASI75_w16")
    m_bime_90 = load_model(BIME_DIR, "PASI90_w16")
    p_bime_75 = predict(m_bime_75, X_bime)
    p_bime_90 = predict(m_bime_90, X_bime)

    # ---------------- RESULTADOS ----------------
    st.markdown("---")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.subheader("Adalimumab")
        st.metric("PASI75", f"{p_ada_75*100:.1f}%")
        st.metric("PASI90", f"{p_ada_90*100:.1f}%")

    with r2:
        st.subheader("Ixekizumab")
        st.metric("PASI75", f"{p_ixe_75*100:.1f}%")
        st.metric("PASI90", f"{p_ixe_90*100:.1f}%")

    with r3:
        st.subheader("Bimekizumab")
        st.metric("PASI75", f"{p_bime_75*100:.1f}%")
        st.metric("PASI90", f"{p_bime_90*100:.1f}%")

    best = max(
        [("Adalimumab", p_ada_90), ("Ixekizumab", p_ixe_90), ("Bimekizumab", p_bime_90)],
        key=lambda x: x[1]
    )[0]

    st.markdown(f"### Mayor probabilidad de PASI90: **{best}**")
