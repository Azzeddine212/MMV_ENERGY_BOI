import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import base64

# Désactiver les avertissements
import warnings
warnings.filterwarnings("ignore")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp  {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Ajout du fond d'écran
add_bg_from_local('interface.jpg')

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Predictions')
    output.seek(0)
    return output.read()

# Sidebar pour l'importation de fichiers et paramètres
st.sidebar.title("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("📂 Téléchargez votre fichier Excel", type=["xlsx"])

model_path = "xgb_model_cb22-23-24_10_param.joblib"
scaler_path = "scaler_cb22-23-24_10_param.pkl"
target_column = "Conso NRJ Usine (kwh/tcossette)"

# Chargement des données
if uploaded_file is not None:
    data_boiry = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ Fichier chargé avec succès !")
    
    # Onglets pour la navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d’ensemble", "🔍 Prédictions", "📈 Visualisation", "💾 Téléchargement"])
    
    with tab1:
        st.header("📊 Vue d’ensemble des données")
        st.dataframe(data_boiry.head())
        st.write("### Statistiques générales")
        st.write(data_boiry.describe())
    
    with tab2:
        st.header("🔍 Prédictions de la Consommation d'Énergie")
        if st.button("🚀 Lancer la prédiction"):
            with st.spinner("📊 Calcul en cours..."):
                # Charger le modèle et scaler
                model = joblib.load(model_path)
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                
                variables = data_boiry.drop(columns=[target_column], errors='ignore')
                X_scaled = scaler.transform(variables)
                predictions = model.predict(X_scaled)
                df_results = pd.DataFrame(predictions, columns=["Prédictions"], index=variables.index)
                df_final = pd.concat([variables, df_results], axis=1)
                st.success("✅ Prédictions terminées !")
                st.dataframe(df_final.head())
                st.session_state['df_results'] = df_final
    
    with tab3:
        if 'df_results' in st.session_state:
            df_results = st.session_state['df_results']
            st.header("📈 Visualisation des Résultats")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Histogramme des Prédictions")
                fig, ax = plt.subplots()
                sns.histplot(df_results["Prédictions"], bins=20, kde=True, color='blue', ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.write("### Distribution des Variables")
                variable = st.selectbox("Sélectionnez une variable", df_results.columns)
                fig, ax = plt.subplots()
                sns.histplot(df_results[variable], bins=20, kde=True, ax=ax)
                st.pyplot(fig)
    
    with tab4:
        if 'df_results' in st.session_state:
            st.header("💾 Télécharger les résultats")
            st.download_button(
                label="💾 Télécharger les prédictions",
                data=convert_df_to_excel(st.session_state['df_results']),
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
