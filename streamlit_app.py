import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns
import io
import warnings
from PIL import Image
import base64

# Désactiver les avertissements
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Tableau de Bord - Prédictions", layout="wide")

# Ajouter l'image en arrière-plan via CSS
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

# Ajout de l'image de fond
add_bg_from_local('interface.jpg')

# Fonction de conversion DataFrame en Excel
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Predictions')
    output.seek(0)
    return output.read()

# Traitement des données
def process_boiry_data(df_boiry):
    def moyenne_pondérée(valeur_1, valeur_2, poid_1, poid_2):
        return (valeur_1 * poid_1 + valeur_2 * poid_2) / (poid_1 + poid_2)
    
    df_boiry['Soutirage_tot'] = df_boiry['Soutirage 9m'] + df_boiry['Soutirage 11m']
    df_boiry['Temp entrée JAE_moy'] = df_boiry.apply(lambda row: moyenne_pondérée(row['Temp entrée JAE A'], row['Temp entrée JAE B'], row['Débit JAE A'], row['Débit JAE B']), axis=1)
    df_boiry['Temp sortie JAE_moy'] = df_boiry.apply(lambda row: moyenne_pondérée(row['Temp sortie JAE A'], row['Temp sortie JAE B'], row['Débit JAE A'], row['Débit JAE B']), axis=1)
    df_boiry['Débit JAE_tot'] = df_boiry['Débit JAE A'] + df_boiry['Débit JAE B']
    df_boiry['Débit vapeur_tot'] = df_boiry['Débit vapeur 140T'] + df_boiry['Débit vapeur 120T']
    df_boiry['Energie kWh 0°C_pci'] = df_boiry['Energie KWh 0°C'] * 0.9
    df_boiry['Conso NRJ Usine (kwh/tcossette)'] = df_boiry['Energie kWh 0°C_pci'] / df_boiry['Tonnage']
    df_boiry.reset_index(drop=True, inplace=True)
    return df_boiry

# Chargement du fichier utilisateur
st.title("🔍 Prédiction de la Consommation d'Énergie")
uploaded_file = st.file_uploader("📂 Téléchargez votre fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    data_boiry = pd.read_excel(uploaded_file)
    st.success("✅ Fichier chargé avec succès !")
    st.dataframe(data_boiry.head())
    
    model_path = "xgb_model_cb22-23-24_10_param.joblib"
    scaler_path = "scaler_cb22-23-24_10_param.pkl"
    target_column = "Conso NRJ Usine (kwh/tcossette)"
    
    if st.button("🚀 Lancer la prédiction"):
        with st.spinner("📊 Calcul en cours..."):
            df_results = process_boiry_data(data_boiry)
            st.success("✅ Prédictions terminées !")
            st.dataframe(df_results.head())
            
            # Affichage du graphique de distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_results[target_column], bins=20, kde=True, color='blue', ax=ax)
            ax.set_title("Distribution des Prédictions de Consommation d'Énergie")
            ax.set_xlabel("Consommation (kWh/tcossette)")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
            
            # Bouton de téléchargement
            st.download_button(
                label="💾 Télécharger les résultats",
                data=convert_df_to_excel(df_results),
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
