import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import io

# Désactiver les avertissements
import warnings
warnings.filterwarnings("ignore")

# Convert DataFrame to Excel
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Predictions')
    output.seek(0)
    return output.read()

# Fonction pour traiter les données
def process_boiry_data(df_boiry):
    """Traitement des données"""
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

# Fonction pour prédire
def process_and_predict(input_data, df_lim, model_path, scaler_path, target_column):
    model = joblib.load(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    data_test = process_boiry_data(input_data)
    data_test = data_test[df_lim.columns.intersection(data_test.columns)]
    
    for col in data_test.columns:
        if col in df_lim.columns:
            data_test = data_test[(data_test[col] >= df_lim.loc['min', col]) & (data_test[col] <= df_lim.loc['max', col])]
    
    if target_column not in data_test.columns:
        st.error(f"La colonne cible '{target_column}' est absente après filtrage.")
        return None, None
    
    variables = data_test.drop(columns=[target_column])
    X_scaled = scaler.transform(variables)
    predictions = model.predict(X_scaled)
    df_pred = pd.DataFrame(predictions, columns=["Prédictions"], index=variables.index)
    df_test = pd.concat([variables, df_pred], axis=1)
    
    return df_test, variables

st.title("🔍 Prédiction et Analyse des Données")

uploaded_file = st.file_uploader("📂 Téléchargez votre fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    data_boiry = pd.read_excel(uploaded_file)
    st.success("✅ Fichier chargé avec succès !")
    st.dataframe(data_boiry.head())
    
    model_path = "xgb_model_cb22-23-24_10_param.joblib"
    scaler_path = "scaler_cb22-23-24_10_param.pkl"
    target_column = "Conso NRJ Usine (kwh/tcossette)"

    df_results = None
    variables = process_boiry_data(data_boiry)  # Extraction des variables pour affichage

    if st.button("🚀 Lancer la prédiction"):
        with st.spinner("📊 Calcul en cours..."):
            df_results, variables = process_and_predict(data_boiry, model_path, scaler_path, target_column)
            if df_results is not None:
                st.success("✅ Prédictions terminées !")
                st.dataframe(df_results.head())

                # Graphique des prédictions
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_results.index, df_results["Prédictions"], color="red", label='Prédiction CB24', alpha=0.6)
                ax.set_title("📉 Prédiction de la Consommation d'Énergie")
                ax.set_xlabel("Index")
                ax.set_ylabel("Conso NRJ (kWh/tcossette)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Bouton de téléchargement
                st.download_button(
                    label="💾 Télécharger les résultats",
                    data=convert_df_to_excel(df_results),
                    file_name="predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # Affichage des tendances des variables
    st.subheader("📊 Tendances des Variables")
    numeric_columns = variables.select_dtypes(include=["number"]).columns

    if len(numeric_columns) > 0:
        num_cols = 2  # Nombre de colonnes pour l'affichage
        num_vars = len(numeric_columns)
        rows = (num_vars // num_cols) + (num_vars % num_cols > 0)

        fig, axes = plt.subplots(rows, num_cols, figsize=(12, 4 * rows))
        axes = axes.flatten()  # Convertir en tableau 1D pour itération facile

        for idx, col in enumerate(numeric_columns):
            axes[idx].plot(variables.index, variables[col], label=col, alpha=0.7)
            axes[idx].set_title(f"Tendance : {col}")
            axes[idx].set_xlabel("Index")
            axes[idx].set_ylabel(col)
            axes[idx].grid(True)
            axes[idx].legend()

        # Supprimer les axes vides si le nombre de variables est impair
        for idx in range(num_vars, len(axes)):
            fig.delaxes(axes[idx])

        st.pyplot(fig)
