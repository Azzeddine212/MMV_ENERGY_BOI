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

# Convert the DataFrame to an Excel file in memory
def convert_df_to_excel(df):
    """Convert DataFrame to Excel format for downloading."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Predictions')
    output.seek(0)  # Reset the pointer to the beginning of the buffer
    return output.read()

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

def process_and_predict(input_data, df_lim, model_path, scaler_path, target_column):
    """Chargement du modèle, prédiction et affichage des résultats"""
    model = joblib.load(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    data_test = process_boiry_data(input_data)
    data_test = data_test[df_lim.columns.intersection(data_test.columns)]
    
    # Identifying out-of-bound values
    valeurs_hors_limites = {}
    for col in data_test.columns:
        if col in df_lim.columns:
            valeurs_hors_min = (data_test[col] < df_lim.loc['min', col]).sum()
            valeurs_hors_max = (data_test[col] > df_lim.loc['max', col]).sum()
            if valeurs_hors_min > 0 or valeurs_hors_max > 0:
                valeurs_hors_limites[col] = (valeurs_hors_min, valeurs_hors_max)
    
    # Filtering the data within the limits
    for col in data_test.columns:
        if col in df_lim.columns:
            data_test = data_test[(data_test[col] >= df_lim.loc['min', col]) & (data_test[col] <= df_lim.loc['max', col])]
    
    if target_column not in data_test.columns:
        st.error(f"La colonne cible '{target_column}' est absente après filtrage.")
        return None
    
    variables = data_test.drop(columns=[target_column])
    X_scaled = scaler.transform(variables)
    predictions = model.predict(X_scaled)
    df_pred = pd.DataFrame(predictions, columns=["Prédictions"], index=variables.index)
    df_test = pd.concat([variables, df_pred], axis=1)
    
    return df_test, variables

st.title("🔍 Prédiction de la Consommation d'Énergie")

uploaded_file = st.file_uploader("📂 Téléchargez votre fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    data_boiry = pd.read_excel(uploaded_file)
    st.success("✅ Fichier chargé avec succès !")
    st.dataframe(data_boiry.head())
    
    model_path = "xgb_model_cb22-23-24_10_param.joblib"
    scaler_path = "scaler_cb22-23-24_10_param.pkl"
    target_column = "Conso NRJ Usine (kwh/tcossette)"
    
    df_lim = pd.DataFrame({
        "Tonnage": [500, 900], "Température": [-2, 50],
        "Richesse cossettes - BOI & ART (g%g)": [14, 20], "Débit JC1": [650, 1250],
        "Pression VE": [2, 3.4], "JAE - Brix poids (g%g)": [11, 20],
        "Sirop sortie évapo-Brix poids (g%g)": [60, 80], "Débit sucre": [40, 136],
        "Débit vapeur_tot": [140, 200], "Temp fumée_moy": [80, 174],
        "Conso NRJ Usine (kwh/tcossette)": [125, 205]
    }, index=["min", "max"])
    
    if st.button("🚀 Lancer la prédiction"):
        with st.spinner("📊 Calcul en cours..."):
            df_results, variables = process_and_predict(data_boiry, df_lim, model_path, scaler_path, target_column)
            if df_results is not None:
                st.success("✅ Prédictions terminées !")
                st.dataframe(df_results.head())
                
                # Plotting the predictions
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_results.index, df_results["Prédictions"], color="red", label='Prédiction CB24', alpha=0.6)
                ax.set_title("Prédiction CB24")
                ax.set_xlabel("Date")
                ax.set_ylabel("Conso NRJ (kWh/tcossette)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Vérification des colonnes numériques disponibles
                numeric_columns = variables.select_dtypes(include=["number"]).columns
                
                if len(numeric_columns) > 0:
                    selected_column = st.selectbox("📌 Sélectionnez une colonne numérique :", numeric_columns)
                
                    # Sélecteur de couleur pour la courbe
                    selected_color = st.color_picker("🎨 Choisissez une couleur pour la courbe :", "#FF0000")  # Rouge par défaut
                
                    # Bouton pour lancer l'affichage
                    if st.button("🚀 Évaluation des tendances des variables"):
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(variables.index, variables[selected_column], color=selected_color, label='Prédiction CB24', alpha=0.6)
                        ax.set_title(f"Tendance de {selected_column}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel(selected_column)
                        ax.legend()
                        ax.grid(True)
                
                        # Affichage du graphique
                        st.pyplot(fig)
else:
    st.warning("⚠️ Aucune colonne numérique disponible dans les données.")

                # Plotting each variable
                #fig, axes = plt.subplots(len(variables.columns), 1, figsize=(10, 5 * len(variables.columns)))
                
                # If there is only one column, axes will be a single object, not an array
                #if len(variables.columns) == 1:
                    #axes = [axes]
                
                #for i, col in enumerate(variables.columns):
                    #axes[i].plot(variables.index, variables[col], color="blue", alpha=0.6, label=col)
                    #axes[i].set_title(col)
                    #axes[i].set_xlabel("Date")
                    #axes[i].set_ylabel(col)
                    #axes[i].legend()
                    #axes[i].grid(True)
                
                #plt.tight_layout()
                #st.pyplot(fig)
                
                # Download button for Excel
                st.download_button(
                    label="💾 Télécharger les résultats",
                    data=convert_df_to_excel(df_results),  # Convert the DataFrame to Excel bytes
                    file_name="predictions.xlsx",  # File name with .xlsx extension
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # MIME type for Excel
                )
