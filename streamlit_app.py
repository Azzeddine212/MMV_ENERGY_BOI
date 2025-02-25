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

st.title("🔍 Prédiction de la Consommation d'Énergie")

uploaded_file = st.file
