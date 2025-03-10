import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from PIL import Image
import base64
import io
import warnings

# Désactiver les avertissements
warnings.filterwarnings("ignore")


# Configuration de la page en mode large
st.set_page_config(page_title="🔍 Prédiction de la Consommation d'Énergie BOIRY", layout="wide")

# Fonction pour ajouter une image en arrière-plan via CSS
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Ajouter l'image en arrière-plan
add_bg_from_local('interface.jpg')

# Convertir un DataFrame en fichier Excel pour téléchargement
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Prédictions')
    output.seek(0)
    return output.read()

# Traitement des données de Boiry
def process_boiry_data(df_boiry):
    def moyenne_pondérée(valeur_1, valeur_2, poid_1, poid_2):
        return (valeur_1 * poid_1 + valeur_2 * poid_2) / (poid_1 + poid_2)
    
    df_boiry['Soutirage_tot'] = df_boiry['Soutirage 9m'] + df_boiry['Soutirage 11m']
    df_boiry['Temp entrée JAE_moy'] = df_boiry.apply(lambda row: moyenne_pondérée(row['Temp entrée JAE A'], row['Temp entrée JAE B'], row['Débit JAE A'], row['Débit JAE B']), axis=1)
    df_boiry['Temp sortie JAE_moy'] = df_boiry.apply(lambda row: moyenne_pondérée(row['Temp sortie JAE A'], row['Temp sortie JAE B'], row['Débit JAE A'], row['Débit JAE B']), axis=1)
    df_boiry['Débit JAE_tot'] = df_boiry['Débit JAE A'] + df_boiry['Débit JAE B']
    df_boiry['Débit vapeur_tot'] = df_boiry['Débit vapeur 140T'] + df_boiry['Débit vapeur 120T']
    
    # Sélection des colonnes moyennées
    df_boiry= df_boiry[['Date','Tonnage', 'Température', 'Soutirage_tot', 'Temp jus TEJC',
       'Débit jus chaulé', 'Temp jus chaulé', 'Débit JC1', 'Temp JC1 ech 1',
       'Temp JC1 ech 2', '% condenseur', 'Temp entrée JAE_moy',
       'Temp sortie JAE_moy', 'Débit JAE_tot', 'Débit sirop 5C',
       'Débit sirop stocké', 'Pression VE', 'Débit SBP', 'Débit refonte',
       'Débit sucre', 'Richesse cossettes - BOI & ART (g%g)',
       'JAE - Brix poids (g%g)', 'Sirop sortie évapo-Brix poids (g%g)',
       'LS1 - Brix poids (g%g)', 'LS1 concentrée - Brix poids (g%g)',
       'SBP - Brix (g%g)', 'SBP instantané - Brix (g%g)',
       'Débit eau_tot', 'Débit vapeur_tot', 'Temp fumée_moy','Energie KWh 0°C']]
    
    df_boiry['Energie kWh 0°C_pci'] = df_boiry['Energie KWh 0°C'] * 0.9
    df_boiry['Conso NRJ Usine (kwh/tcossette)'] = df_boiry['Energie kWh 0°C_pci'] / df_boiry['Tonnage']
    
    df_boiry.reset_index(drop=True, inplace=True)

    # Ajout des données de la chaufferie
    df_boiry['Temp fumée_moy'] = df_boiry['Temp fumée_moy']
    df_boiry.reset_index(drop=True, inplace=True)

    return df_boiry

# Chargement du modèle et prédiction
def process_and_predict(input_data, df_lim, model_path, scaler_path, target_column):
    model = joblib.load(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    data_test = process_boiry_data(input_data)
    data_test["Date"] = pd.to_datetime(data_test["Date"])
    data_test.set_index("Date", inplace=True)
    data_test = data_test[df_lim.columns.intersection(data_test.columns)]
    
    valeurs_hors_limites = {}
    for col in data_test.columns:
        if col in df_lim.columns:
            valeurs_hors_min = (data_test[col] < df_lim.loc['min', col]).sum()
            valeurs_hors_max = (data_test[col] > df_lim.loc['max', col]).sum()
            if valeurs_hors_min > 0 or valeurs_hors_max > 0:
                valeurs_hors_limites[col] = (valeurs_hors_min, valeurs_hors_max)
    
    for col in data_test.columns:
        if col in df_lim.columns:
            data_test = data_test[(data_test[col] >= df_lim.loc['min', col]) & (data_test[col] <= df_lim.loc['max', col])]
    
    if target_column not in data_test.columns:
        st.error(f"La colonne cible '{target_column}' est absente après filtrage.")
        return None
    
    variables = data_test.drop(columns=[target_column])
    X_scaled = scaler.transform(variables)
    predictions = model.predict(X_scaled)
    df_pred = pd.DataFrame(predictions, columns=["Prédictions"], index= variables.index)
    df_test = pd.concat([variables, df_pred], axis=1)
    
    return df_test, variables

# Ajout d'un panneau latéral
st.sidebar.title("🔍 Entrainement Analyse et Prédiction")

# Téléchargement du fichier Excel
uploaded_file = st.sidebar.file_uploader("📂 Téléchargez votre fichier Excel", type=["xlsx"])


# Titre de l'application
#st.title("Prédiction & Analyse de la Consommation d'Énergie BOIRY")
#st.markdown("""<h2 style="text-align: center; font-size: 42px;">PREDICTION ET ANALYSE DE LA CONSOMMATION ENERGETIQUE A TEREOS BOIRY</h2>""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #003366;;'>PRÉDICTION & ANALYSE DE LA CONSOMMATION ÉNERGÉTIQUE – TEREOS BOIRY</h1>", unsafe_allow_html=True)

if uploaded_file is not None:

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
    
    #data_boiry = pd.read_excel(uploaded_file, index_col='Date')
    data_boiry = pd.read_excel(uploaded_file)
    df_results, variables = process_and_predict(data_boiry, df_lim, model_path, scaler_path, target_column)
    st.sidebar.success("✅ Fichier chargé avec succès !")
    st.sidebar.success("✅ Exploration et traitement des données effectués avec succès !")
    
    # Input pour définir l'objectif
    objectif = st.sidebar.number_input("🔢 Entrez l'objectif de consommation énergétique (kWh)", min_value=100, max_value=250)
    prix_gn = st.sidebar.number_input("🔢 Entrez l'objectif le prix du Mwh Gaz Naturel (€/MWh)", min_value=0, max_value=250)
    df_results, variables = process_and_predict(data_boiry, df_lim, model_path, scaler_path, target_column)
    if st.sidebar.button("🚀 Lancer la prédiction"):
        with st.spinner("📊 Calcul en cours..."):
                 
            #df_results, variables = process_and_predict(data_boiry, df_lim, model_path, scaler_path, target_column)
            if df_results is not None:
                
                st.sidebar.success("✅ Prédictions terminées !")
                
    page = st.sidebar.radio("Sélectionnez une page :", ["📈 Tableau de Bord", "📥 Télécharger"])
    
    if page == "📈 Tableau de Bord":
        
        col1, col2, = st.columns([3, 2]) # 2 colonnes avec un ratio de largeur
          
        with col1:
            #st.header("📊 Prédiction & Analyse ")
            st.markdown("<h1 style='text-align: center; color: #003366; font-size: 28px;'>🔍 Prédiction & Résultats</h1>", unsafe_allow_html=True)
        
            # Affichage des statistiques
            #moyenne = df_results["Prédictions"].mean()
            #mediane = df_results["Prédictions"].median()
            #ecart_type = df_results["Prédictions"].std()
            #st.write(f"**Moyenne:** {moyenne:.2f} kWh")
            #st.write(f"**Médiane:** {mediane:.2f} kWh")
            #st.write(f"**Écart-type:** {ecart_type:.2f} kWh")

            # Plotting the predictions
            fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
            mean = df_results["Prédictions"].mean()
            std_dev = df_results["Prédictions"].std()
            upper_limit = mean + 2 * std_dev
            lower_limit = mean - 2 * std_dev
    
            # Ajouter une ligne horizontale représentant l'objectif
            ax.axhline(y=objectif, color="red", linestyle="--", linewidth=2, label=f'Objectif : {objectif} kWh')
    
            # Identifier et marquer les points au-dessus de l'objectif
            au_dessus = df_results["Prédictions"] > objectif  # Masque booléen
            ax.scatter(df_results.index[au_dessus], df_results["Prédictions"][au_dessus], color="red", label="Au-dessus de l'objectif", zorder=3)
    
            ax.axhline(upper_limit, color="green", linestyle="dashed", linewidth=1, label=f"Mean + 2σ = {upper_limit:.2f}")
            ax.axhline(lower_limit, color="green", linestyle="dashed", linewidth=1, label=f"Mean - 2σ = {lower_limit:.2f}")
            ax.plot(df_results.index, df_results["Prédictions"], color="blue", label='Prédiction CB24', alpha=0.6)
            #ax.bar(df_results.index, df_results["Prédictions"], color="red", label='Prédiction CB24', alpha=0.6)
            ax.set_title("Prédiction CB24")
            ax.set_xlabel("Date")
            ax.set_ylabel("Conso NRJ (kWh/tcossette)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig,use_container_width=False)

            # Vérifier que la colonne "Prédictions" existe
            if "Prédictions" in df_results.columns:
                # Calcul des statistiques
                moyenne = df_results["Prédictions"].mean()
                mediane = df_results["Prédictions"].median()
                ecart_type = df_results["Prédictions"].std()
                
                # Affichage des statistiques
                #st.write(f"**Moyenne:** {moyenne:.2f} kWh")
                #st.write(f"**Médiane:** {mediane:.2f} kWh")
                #st.write(f"**Écart-type:** {ecart_type:.2f} kWh")
                
                # Tracer l'histogramme avec KDE
                fig, ax = plt.subplots(figsize=(20, 10))
                sns.histplot(df_results["Prédictions"], bins=20, kde=True, color='blue', ax=ax)
                
                # Ajouter les statistiques sur le graphique
                ax.axvline(moyenne, color='red', linestyle='--', label=f'Moyenne: {moyenne:.2f} kWh')
                ax.axvline(mediane, color='green', linestyle='--', label=f'Médiane: {mediane:.2f} kWh')
                ax.axvline(moyenne + ecart_type, color='orange', linestyle=':', label=f'Écart-type: {ecart_type:.2f} kWh')
    
                total = df_results["Prédictions"].shape[0]
                for patch in ax.patches:
                    height = patch.get_height()
                    width = patch.get_width()
                    x_position = patch.get_x() + width / 2
                    percentage = (height / total) * 100
                    ax.text(x_position, height + 5, f'{percentage:.1f}%', ha='center', fontsize=7)
                
                # Ajouter des titres et labels
                ax.set_title("Histogramme des Prédictions de Consommation Énergétique", fontsize=14)
                ax.set_xlabel("Consommation Énergétique (kWh)", fontsize=12)
                ax.set_ylabel("Densité", fontsize=12)
                ax.legend()
                
                # Affichage du graphique dans Streamlit
                st.pyplot(fig,use_container_width=False)
            else:
                st.error("Le fichier ne contient pas de colonne 'Prédictions'. Veuillez vérifier vos données.")

   
                    
        with col2:
            #st.header("📈 Tendances des Variables ")
            st.markdown("<h1 style='text-align: center; color: #003366; font-size: 28px;'>📈 Analyse & Tendance</h1>", unsafe_allow_html=True)
            
            # Définir 'available_vars' comme étant les colonnes du DataFrame df_results
            available_vars = df_results.columns.tolist()
        
            # Sélection de 2 variables via sidebar
            st.sidebar.header("🔧 Sélection des Variables")
            selected_vars = st.sidebar.multiselect("Choisissez **six** variables :", available_vars, default=available_vars[:6])
            
            # Assurer toujours deux éléments (None si insuffisants)
            selected_vars = selected_vars[:6] + [None] * (6 - len(selected_vars))
        
            #st.subheader("📊 Tendances des Variables avec Seuils ± 3σ")
            # Création de la figure avec 3 lignes et 2 colonnes
            fig, axes = plt.subplots(3, 2, figsize=(15, 20))  # Hauteur augmentée pour un meilleur affichage
            axes = axes.flatten()  # Aplatir en 1D pour indexation plus facile
            
            for idx, col in enumerate(selected_vars):
                if col is not None:
                    mean = df_results[col].mean()
                    std_dev = df_results[col].std()
                    upper_limit = mean + 3 * std_dev
                    lower_limit = mean - 3 * std_dev
            
                    axes[idx].plot(df_results.index, df_results[col], color="blue", alpha=0.6, label=col)
                    axes[idx].axhline(upper_limit, color="red", linestyle="dashed", linewidth=1, label=f"Mean + 3σ = {upper_limit:.2f}")
                    axes[idx].axhline(lower_limit, color="red", linestyle="dashed", linewidth=1, label=f"Mean - 3σ = {lower_limit:.2f}")
                    axes[idx].set_title(f"Tendance : {col}")
                    axes[idx].set_xlabel("Date")
                    axes[idx].set_ylabel(col)
                    axes[idx].legend()
                    axes[idx].grid(True)
                    axes[idx].tick_params(axis="x", rotation=45)
                else:
                    axes[idx].set_visible(False)  # Masquer proprement l'axe vide
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            #st.header("📊 Bilan & Résultats")
            st.dataframe(df_results["Prédictions"].describe().to_frame().T)
            
            # Filtrer les lignes où "Prédictions" est supérieure à l'objectif
            df_surco = df_results[df_results["Prédictions"] > objectif].copy()
            # Filtrer les lignes où "Prédictions" est inférieure à l'objectif
            df_sousco = df_results[df_results["Prédictions"] < objectif].copy()
        
            # Calculer la surconsommation d'énergie
            df_surco["NRJ_surconsommée"] = abs(df_surco["Prédictions"] - objectif) * df_surco["Tonnage"]
            
            # Calculer la sousconsommation d'énergie
            df_sousco["NRJ_sousconsommée"] = abs(df_sousco["Prédictions"] - objectif) * df_sousco["Tonnage"]
            
            # Calculer le total des surconsommations et sousconsommations
            surenergie_totale = df_surco["NRJ_surconsommée"].sum() / 1000
            sousenergie_totale = df_sousco["NRJ_sousconsommée"].sum() / 1000
            
            # Calculer les coûts associés
            surcout_totale = (df_surco["NRJ_surconsommée"].sum() / 1000) * prix_gn / 1000
            souscout_totale = (df_sousco["NRJ_sousconsommée"].sum() / 1000) * prix_gn / 1000
            
            # Calcul des totaux nets
            energie_totale = surenergie_totale - sousenergie_totale
            cout_NRJ = surcout_totale - souscout_totale
            
            # Messages à afficher
            if energie_totale > 0:
                message_5 = f"⚡ La quantité d'énergie surconsommée par rapport à l'objectif est : {energie_totale:.2f} MWh 📈"
                message_6 = f"💰 Le coût total de sur-consommation d'énergie est : {cout_NRJ:.2f} k€ 📉"
            elif energie_totale < 0:
                message_5 = f"⚡ La quantité d'énergie sous-consommée par rapport à l'objectif est : {abs(energie_totale):.2f} MWh 📉"
                message_6 = f"💰 Le coût total de sous-consommation d'énergie est : {abs(cout_NRJ):.2f} k€ 📉"
            else:
                message_5 = f"⚡ La quantité d'énergie consommée est égale à l'objectif : {energie_totale:.2f} MWh ✅"
                message_6 = f"💰 Le coût total d'énergie consommée est égale à l'objectif : {cout_NRJ:.2f} k€ ✅"
            
            # Afficher les résultats dans un cadre blanc
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 8px; 
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #2F4F4F; font-size: 16px;">{message_5} <br><br><br> {message_6}</h3>
            </div>
            """, unsafe_allow_html=True)
                    
    # --- Page Téléchargement ---
    elif page == "📥 Télécharger":
        st.title("📥 Télécharger les Résultats")
    
        # Download button for Excel
        st.download_button(
            label="💾 Télécharger les résultats",
            data=convert_df_to_excel(df_results),  # Convert the DataFrame to Excel bytes
            file_name="predictions.xlsx",  # File name with .xlsx extension
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # MIME type for Excel
        )
