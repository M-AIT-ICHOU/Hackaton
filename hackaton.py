# Dans le terminal exécuter : streamlit run 2.py

import io
import os
import time
import random
import pstats
import leafmap
import rasterio
import cProfile
import scipy.stats
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from numba import njit
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import leafmap.foliumap as leafmap
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from shapely.ops import unary_union
from shapely.strtree import STRtree
from rasterio.features import shapes
from streamlit_folium import st_folium
from shapely.geometry import LineString
from math import radians, sin, cos, sqrt
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import leafmap.foliumap as leafmap

# --- Détection de localtileserver + fonction de fallback (DOIT être définie avant l'usage) ---
try:
    import localtileserver as _lts  # noqa: F401
    LTS_AVAILABLE = True
    LTS_VERSION = getattr(_lts, "__version__", None)
except Exception:
    LTS_AVAILABLE = False
    LTS_VERSION = None

def _preview_raster_statique(raster_path, title=None, colormap="viridis"):
    """Affiche un aperçu statique d'un raster via rasterio + matplotlib (fallback si localtileserver absent)."""
    try:
        import rasterio
        import matplotlib.pyplot as _plt
        import numpy as _np
        if not os.path.exists(raster_path):
            st.error(f"Fichier introuvable : {raster_path}")
            return
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype(float)
            arr[~_np.isfinite(arr)] = _np.nan
            fig, ax = _plt.subplots(figsize=(8, 5))
            im = ax.imshow(arr, cmap=colormap, origin="upper")
            ax.set_title(title or f"Aperçu : {os.path.basename(raster_path)}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            _plt.close(fig)
    except Exception as e:
        st.error(f"Aperçu statique impossible : {e}")

st.set_page_config(layout="wide")
data_dir = "/workspaces/Hackaton"

# --- Gestion centralisée des fonds de carte ---
fond_leafmap = {
    "OpenStreetMap": "OpenStreetMap",
    "Esri Satellite": "Esri Satellite",
    "Google Satellite": "HYBRID",
    "OpenTopoMap": "OpenTopoMap",
    "CartoDB Positron": "CartoDB.Positron",
    "CartoDB dark_matter": "CartoDB.DarkMatter"
}

# --- Définition unique des fonds de carte ---
tiles_dict = {
    "OpenStreetMap": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors"
    },
        "Esri Satellite": {  
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
    },
    "Google Satellite": {
        "tiles": "http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "© Google"
    },
    "OpenTopoMap": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "© OpenTopoMap contributors"
    },
    "CartoDB Positron": {
        "tiles": "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors © CARTO"
    },
    "CartoDB dark_matter": {
        "tiles": "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors © CARTO"
    }
}
fond_options = list(tiles_dict.keys())

# --- Initialisation du fond de carte dans session_state ---
if "fond_carte" not in st.session_state:
    st.session_state["fond_carte"] = fond_options[0]

# --- Ajoute un espace vertical avant les logos et le titre ---
st.markdown("<div style='height:4em;'></div>", unsafe_allow_html=True)

# --- Chemin des données ---
data_dir = "/workspaces/Hackaton"

# --- Logo et titre ---
col1, col2, col3 = st.columns([2, 6, 2])
with col1:
    logo1_path = os.path.join(data_dir, "Documents d'aide", "CrigeBaseline_redim.png")
    if os.path.exists(logo1_path):
        st.image(logo1_path, use_container_width=False, width=150)
    else:
        # évite l'erreur si le fichier est absent
        st.write("") 

with col2:
    st.markdown("""
    <div style='display:flex;align-items:center;justify-content:center;width:100%;'>
    <span style='font-size:2.8em;font-weight:bold;color:#fff;text-align:center;line-height:50px;flex:1;'>
        Dépérissement forestier, une histoire de marges
    </span>""",
        unsafe_allow_html=True
    )
with col3:
    logo2_path = os.path.join(data_dir, "Documents d'aide", "Logo geodatalab 2025.png")
    if os.path.exists(logo2_path):
        st.image(logo2_path, use_container_width=False, width=150)
    else:
        st.write("")


# --- Paramètre titre ---
st.markdown("""
<style>
.titre-principal {
    /* background: #181818 !important; */
    border-radius: 12px !important;
    box-shadow: none !important;
    margin-bottom: 0.7 em !important;
    padding: 1.5em 0 1em 0 !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# --- Styles personnalisés pour Streamlit ---
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
h1, h2, h3 {color: #b22222;}
.stButton>button {background-color: #A0C989; color: white;}
/* Barre d'onglets fond noir, sans contour, largeur quasi totale, hauteur réduite */
.stTabs [data-baseweb="tab-list"] {
    border-radius: 16px !important;
    margin-top: 0.5em !important;
    margin-left: -1vw !important;
    margin-right: -1vw !important;
    width: 102vw !important;
    min-width: 0 !important;
    left: 0 !important;
    transform: none !important;
    position: relative;
    z-index: 10;
    padding-top: 0.1em !important;
    padding-bottom: 0.1em !important;
    height: 3.2em !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.35em;      /* Augmente la taille */
    color: white !important;
    font-weight: bold;      /* Met en gras */
    border-radius: 12px !important;
    margin: 0 0.2em;
    padding: 0.5em 1.5em;   /* Optionnel : agrandit la zone cliquable */
    transition: background 0.2s, border 0.2s;
    height: 2.8em !important;
    line-height: 2.6em !important;
    min-width: 130px;
}
.stTabs [aria-selected="true"] {
    color: white !important;
    border-radius: 12px 12px 0 0 !important;
    z-index: 11;
}
</style>
""", unsafe_allow_html=True)

# --- Barre d'onglets personnalisée ---
st.markdown("""
<style>
.barre-onglets {
    background: #181818 !important;
    border-radius: 8px !important;
    padding: 0.5em 0 !important;
    margin-bottom: 1em !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# --- Fonction pour espace entre le label et la boîte de sélection  ---
st.markdown("""
<style>
[data-testid="stNumberInput"] > label {margin-bottom: -0.7em;}
</style>
""", unsafe_allow_html=True)


# AJOUTE ICI l'initialisation du fond de carte
if "fond_carte" not in st.session_state:
    st.session_state["fond_carte"] = "OpenStreetMap"

# --- Sidebar pour les paramètres de simulation ---
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;'>
        <div style='font-size:1.9em;font-weight:bold;margin-bottom:1.2em;letter-spacing:0.04em;'>Paramètres de simulation</div>
    </div>
    <div style='font-size:1.15em;color:#b22222;margin-bottom:1.2em;margin-left:0.1em;text-align:left;'>
        Dans la région Provences-Alpes-Côtes-d'Azur
    </div>
    """, unsafe_allow_html=True)
    
    # Ajoute le sélecteur de fond de carte
    fond_selected = st.selectbox(
        "Choix du fond de carte",
        fond_options,
        index=fond_options.index(st.session_state.get("fond_carte", fond_options[0])),
        key="fond_carte",
    )

    couche_options = [
        "Espèces forestières occurences - PACA",
        "Inventaire forestier national PACA",
        # Ajoute ici les autres couches si besoin
    ]
    selected_couches = st.multiselect(
        "Afficher des couches supplémentaires",
        options=couche_options,
        default=[]
    )
    # Liste des sous-dossiers d'espèces
    especes_dir = os.path.join(data_dir, "Niche écologique par espèces")
    # Liste des sous-dossiers d'espèces (affichage propre, sans "niche_")
    especes_raw = [d for d in os.listdir(especes_dir) if os.path.isdir(os.path.join(especes_dir, d))]
    especes_list = [
        " ".join([w.capitalize() for w in d.replace("niche_", "").split("_")])
        for d in especes_raw
    ]
    espece_selected_label = st.selectbox(
        "Sélectionnez une espèce à visualiser",
        options=especes_list,
        key="espece_selected"
    )
    # Pour retrouver le nom du dossier sélectionné :
    espece_selected = especes_raw[especes_list.index(espece_selected_label)]

    # Liste des rasters dans le sous-dossier sélectionné
    raster_dir = os.path.join(especes_dir, espece_selected)
    raster_files = [f for f in os.listdir(raster_dir) if f.endswith(".tif")]

    # Mapping suffixe -> label
    suffix_map = {
        "dif": "diff",
        "present": "present",
        "futur": "futur", 
        "marge": "marge"
    }
    raster_options = {}
    for f in raster_files:
        for suf, label in suffix_map.items():
            if f"_{suf}." in f:
                raster_options[label] = f
    # Ordre d'affichage
    radio_labels = ["diff", "present", "futur", "marge"]
    raster_labels = [lbl for lbl in radio_labels if lbl in raster_options]
    raster_selected_label = st.radio(
        "Choisissez le scénario de niche à afficher",
        options=raster_labels,
        key="raster_selected_label",
        horizontal=True
    )
    if raster_labels and raster_selected_label is not None:
        raster_selected = raster_options[raster_selected_label]
    else:
        raster_selected = None

    from matplotlib.colors import LinearSegmentedColormap
    
    # Dégradé personnalisé (rouge → orange → gris → bleu → vert)
    foret_marges_cmap = LinearSegmentedColormap.from_list(
        "ForetMarges",
        ["#d32f2f", "#fbc02d", "#bdbdbd", "#1976d2", "#388e3c"],
        N=256
    )

    # Sélecteur de palette de couleurs pour la carte raster
    colormap_options = [
        "ForetMarges", "viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu", "YlOrRd", "Greens", "Blues", "Reds", "Purples", "ForetMarges"
    ]
    colormap_selected = st.selectbox(
        "Palette de couleurs pour la carte raster",
        options=colormap_options,
        index=0,
        key="colormap_selected"
    )

    # Slider pour régler la transparence des couches raster
    opacity_value = st.slider(
        "Transparence des couches raster",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        key="opacity_value"
    )

    # Surveillance mémoire
    import psutil
    mem = psutil.virtual_memory()
    if mem.percent > 80:
        st.warning(f"Attention : la mémoire utilisée par le système est élevée ({mem.percent}%). Cela peut ralentir ou interrompre la simulation.")
    
        st.success("Paramètres validés")
        st.session_state["param_valides"] = True
        st.session_state["recalc"] = True
    

# --- Partie simulation ---
import gc
gc.collect()
tab_sim, tab_analyse, tab_export, tab_aide = st.tabs(["Simulation", "Analyse", "Export", "Aide"])

# --- Onglet Simulation ---
with tab_sim:
    # Initialisation des états Streamlit pour les contrôles
    if "running" not in st.session_state:
        st.session_state["running"] = False
    if "temps" not in st.session_state:
        st.session_state["temps"] = 0

    # Fonction de synchronisation du slider
    def maj_temps():
        st.session_state["temps"] = st.session_state["slider_temps"]

    spinner_placeholder = st.empty()
    carte_placeholder = st.empty()

    def afficher_carte_leafmap(temps, fond, afficher_polygones=True):
        fond_name = fond_leafmap.get(fond, "OpenStreetMap")
        m = leafmap.Map(center=[43.5, 5.5], zoom=9, tiles=None)
        m.add_basemap(fond_name)
    
        # Ajout des couches supplémentaires
        for couche in selected_couches:
            try:
                if couche == "Espèces forestières occurences - PACA":
                    gdf_foret = gpd.read_file(f"{data_dir}occurrences/Espèces forestières occurences - PACA.gpkg").to_crs(epsg=4326)
                    unique_names = gdf_foret["Name"].unique()
                    print("Espèces uniques :", unique_names)
                    print("Nombre d'occurrences par espèce :", gdf_foret["Name"].value_counts())
                    color_map = px.colors.qualitative.Dark24  # Plus de couleurs
                    color_dict = {name: color_map[i % len(color_map)] for i, name in enumerate(unique_names)}
                    for name in unique_names:
                        subset = gdf_foret[gdf_foret["Name"] == name]
                        m.add_gdf(
                            subset,
                            layer_name=f"Espèce : {name}",
                            style={
                                "color": color_dict[name],
                                "fillColor": color_dict[name],
                                "fillOpacity": 0.3,
                                "weight": 1
                            }
                        )

                elif couche == "Inventaire forestier national PACA":
                    gdf_ifn_visu = gpd.read_file(f"{data_dir}Inventaire foret national/ifn_paca.gpkg", layer="arbre_paca").to_crs(epsg=4326)
                    m.add_gdf(
                        gdf_ifn_visu,
                        layer_name="IFN PACA",
                        style={"color": "#1976D2", "fillColor": "#1976D2", "fillOpacity": 0.1, "weight": 1}
                    )
            except Exception as e:
                st.warning(f"Impossible d'afficher la couche {couche} : {e}")

        # Ajout du raster du sous-dossier sélectionné sur la carte Leafmap
        raster_dir = os.path.join(data_dir, "Niche écologique par espèces", espece_selected)
        raster_files = [f for f in os.listdir(raster_dir) if f.endswith(".tif")]
        
        if raster_selected:
            raster_path = os.path.join(data_dir, "Niche écologique par espèces", espece_selected, raster_selected)

            # --- helper de debug pour le raster ---
            def _check_raster_info(path):
                info = {}
                try:
                    import rasterio
                    if not os.path.exists(path):
                        info["error"] = "Fichier introuvable"
                        return info
                    with rasterio.open(path) as src:
                        info["exists"] = True
                        info["dtype"] = str(src.dtypes[0]) if src.dtypes else None
                        info["shape"] = (src.count, src.height, src.width)
                        info["crs"] = str(src.crs) if src.crs else None
                        info["bounds"] = tuple(src.bounds)
                        try:
                            arr = src.read(1, masked=True).astype(float)
                            mask = ~np.isfinite(arr)
                            info["valid_pixels"] = int(np.ma.count_masked(arr) == 0 and np.count_nonzero(~mask) or np.count_nonzero(~mask))
                            info["min"] = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else None
                            info["max"] = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else None
                            info["nodata"] = src.nodata
                        except Exception as ex:
                            info["read_error"] = str(ex)
                except Exception as ex:
                    info["error"] = str(ex)
                return info

            info = _check_raster_info(raster_path)
            st.sidebar.markdown("**Debug raster**")
            st.sidebar.json(info)

            try:
                colormap_leafmap = colormap_selected
                if colormap_selected == "ForetMarges":
                    colormap_leafmap = "viridis"

                # Si pas de pixels valides, afficher aperçu statique et avertir
                if info.get("valid_pixels", 0) == 0:
                    st.warning("Le raster contient peu ou pas de pixels valides — affichage statique à la place.")
                    _preview_raster_statique(raster_path, title=raster_selected, colormap=colormap_leafmap)
                else:
                    # tentative normale via leafmap / localtileserver
                    try:
                        m.add_raster(
                            raster_path,
                            layer_name=raster_selected,
                            colormap=colormap_leafmap,
                            opacity=opacity_value
                        )
                    except Exception as e:
                        # log erreur et fallback statique
                        st.error(f"Erreur lors de m.add_raster : {e}")
                        st.warning("Affichage statique de secours (rasterio).")
                        _preview_raster_statique(raster_path, title=raster_selected, colormap=colormap_leafmap)
            except ModuleNotFoundError as me:
                st.error(f"Module manquant : {me}")
                raise me
            except Exception as e:
                estr = str(e).lower()
                if "localtileserver" in estr:
                    st.error(
                        "Le package localtileserver est requis pour l'affichage tuilé interactif via leafmap.\n"
                        "Installez-le : `source .venv/bin/activate && pip install --prefer-binary localtileserver`"
                    )
                else:
                    st.warning(f"Impossible d'afficher le raster {raster_selected} : {e}")
        else:
            st.warning("Aucun raster sélectionné ou disponible pour cette espèce et ce scénario.")
        
        raster_dir = os.path.join(data_dir, "Niche écologique par espèces", espece_selected)
        if raster_selected:
            raster_path = os.path.join(raster_dir, raster_selected)
            # Cherche le fichier _aire.txt associé au raster sélectionné
            base_name = os.path.splitext(raster_selected)[0]
            aire_txt_path = os.path.join(raster_dir, f"{base_name}_aire.txt")
            if os.path.exists(aire_txt_path):
                with open(aire_txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                classe_vals = []
                for line in lines:
                    if line.startswith("Classe"):
                        parts = line.strip().split(":")
                        classe = parts[0].replace("Classe ", "")
                        aire = parts[1].replace("km²", "").strip()
                        classe_vals.append({"Classe": classe, "Aire bioclimatique (km²)": aire})
                df_air = pd.DataFrame(classe_vals)
                st.markdown("#### Aire bioclimatique par classe")
                st.dataframe(df_air, use_container_width=True)
            else:
                if raster_selected and raster_selected.endswith("_marge.tif"):
                    st.warning("Aucun fichier d'aire bioclimatique associé au raster sélectionné.")
        else:
            st.info("Sélectionnez un raster dans la barre latérale pour afficher le tableau d'aire bioclimatique.")
        
        # Ajout des arbres morts (points) sur la carte, style professionnel
        gpkg_dir = os.path.join(data_dir, "Niche écologique par espèces", espece_selected)
        gpkg_files = [f for f in os.listdir(gpkg_dir) if f.endswith(".gpkg")]
        
        for gpkg_file in gpkg_files:
            gpkg_path = os.path.join(gpkg_dir, gpkg_file)
            try:
                gdf = gpd.read_file(gpkg_path)
                # Conversion MultiPoint → Points individuels pour affichage en cercles
                if any(gdf.geom_type == "MultiPoint"):
                    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
                
                # Vérifie que la géométrie finale est bien de type Point
                if not all(gdf.geom_type == "Point"):
                    gdf = gdf[gdf.geom_type == "Point"].copy()
                
                # Ajoute les points sur la carte, petits cercles noirs sans attributs
                m.add_gdf(
                    gdf,
                    layer_name="Arbres morts",
                    style={
                        "color": "#000",         # Contour noir (identique au remplissage)
                        "fillColor": "#000",     # Noir
                        "fillOpacity": 1.0,      # Opaque
                        "weight": 0,             # Pas de contour
                        "radius": 4              # Petit cercle
                    },
                    info_mode=None
                )
            except Exception as e:
                st.warning(f"Impossible d'afficher {gpkg_file} sur la carte : {e}")

        # Affiche la carte dans Streamlit (une seule fois, tous les rasters superposés)
        m.to_streamlit(height=600)

    # Appel de la fonction d'affichage dans l'onglet Simulation
    afficher_carte_leafmap(
        st.session_state["temps"],
        st.session_state["fond_carte"],
        afficher_polygones=True
    )

with tab_analyse:
    st.header("Analyse des résultats")
    st.markdown("""
    ### Modélisation de la niche écologique
    - **SDM (Bioclim, MaxEnt, Random Forest, etc.)** : Utilisation des données BD_Forêt, BD TOPO, IFN, occupation du sol, RGE alti 5m et variables bioclimatiques WorldClim pour modéliser la niche écologique des espèces forestières.
    - **Calcul de la distance au centre de niche** : Méthodes statistiques pour estimer la position des peuplements par rapport au centre ou à la marge de la niche.
    - **Calcul de l’aire bioclimatique** : Détermination des zones bioclimatiques favorables par département.
    - **Régression mortalité vs distance au centre de niche** : Analyse statistique de la relation entre mortalité observée (IFN) et distance à la niche optimale.
    - **Visualisation des zones de mortalité accrue** : Cartographie des marges écologiques et des zones à risque.
    - **Comparaison entre espèces** : Analyse comparative multi-espèces sur la région PACA.
    """)

    st.markdown("#### Sources et variables utilisées pour la modélisation")
    variables_df = pd.DataFrame({
        "Source": [
            "BD Forêt PACA",
            "IFN (Arbres)",
            "IFN (Placettes)",
            "Occupation du sol",
            "BIO1 - Température annuelle moyenne",
            "BIO2 - Plage diurne moyenne",
            "BIO3 - Isothermie",
            "BIO4 - Saisonnalité de la température",
            "BIO5 - Température max du mois le plus chaud",
            "BIO6 - Température min du mois le plus froid",
            "BIO7 - Plage annuelle de température",
            "BIO8 - Température moyenne du trimestre le plus humide",
            "BIO9 - Température moyenne du trimestre le plus sec",
            "BIO10 - Température moyenne du trimestre le plus chaud",
            "BIO11 - Température moyenne du trimestre le plus froid",
            "BIO12 - Précipitations annuelles",
            "BIO13 - Précipitations du mois le plus humide",
            "BIO14 - Précipitations du mois le plus sec",
            "BIO15 - Saisonnalité des précipitations",
            "BIO16 - Précipitations du trimestre le plus humide",
            "BIO17 - Précipitations du trimestre le plus sec",
            "BIO18 - Précipitations du trimestre le plus chaud",
            "BIO19 - Précipitations du trimestre le plus froid"
        ],
        "Format": [
            "Vecteur (polygones)",
            "Vecteur (points)",
            "Vecteur (points)",
            "Raster (.tif)",
            *["Raster (.tif)"]*19
        ],
        "Rôle dans le modèle": [
            "Délimitation des peuplements",
            "Présence et mortalité des arbres",
            "Coordonnées et département",
            "Variable environnementale",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique",
            "Variable bioclimatique"
        ],
        "Fichier": [
            "BD_Foret_PACA/bd_foret_paca.gpkg",
            "Inventaire foret national/ifn_paca.gpkg (arbre_paca)",
            "Inventaire foret national/ifn_paca.gpkg (placette_sans_doublons)",
            "OCS_PACA_cog/ocs_paca_cog.tif",
            "world_clim/BIO1.tif",
            "world_clim/BIO2.tif",
            "world_clim/BIO3.tif",
            "world_clim/BIO4.tif",
            "world_clim/BIO5.tif",
            "world_clim/BIO6.tif",
            "world_clim/BIO7.tif",
            "world_clim/BIO8.tif",
            "world_clim/BIO9.tif",
            "world_clim/BIO10.tif",
            "world_clim/BIO11.tif",
            "world_clim/BIO12.tif",
            "world_clim/BIO13.tif",
            "world_clim/BIO14.tif",
            "world_clim/BIO15.tif",
            "world_clim/BIO16.tif",
            "world_clim/BIO17.tif",
            "world_clim/BIO18.tif",
            "world_clim/BIO19.tif"
        ]
    })
    st.dataframe(variables_df, use_container_width=True)
    
    # Distribution des pixels du raster sélectionné
    st.markdown("#### Distribution des valeurs du raster sélectionné")
    
    raster_dir = os.path.join(data_dir, "Niche écologique par espèces", espece_selected)
    if raster_selected:
        raster_path = os.path.join(raster_dir, raster_selected)
        if os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                arr = src.read(1)
                arr = arr[np.isfinite(arr)]  # Garde uniquement les valeurs finies
                fig, ax = plt.subplots(figsize=(7, 3))
                # Courbe de densité (distribution)
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(arr)
                x_vals = np.linspace(np.nanmin(arr), np.nanmax(arr), 200)
                ax.plot(x_vals, kde(x_vals), color="#1976D2", lw=2)
                ax.fill_between(x_vals, kde(x_vals), color="#1976D2", alpha=0.2)
                ax.set_title(f"Densité des valeurs du raster : {raster_selected_label}", fontsize=14)
                ax.set_xlabel("Valeur du pixel")
                ax.set_ylabel("Densité")
                st.pyplot(fig)
                st.markdown(f"**Min :** {np.nanmin(arr):.3f} &nbsp;&nbsp; **Max :** {np.nanmax(arr):.3f} &nbsp;&nbsp; **Moyenne :** {np.nanmean(arr):.3f}")
        else:
            st.warning("Le fichier raster sélectionné n'existe pas.")
    else:
        st.info("Sélectionnez un raster dans la barre latérale pour afficher sa distribution.")

with tab_export:
    st.header("Export de la couche sélectionnée")
    st.markdown("Exportez une carte prête à partager avec une sémiologie nette et universelle.")

    # --- chemins
    raster_dir = os.path.join(data_dir, "Niche écologique par espèces", espece_selected)

    # --- palette discrète (5 classes)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Rectangle
    from math import cos, radians

    couleurs = ["#d32f2f", "#fbc02d", "#bdbdbd", "#1976d2", "#388e3c"]
    labels_5 = [
        "Marge inférieure",
        "Marge inf. intermédiaire",
        "Centre de la niche",
        "Marge sup. intermédiaire",
        "Marge supérieure"
    ]
    cmap5 = ListedColormap(couleurs)

    # ---------- helpers échelle (graduée) ----------
    def _metres_to_pixels(src, metres: float) -> int:
        """Convertit des mètres en pixels sur l'axe X du raster."""
        t = src.transform
        px_w = abs(t.a)  # taille pixel horizontale (unités CRS)
        if src.crs and src.crs.is_projected:
            # unités en mètres
            return max(1, int(round(metres / px_w)))
        # CRS géographique (degrés) -> approx mètres/° en lon à la latitude médiane
        bounds = src.bounds
        mid_lat = (bounds.bottom + bounds.top) / 2.0
        metres_par_deg_lon = 111320.0 * cos(radians(mid_lat))
        deg_lon = metres / max(1e-9, metres_par_deg_lon)
        return max(1, int(round(deg_lon / px_w)))

    def draw_scalebar(ax, src, total_km: float = 60, n_div: int = 4,
                      pad_x: float = 0.08, pad_y: float = 0.08,
                      h_frac: float = 0.012, font_size: int = 10):
        """
        Barre d’échelle à cases alternées (type carto pro).
        - total_km : longueur totale en kilomètres
        - n_div    : nombre de graduations (cases)
        - pad_x, pad_y : marges (fraction de l’étendue)
        - h_frac   : hauteur (fraction de l’étendue Y)
        """
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        W = extent[1] - extent[0]
        H = extent[3] - extent[2]

        total_m = total_km * 1000.0
        px_total = _metres_to_pixels(src, total_m)

        # largeur en unités du CRS (px -> unités carte via transform)
        t = src.transform
        unit_per_px = abs(t.a)
        L_units = px_total * unit_per_px

        # origine (gauche-bas)
        x0 = extent[0] + W * pad_x
        y0 = extent[2] + H * pad_y
        height = H * h_frac
        dx = L_units / n_div

        # cases alternées + contour
        for i in range(n_div):
            fc = "#000000" if i % 2 == 0 else "#ffffff"
            rect = Rectangle((x0 + i * dx, y0), dx, height,
                             facecolor=fc, edgecolor="#111111", linewidth=1.0)
            ax.add_patch(rect)

        # ticks & labels (0, 15, 30, …)
        for i in range(n_div + 1):
            x = x0 + i * dx
            ax.plot([x, x], [y0 + height, y0 + height * 1.25], color="#111111", lw=1.2)
            km_label = int(round((total_km / n_div) * i))
            ax.text(x, y0 + height * 1.6, f"{km_label}",
                    ha="center", va="bottom", fontsize=font_size, color="#111")

        # intitulé centré sous la barre
        ax.text(x0 + L_units / 2, y0 - height * 0.6, "Km",
                ha="center", va="top", fontsize=font_size, color="#111")

    if raster_selected:
        raster_path = os.path.join(raster_dir, raster_selected)
        if os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                arr = src.read(1).astype(float)
                # nodata / hors zone
                arr[~np.isfinite(arr)] = np.nan
                arr = np.ma.masked_where((arr == 0) | np.isnan(arr), arr)
                if np.ma.count(arr) == 0:
                    st.warning("Le raster ne contient aucune valeur exploitable.")
                else:
                    # --- classification en 5 classes (quantiles robustes)
                    qs = np.nanpercentile(arr.compressed(), [0, 20, 40, 60, 80, 100])
                    # évite bornes identiques
                    for i in range(1, len(qs)):
                        if qs[i] <= qs[i - 1]:
                            qs[i] = qs[i - 1] + 1e-9
                    norm = BoundaryNorm(qs, cmap5.N, clip=True)

                    # --- figure (A4 paysage)
                    fig = plt.figure(figsize=(11.7, 8.3), dpi=150)
                    fig.patch.set_facecolor("white")
                    # zone carte + léger encadrement
                    ax = fig.add_axes([0.06, 0.10, 0.68, 0.8])   # carte
                    ax_bg = fig.add_axes([0.05, 0.09, 0.70, 0.82])  # fond doux
                    ax_bg.axis("off")
                    ax_bg.set_facecolor("#f2f2f2")
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.6)
                        spine.set_edgecolor("#222")

                    # --- extent géoréférencé
                    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

                    # --- trame raster
                    ax.imshow(arr, cmap=cmap5, norm=norm, extent=extent, origin="upper")

                    # --- titre & sous-titre
                    titre = f"Niche écologique : {espece_selected_label}  —  {raster_selected_label.upper()}"
                    ax.set_title(titre, fontsize=18, fontweight="bold", pad=12)
                    fig.text(0.40, 0.93, "Région Provence–Alpes–Côte d’Azur",
                             fontsize=11, color="#444", ha="center")

                    # --- flèche du nord (discrète)
                    def north_arrow(ax, x, y, size=0.06):
                        ax.annotate("N", xy=(x, y + size * 0.9), xytext=(x, y + size * 0.9),
                                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                                    xycoords=ax.transAxes)
                        ax.annotate("", xy=(x, y + size), xytext=(x, y),
                                    arrowprops=dict(arrowstyle="-|>", lw=2, color="#111"),
                                    xycoords=ax.transAxes)
                    north_arrow(ax, 0.08, 0.78)

                    # --- barre d’échelle graduée : 4 graduations × 15 km = 60 km
                    draw_scalebar(ax, src, total_km=60, n_div=4)

                    # --- légende discrète
                    handles = [Patch(facecolor=c, edgecolor="#222", label=lab) for c, lab in zip(couleurs, labels_5)]
                    leg = ax.legend(handles=handles, loc="lower right", title="Marge écologique",
                                    frameon=True, framealpha=0.96, fancybox=True, borderpad=0.8)
                    leg.get_frame().set_edgecolor("#222")
                    leg.get_title().set_fontweight("bold")

                    # --- encart statistiques
                    vmin, vmax, vmean = np.nanmin(arr), np.nanmax(arr), np.nanmean(arr)
                    txt = f"Min {vmin:.3f}   •   Moy {vmean:.3f}   •   Max {vmax:.3f}"
                    fig.text(0.06, 0.06, txt, fontsize=9, color="#555")

                    # --- source & auteur
                    fig.text(0.76, 0.06, "Source : WorldClim 1970–2000   •   Auteur : Team Défis 1 (2025)",
                             ha="right", fontsize=9, color="#555")

                    # --- habillage
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticks([])
                    ax.set_yticks([])

                    st.pyplot(fig)

                    # --- export PNG HD + export GeoTIFF d’origine
                    png_path = os.path.join(raster_dir, f"carte_{os.path.splitext(raster_selected)[0]}.png")
                    fig.savefig(png_path, dpi=300, bbox_inches="tight")
                    with open(png_path, "rb") as f:
                        st.download_button("⬇️ Télécharger la carte (PNG, 300 dpi)", f,
                                           file_name=os.path.basename(png_path), mime="image/png")

                    with open(raster_path, "rb") as f:
                        st.download_button(f"⬇️ Télécharger le raster : {raster_selected}",
                                           f, file_name=raster_selected, mime="image/tiff")
                    plt.close(fig)
        else:
            st.warning("Le fichier raster sélectionné n'existe pas.")
    else:
        st.info("Sélectionnez un raster dans la barre latérale pour générer la carte.")

with tab_aide:
    st.header("Aide & Documentation")
    st.markdown("""
    **Mode d'emploi :**
    - Sélectionnez les paramètres dans la colonne de gauche.
    - Lancez la simulation pour visualiser la propagation.
    - Utilisez l'onglet Analyse pour explorer les résultats.
    - Exportez ou rechargez un scénario dans l'onglet Export.
    """)