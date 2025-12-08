# Dans le terminal exécuter : streamlit run hackaton.py

import os
import gc
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    BoundaryNorm,
)
from matplotlib.patches import Patch, Rectangle
from math import cos, radians
from scipy.stats import gaussian_kde

import leafmap.foliumap as leafmap
from folium.raster_layers import ImageOverlay
import psutil

# -------------------------------------------------------------------
# CONFIG STREAMLIT + CHEMINS RELATIFS
# -------------------------------------------------------------------

st.set_page_config(layout="wide")

# Dossier où se trouve ce fichier hackaton.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = BASE_DIR  # on garde ce nom pour le reste du code

# -------------------------------------------------------------------
# FONCTIONS RASTER
# -------------------------------------------------------------------

def _preview_raster_statique(raster_path, title=None, colormap="viridis"):
    """Affiche un aperçu statique d'un raster via rasterio + matplotlib."""
    try:
        if not os.path.exists(raster_path):
            st.error(f"Fichier introuvable : {raster_path}")
            return
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype(float)
            arr[~np.isfinite(arr)] = np.nan
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(arr, cmap=colormap, origin="upper")
            ax.set_title(title or f"Aperçu : {os.path.basename(raster_path)}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.error(f"Aperçu statique impossible : {e}")


def ajouter_raster_overlay(m, raster_path, opacity=1.0, colormap_name="viridis"):
    """
    Superpose un raster sur la carte Leafmap via ImageOverlay (compatible Streamlit Cloud).
    Utilise une colormap continue et encode l'image en PNG base64.
    """
    try:
        if not os.path.exists(raster_path):
            st.error(f"Raster introuvable : {raster_path}")
            return False

        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype(float)

            # Gestion nodata / NaN
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[~np.isfinite(arr)] = np.nan

            if np.all(np.isnan(arr)):
                st.warning("Le raster ne contient aucune valeur valide.")
                return False

            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)
            if vmax - vmin == 0:
                st.warning("Le raster est constant, impossible de normaliser.")
                return False

            # Normalisation 0–1 pour la colormap
            norm_arr = (arr - vmin) / (vmax - vmin)
            norm_arr = np.clip(norm_arr, 0, 1)

            # Choix colormap
            if colormap_name == "ForetMarges":
                cmap = foret_marges_cmap
            else:
                cmap = plt.get_cmap(colormap_name)

            rgba = cmap(norm_arr)  # shape (H, W, 4), valeurs [0,1]
            rgba[np.isnan(norm_arr)] = [0, 0, 0, 0]  # transparence sur les NaN
            rgba_uint8 = (rgba * 255).astype("uint8")

            # Sauvegarde en PNG en mémoire
            buf = BytesIO()
            plt.imsave(buf, rgba_uint8, format="png")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("utf-8")

            # Bornes géo (ymin, xmin, ymax, xmax)
            bounds = [
                [src.bounds.bottom, src.bounds.left],
                [src.bounds.top, src.bounds.right],
            ]

            overlay = ImageOverlay(
                image=f"data:image/png;base64,{encoded}",
                bounds=bounds,
                opacity=opacity,
                interactive=True,
                cross_origin=False,
                zindex=5,
            )
            # leafmap.Map hérite de folium.Map -> on peut ajouter directement
            overlay.add_to(m)

        return True

    except Exception as e:
        st.error(f"Erreur overlay raster : {e}")
        return False

# -------------------------------------------------------------------
# FONDS DE CARTE
# -------------------------------------------------------------------

fond_leafmap = {
    "OpenStreetMap": "OpenStreetMap",
    "Esri Satellite": "Esri Satellite",
    "Google Satellite": "HYBRID",
    "OpenTopoMap": "OpenTopoMap",
    "CartoDB Positron": "CartoDB.Positron",
    "CartoDB dark_matter": "CartoDB.DarkMatter",
}

tiles_dict = {
    "OpenStreetMap": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors",
    },
    "Esri Satellite": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": (
            "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
    },
    "Google Satellite": {
        "tiles": "http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "© Google",
    },
    "OpenTopoMap": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "© OpenTopoMap contributors",
    },
    "CartoDB Positron": {
        "tiles": (
            "https://cartodb-basemaps-a.global.ssl.fastly.net/"
            "light_all/{z}/{x}/{y}.png"
        ),
        "attr": "© OpenStreetMap contributors © CARTO",
    },
    "CartoDB dark_matter": {
        "tiles": (
            "https://cartodb-basemaps-a.global.ssl.fastly.net/"
            "dark_all/{z}/{x}/{y}.png"
        ),
        "attr": "© OpenStreetMap contributors © CARTO",
    },
}
fond_options = list(tiles_dict.keys())

if "fond_carte" not in st.session_state:
    st.session_state["fond_carte"] = fond_options[0]

# -------------------------------------------------------------------
# STYLES UI
# -------------------------------------------------------------------

st.markdown("<div style='height:4em;'></div>", unsafe_allow_html=True)

# Logo et titre
col1, col2, col3 = st.columns([2, 6, 2])

with col1:
    logo1_path = os.path.join(data_dir, "Documents d'aide",
                              "CrigeBaseline_redim.png")
    if os.path.exists(logo1_path):
        st.image(logo1_path, use_container_width=False, width=150)
    else:
        st.write("")

with col2:
    st.markdown(
        """
        <div style='display:flex;align-items:center;justify-content:center;width:100%;'>
        <span style='font-size:2.8em;font-weight:bold;color:#fff;text-align:center;line-height:50px;flex:1;'>
            Dépérissement forestier, une histoire de marges
        </span>
        """,
        unsafe_allow_html=True,
    )

with col3:
    logo2_path = os.path.join(data_dir, "Documents d'aide",
                              "Logo geodatalab 2025.png")
    if os.path.exists(logo2_path):
        st.image(logo2_path, use_container_width=False, width=150)
    else:
        st.write("")

# Styles CSS
st.markdown(
    """
<style>
.titre-principal {
    border-radius: 12px !important;
    box-shadow: none !important;
    margin-bottom: 0.7em !important;
    padding: 1.5em 0 1em 0 !important;
    width: 100% !important;
}
.block-container {padding-top: 1rem;}
h1, h2, h3 {color: #b22222;}
.stButton>button {background-color: #A0C989; color: white;}
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
    font-size: 1.35em;
    color: white !important;
    font-weight: bold;
    border-radius: 12px !important;
    margin: 0 0.2em;
    padding: 0.5em 1.5em;
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
.barre-onglets {
    background: #181818 !important;
    border-radius: 8px !important;
    padding: 0.5em 0 !important;
    margin-bottom: 1em !important;
    width: 100% !important;
}
[data-testid="stNumberInput"] > label {margin-bottom: -0.7em;}
</style>
""",
    unsafe_allow_html=True,
)

if "fond_carte" not in st.session_state:
    st.session_state["fond_carte"] = "OpenStreetMap"

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center;'>
        <div style='font-size:1.9em;font-weight:bold;margin-bottom:1.2em;letter-spacing:0.04em;'>
            Paramètres de simulation
        </div>
    </div>
    <div style='font-size:1.15em;color:#b22222;margin-bottom:1.2em;margin-left:0.1em;text-align:left;'>
        Dans la région Provences-Alpes-Côtes-d'Azur
    </div>
    """,
        unsafe_allow_html=True,
    )

    fond_selected = st.selectbox(
        "Choix du fond de carte",
        fond_options,
        index=fond_options.index(
            st.session_state.get("fond_carte", fond_options[0])
        ),
        key="fond_carte",
    )

    couche_options = [
        "Espèces forestières occurences - PACA",
        "Inventaire forestier national PACA",
    ]
    selected_couches = st.multiselect(
        "Afficher des couches supplémentaires",
        options=couche_options,
        default=[],
    )

    # -------------------------------
    # ESPÈCES : chemins robustes
    # -------------------------------
    especes_dir = os.path.join(data_dir, "Niche écologique par espèces")

    if not os.path.exists(especes_dir):
        st.error(
            f"❌ Le dossier '{especes_dir}' est introuvable.\n\n"
            "Vérifie qu'il existe bien dans ton repo GitHub."
        )
        st.stop()

    try:
        especes_raw = [
            d
            for d in os.listdir(especes_dir)
            if os.path.isdir(os.path.join(especes_dir, d))
        ]
    except Exception as e:
        st.error(f"❌ Impossible de lire le dossier des espèces : {e}")
        st.stop()

    if not especes_raw:
        st.error(
            "❌ Aucun sous-dossier d'espèce trouvé dans "
            "'Niche écologique par espèces/'."
        )
        st.stop()

    especes_list = [
        " ".join(
            [w.capitalize() for w in d.replace("niche_", "").split("_")]
        )
        for d in especes_raw
    ]

    espece_selected_label = st.selectbox(
        "Sélectionnez une espèce à visualiser",
        options=especes_list,
        key="espece_selected",
    )

    espece_selected = especes_raw[especes_list.index(espece_selected_label)]

    raster_dir = os.path.join(especes_dir, espece_selected)

    if not os.path.exists(raster_dir):
        st.error(
            f"❌ Le dossier raster pour l'espèce '{espece_selected}' est "
            f"introuvable : {raster_dir}"
        )
        st.stop()

    try:
        raster_files = [
            f for f in os.listdir(raster_dir) if f.endswith(".tif")
        ]
    except Exception as e:
        st.error(f"❌ Impossible de lire les rasters : {e}")
        st.stop()

    suffix_map = {
        "dif": "diff",
        "present": "present",
        "futur": "futur",
        "marge": "marge",
    }
    raster_options = {}
    for f in raster_files:
        for suf, label in suffix_map.items():
            if f"_{suf}." in f:
                raster_options[label] = f

    radio_labels = ["diff", "present", "futur", "marge"]
    raster_labels = [lbl for lbl in radio_labels if lbl in raster_options]

    raster_selected_label = st.radio(
        "Choisissez le scénario de niche à afficher",
        options=raster_labels,
        key="raster_selected_label",
        horizontal=True,
    )

    if raster_labels and raster_selected_label is not None:
        raster_selected = raster_options[raster_selected_label]
    else:
        raster_selected = None

    # Colormap personnalisée
    foret_marges_cmap = LinearSegmentedColormap.from_list(
        "ForetMarges",
        ["#d32f2f", "#fbc02d", "#bdbdbd", "#1976d2", "#388e3c"],
        N=256,
    )

    colormap_options = [
        "ForetMarges",
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "YlGnBu",
        "YlOrRd",
        "Greens",
        "Blues",
        "Reds",
        "Purples",
    ]
    colormap_selected = st.selectbox(
        "Palette de couleurs pour la carte raster",
        options=colormap_options,
        index=0,
        key="colormap_selected",
    )

    opacity_value = st.slider(
        "Transparence des couches raster",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        key="opacity_value",
    )

    mem = psutil.virtual_memory()
    if mem.percent > 80:
        st.warning(
            f"Attention : la mémoire utilisée par le système est élevée "
            f"({mem.percent}%). Cela peut ralentir ou interrompre la simulation."
        )

    st.success("Paramètres validés")
    st.session_state["param_valides"] = True
    st.session_state["recalc"] = True

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------

gc.collect()
tab_sim, tab_analyse, tab_export, tab_aide = st.tabs(
    ["Simulation", "Analyse", "Export", "Aide"]
)

# -------------------------------------------------------------------
# FONCTION D'AFFICHAGE CARTE
# -------------------------------------------------------------------

def afficher_carte_leafmap(temps, fond, afficher_polygones=True):
    # 1) Carte Leafmap pour vecteurs + raster superposé
    fond_name = fond_leafmap.get(fond, "OpenStreetMap")
    m = leafmap.Map(center=[43.5, 5.5], zoom=8, tiles=None)
    m.add_basemap(fond_name)

    # Couches vectorielles
    for couche in selected_couches:
        try:
            if couche == "Espèces forestières occurences - PACA":
                foret_path = os.path.join(
                    data_dir,
                    "occurrences",
                    "Espèces forestières occurences - PACA.gpkg",
                )
                gdf_foret = gpd.read_file(foret_path).to_crs(epsg=4326)
                unique_names = gdf_foret["Name"].unique()
                color_map = px.colors.qualitative.Dark24
                color_dict = {
                    name: color_map[i % len(color_map)]
                    for i, name in enumerate(unique_names)
                }
                for name in unique_names:
                    subset = gdf_foret[gdf_foret["Name"] == name]
                    m.add_gdf(
                        subset,
                        layer_name=f"Espèce : {name}",
                        style={
                            "color": color_dict[name],
                            "fillColor": color_dict[name],
                            "fillOpacity": 0.3,
                            "weight": 1,
                        },
                    )

            elif couche == "Inventaire forestier national PACA":
                ifn_path = os.path.join(
                    data_dir,
                    "Inventaire foret national",
                    "ifn_paca.gpkg",
                )
                gdf_ifn_visu = gpd.read_file(
                    ifn_path, layer="arbre_paca"
                ).to_crs(epsg=4326)
                m.add_gdf(
                    gdf_ifn_visu,
                    layer_name="IFN PACA",
                    style={
                        "color": "#1976D2",
                        "fillColor": "#1976D2",
                        "fillOpacity": 0.1,
                        "weight": 1,
                    },
                )
        except Exception as e:
            st.warning(f"Impossible d'afficher la couche {couche} : {e}")

    # Raster superposé via ImageOverlay
    raster_dir_local = os.path.join(
        data_dir, "Niche écologique par espèces", espece_selected
    )
    if raster_selected:
        raster_path = os.path.join(raster_dir_local, raster_selected)
        ajouter_raster_overlay(
            m,
            raster_path,
            opacity=opacity_value,
            colormap_name=colormap_selected,
        )
    else:
        st.warning("Aucun raster sélectionné pour cette espèce/scénario.")

    # Affichage de la carte
    m.to_streamlit(height=500)

    # Tableau d'aire bioclimatique
    if raster_selected:
        base_name = os.path.splitext(raster_selected)[0]
        aire_txt_path = os.path.join(
            raster_dir_local,
            f"{base_name}_aire.txt",
        )
        if os.path.exists(aire_txt_path):
            with open(aire_txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            classe_vals = []
            for line in lines:
                if line.startswith("Classe"):
                    parts = line.strip().split(":")
                    classe = parts[0].replace("Classe ", "")
                    aire = parts[1].replace("km²", "").strip()
                    classe_vals.append(
                        {
                            "Classe": classe,
                            "Aire bioclimatique (km²)": aire,
                        }
                    )
            df_air = pd.DataFrame(classe_vals)
            st.markdown("#### Aire bioclimatique par classe")
            st.dataframe(df_air, use_container_width=True)
        else:
            if raster_selected.endswith("_marge.tif"):
                st.warning(
                    "Aucun fichier d'aire bioclimatique associé au "
                    "raster sélectionné."
                )
    else:
        st.info(
            "Sélectionnez un raster dans la barre latérale pour "
            "afficher le tableau d'aire bioclimatique."
        )

# -------------------------------------------------------------------
# ONGLET SIMULATION
# -------------------------------------------------------------------

with tab_sim:
    if "running" not in st.session_state:
        st.session_state["running"] = False
    if "temps" not in st.session_state:
        st.session_state["temps"] = 0

    def maj_temps():
        st.session_state["temps"] = st.session_state["slider_temps"]

    spinner_placeholder = st.empty()
    carte_placeholder = st.empty()

    afficher_carte_leafmap(
        st.session_state["temps"],
        st.session_state["fond_carte"],
        afficher_polygones=True,
    )

# -------------------------------------------------------------------
# ONGLET ANALYSE
# -------------------------------------------------------------------

with tab_analyse:
    st.header("Analyse des résultats")
    st.markdown(
        """
    ### Modélisation de la niche écologique
    - **SDM (Bioclim, MaxEnt, Random Forest, etc.)** : Utilisation des données BD_Forêt, BD TOPO, IFN, occupation du sol, RGE alti 5m et variables bioclimatiques WorldClim pour modéliser la niche écologique des espèces forestières.
    - **Calcul de la distance au centre de niche** : Méthodes statistiques pour estimer la position des peuplements par rapport au centre ou à la marge de la niche.
    - **Calcul de l’aire bioclimatique** : Détermination des zones bioclimatiques favorables par département.
    - **Régression mortalité vs distance au centre de niche** : Analyse statistique de la relation entre mortalité observée (IFN) et distance à la niche optimale.
    - **Visualisation des zones de mortalité accrue** : Cartographie des marges écologiques et des zones à risque.
    - **Comparaison entre espèces** : Analyse comparative multi-espèces sur la région PACA.
    """
    )

    variables_df = pd.DataFrame(
        {
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
                "BIO19 - Précipitations du trimestre le plus froid",
            ],
            "Format": [
                "Vecteur (polygones)",
                "Vecteur (points)",
                "Vecteur (points)",
                "Raster (.tif)",
                *["Raster (.tif)"] * 19,
            ],
            "Rôle dans le modèle": [
                "Délimitation des peuplements",
                "Présence et mortalité des arbres",
                "Coordonnées et département",
                "Variable environnementale",
                *["Variable bioclimatique"] * 19,
            ],
            "Fichier": [
                "BD_Foret_PACA/bd_foret_paca.gpkg",
                "Inventaire foret national/ifn_paca.gpkg (arbre_paca)",
                "Inventaire foret national/ifn_paca.gpkg "
                "(placette_sans_doublons)",
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
                "world_clim/BIO19.tif",
            ],
        }
    )
    st.dataframe(variables_df, use_container_width=True)

    st.markdown("#### Distribution des valeurs du raster sélectionné")

    raster_dir_local = os.path.join(
        data_dir, "Niche écologique par espèces", espece_selected
    )
    if raster_selected:
        raster_path = os.path.join(raster_dir_local, raster_selected)
        if os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                arr = src.read(1)
                arr = arr[np.isfinite(arr)]
                fig, ax = plt.subplots(figsize=(7, 3))
                kde = gaussian_kde(arr)
                x_vals = np.linspace(np.nanmin(arr), np.nanmax(arr), 200)
                ax.plot(x_vals, kde(x_vals), color="#1976D2", lw=2)
                ax.fill_between(
                    x_vals, kde(x_vals), color="#1976D2", alpha=0.2
                )
                ax.set_title(
                    f"Densité des valeurs du raster : "
                    f"{raster_selected_label}",
                    fontsize=14,
                )
                ax.set_xlabel("Valeur du pixel")
                ax.set_ylabel("Densité")
                st.pyplot(fig)
                st.markdown(
                    f"**Min :** {np.nanmin(arr):.3f} &nbsp;&nbsp; "
                    f"**Max :** {np.nanmax(arr):.3f} &nbsp;&nbsp; "
                    f"**Moyenne :** {np.nanmean(arr):.3f}"
                )
        else:
            st.warning("Le fichier raster sélectionné n'existe pas.")
    else:
        st.info(
            "Sélectionnez un raster dans la barre latérale pour "
            "afficher sa distribution."
        )

# -------------------------------------------------------------------
# ONGLET EXPORT
# -------------------------------------------------------------------

with tab_export:
    st.header("Export de la couche sélectionnée")
    st.markdown(
        "Exportez une carte prête à partager avec une sémiologie "
        "nette et universelle."
    )

    raster_dir_local = os.path.join(
        data_dir, "Niche écologique par espèces", espece_selected
    )

    couleurs = ["#d32f2f", "#fbc02d", "#bdbdbd", "#1976d2", "#388e3c"]
    labels_5 = [
        "Marge inférieure",
        "Marge inf. intermédiaire",
        "Centre de la niche",
        "Marge sup. intermédiaire",
        "Marge supérieure",
    ]
    cmap5 = ListedColormap(couleurs)

    def _metres_to_pixels(src, metres: float) -> int:
        t = src.transform
        px_w = abs(t.a)
        if src.crs and src.crs.is_projected:
            return max(1, int(round(metres / px_w)))
        bounds = src.bounds
        mid_lat = (bounds.bottom + bounds.top) / 2.0
        metres_par_deg_lon = 111320.0 * cos(radians(mid_lat))
        deg_lon = metres / max(1e-9, metres_par_deg_lon)
        return max(1, int(round(deg_lon / px_w)))

    def draw_scalebar(
        ax,
        src,
        total_km: float = 60,
        n_div: int = 4,
        pad_x: float = 0.08,
        pad_y: float = 0.08,
        h_frac: float = 0.012,
        font_size: int = 10,
    ):
        extent = (
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        )
        W = extent[1] - extent[0]
        H = extent[3] - extent[2]

        total_m = total_km * 1000.0
        px_total = _metres_to_pixels(src, total_m)

        t = src.transform
        unit_per_px = abs(t.a)
        L_units = px_total * unit_per_px

        x0 = extent[0] + W * pad_x
        y0 = extent[2] + H * pad_y
        height = H * h_frac
        dx = L_units / n_div

        for i in range(n_div):
            fc = "#000000" if i % 2 == 0 else "#ffffff"
            rect = Rectangle(
                (x0 + i * dx, y0),
                dx,
                height,
                facecolor=fc,
                edgecolor="#111111",
                linewidth=1.0,
            )
            ax.add_patch(rect)

        for i in range(n_div + 1):
            x = x0 + i * dx
            ax.plot(
                [x, x],
                [y0 + height, y0 + height * 1.25],
                color="#111111",
                lw=1.2,
            )
            km_label = int(round((total_km / n_div) * i))
            ax.text(
                x,
                y0 + height * 1.6,
                f"{km_label}",
                ha="center",
                va="bottom",
                fontsize=font_size,
                color="#111",
            )

        ax.text(
            x0 + L_units / 2,
            y0 - height * 0.6,
            "Km",
            ha="center",
            va="top",
            fontsize=font_size,
            color="#111",
        )

    if raster_selected:
        raster_path = os.path.join(raster_dir_local, raster_selected)
        if os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                arr = src.read(1).astype(float)
                arr[~np.isfinite(arr)] = np.nan
                arr = np.ma.masked_where((arr == 0) | np.isnan(arr), arr)
                if np.ma.count(arr) == 0:
                    st.warning(
                        "Le raster ne contient aucune valeur exploitable."
                    )
                else:
                    qs = np.nanpercentile(
                        arr.compressed(), [0, 20, 40, 60, 80, 100]
                    )
                    for i in range(1, len(qs)):
                        if qs[i] <= qs[i - 1]:
                            qs[i] = qs[i - 1] + 1e-9
                    norm = BoundaryNorm(qs, cmap5.N, clip=True)

                    fig = plt.figure(figsize=(11.7, 8.3), dpi=150)
                    fig.patch.set_facecolor("white")
                    ax = fig.add_axes([0.06, 0.10, 0.68, 0.8])
                    ax_bg = fig.add_axes([0.05, 0.09, 0.70, 0.82])
                    ax_bg.axis("off")
                    ax_bg.set_facecolor("#f2f2f2")
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.6)
                        spine.set_edgecolor("#222")

                    extent = (
                        src.bounds.left,
                        src.bounds.right,
                        src.bounds.bottom,
                        src.bounds.top,
                    )

                    ax.imshow(
                        arr,
                        cmap=cmap5,
                        norm=norm,
                        extent=extent,
                        origin="upper",
                    )

                    titre = (
                        f"Niche écologique : {espece_selected_label}  —  "
                        f"{raster_selected_label.upper()}"
                    )
                    ax.set_title(
                        titre, fontsize=18, fontweight="bold", pad=12
                    )
                    fig.text(
                        0.40,
                        0.93,
                        "Région Provence–Alpes–Côte d’Azur",
                        fontsize=11,
                        color="#444",
                        ha="center",
                    )

                    def north_arrow(ax, x, y, size=0.06):
                        ax.annotate(
                            "N",
                            xy=(x, y + size * 0.9),
                            xytext=(x, y + size * 0.9),
                            ha="center",
                            va="bottom",
                            fontsize=12,
                            fontweight="bold",
                            xycoords=ax.transAxes,
                        )
                        ax.annotate(
                            "",
                            xy=(x, y + size),
                            xytext=(x, y),
                            arrowprops=dict(
                                arrowstyle="-|>", lw=2, color="#111"
                            ),
                            xycoords=ax.transAxes,
                        )

                    north_arrow(ax, 0.08, 0.78)

                    handles = [
                        Patch(
                            facecolor=c,
                            edgecolor="#222",
                            label=lab,
                        )
                        for c, lab in zip(couleurs, labels_5)
                    ]
                    leg = ax.legend(
                        handles=handles,
                        loc="lower right",
                        title="Marge écologique",
                        frameon=True,
                        framealpha=0.96,
                        fancybox=True,
                        borderpad=0.8,
                    )
                    leg.get_frame().set_edgecolor("#222")
                    leg.get_title().set_fontweight("bold")

                    vmin, vmax, vmean = (
                        np.nanmin(arr),
                        np.nanmax(arr),
                        np.nanmean(arr),
                    )
                    txt = (
                        f"Min {vmin:.3f}   •   Moy {vmean:.3f}   •   "
                        f"Max {vmax:.3f}"
                    )
                    fig.text(
                        0.06,
                        0.06,
                        txt,
                        fontsize=9,
                        color="#555",
                    )

                    fig.text(
                        0.76,
                        0.06,
                        "Source : WorldClim 1970–2000   •   "
                        "Auteur : Team Défis 1 (2025)",
                        ha="right",
                        fontsize=9,
                        color="#555",
                    )

                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticks([])
                    ax.set_yticks([])

                    st.pyplot(fig)

                    png_path = os.path.join(
                        raster_dir_local,
                        f"carte_{os.path.splitext(raster_selected)[0]}.png",
                    )
                    fig.savefig(
                        png_path, dpi=300, bbox_inches="tight"
                    )
                    with open(png_path, "rb") as f:
                        st.download_button(
                            "⬇️ Télécharger la carte (PNG, 300 dpi)",
                            f,
                            file_name=os.path.basename(png_path),
                            mime="image/png",
                        )

                    with open(raster_path, "rb") as f:
                        st.download_button(
                            f"⬇️ Télécharger le raster : {raster_selected}",
                            f,
                            file_name=raster_selected,
                            mime="image/tiff",
                        )
                    plt.close(fig)
        else:
            st.warning("Le fichier raster sélectionné n'existe pas.")
    else:
        st.info(
            "Sélectionnez un raster dans la barre latérale pour "
            "générer la carte."
        )

# -------------------------------------------------------------------
# ONGLET AIDE
# -------------------------------------------------------------------

with tab_aide:
    st.header("Aide & Documentation")
    st.markdown(
        """
    **Mode d'emploi :**
    - Sélectionnez les paramètres dans la colonne de gauche.
    - Visualisez la carte (fond + raster superposé) dans l’onglet Simulation.
    - Utilisez l'onglet Analyse pour explorer les résultats.
    - Exportez ou partagez une carte dans l'onglet Export.
    """
    )
