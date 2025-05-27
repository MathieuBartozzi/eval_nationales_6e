# 📦 Imports
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import imageio.v2 as imageio

# 📂 Chargement des données
df = pd.read_csv("../data/raw/evaluations_6e.csv", sep=";")
df_geoloc = pd.read_csv("../data/raw/annuaire.csv", sep=";")



# 🧼 Nettoyage et agrégation
def preparer_donnees_agrandies(df: pd.DataFrame) -> pd.DataFrame:
    col_to_keep = [
        'Année', 'Libellé région académique', 'Libellé académie',
        'Libellé département', 'UAI', 'Libellé secteur',
        'Matière', 'Caractéristique', 'Effectif', 'Score moyen', 'Ecart type',
        'Groupe 1', 'Groupe 2', 'Groupe 3', 'Groupe 4', 'Groupe 5', 'Groupe 6'
    ]
    df = df[col_to_keep].copy()
    df["pond_score"] = df["Score moyen"] * df["Effectif"]
    group_cols = [
        "Année", "UAI", "Matière", "Caractéristique",
        "Libellé académie", "Libellé département", "Libellé région académique", "Libellé secteur"
    ]
    df_grouped = df.groupby(group_cols, as_index=False).agg({
        "Effectif": "sum",
        "pond_score": "sum",
        "Ecart type": "mean",
        **{f"Groupe {i}": "sum" for i in range(1, 7)}
    })
    df_grouped["Score moyen"] = df_grouped["pond_score"] / df_grouped["Effectif"]
    return df_grouped.drop(columns=["pond_score"])

df_grouped = preparer_donnees_agrandies(df)

# 🌍 Filtrage géographique
def filtrer_france_metropolitaine(df, lat_col="latitude", lon_col="longitude"):
    return df[df[lat_col].between(41, 51) & df[lon_col].between(-6, 10)].copy()

df_geoloc = filtrer_france_metropolitaine(df_geoloc)

# 📍 Localisation des établissements
def localiser_etablissements(df_grouped: pd.DataFrame, df_geoloc: pd.DataFrame, keep_columns=None) -> pd.DataFrame:
    if keep_columns is None:
        keep_columns = ["latitude", "longitude", "Nom_etablissement"]
    df_geo_college = df_geoloc[df_geoloc["Type_etablissement"].str.lower().str.contains("collège", na=False)].copy()
    df_geo_college = df_geo_college.rename(columns={"Identifiant_de_l_etablissement": "UAI"})
    df_geo_clean = df_geo_college.drop_duplicates(subset="UAI")
    cols_to_merge = ["UAI"] + [col for col in keep_columns if col in df_geo_clean.columns]
    return df_grouped.merge(df_geo_clean[cols_to_merge], on="UAI", how="left")

df_localise = localiser_etablissements(df_grouped, df_geoloc, keep_columns=["latitude", "longitude", "Nom_etablissement"])

# 👧👦 Fusion filles/garçons
def fusion_filles_garcons(df: pd.DataFrame) -> pd.DataFrame:
    filles = df[df["Caractéristique"].str.lower() == "fille"].copy()
    garcons = df[df["Caractéristique"].str.lower() == "garçon"].copy()
    filles = filles.rename(columns={"Score moyen": "Score moyen_fille"})
    garcons = garcons.rename(columns={"Score moyen": "Score moyen_garçon"})
    fusion = pd.merge(
        filles[["UAI", "Année", "Matière", "Score moyen_fille"]],
        garcons[["UAI", "Année", "Matière", "Score moyen_garçon"]],
        on=["UAI", "Année", "Matière"],
        how="inner"
    )
    fusion["Ecart_score"] = fusion["Score moyen_fille"] - fusion["Score moyen_garçon"]
    return fusion

df_ecarts = fusion_filles_garcons(df_grouped)
df_ecarts_geo = df_ecarts.merge(df_localise, on=["UAI", "Année", "Matière"], how="left")

# 📸 Export des heatmaps par année
def export_heatmaps_par_annee(df, matiere="Mathématiques", dossier="heatmaps"):
    os.makedirs(dossier, exist_ok=True)
    annees = sorted(df["Année"].unique())

    for annee in annees:
        dff = df[
            (df["Année"] == annee) &
            (df["Matière"] == matiere) &
            df["latitude"].notna() &
            df["longitude"].notna()
        ].copy()

        dff["Recul_filles"] = dff["Ecart_score"].apply(lambda x: abs(x) if x < 0 else 0)

        fig = px.density_map(
            dff,
            lat="latitude",
            lon="longitude",
            z="Recul_filles",
            radius=8,
            center={"lat": 46.5, "lon": 2.5},
            zoom=5,
            color_continuous_scale="Tealgrn",
            range_color=(0, 20),
            height=900,
            title=f"Évaluations Nationales 6e – Écart Filles/Garçons en {matiere}"

        )
        fig.update_layout(coloraxis_colorbar=dict(
            title="moy. Garçons - moy. Filles<br>(en points)",
            # titlefont=dict(size=14),
            ticksuffix=" pts"
        ))

        fig.add_annotation(
            text=str(annee),
            x=0.025,
            y=0.90,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=25, color="black"),
            align="right"
        )
        fig.add_annotation(
        text="Un point = un collège<br>Couleur = avantage garçons plus marqué",
        xref="paper", yref="paper",
        x=0.5, y=0.05,
        align='center',
        showarrow=False,
        xanchor="center",
        font=dict(size=15, color="gray"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        borderpad=4
    )



        filepath = os.path.join(dossier, f"heatmap_{annee}.png")
        pio.write_image(fig, filepath, format="png", width=1200, height=900)

# 🎞️ Création d'un GIF animé
def creer_gif_heatmap(dossier="heatmaps", gif_path="recul_filles.gif", fps=1):
    fichiers = sorted([f for f in os.listdir(dossier) if f.endswith(".png")])
    images = [imageio.imread(os.path.join(dossier, f)) for f in fichiers]
    imageio.mimsave(gif_path, images, fps=fps)


if __name__ == "__main__":
    df_grouped = preparer_donnees_agrandies(df)
    df_geoloc = filtrer_france_metropolitaine(df_geoloc)
    df_localise = localiser_etablissements(df_grouped, df_geoloc, keep_columns=["latitude", "longitude", "Nom_etablissement"])
    df_ecarts = fusion_filles_garcons(df_grouped)
    df_ecarts_geo = df_ecarts.merge(df_localise, on=["UAI", "Année", "Matière"], how="left")

    export_heatmaps_par_annee(df_ecarts_geo, matiere="Mathématiques")
    creer_gif_heatmap()

