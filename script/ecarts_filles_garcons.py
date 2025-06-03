# 📦 Imports
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt



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


def export_heatmaps_duales_par_annee(df, matiere="Mathématiques", dossier="heatmaps_dual"):
    os.makedirs(dossier, exist_ok=True)
    annees = sorted(df["Année"].unique())

    for annee in annees:
        dff = df[
            (df["Année"] == annee) &
            (df["Matière"] == matiere) &
            df["latitude"].notna() &
            df["longitude"].notna()
        ].copy()

        df_recul = dff.copy()
        df_avantage = dff.copy()
        df_recul["Avantage_garcons"] = df_recul["Ecart_score"].apply(lambda x: abs(x) if x < 0 else 0)
        df_avantage["Avantage_filles"] = df_avantage["Ecart_score"].apply(lambda x: abs(x) if x > 0 else 0)

        fig_recul = px.density_map(
            df_recul,
            lat="latitude", lon="longitude", z="Avantage_garcons",
            radius=8, center={"lat": 46.5, "lon": 2.5}, zoom=4.7,
            color_continuous_scale="Tealgrn", range_color=(0, 20),
            title=f"{annee} – Etablissements avec avantage garçons (score moyen G > F)"
        )
        fig_recul.update_layout(mapbox_style="open-street-map")
        fig_recul.update_layout(coloraxis_colorbar=dict(title="Écart (G - F)"))


        fig_avantage = px.density_map(
            df_avantage,
            lat="latitude", lon="longitude", z="Avantage_filles",
            radius=8, center={"lat": 46.5, "lon": 2.5}, zoom=4.7,
            color_continuous_scale="OrRd", range_color=(0, 20),
            title=f"{annee} – Etablissements avec avantage filles (score moyen F > G)"
        )
        fig_avantage.update_layout(mapbox_style="open-street-map")
        fig_avantage.update_layout(coloraxis_colorbar=dict(title="Écart (F - G)"))


        path_recul = os.path.join(dossier, f"recul_{annee}.png")
        path_avantage = os.path.join(dossier, f"avantage_{annee}.png")
        pio.write_image(fig_recul, path_recul, width=800, height=900)
        pio.write_image(fig_avantage, path_avantage, width=800, height=900)

        # # 🧩 Fusion des deux images horizontalement
        img1 = Image.open(path_recul)
        img2 = Image.open(path_avantage)
        fusion = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
        fusion.paste(img1, (0, 0))
        fusion.paste(img2, (img1.width, 0))


 # Génère la courbe de suivi jusqu'à l’année courante
        generer_graphe_proportions_par_annee(df, annee, matiere, dossier="graph_proportions")
        graph_img = Image.open(f"graph_proportions/courbe_{annee}.png")


        final = Image.new("RGB", (fusion.width, fusion.height + graph_img.height), color="white")
        final.paste(fusion, (0, 0))
        final.paste(graph_img, ((fusion.width - graph_img.width) // 2, fusion.height))

        # Sauvegarde l'image pour le GIF
        fusion_path = os.path.join(dossier, f"heatmap_{annee}.png")
        final.save(fusion_path)

        # Supprime les intermédiaires
        os.remove(path_recul)
        os.remove(path_avantage)
        os.remove(f"graph_proportions/courbe_{annee}.png")


def generer_graphe_proportions_par_annee(df_ecarts, annee, matiere="Mathématiques", dossier="graph_proportions"):
    import matplotlib.pyplot as plt
    os.makedirs(dossier, exist_ok=True)

    # Préparation des données
    df = df_ecarts[df_ecarts["Matière"] == matiere].copy()
    df["F_>_G"] = df["Ecart_score"] > 0
    df["G_>_F"] = df["Ecart_score"] < 0

    df_counts = df.groupby("Année").apply(lambda x: pd.Series({
        "total": x["UAI"].nunique(),
        "F_sup_G": x.loc[x["Ecart_score"] > 0, "UAI"].nunique(),
        "G_sup_F": x.loc[x["Ecart_score"] < 0, "UAI"].nunique()
    })).reset_index()
    df_counts["% F > G"] = 100 * df_counts["F_sup_G"] / df_counts["total"]
    df_counts["% G > F"] = 100 * df_counts["G_sup_F"] / df_counts["total"]

    # Crée une structure d'années fixes 2017-2024
    full_years = pd.DataFrame({"Année": list(range(2017, 2025))})
    df_all = full_years.merge(df_counts, on="Année", how="left")
    df_all["Année"] = df_all["Année"].astype(str)
    df_plot = df_all[df_all["Année"].astype(int) <= annee]

    # Création du graphique
    plt.figure(figsize=(8, 3))
    plt.plot(df_plot["Année"], df_plot["% F > G"], marker="o", label="% F > G", color="orangered")
    plt.plot(df_plot["Année"], df_plot["% G > F"], marker="o", label="% G > F", color="steelblue")

    # Ajout de points invisibles pour forcer l'axe complet 2017-2024
    plt.plot(df_all["Année"], [0]*len(df_all), alpha=0)  # bas
    plt.plot(df_all["Année"], [100]*len(df_all), alpha=0)  # haut

    plt.title(f"Évolution de la proportion de collèges selon l’avantage de performance en mathématiques (filles vs garçons)", wrap=True)
    plt.ylabel("Proportion (%)")
    plt.xlabel("Année")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="center right")
    plt.tight_layout()

    path = os.path.join(dossier, f"courbe_{annee}.png")
    plt.savefig(path, dpi=100)
    plt.close()




def creer_gif_dual(dossier="heatmaps_dual", gif_path="ecarts_dual.gif", fps=1):
    fichiers = sorted([f for f in os.listdir(dossier) if f.startswith("heatmap_") and f.endswith(".png")])
    images = [imageio.imread(os.path.join(dossier, f)) for f in fichiers]
    imageio.mimsave(gif_path, images, fps=fps)


if __name__ == "__main__":
    df_grouped = preparer_donnees_agrandies(df)
    df_geoloc = filtrer_france_metropolitaine(df_geoloc)
    df_localise = localiser_etablissements(df_grouped, df_geoloc, keep_columns=["latitude", "longitude", "Nom_etablissement"])
    df_ecarts = fusion_filles_garcons(df_grouped)
    df_ecarts_geo = df_ecarts.merge(df_localise, on=["UAI", "Année", "Matière"], how="left")

    export_heatmaps_duales_par_annee(df_ecarts_geo, matiere="Mathématiques")
    creer_gif_dual()
