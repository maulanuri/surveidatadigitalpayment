import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr, chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# ----------------------------------------------------------------
# NLTK setup (akan mencoba download resource jika belum ada)
# ----------------------------------------------------------------
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")

# ----------------------------------------------------------------
# Multi-language dictionary
# ----------------------------------------------------------------

LANG_TEXT = {
    "id": {
        "language_name": "Bahasa Indonesia",
        "title": "Analisis Data Survei – Aplikasi Web Streamlit",
        "language_select_label": "Pilih Bahasa / Select Language",
        "upload_subheader": "Upload Data",
        "upload_label": "Upload file data survei (CSV atau Excel)",
        "no_file": "Belum ada file yang diupload.",
        "data_preview": "Pratinjau Data",
        "text_processing_subheader": "Pra-pemrosesan Teks",
        "text_columns_detected": "Kolom teks terdeteksi:",
        "no_text_columns": "Tidak ada kolom teks yang terdeteksi.",
        "text_processing_note": "Pra-pemrosesan dasar: lowercasing, hapus tanda baca, tokenisasi, dan stopword removal (English).",
        "sample_tokens": "Contoh token dari beberapa baris:",
        "top_words": "Top 10 kata terbanyak:",
        "stats_subheader": "Statistik Deskriptif",
        "select_numeric_col": "Pilih kolom numerik untuk analisis:",
        "no_numeric_cols": "Tidak ada kolom numerik yang tersedia.",
        "desc_stats": "Statistik deskriptif:",
        "freq_table_subheader": "Tabel Frekuensi",
        "select_categorical_col": "Pilih kolom kategorikal untuk tabel frekuensi:",
        "no_categorical_cols": "Tidak ada kolom kategorikal yang tersedia.",
        "freq_count": "Frekuensi (count)",
        "freq_percent": "Frekuensi relatif (%)",
        "visual_subheader": "Visualisasi Data",
        "histogram": "Histogram",
        "boxplot": "Boxplot",
        "correlation_subheader": "Analisis Korelasi",
        "pearson_header": "Korelasi Pearson",
        "spearman_header": "Korelasi Spearman",
        "select_x_numeric": "Pilih variabel numerik X:",
        "select_y_numeric": "Pilih variabel numerik Y:",
        "not_enough_numeric": "Butuh minimal dua kolom numerik untuk analisis korelasi.",
        "pearson_result": "Hasil Korelasi Pearson",
        "spearman_result": "Hasil Korelasi Spearman",
        "corr_coef": "Koefisien korelasi (r):",
        "p_value": "p-value:",
        "interpretation": "Interpretasi:",
        "chi_square_subheader": "Uji Chi-square (Asosiasi Kategorikal)",
        "select_x_cat": "Pilih variabel kategorikal X:",
        "select_y_cat": "Pilih variabel kategorikal Y:",
        "not_enough_categorical": "Butuh minimal dua kolom kategorikal untuk uji Chi-square.",
        "chi_square_result": "Hasil Uji Chi-square",
        "chi_square_stat": "Nilai Chi-square:",
        "chi_square_df": "Derajat kebebasan (df):",
        "chi_square_p": "p-value:",
        "chi_square_interpret": "Interpretasi:",
        "alpha_note": "Gunakan alpha = 0.05 untuk menentukan signifikansi.",
        "significant_assoc": "Ada asosiasi yang signifikan antara variabel (tolak H0).",
        "no_significant_assoc": "Tidak ada asosiasi yang signifikan antara variabel (gagal tolak H0).",
        "corr_positive": "Arah korelasi: positif.",
        "corr_negative": "Arah korelasi: negatif.",
        "corr_zero": "Tidak ada korelasi linier yang jelas.",
        "corr_weak": "Kekuatan korelasi: lemah.",
        "corr_moderate": "Kekuatan korelasi: moderat.",
        "corr_strong": "Kekuatan korelasi: kuat.",
        "warning_select_valid": "Silakan pilih kolom yang sesuai dengan tipe data yang diminta.",
    },
    "en": {
        "language_name": "English",
        "title": "Survey Data Analysis – Streamlit Web App",
        "language_select_label": "Select Language",
        "upload_subheader": "Upload Data",
        "upload_label": "Upload survey dataset (CSV or Excel)",
        "no_file": "No file uploaded yet.",
        "data_preview": "Data Preview",
        "text_processing_subheader": "Text Preprocessing",
        "text_columns_detected": "Detected text columns:",
        "no_text_columns": "No text columns detected.",
        "text_processing_note": "Basic preprocessing: lowercasing, remove punctuation, tokenization, and stopword removal (English).",
        "sample_tokens": "Sample tokens from a few rows:",
        "top_words": "Top 10 most frequent words:",
        "stats_subheader": "Descriptive Statistics",
        "select_numeric_col": "Select numeric column for analysis:",
        "no_numeric_cols": "No numeric columns available.",
        "desc_stats": "Descriptive statistics:",
        "freq_table_subheader": "Frequency Tables",
        "select_categorical_col": "Select categorical column for frequency table:",
        "no_categorical_cols": "No categorical columns available.",
        "freq_count": "Frequency (count)",
        "freq_percent": "Relative frequency (%)",
        "visual_subheader": "Data Visualization",
        "histogram": "Histogram",
        "boxplot": "Boxplot",
        "correlation_subheader": "Correlation Analysis",
        "pearson_header": "Pearson Correlation",
        "spearman_header": "Spearman Correlation",
        "select_x_numeric": "Select numeric variable X:",
        "select_y_numeric": "Select numeric variable Y:",
        "not_enough_numeric": "At least two numeric columns are required for correlation analysis.",
        "pearson_result": "Pearson Correlation Result",
        "spearman_result": "Spearman Correlation Result",
        "corr_coef": "Correlation coefficient (r):",
        "p_value": "p-value:",
        "interpretation": "Interpretation:",
        "chi_square_subheader": "Chi-square Test (Categorical Association)",
        "select_x_cat": "Select categorical variable X:",
        "select_y_cat": "Select categorical variable Y:",
        "not_enough_categorical": "At least two categorical columns are required for Chi-square test.",
        "chi_square_result": "Chi-square Test Result",
        "chi_square_stat": "Chi-square statistic:",
        "chi_square_df": "Degrees of freedom (df):",
        "chi_square_p": "p-value:",
        "chi_square_interpret": "Interpretation:",
        "alpha_note": "Use alpha = 0.05 to assess significance.",
        "significant_assoc": "There is a significant association between the variables (reject H0).",
        "no_significant_assoc": "There is no significant association between the variables (fail to reject H0).",
        "corr_positive": "Direction: positive correlation.",
        "corr_negative": "Direction: negative correlation.",
        "corr_zero": "No clear linear correlation.",
        "corr_weak": "Strength: weak correlation.",
        "corr_moderate": "Strength: moderate correlation.",
        "corr_strong": "Strength: strong correlation.",
        "warning_select_valid": "Please select columns with appropriate data types for this analysis.",
    },
    "es": {
        "language_name": "Español",
        "title": "Análisis de Datos de Encuestas – Aplicación Web Streamlit",
        "language_select_label": "Seleccionar idioma",
        "upload_subheader": "Cargar datos",
        "upload_label": "Cargar archivo de encuesta (CSV o Excel)",
        "no_file": "Todavía no se ha cargado ningún archivo.",
        "data_preview": "Vista previa de datos",
        "text_processing_subheader": "Preprocesamiento de texto",
        "text_columns_detected": "Columnas de texto detectadas:",
        "no_text_columns": "No se detectaron columnas de texto.",
        "text_processing_note": "Preprocesamiento básico: minúsculas, eliminación de puntuación, tokenización y eliminación de stopwords (inglés).",
        "sample_tokens": "Ejemplos de tokens de algunas filas:",
        "top_words": "Top 10 palabras más frecuentes:",
        "stats_subheader": "Estadísticos descriptivos",
        "select_numeric_col": "Seleccione una columna numérica para el análisis:",
        "no_numeric_cols": "No hay columnas numéricas disponibles.",
        "desc_stats": "Estadísticos descriptivos:",
        "freq_table_subheader": "Tablas de frecuencia",
        "select_categorical_col": "Seleccione una columna categórica para la tabla de frecuencia:",
        "no_categorical_cols": "No hay columnas categóricas disponibles.",
        "freq_count": "Frecuencia (recuento)",
        "freq_percent": "Frecuencia relativa (%)",
        "visual_subheader": "Visualización de datos",
        "histogram": "Histograma",
        "boxplot": "Diagrama de caja",
        "correlation_subheader": "Análisis de correlación",
        "pearson_header": "Correlación de Pearson",
        "spearman_header": "Correlación de Spearman",
        "select_x_numeric": "Seleccione variable numérica X:",
        "select_y_numeric": "Seleccione variable numérica Y:",
        "not_enough_numeric": "Se requieren al menos dos columnas numéricas para el análisis de correlación.",
        "pearson_result": "Resultado de la correlación de Pearson",
        "spearman_result": "Resultado de la correlación de Spearman",
        "corr_coef": "Coeficiente de correlación (r):",
        "p_value": "p-valor:",
        "interpretation": "Interpretación:",
        "chi_square_subheader": "Prueba Chi-cuadrado (Asociación categórica)",
        "select_x_cat": "Seleccione variable categórica X:",
        "select_y_cat": "Seleccione variable categórica Y:",
        "not_enough_categorical": "Se requieren al menos dos columnas categóricas para la prueba Chi-cuadrado.",
        "chi_square_result": "Resultado de la prueba Chi-cuadrado",
        "chi_square_stat": "Estadístico Chi-cuadrado:",
        "chi_square_df": "Grados de libertad (df):",
        "chi_square_p": "p-valor:",
        "chi_square_interpret": "Interpretación:",
        "alpha_note": "Use alfa = 0.05 para evaluar la significancia.",
        "significant_assoc": "Existe una asociación significativa entre las variables (rechazar H0).",
        "no_significant_assoc": "No existe una asociación significativa entre las variables (no se rechaza H0).",
        "corr_positive": "Dirección: correlación positiva.",
        "corr_negative": "Dirección: correlación negativa.",
        "corr_zero": "No hay correlación lineal clara.",
        "corr_weak": "Fuerza: correlación débil.",
        "corr_moderate": "Fuerza: correlación moderada.",
        "corr_strong": "Fuerza: correlación fuerte.",
        "warning_select_valid": "Seleccione columnas con tipos de datos apropiados para este análisis.",
    },
    "fr": {
        "language_name": "Français",
        "title": "Analyse de Données d’Enquête – Application Web Streamlit",
        "language_select_label": "Choisir la langue",
        "upload_subheader": "Téléverser des données",
        "upload_label": "Téléverser le fichier d’enquête (CSV ou Excel)",
        "no_file": "Aucun fichier téléversé pour le moment.",
        "data_preview": "Aperçu des données",
        "text_processing_subheader": "Prétraitement du texte",
        "text_columns_detected": "Colonnes de texte détectées :",
        "no_text_columns": "Aucune colonne de texte détectée.",
        "text_processing_note": "Prétraitement basique : minuscules, suppression de la ponctuation, tokenisation et suppression des stopwords (anglais).",
        "sample_tokens": "Exemples de tokens de quelques lignes :",
        "top_words": "Top 10 des mots les plus fréquents :",
        "stats_subheader": "Statistiques descriptives",
        "select_numeric_col": "Sélectionner une colonne numérique pour l’analyse :",
        "no_numeric_cols": "Aucune colonne numérique disponible.",
        "desc_stats": "Statistiques descriptives :",
        "freq_table_subheader": "Tableaux de fréquences",
        "select_categorical_col": "Sélectionner une colonne catégorielle pour le tableau de fréquences :",
        "no_categorical_cols": "Aucune colonne catégorielle disponible.",
        "freq_count": "Fréquence (compte)",
        "freq_percent": "Fréquence relative (%)",
        "visual_subheader": "Visualisation des données",
        "histogram": "Histogramme",
        "boxplot": "Boîte à moustaches",
        "correlation_subheader": "Analyse de corrélation",
        "pearson_header": "Corrélation de Pearson",
        "spearman_header": "Corrélation de Spearman",
        "select_x_numeric": "Sélectionner la variable numérique X :",
        "select_y_numeric": "Sélectionner la variable numérique Y :",
        "not_enough_numeric": "Au moins deux colonnes numériques sont nécessaires pour l’analyse de corrélation.",
        "pearson_result": "Résultat de la corrélation de Pearson",
        "spearman_result": "Résultat de la corrélation de Spearman",
        "corr_coef": "Coefficient de corrélation (r) :",
        "p_value": "p-value :",
        "interpretation": "Interprétation :",
        "chi_square_subheader": "Test du Chi-deux (Association catégorielle)",
        "select_x_cat": "Sélectionner la variable catégorielle X :",
        "select_y_cat": "Sélectionner la variable catégorielle Y :",
        "not_enough_categorical": "Au moins deux colonnes catégorielles sont nécessaires pour le test du Chi-deux.",
        "chi_square_result": "Résultat du test du Chi-deux",
        "chi_square_stat": "Statistique du Chi-deux :",
        "chi_square_df": "Degrés de liberté (df) :",
        "chi_square_p": "p-value :",
        "chi_square_interpret": "Interprétation :",
        "alpha_note": "Utilisez alpha = 0.05 pour évaluer la signification.",
        "significant_assoc": "Il existe une association significative entre les variables (rejet de H0).",
        "no_significant_assoc": "Il n’existe pas d’association significative entre les variables (échec du rejet de H0).",
        "corr_positive": "Direction : corrélation positive.",
        "corr_negative": "Direction : corrélation négative.",
        "corr_zero": "Pas de corrélation linéaire claire.",
        "corr_weak": "Intensité : corrélation faible.",
        "corr_moderate": "Intensité : corrélation modérée.",
        "corr_strong": "Intensité : corrélation forte.",
        "warning_select_valid": "Veuillez sélectionner des colonnes avec des types de données appropriés pour cette analyse.",
    },
    "de": {
        "language_name": "Deutsch",
        "title": "Umfragedatenanalyse – Streamlit Web App",
        "language_select_label": "Sprache wählen",
        "upload_subheader": "Daten hochladen",
        "upload_label": "Umfragedatei hochladen (CSV oder Excel)",
        "no_file": "Noch keine Datei hochgeladen.",
        "data_preview": "Datenvorschau",
        "text_processing_subheader": "Text-Vorverarbeitung",
        "text_columns_detected": "Erkannte Textspalten:",
        "no_text_columns": "Keine Textspalten erkannt.",
        "text_processing_note": "Einfache Vorverarbeitung: Kleinschreibung, Entfernen von Satzzeichen, Tokenisierung und Entfernen von Stoppwörtern (Englisch).",
        "sample_tokens": "Beispieltokens aus einigen Zeilen:",
        "top_words": "Top 10 der häufigsten Wörter:",
        "stats_subheader": "Deskriptive Statistik",
        "select_numeric_col": "Numerische Spalte für Analyse wählen:",
        "no_numeric_cols": "Keine numerischen Spalten verfügbar.",
        "desc_stats": "Deskriptive Kennzahlen:",
        "freq_table_subheader": "Häufigkeitstabellen",
        "select_categorical_col": "Kategoriale Spalte für Häufigkeitstabelle wählen:",
        "no_categorical_cols": "Keine kategorialen Spalten verfügbar.",
        "freq_count": "Häufigkeit (Anzahl)",
        "freq_percent": "Relative Häufigkeit (%)",
        "visual_subheader": "Datenvisualisierung",
        "histogram": "Histogramm",
        "boxplot": "Boxplot",
        "correlation_subheader": "Korrelationsanalyse",
        "pearson_header": "Pearson-Korrelation",
        "spearman_header": "Spearman-Korrelation",
        "select_x_numeric": "Numerische Variable X wählen:",
        "select_y_numeric": "Numerische Variable Y wählen:",
        "not_enough_numeric": "Mindestens zwei numerische Spalten werden für die Korrelationsanalyse benötigt.",
        "pearson_result": "Ergebnis der Pearson-Korrelation",
        "spearman_result": "Ergebnis der Spearman-Korrelation",
        "corr_coef": "Korrelationskoeffizient (r):",
        "p_value": "p-Wert:",
        "interpretation": "Interpretation:",
        "chi_square_subheader": "Chi-Quadrat-Test (kategoriale Assoziation)",
        "select_x_cat": "Kategoriale Variable X wählen:",
        "select_y_cat": "Kategoriale Variable Y wählen:",
        "not_enough_categorical": "Mindestens zwei kategoriale Spalten werden für den Chi-Quadrat-Test benötigt.",
        "chi_square_result": "Ergebnis des Chi-Quadrat-Tests",
        "chi_square_stat": "Chi-Quadrat-Wert:",
        "chi_square_df": "Freiheitsgrade (df):",
        "chi_square_p": "p-Wert:",
        "chi_square_interpret": "Interpretation:",
        "alpha_note": "Verwenden Sie Alpha = 0,05 zur Beurteilung der Signifikanz.",
        "significant_assoc": "Es besteht ein signifikanter Zusammenhang zwischen den Variablen (H0 wird verworfen).",
        "no_significant_assoc": "Es besteht kein signifikanter Zusammenhang zwischen den Variablen (H0 wird nicht verworfen).",
        "corr_positive": "Richtung: positive Korrelation.",
        "corr_negative": "Richtung: negative Korrelation.",
        "corr_zero": "Keine klare lineare Korrelation.",
        "corr_weak": "Stärke: schwache Korrelation.",
        "corr_moderate": "Stärke: mittlere Korrelation.",
        "corr_strong": "Stärke: starke Korrelation.",
        "warning_select_valid": "Bitte wählen Sie Spalten mit geeigneten Datentypen für diese Analyse.",
    },
}


# ----------------------------------------------------------------
# Helper: get translations
# ----------------------------------------------------------------
def get_text(lang_code: str, key: str) -> str:
    """Return UI text for given language and key, fallback to English."""
    if lang_code not in LANG_TEXT:
        lang_code = "en"
    return LANG_TEXT.get(lang_code, LANG_TEXT["en"]).get(
        key, LANG_TEXT["en"].get(key, key)
    )


# ----------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel file into pandas DataFrame."""
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xls") or name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")
    return df


# ----------------------------------------------------------------
# Text preprocessing
# ----------------------------------------------------------------
def preprocess_text_series(series: pd.Series):
    """
    Basic preprocessing for a pandas Series of text:
    - lowercasing
    - remove punctuation
    - tokenization
    - English stopword removal
    Returns:
      - list of token lists per row
      - global frequency (value_counts) for all tokens
    """
    eng_stop = set(stopwords.words("english"))
    tokens_per_row = []
    all_tokens = []

    for text in series.fillna("").astype(str):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha() and t not in eng_stop]
        tokens_per_row.append(tokens)
        all_tokens.extend(tokens)

    freq = pd.Series(all_tokens).value_counts()
    return tokens_per_row, freq


def preprocess_text(df: pd.DataFrame):
    """Detect text columns and run preprocessing on the first one (for demo)."""
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_cols:
        return [], None, None, None

    first_col = text_cols[0]
    tokens_per_row, freq = preprocess_text_series(df[first_col])

    # Prepare sample tokens (first few rows)
    sample_rows = min(5, len(tokens_per_row))
    sample_tokens = tokens_per_row[:sample_rows]
    top_words = freq.head(10)

    return text_cols, first_col, sample_tokens, top_words


# ----------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame, col: str):
    series = df[col].dropna()
    if series.empty:
        return None
    desc = {
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "min": series.min(),
        "max": series.max(),
        "std": series.std(),
    }
    return pd.DataFrame.from_dict(desc, orient="index", columns=[col])


def frequency_tables(df: pd.DataFrame, col: str):
    series = df[col].astype("category")
    counts = series.value_counts(dropna=False)
    perc = series.value_counts(normalize=True, dropna=False) * 100
    freq_df = pd.DataFrame(
        {get_text(current_lang(), "freq_count"): counts,
         get_text(current_lang(), "freq_percent"): perc.round(2)}
    )
    return freq_df


# ----------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------
def visualize_data(df: pd.DataFrame, col: str, lang: str):
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, ax=ax_hist)
    ax_hist.set_title(f"{get_text(lang, 'histogram')} - {col}")
    st.pyplot(fig_hist)

    fig_box, ax_box = plt.subplots()
    sns.boxplot(x=df[col].dropna(), ax=ax_box)
    ax_box.set_title(f"{get_text(lang, 'boxplot')} - {col}")
    st.pyplot(fig_box)


# ----------------------------------------------------------------
# Correlation helpers
# ----------------------------------------------------------------
def interpret_strength(r: float, lang: str) -> str:
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = get_text(lang, "corr_zero")
    elif abs_r < 0.3:
        strength = get_text(lang, "corr_weak")
    elif abs_r < 0.5:
        strength = get_text(lang, "corr_moderate")
    else:
        strength = get_text(lang, "corr_strong")

    if r > 0.1:
        direction = get_text(lang, "corr_positive")
    elif r < -0.1:
        direction = get_text(lang, "corr_negative")
    else:
        direction = get_text(lang, "corr_zero")

    return f"{direction} {strength}"


def correlation_analysis(df: pd.DataFrame, x_col: str, y_col: str, method: str):
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return r, p


# ----------------------------------------------------------------
# Chi-square test
# ----------------------------------------------------------------
def chi_square_test(df: pd.DataFrame, x_col: str, y_col: str):
    table = pd.crosstab(df[x_col], df[y_col])
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, table, expected


# ----------------------------------------------------------------
# Utility to access current language inside functions
# (Streamlit state workaround)
# ----------------------------------------------------------------
def current_lang():
    return st.session_state.get("lang_code", "en")


# ----------------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------------
def main():
    # Language selection
    lang_options = {
        "id": LANG_TEXT["id"]["language_name"],
        "en": LANG_TEXT["en"]["language_name"],
        "es": LANG_TEXT["es"]["language_name"],
        "fr": LANG_TEXT["fr"]["language_name"],
        "de": LANG_TEXT["de"]["language_name"],
    }
    # Selectbox shows language names, but we map back to codes
    st.sidebar.markdown("## Language / Bahasa")
    selected_lang_name = st.sidebar.selectbox(
        "Language / Bahasa",
        options=list(lang_options.values()),
        index=0,
    )
    # Find code from name
    lang_code = [code for code, name in lang_options.items() if name == selected_lang_name][0]
    st.session_state["lang_code"] = lang_code

    # Title
    st.title(get_text(lang_code, "title"))

    # Upload section
    st.subheader(get_text(lang_code, "upload_subheader"))
    uploaded_file = st.file_uploader(
        get_text(lang_code, "upload_label"),
        type=["csv", "xls", "xlsx"],
    )

    if uploaded_file is None:
        st.info(get_text(lang_code, "no_file"))
        return

    # Load data
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error: {e}")
        return

    if df is None or df.empty:
        st.warning(get_text(lang_code, "no_file"))
        return

    # Preview
    st.subheader(get_text(lang_code, "data_preview"))
    st.dataframe(df.head())

    # ---------------- Text preprocessing ----------------
    st.subheader(get_text(lang_code, "text_processing_subheader"))
    text_cols, first_text_col, sample_tokens, top_words = preprocess_text(df)
    if not text_cols:
        st.write(get_text(lang_code, "no_text_columns"))
    else:
        st.write(f"{get_text(lang_code, 'text_columns_detected')} {', '.join(text_cols)}")
        st.caption(get_text(lang_code, "text_processing_note"))

        if first_text_col is not None:
            st.write(f"Column used for preview: {first_text_col}")
            st.write(get_text(lang_code, "sample_tokens"))
            if sample_tokens:
                for i, toks in enumerate(sample_tokens):
                    st.write(f"Row {i}: {toks}")
            st.write(get_text(lang_code, "top_words"))
            if top_words is not None:
                st.table(top_words.reset_index().rename(columns={"index": "word", 0: "freq"}))

    # ---------------- Descriptive statistics ----------------
    st.subheader(get_text(lang_code, "stats_subheader"))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.write(get_text(lang_code, "no_numeric_cols"))
    else:
        selected_num_col = st.selectbox(
            get_text(lang_code, "select_numeric_col"),
            options=numeric_cols,
        )
        if selected_num_col:
            desc_df = descriptive_stats(df, selected_num_col)
            if desc_df is not None:
                st.write(get_text(lang_code, "desc_stats"))
                st.table(desc_df)

    # ---------------- Frequency tables ----------------
    st.subheader(get_text(lang_code, "freq_table_subheader"))
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not categorical_cols:
        st.write(get_text(lang_code, "no_categorical_cols"))
    else:
        selected_cat_col = st.selectbox(
            get_text(lang_code, "select_categorical_col"),
            options=categorical_cols,
        )
        if selected_cat_col:
            freq_df = frequency_tables(df, selected_cat_col)
            st.table(freq_df)

    # ---------------- Visualization ----------------
    st.subheader(get_text(lang_code, "visual_subheader"))
    if numeric_cols:
        selected_vis_col = st.selectbox(
            get_text(lang_code, "select_numeric_col") + " (visualisation)",
            options=numeric_cols,
            key="vis_num_col",
        )
        if selected_vis_col:
            visualize_data(df, selected_vis_col, lang_code)

    # ---------------- Correlation Analysis ----------------
    st.subheader(get_text(lang_code, "correlation_subheader"))
    if len(numeric_cols) < 2:
        st.write(get_text(lang_code, "not_enough_numeric"))
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox(
                get_text(lang_code, "select_x_numeric"),
                options=numeric_cols,
                key="pearson_x",
            )
        with col2:
            y_col = st.selectbox(
                get_text(lang_code, "select_y_numeric"),
                options=[c for c in numeric_cols if c != x_col] or numeric_cols,
                key="pearson_y",
            )

        if x_col and y_col:
            # Pearson
            st.markdown(f"### {get_text(lang_code, 'pearson_header')}")
            try:
                r_p, p_p = correlation_analysis(df, x_col, y_col, method="pearson")
                st.write(f"{get_text(lang_code, 'corr_coef')} {r_p:.4f}")
                st.write(f"{get_text(lang_code, 'p_value')} {p_p:.4g}")
                interp = interpret_strength(r_p, lang_code)
                st.write(f"{get_text(lang_code, 'interpretation')} {interp}")
            except Exception:
                st.warning(get_text(lang_code, "warning_select_valid"))

            # Spearman
            st.markdown(f"### {get_text(lang_code, 'spearman_header')}")
            try:
                r_s, p_s = correlation_analysis(df, x_col, y_col, method="spearman")
                st.write(f"{get_text(lang_code, 'corr_coef')} {r_s:.4f}")
                st.write(f"{get_text(lang_code, 'p_value')} {p_s:.4g}")
                interp_s = interpret_strength(r_s, lang_code)
                st.write(f"{get_text(lang_code, 'interpretation')} {interp_s}")
            except Exception:
                st.warning(get_text(lang_code, "warning_select_valid"))

    # ---------------- Chi-square test ----------------
    st.subheader(get_text(lang_code, "chi_square_subheader"))
    if len(categorical_cols) < 2:
        st.write(get_text(lang_code, "not_enough_categorical"))
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_cat = st.selectbox(
                get_text(lang_code, "select_x_cat"),
                options=categorical_cols,
                key="chi_x",
            )
        with col2:
            y_cat = st.selectbox(
                get_text(lang_code, "select_y_cat"),
                options=[c for c in categorical_cols if c != x_cat] or categorical_cols,
                key="chi_y",
            )

        if x_cat and y_cat:
            try:
                chi2, p, dof, table, expected = chi_square_test(df, x_cat, y_cat)
                st.markdown(f"### {get_text(lang_code, 'chi_square_result')}")
                st.write(f"{get_text(lang_code, 'chi_square_stat')} {chi2:.4f}")
                st.write(f"{get_text(lang_code, 'chi_square_df')} {dof}")
                st.write(f"{get_text(lang_code, 'chi_square_p')} {p:.4g}")
                st.caption(get_text(lang_code, "alpha_note"))

                if p < 0.05:
                    st.write(get_text(lang_code, "chi_square_interpret"))
                    st.write(get_text(lang_code, "significant_assoc"))
                else:
                    st.write(get_text(lang_code, "chi_square_interpret"))
                    st.write(get_text(lang_code, "no_significant_assoc"))

                st.write("Observed counts:")
                st.table(table)
                st.write("Expected counts:")
                st.table(pd.DataFrame(expected, index=table.index, columns=table.columns))
            except Exception:
                st.warning(get_text(lang_code, "warning_select_valid"))


if __name__ == "__main__":
    main()
