import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, chi2_contingency
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter

# Inisialisasi resource NLTK (boleh dikomentari jika sudah di-download)
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
try:
    _ = word_tokenize("test")
except LookupError:
    nltk.download("punkt")

# --------------------------- PAGE CONFIG & CSS ---------------------------

st.set_page_config(page_title="Survey Data Analyzer", layout="wide")

CUSTOM_CSS = """
<style>
body {
    background: linear-gradient(135deg, #ECFDF5 0%, #A7F3D0 35%, #6EE7B7 70%, #10B981 100%);
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.main-card {
    background-color: rgba(240, 253, 250, 0.94);
    border-radius: 24px;
    padding: 2.0rem 2.4rem;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}
.hero-card {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 28px;
    padding: 2.2rem 2.6rem;
    box-shadow: 0 24px 60px rgba(16, 185, 129, 0.35);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.upload-card {
    background-color: #FFFFFF;
    border-radius: 24px;
    padding: 1.6rem 2.2rem;
    border: 2px dashed #22c55e;
    text-align: center;
    box-shadow: 0 12px 30px rgba(34, 197, 94, 0.35);
}
.feature-card {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 12px 28px rgba(16, 185, 129, 0.35);
    border: 1px solid rgba(34, 197, 94, 0.30);
}
.lang-pill {
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    border: 1px solid #22c55e;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    background: #ffffff;
    color: #15803d;
    margin-left: 0.18rem;
}
.lang-pill-active {
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    border: none;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: default;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #ffffff;
    box-shadow: 0 10px 25px rgba(22, 163, 74, 0.5);
}
.helper-text {
    font-size: 0.82rem;
    color: #047857;
}
.decorative-divider {
    height: 1px;
    width: 100%;
    margin: 0.7rem 0 1.3rem 0;
    background: linear-gradient(to right, transparent, #22c55e, transparent);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------- MULTI LANGUAGE DICTIONARY ---------------------------

LANG_TEXT = {
    "id": {
        "language_name": "Indonesia",
        "title": "📊 Analisis Data Survei",
        "subtitle": "Unggah file survei Anda (CSV/Excel) dan lakukan analisis deskriptif, visualisasi, serta uji korelasi secara interaktif.",
        "upload_subheader": "📁 Unggah Data Survei",
        "upload_label": "Tarik & letakkan file di sini atau klik untuk memilih (CSV, XLS, XLSX)",
        "no_file": "Belum ada file yang diunggah. Silakan unggah file untuk memulai analisis.",
        "data_preview": "Pratinjau Data (maksimal 1000 baris pertama)",
        "text_processing_subheader": "📝 Pra-pemrosesan Teks",
        "text_columns_detected": "Kolom teks terdeteksi:",
        "select_text_col": "Pilih kolom teks untuk diproses",
        "no_text_columns": "Tidak ada kolom bertipe teks yang terdeteksi.",
        "text_processing_note": "Teks akan diubah menjadi huruf kecil, dihapus tanda baca, ditokenisasi, dan dihapus stopword bahasa Inggris.",
        "sample_tokens": "Contoh token yang telah diproses",
        "top_words": "10 Kata Teratas Berdasarkan Frekuensi",
        "stats_subheader": "📈 Statistik Deskriptif & Distribusi",
        "select_numeric_col": "Pilih kolom numerik untuk statistik & grafik",
        "no_numeric_cols": "Tidak ada kolom numerik yang tersedia.",
        "desc_stats": "Statistik deskriptif untuk kolom terpilih",
        "freq_table_subheader": "📊 Tabel Frekuensi Kategorikal",
        "select_categorical_col": "Pilih kolom kategorikal untuk tabel frekuensi",
        "no_categorical_cols": "Tidak ada kolom kategorikal yang tersedia.",
        "freq_count": "Jumlah",
        "freq_percent": "Persentase (%)",
        "visual_subheader": "📉 Visualisasi Data",
        "histogram": "Histogram",
        "boxplot": "Boxplot",
        "visual_select": "Pilih jenis visualisasi",
        "correlation_subheader": "🔗 Analisis Korelasi & Uji Statistik",
        "pearson_header": "Korelasi Pearson",
        "spearman_header": "Korelasi Spearman (Rank)",
        "chi_header": "Uji Chi-square",
        "select_x_numeric": "Pilih variabel X (numerik)",
        "select_y_numeric": "Pilih variabel Y (numerik)",
        "not_enough_numeric": "Jumlah kolom numerik tidak mencukupi untuk analisis ini.",
        "pearson_result": "Hasil Korelasi Pearson",
        "spearman_result": "Hasil Korelasi Spearman",
        "corr_coef": "Koefisien korelasi (r)",
        "p_value": "Nilai p",
        "interpretation": "Interpretasi",
        "chi_square_subheader": "Uji Chi-square untuk Keterkaitan Kategorikal",
        "select_x_cat": "Pilih variabel X (kategorikal)",
        "select_y_cat": "Pilih variabel Y (kategorikal)",
        "not_enough_categorical": "Jumlah kolom kategorikal tidak mencukupi untuk uji Chi-square.",
        "chi_square_result": "Hasil Uji Chi-square",
        "chi_square_stat": "Statistik Chi-square",
        "chi_square_df": "Derajat kebebasan (df)",
        "chi_square_p": "Nilai p",
        "chi_square_interpret": "Interpretasi hasil Chi-square:",
        "alpha_note": "Uji signifikansi menggunakan α = 0.05.",
        "significant_assoc": "Terdapat asosiasi yang signifikan secara statistik antara kedua variabel.",
        "no_significant_assoc": "Tidak terdapat asosiasi yang signifikan secara statistik antara kedua variabel.",
        "corr_direction_positive": "Hubungan positif: ketika X meningkat, Y cenderung meningkat.",
        "corr_direction_negative": "Hubungan negatif: ketika X meningkat, Y cenderang menurun.",
        "corr_direction_zero": "Tidak ada arah hubungan yang jelas (mendekati nol).",
        "corr_strength_none": "Kekuatan hubungan hampir tidak ada.",
        "corr_strength_weak": "Hubungan lemah.",
        "corr_strength_moderate": "Hubungan moderat.",
        "corr_strength_strong": "Hubungan kuat.",
        "warning_select_valid": "Silakan pilih kombinasi kolom yang valid.",
        "header_github": "Fork di GitHub",
        "feature_title_1": "Analisis Deskriptif",
        "feature_desc_1": "Ringkas data numerik dengan statistik utama seperti mean, median, dan deviasi standar.",
        "feature_title_2": "Grafik Visual",
        "feature_desc_2": "Bangun histogram dan boxplot interaktif untuk melihat pola dan outlier.",
        "feature_title_3": "Analisis Korelasi",
        "feature_desc_3": "Gunakan Pearson, Spearman, dan Chi-square untuk mengecek hubungan antar variabel."
    },
    "en": {
        "language_name": "English",
        "title": "📊 Survey Data Analysis",
        "subtitle": "Upload your survey file (CSV/Excel) and explore descriptive statistics, visualizations, and correlation tests interactively.",
        "upload_subheader": "📁 Upload Survey Data",
        "upload_label": "Drag & drop file here or click to browse (CSV, XLS, XLSX)",
        "no_file": "No file uploaded yet. Please upload a file to start the analysis.",
        "data_preview": "Data Preview (up to first 1000 rows)",
        "text_processing_subheader": "📝 Text Preprocessing",
        "text_columns_detected": "Detected text columns:",
        "select_text_col": "Select a text column to process",
        "no_text_columns": "No text-type columns detected.",
        "text_processing_note": "Text will be lowercased, punctuation removed, tokenized, and English stopwords removed.",
        "sample_tokens": "Sample of processed tokens",
        "top_words": "Top 10 Words by Frequency",
        "stats_subheader": "📈 Descriptive Statistics & Distribution",
        "select_numeric_col": "Select a numeric column for statistics & plots",
        "no_numeric_cols": "No numeric columns available.",
        "desc_stats": "Descriptive statistics for the selected column",
        "freq_table_subheader": "📊 Categorical Frequency Table",
        "select_categorical_col": "Select a categorical column for frequency table",
        "no_categorical_cols": "No categorical columns available.",
        "freq_count": "Count",
        "freq_percent": "Percent (%)",
        "visual_subheader": "📉 Data Visualizations",
        "histogram": "Histogram",
        "boxplot": "Boxplot",
        "visual_select": "Select visualization type",
        "correlation_subheader": "🔗 Correlation & Statistical Tests",
        "pearson_header": "Pearson Correlation",
        "spearman_header": "Spearman Rank Correlation",
        "chi_header": "Chi-square Test",
        "select_x_numeric": "Select X variable (numeric)",
        "select_y_numeric": "Select Y variable (numeric)",
        "not_enough_numeric": "Not enough numeric columns for this analysis.",
        "pearson_result": "Pearson Correlation Result",
        "spearman_result": "Spearman Correlation Result",
        "corr_coef": "Correlation coefficient (r)",
        "p_value": "p-value",
        "interpretation": "Interpretation",
        "chi_square_subheader": "Chi-square Test for Categorical Association",
        "select_x_cat": "Select X variable (categorical)",
        "select_y_cat": "Select Y variable (categorical)",
        "not_enough_categorical": "Not enough categorical columns for Chi-square test.",
        "chi_square_result": "Chi-square Test Result",
        "chi_square_stat": "Chi-square statistic",
        "chi_square_df": "Degrees of freedom (df)",
        "chi_square_p": "p-value",
        "chi_square_interpret": "Chi-square result interpretation:",
        "alpha_note": "Significance tested at α = 0.05.",
        "significant_assoc": "There is a statistically significant association between the two variables.",
        "no_significant_assoc": "There is no statistically significant association between the two variables.",
        "corr_direction_positive": "Positive relationship: as X increases, Y tends to increase.",
        "corr_direction_negative": "Negative relationship: as X increases, Y tends to decrease.",
        "corr_direction_zero": "No clear direction of relationship (near zero).",
        "corr_strength_none": "Virtually no relationship.",
        "corr_strength_weak": "Weak relationship.",
        "corr_strength_moderate": "Moderate relationship.",
        "corr_strength_strong": "Strong relationship.",
        "warning_select_valid": "Please select a valid combination of columns.",
        "header_github": "Fork on GitHub",
        "feature_title_1": "Descriptive Analytics",
        "feature_desc_1": "Summarize numeric data with key statistics like mean, median, and standard deviation.",
        "feature_title_2": "Visual Charts",
        "feature_desc_2": "Build interactive histograms and boxplots to reveal patterns and outliers.",
        "feature_title_3": "Correlation Analysis",
        "feature_desc_3": "Use Pearson, Spearman, and Chi-square to assess relationships between variables."
    },
    "zh": {
        "language_name": "中文",
        "title": "📊 调查数据分析",
        "subtitle": "上传问卷数据文件（CSV/Excel），交互式地探索描述性统计、可视化和相关性检验。",
    },
    "ja": {
        "language_name": "日本語",
        "title": "📊 アンケートデータ分析",
        "subtitle": "アンケートファイル（CSV/Excel）をアップロードして、記述統계・可视化・相関分析を行います。",
    },
    "ko": {
        "language_name": "한국어",
        "title": "📊 설문 데이터 분석",
        "subtitle": "설문 파일(CSV/Excel)을 업로드하고 기술 통계, 시각화, 상관 분석을 수행합니다.",
    },
    "es": {
        "language_name": "Español",
        "title": "📊 Análisis de Datos de Encuestas",
        "subtitle": "Cargue su archivo de encuesta (CSV/Excel) y explore estadísticas descriptivas, visualizaciones y pruebas de correlación.",
    },
    "ar": {
        "language_name": "العربية",
        "title": "📊 تحليل بيانات الاستبيان",
        "subtitle": "قم برفع ملف الاستبيان (CSV/Excel) لاستكشاف الإحصاءات الوصفية والرسوم البيانية واختبارات الارتباط.",
    },
    "it": {
        "language_name": "Italiano",
        "title": "📊 Analisi dei Dati di Sondaggio",
        "subtitle": "Carica il file del sondaggio (CSV/Excel) ed esplora statistiche descrittive, visualizzazioni e test di correlazione.",
    },
}

for code in ["zh", "ja", "ko", "es", "ar", "it"]:
    base = LANG_TEXT["en"]
    LANG_TEXT[code] = {**base, **LANG_TEXT[code]}

def get_text(lang: str, key: str) -> str:
    if lang not in LANG_TEXT:
        lang = "en"
    return LANG_TEXT.get(lang, LANG_TEXT["en"]).get(
        key, LANG_TEXT["en"].get(key, key)
    )

# --------------------------- HELPER FUNCTIONS ---------------------------

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xls") or name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
    except Exception:
        return None
    return None


def preprocess_text_series(series: pd.Series) -> pd.Series:
    eng_stop = set(stopwords.words("english"))
    punct_table = str.maketrans("", "", string.punctuation)

    def _clean(text):
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = text.translate(punct_table)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha() and t not in eng_stop]
        return tokens

    return series.apply(_clean)


def descriptive_stats(series: pd.Series) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    stats_dict = {
        "mean": s.mean(),
        "median": s.median(),
        "mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
        "min": s.min(),
        "max": s.max(),
        "std": s.std(),
    }
    return pd.DataFrame(stats_dict, index=[0]).T.rename(columns={0: "value"})


def frequency_tables(series: pd.Series) -> pd.DataFrame:
    freq = series.value_counts(dropna=False)
    pct = series.value_counts(normalize=True, dropna=False) * 100
    return pd.DataFrame({"count": freq, "percent": pct})


def visualize_data(df: pd.DataFrame, col: str, lang: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        st.warning(get_text(lang, "warning_select_valid"))
        return
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(s, kde=True, ax=ax, color="#16a34a")
        ax.set_title(get_text(lang, "histogram"))
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.boxplot(x=s, ax=ax2, color="#22c55e")
        ax2.set_title(get_text(lang, "boxplot"))
        st.pyplot(fig2)


def interpret_strength(r: float, lang: str) -> str:
    if r is None or np.isnan(r):
        return get_text(lang, "corr_strength_none")
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = get_text(lang, "corr_strength_none")
    elif abs_r < 0.3:
        strength = get_text(lang, "corr_strength_weak")
    elif abs_r < 0.5:
        strength = get_text(lang, "corr_strength_moderate")
    else:
        strength = get_text(lang, "corr_strength_strong")
    if r > 0.05:
        direction = get_text(lang, "corr_direction_positive")
    elif r < -0.05:
        direction = get_text(lang, "corr_direction_negative")
    else:
        direction = get_text(lang, "corr_direction_zero")
    return f"{strength} {direction}"


def correlation_analysis(df: pd.DataFrame, x_col: str, y_col: str, method: str = "pearson"):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 2:
        return np.nan, np.nan
    if method == "spearman":
        r, p = spearmanr(x_clean, y_clean)
    else:
        r, p = pearsonr(x_clean, y_clean)
    return r, p


def chi_square_test(df: pd.DataFrame, x_col: str, y_col: str):
    table = pd.crosstab(df[x_col], df[y_col])
    if table.size == 0:
        return None, None, None, None
    chi2, p, dof, expected = chi2_contingency(table)
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    return chi2, p, dof, expected_df

# --------------------------- SESSION STATE ---------------------------

if "lang" not in st.session_state:
    st.session_state["lang"] = "id"
if "active_feature" not in st.session_state:
    st.session_state["active_feature"] = None  # "desc", "visual", "corr"

current_lang = st.session_state["lang"]

# --------------------------- HEADER BAR ---------------------------

st.markdown(
    """
    <div style="
        width:100%;
        padding:0.40rem 0.9rem 0.35rem 0.9rem;
        display:flex;
        align-items:center;
        justify-content:space-between;
        border-radius:18px;
        background:rgba(240, 253, 250, 0.96);
        box-shadow:0 10px 25px rgba(15, 118, 110, 0.15);
        border:1px solid rgba(45, 212, 191, 0.55);
        margin-bottom:0.9rem;
    ">
      <div style="font-weight:650; color:#047857; font-size:0.95rem; letter-spacing:0.02em;">
        Survey Data Analyzer
      </div>
      <div style="display:flex; align-items:center; gap:1.2rem;">
        <div>
          <a href="https://github.com/yourusername/survey-analyzer"
             style="text-decoration:none; font-weight:600; color:#047857; font-size:0.9rem;">
             🐙 """ + get_text(current_lang, "header_github") + """
          </a>
        </div>
        <div style="display:flex; align-items:center; gap:0.25rem; flex-wrap:wrap;">
    """,
    unsafe_allow_html=True,
)

lang_options = [
    ("id", "ID"),
    ("en", "EN"),
    ("zh", "ZH"),
    ("ja", "JA"),
    ("ko", "KO"),
    ("es", "ES"),
    ("ar", "AR"),
    ("it", "IT"),
]

cols = st.columns(len(lang_options))

for (code, label), col in zip(lang_options, cols):
    with col:
        is_active = (current_lang == code)
        btn_label = f"● {label}" if is_active else label
        if st.button(btn_label, key=f"lang_btn_{code}", help=LANG_TEXT[code]["language_name"]):
            st.session_state["lang"] = code
            current_lang = code

st.markdown("</div></div></div>", unsafe_allow_html=True)

st.markdown("<div class='decorative-divider'></div>", unsafe_allow_html=True)

# --------------------------- HERO SECTION ---------------------------

c1, c2, c3 = st.columns([1, 3, 1])
with c2:
    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    st.markdown(f"## {get_text(current_lang, 'title')}")
    st.markdown(
        f"<p style='font-size:0.95rem; color:#065f46;'>{get_text(current_lang, 'subtitle')}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# --------------------------- MAIN CARD ---------------------------

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Upload section (drag & drop + Excel)
st.markdown(f"### {get_text(current_lang, 'upload_subheader')}")

u1, u2, u3 = st.columns([1, 2, 1])
with u2:
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-weight:600; margin-bottom:0.2rem;'>📤</p>"
        "<p style='margin-bottom:0.1rem;'>Drag & drop file di area di bawah ini atau klik untuk memilih file</p>"
        "<p class='helper-text'>Limit 200MB • CSV, XLS, XLSX</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        label=get_text(current_lang, "upload_label"),
        type=["csv", "xls", "xlsx"],
        label_visibility="visible",
        accept_multiple_files=False,
    )

    st.markdown("</div>", unsafe_allow_html=True)

df = load_data(uploaded)
if df is None:
    st.info(get_text(current_lang, "no_file"))
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Data preview sampai 1000 baris
st.markdown(f"#### {get_text(current_lang, 'data_preview')}")
max_rows_preview = 1000
df_preview = df.head(max_rows_preview)
st.dataframe(df_preview, height=500, use_container_width=True)

# --------------------------- FEATURE CARDS INTERAKTIF ---------------------------

st.markdown("### ✨ Fitur Utama / Key Features")
fc1, fc2, fc3 = st.columns(3)

with fc1:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("#### 📈 " + get_text(current_lang, "feature_title_1"))
    st.markdown(
        f"<p class='helper-text'>{get_text(current_lang, 'feature_desc_1')}</p>",
        unsafe_allow_html=True,
    )
    if st.button("Lihat / Show", key="feat_desc"):
        st.session_state["active_feature"] = "desc"
    st.markdown("</div>", unsafe_allow_html=True)

with fc2:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 " + get_text(current_lang, "feature_title_2"))
    st.markdown(
        f"<p class='helper-text'>{get_text(current_lang, 'feature_desc_2')}</p>",
        unsafe_allow_html=True,
    )
    if st.button("Lihat / Show", key="feat_visual"):
        st.session_state["active_feature"] = "visual"
    st.markdown("</div>", unsafe_allow_html=True)

with fc3:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔗 " + get_text(current_lang, "feature_title_3"))
    st.markdown(
        f"<p class='helper-text'>{get_text(current_lang, 'feature_desc_3')}</p>",
        unsafe_allow_html=True,
    )
    if st.button("Lihat / Show", key="feat_corr"):
        st.session_state["active_feature"] = "corr"
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# --------------------------- TEXT PREPROCESSING ---------------------------

with st.expander(get_text(current_lang, "text_processing_subheader"), expanded=False):
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_cols:
        st.warning(get_text(current_lang, "no_text_columns"))
    else:
        st.markdown(get_text(current_lang, "text_columns_detected") + f" `{', '.join(text_cols)}`")
        text_col = st.selectbox(get_text(current_lang, "select_text_col"), options=text_cols)
        st.markdown(
            f"<p class='helper-text'>{get_text(current_lang, 'text_processing_note')}</p>",
            unsafe_allow_html=True,
        )
        processed = preprocess_text_series(df[text_col])
        st.markdown(f"**{get_text(current_lang, 'sample_tokens')}**")
        st.write(processed.head(5).tolist())
        all_tokens = [t for row in processed for t in row]
        counter = Counter(all_tokens)
        top10 = counter.most_common(10)
        if top10:
            top_df = pd.DataFrame(top10, columns=["word", "count"])
            st.markdown(f"**{get_text(current_lang, 'top_words')}**")
            st.table(top_df)

# --------------------------- DESCRIPTIVE STATS & FREQUENCY ---------------------------

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

num_col = None
cat_col = None

if st.session_state["active_feature"] in (None, "desc", "visual"):
    st.markdown(f"### {get_text(current_lang, 'stats_subheader')}")
    if not numeric_cols:
        st.warning(get_text(current_lang, "no_numeric_cols"))
    else:
        num_col = st.selectbox(get_text(current_lang, "select_numeric_col"), options=numeric_cols)
        stats_df = descriptive_stats(df[num_col])
        st.markdown(f"**{get_text(current_lang, 'desc_stats')}**")
        st.table(stats_df)

    if not cat_cols:
        st.info(get_text(current_lang, "no_categorical_cols"))
    else:
        cat_col = st.selectbox(get_text(current_lang, "select_categorical_col"), options=cat_cols)
        freq_df = frequency_tables(df[cat_col])
        freq_df.columns = [get_text(current_lang, "freq_count"),
                           get_text(current_lang, "freq_percent")]
        st.markdown(f"### {get_text(current_lang, 'freq_table_subheader')}")
        st.table(freq_df)

# --------------------------- VISUALISASI ---------------------------

if num_col and st.session_state["active_feature"] in (None, "visual", "desc"):
    st.markdown(f"### {get_text(current_lang, 'visual_subheader')}")
    visualize_data(df, num_col, current_lang)

# --------------------------- KORELASI & UJI STATISTIK ---------------------------

if st.session_state["active_feature"] == "corr":
    st.markdown(f"### {get_text(current_lang, 'correlation_subheader')}")

    # Pearson
    with st.expander(get_text(current_lang, "pearson_header"), expanded=True):
        if len(numeric_cols) < 2:
            st.info(get_text(current_lang, "not_enough_numeric"))
        else:
            c1p, c2p = st.columns(2)
            with c1p:
                x_num = st.selectbox(get_text(current_lang, "select_x_numeric"),
                                     options=numeric_cols, key="pearson_x")
            with c2p:
                y_num = st.selectbox(get_text(current_lang, "select_y_numeric"),
                                     options=[c for c in numeric_cols if c != x_num],
                                     key="pearson_y")
            if x_num and y_num:
                try:
                    r, p = correlation_analysis(df, x_num, y_num, method="pearson")
                    if np.isnan(r):
                        st.warning(get_text(current_lang, "warning_select_valid"))
                    else:
                        st.markdown(f"**{get_text(current_lang, 'pearson_result')}**")
                        out = pd.DataFrame({
                            get_text(current_lang, "corr_coef"): [r],
                            get_text(current_lang, "p_value"): [p],
                        })
                        st.table(out)
                        st.markdown(
                            f"**{get_text(current_lang, 'interpretation')}:** "
                            f"{interpret_strength(r, current_lang)}"
                        )
                except Exception:
                    st.warning(get_text(current_lang, "warning_select_valid"))

    # Spearman
    with st.expander(get_text(current_lang, "spearman_header"), expanded=True):
        if len(numeric_cols) < 2:
            st.info(get_text(current_lang, "not_enough_numeric"))
        else:
            c1s, c2s = st.columns(2)
            with c1s:
                x_s = st.selectbox(get_text(current_lang, "select_x_numeric"),
                                   options=numeric_cols, key="spearman_x")
            with c2s:
                y_s = st.selectbox(get_text(current_lang, "select_y_numeric"),
                                   options=[c for c in numeric_cols if c != x_s],
                                   key="spearman_y")
            if x_s and y_s:
                try:
                    r_s, p_s = correlation_analysis(df, x_s, y_s, method="spearman")
                    if np.isnan(r_s):
                        st.warning(get_text(current_lang, "warning_select_valid"))
                    else:
                        st.markdown(f"**{get_text(current_lang, 'spearman_result')}**")
                        out_s = pd.DataFrame({
                            get_text(current_lang, "corr_coef"): [r_s],
                            get_text(current_lang, "p_value"): [p_s],
                        })
                        st.table(out_s)
                        st.markdown(
                            f"**{get_text(current_lang, 'interpretation')}:** "
                            f"{interpret_strength(r_s, current_lang)}"
                        )
                except Exception:
                    st.warning(get_text(current_lang, "warning_select_valid"))

    # Chi-square
    with st.expander(get_text(current_lang, "chi_header"), expanded=True):
        if len(cat_cols) < 2:
            st.info(get_text(current_lang, "not_enough_categorical"))
        else:
            c1c, c2c = st.columns(2)
            with c1c:
                x_cat = st.selectbox(get_text(current_lang, "select_x_cat"),
                                     options=cat_cols, key="chi_x")
            with c2c:
                y_cat = st.selectbox(get_text(current_lang, "select_y_cat"),
                                     options=[c for c in cat_cols if c != x_cat],
                                     key="chi_y")
            if x_cat and y_cat:
                try:
                    chi2, p_val, dof_val, expected_df = chi_square_test(df, x_cat, y_cat)
                    if chi2 is None:
                        st.warning(get_text(current_lang, "warning_select_valid"))
                    else:
                        st.markdown(f"**{get_text(current_lang, 'chi_square_result')}**")
                        out_c = pd.DataFrame({
                            get_text(current_lang, "chi_square_stat"): [chi2],
                            get_text(current_lang, "chi_square_df"): [dof_val],
                            get_text(current_lang, "chi_square_p"): [p_val],
                        })
                        st.table(out_c)

                        st.markdown("**Observed**")
                        observed_table = pd.crosstab(df[x_cat], df[y_cat])
                        st.dataframe(
                            observed_table,
                            height=200,
                            use_container_width=True
                        )

                        st.markdown("**Expected**")
                        st.dataframe(
                            expected_df,
                            height=200,
                            use_container_width=True
                        )

                        st.markdown(f"_{get_text(current_lang, 'alpha_note')}_")
                        if p_val < 0.05:
                            st.success(get_text(current_lang, "significant_assoc"))
                        else:
                            st.info(get_text(current_lang, "no_significant_assoc"))
                except Exception:
                    st.warning(get_text(current_lang, "warning_select_valid"))

st.markdown("</div>", unsafe_allow_html=True)
