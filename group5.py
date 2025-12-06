import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, chi2_contingency, normaltest
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
import time

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

# --------------------------- NLTK INIT ---------------------------
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# --------------------------- SESSION STATE ---------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "language" not in st.session_state:
    st.session_state["language"] = "EN"

# --------------------------- PAGE CONFIG & CSS ---------------------------
st.set_page_config(page_title="Digital Payment Usage & Financial Discipline Survey", layout="wide")

# Aurora background animation
st.markdown("""
<div class="aurora-container">
    <div class="aurora-layer"></div>
    <div class="aurora-layer"></div>
    <div class="aurora-layer"></div>
</div>
""", unsafe_allow_html=True)

top_col1, top_col2 = st.columns([3, 3])
with top_col1:
    dm = st.toggle("🌙 Dark mode", value=st.session_state["dark_mode"])
    st.session_state["dark_mode"] = dm
with top_col2:
    lang = st.radio(
        "Language",
        options=["EN", "ID", "JP", "KR", "CN", "AR"],
        horizontal=True,
        index=["EN", "ID", "JP", "KR", "CN", "AR"].index(st.session_state["language"]),
    )
    st.session_state["language"] = lang

CUSTOM_CSS = """
<style>
body {
    background: #000;
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
    border-radius: 14px;
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
.summary-badge {
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.4);
    font-size: 0.8rem;
    color: #047857;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-right: 0.4rem;
}
.summary-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #22c55e;
}
.section-card {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 1.0rem 1.4rem;
    border: 1px solid rgba(34, 197, 94, 0.35);
    box-shadow: 0 10px 26px rgba(16, 185, 129, 0.30);
    margin: 0.6rem 0 0.9rem 0;
}
.section-title {
    font-weight: 700;
    font-size: 1.0rem;
    margin-bottom: 0.25rem;
}
.section-subtitle {
    font-size: 0.85rem;
    color: #047857;
    margin-bottom: 0;
}
.aurora-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
}
.aurora-layer {
    position: absolute;
    width: 100%;
    height: 100%;
    background: radial-gradient(ellipse at 50% 50%, rgba(0, 100, 255, 0.3) 0%, transparent 50%);
    animation: wave 15s infinite linear;
}
.aurora-layer:nth-child(2) {
    background: radial-gradient(ellipse at 30% 70%, rgba(255, 100, 0, 0.2) 0%, transparent 50%);
    animation: wave 20s infinite linear 5s;
}
.aurora-layer:nth-child(3) {
    background: radial-gradient(ellipse at 70% 30%, rgba(100, 255, 0, 0.25) 0%, transparent 50%);
    animation: wave 25s infinite linear 10s;
}
@keyframes wave {
    0% {
        transform: scale(1) rotate(0deg) translateX(-20%) translateY(0%);
    }
    25% {
        transform: scale(1.05) rotate(90deg) translateX(0%) translateY(-10%);
    }
    50% {
        transform: scale(1.1) rotate(180deg) translateX(20%) translateY(0%);
    }
    75% {
        transform: scale(1.05) rotate(270deg) translateX(0%) translateY(10%);
    }
    100% {
        transform: scale(1) rotate(360deg) translateX(-20%) translateY(0%);
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if st.session_state["dark_mode"]:
    st.markdown(
        """
        <style>
        body {
            background: radial-gradient(circle at top, #0f172a 0%, #020617 55%, #000000 100%) !important;
            color: #e5e7eb !important;
        }
        .main-card, .hero-card, .upload-card {
            background-color: rgba(15, 23, 42, 0.96) !important;
            color: #e5e7eb !important;
        }
        .helper-text {
            color: #a7f3d0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------- MULTI-LANGUAGE TEXTS ---------------------------
TEXTS = {
    "EN": {
        "title": "📊 Digital Payment Usage & Financial Discipline Survey",
        "subtitle": "survey data analysis",
        "upload_subheader": "📁 Upload Survey Data",
        "upload_label": "Drag & drop file here or click to browse (CSV, XLS, XLSX)",
        "data_preview": "Data Preview (up to first 1000 rows)",
        "text_processing_subheader": "📝 Text Preprocessing",
        "text_columns_detected": "Detected text columns:",
        "select_text_col": "Select a text column to process",
        "no_text_columns": "No text-type columns detected.",
        "text_processing_note": "Text will be lowercased, punctuation removed, tokenized (split by spaces), and English stopwords removed.",
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
        "correlation_subheader": "🔗 Correlation & Statistical Tests",
        "pearson_header": "Pearson Correlation",
        "spearman_header": "Spearman Rank Correlation",
        "chi_header": "Chi-square Test",
        "select_x_numeric": "Select X variable (numeric)",
        "select_y_numeric": "Select Y variable (numeric)",
        "not_enough_numeric": "Not enough numeric columns for this analysis.",
        "pearson_result": "Pearson Correlation Result",
        "spearman_result": "Spearman Rank Correlation Result",
        "corr_coef": "Correlation coefficient (r)",
        "p_value": "p-value",
        "interpretation": "Interpretation",
        "select_x_cat": "Select X variable (categorical)",
        "select_y_cat": "Select Y variable (categorical)",
        "not_enough_categorical": "Not enough categorical columns for Chi-square test.",
        "chi_square_result": "Chi-square Test Result",
        "chi_square_stat": "Chi-square statistic",
        "chi_square_df": "Degrees of freedom (df)",
        "chi_square_p": "p-value",
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
        "nav_desc": "Descriptive Stats",
        "nav_visual": "Visualizations",
        "nav_corr": "Correlations & Tests",
        "nav_text": "Text Processing",
        "export_title": "Export Report",
        "export_desc": "Generate a complete PDF with all descriptive stats, normality test, histograms, boxplots, correlations, and text analysis summary.",
        "export_button": "Generate PDF report",
        "export_filename": "survey_full_report.pdf",
        "pdf_title": "Survey Data Full Report",
        "pdf_section_numdist": "1. Numeric Variables - Distributions",
        "pdf_section_scatter": "2. Scatter Plots - Relationships",
        "pdf_section_catbar": "3. Categorical Variables - Bar Charts",
        "pdf_section_numfull": "4. Numeric Variables - Full Statistics",
        "pdf_section_catfreq": "5. Categorical Variables - Frequency Tables",
        "pdf_section_corr": "6. Correlation Analysis",
        "pdf_section_text": "7. Text Analysis - Top Words",
        "pdf_notext": "No text data to analyze.",
        "filter_data_optional": "Filter data (optional)",
        "filter_column": "Filter column",
        "no_filter": "(No filter)",
        "select_values": "Select values",
        "summary_normality": "Summary & Normality",
        "distribution": "Distribution",
        "select_column_distribution": "Select column for distribution",
        "normality_test": "Normality test (D’Agostino-Pearson)",
        "statistic": "Statistic",
        "deviate_normal": "Data deviate significantly from normal distribution (reject H0 at α = 0.05).",
        "no_deviate_normal": "No significant deviation from normal distribution (fail to reject H0 at α = 0.05).",
        "not_enough_normality": "Not enough data points for normality test (need at least 8 non-missing values).",
        "histogram_boxplot": "Histogram / Boxplot",
        "scatter_bar": "Scatter & Bar",
        "x_variable_numeric": "X variable (numeric)",
        "y_variable_numeric": "Y variable (numeric)",
        "scatter_plot": "Scatter plot",
        "not_enough_scatter": "Not enough valid data for scatter plot.",
        "need_2_numeric": "Need at least 2 numeric columns for scatter plot.",
        "categorical_bar": "Categorical column for bar chart",
        "bar_chart": "Bar chart (top 20)",
        "no_categorical_bar": "No categorical columns for bar chart.",
        "independent_variable": "Independent variable",
        "dependent_variable": "Dependent variable",
        "observed": "Observed",
        "expected": "Expected",
        "no_file": "Please upload a file to get started.",
        "data_preview_subtitle": "survey data analysis",
    },
    "ID": {
        "title": "📊 Digital Payment Usage & Financial Discipline Survey",
        "subtitle": "survey data analysis",
        "upload_subheader": "📁 Unggah Data Survei",
        "upload_label": "Tarik & letakkan file di sini atau klik untuk memilih (CSV, XLS, XLSX)",
        "data_preview": "Pratinjau Data (maksimal 1000 baris pertama)",
        "text_processing_subheader": "📝 Pemrosesan Teks",
        "text_columns_detected": "Kolom teks terdeteksi:",
        "select_text_col": "Pilih kolom teks untuk diproses",
        "no_text_columns": "Tidak ada kolom bertipe teks.",
        "text_processing_note": "Teks akan di-lowercase, tanda baca dihapus, dipisah per kata, dan stopwords bahasa Inggris dihapus.",
        "sample_tokens": "Contoh token yang telah diproses",
        "top_words": "10 Kata Teratas berdasarkan Frekuensi",
        "stats_subheader": "📈 Statistik Deskriptif & Distribusi",
        "select_numeric_col": "Pilih kolom numerik untuk statistik & grafik",
        "no_numeric_cols": "Tidak ada kolom numerik.",
        "desc_stats": "Statistik deskriptif untuk kolom yang dipilih",
        "freq_table_subheader": "📊 Tabel Frekuensi Kategorikal",
        "select_categorical_col": "Pilih kolom kategorikal untuk tabel frekuensi",
        "no_categorical_cols": "Tidak ada kolom kategorikal.",
        "freq_count": "Frekuensi",
        "freq_percent": "Persentase (%)",
        "visual_subheader": "📉 Visualisasi Data",
        "histogram": "Histogram",
        "boxplot": "Boxplot",
        "correlation_subheader": "🔗 Korelasi & Uji Statistik",
        "pearson_header": "Korelasi Pearson",
        "spearman_header": "Korelasi Spearman",
        "chi_header": "Uji Chi-square",
        "select_x_numeric": "Pilih variabel X (numerik)",
        "select_y_numeric": "Pilih variabel Y (numerik)",
        "not_enough_numeric": "Kolom numerik tidak mencukupi untuk analisis ini.",
        "pearson_result": "Hasil Korelasi Pearson",
        "spearman_result": "Hasil Korelasi Spearman",
        "corr_coef": "Koefisien korelasi (r)",
        "p_value": "p-value",
        "interpretation": "Interpretasi",
        "select_x_cat": "Pilih variabel X (kategorikal)",
        "select_y_cat": "Pilih variabel Y (kategorikal)",
        "not_enough_categorical": "Kolom kategorikal tidak mencukupi untuk uji Chi-square.",
        "chi_square_result": "Hasil Uji Chi-square",
        "chi_square_stat": "Statistik Chi-square",
        "chi_square_df": "Derajat bebas (df)",
        "chi_square_p": "p-value",
        "alpha_note": "Signifikansi diuji pada α = 0,05.",
        "significant_assoc": "Terdapat hubungan yang signifikan secara statistik antara kedua variabel.",
        "no_significant_assoc": "Tidak terdapat hubungan yang signifikan secara statistik antara kedua variabel.",
        "corr_direction_positive": "Hubungan positif: ketika X naik, Y cenderung naik.",
        "corr_direction_negative": "Hubungan negatif: ketika X naik, Y cenderung turun.",
        "corr_direction_zero": "Tidak ada arah hubungan yang jelas (mendekati nol).",
        "corr_strength_none": "Hampir tidak ada hubungan.",
        "corr_strength_weak": "Hubungan lemah.",
        "corr_strength_moderate": "Hubungan sedang.",
        "corr_strength_strong": "Hubungan kuat.",
        "warning_select_valid": "Silakan pilih kombinasi kolom yang valid.",
        "header_github": "Fork di GitHub",
        "nav_desc": "Statistik Deskriptif",
        "nav_visual": "Visualisasi",
        "nav_corr": "Korelasi & Uji",
        "nav_text": "Pemrosesan Teks",
        "export_title": "Ekspor Laporan",
        "export_desc": "Buat PDF lengkap berisi statistik deskriptif, uji normalitas, histogram, boxplot, korelasi, dan ringkasan analisis teks.",
        "export_button": "Buat laporan PDF",
        "export_filename": "laporan_survei_lengkap.pdf",
        "pdf_title": "Laporan Lengkap Data Survei",
        "pdf_section_numdist": "1. Variabel Numerik - Distribusi",
        "pdf_section_scatter": "2. Scatter Plot - Hubungan",
        "pdf_section_catbar": "3. Variabel Kategorikal - Diagram Batang",
        "pdf_section_numfull": "4. Variabel Numerik - Statistik Lengkap",
        "pdf_section_catfreq": "5. Variabel Kategorikal - Tabel Frekuensi",
        "pdf_section_corr": "6. Analisis Korelasi",
        "pdf_section_text": "7. Analisis Teks - Kata Teratas",
        "pdf_notext": "Tidak ada data teks untuk dianalisis.",
    },
    "JP": {  # Japanese
        "title": "📊 Digital Payment Usage & Financial Discipline Survey",
        "subtitle": "survey data analysis",
        "upload_subheader": "📁 アンケートデータのアップロード",
        "upload_label": "ここにファイルをドラッグ＆ドロップ、またはクリックして選択（CSV, XLS, XLSX）",
        "data_preview": "データプレビュー（先頭1000行まで）",
        "text_processing_subheader": "📝 テキスト前処理",
        "text_columns_detected": "検出されたテキスト列：",
        "select_text_col": "前処理するテキスト列を選択",
        "no_text_columns": "テキスト型の列が見つかりません。",
        "text_processing_note": "テキストは小文字化され、句読点が削除され、スペースで分割され、英語のストップワードが除去されます。",
        "sample_tokens": "前処理されたトークンのサンプル",
        "top_words": "出現頻度トップ10の単語",
        "stats_subheader": "📈 記述統計と分布",
        "select_numeric_col": "統計・グラフ用の数値列を選択",
        "no_numeric_cols": "利用可能な数値列がありません。",
        "desc_stats": "選択された列の記述統計",
        "freq_table_subheader": "📊 カテゴリ頻度表",
        "select_categorical_col": "頻度表を作成するカテゴリ列を選択",
        "no_categorical_cols": "カテゴリ列がありません。",
        "freq_count": "度数",
        "freq_percent": "割合（％）",
        "visual_subheader": "📉 データの可視化",
        "histogram": "ヒストグラム",
        "boxplot": "箱ひげ図",
        "correlation_subheader": "🔗 相関と統計的検定",
        "pearson_header": "ピアソンの相関",
        "spearman_header": "スピアマンの順位相関",
        "chi_header": "カイ二乗検定",
        "select_x_numeric": "X変数（数値）を選択",
        "select_y_numeric": "Y変数（数値）を選択",
        "not_enough_numeric": "この分析に必要な数値列が不足しています。",
        "pearson_result": "ピアソン相関の結果",
        "spearman_result": "スピアマン相関の結果",
        "corr_coef": "相関係数 (r)",
        "p_value": "p値",
        "interpretation": "解釈",
        "select_x_cat": "X変数（カテゴリ）を選択",
        "select_y_cat": "Y変数（カテゴリ）を選択",
        "not_enough_categorical": "カイ二乗検定に必要なカテゴリ列が不足しています。",
        "chi_square_result": "カイ二乗検定の結果",
        "chi_square_stat": "カイ二乗統計量",
        "chi_square_df": "自由度 (df)",
        "chi_square_p": "p値",
        "alpha_note": "有意水準 α = 0.05 で検定しています。",
        "significant_assoc": "2つの変数の間に統計的に有意な関係があります。",
        "no_significant_assoc": "2つの変数の間に統計的に有意な関係はありません。",
        "corr_direction_positive": "正の関係：Xが増加するとYも増加する傾向があります。",
        "corr_direction_negative": "負の関係：Xが増加するとYは減少する傾向があります。",
        "corr_direction_zero": "明確な関係の方向がありません（ほぼ0）。",
        "corr_strength_none": "ほとんど関係がありません。",
        "corr_strength_weak": "弱い関係です。",
        "corr_strength_moderate": "中程度の関係です。",
        "corr_strength_strong": "強い関係です。",
        "warning_select_valid": "有効な列の組み合わせを選択してください。",
        "header_github": "GitHubでフォーク",
        "nav_desc": "記述統計",
        "nav_visual": "可視化",
        "nav_corr": "相関・検定",
        "nav_text": "テキスト処理",
        "export_title": "レポートのエクスポート",
        "export_desc": "記述統計・正規性検定・ヒストグラム・箱ひげ図・相関・テキスト分析サマリーを含むPDFレポートを生成します。",
        "export_button": "PDFレポートを生成",
        "export_filename": "survey_full_report_jp.pdf",
        "pdf_title": "アンケート完全レポート",
        "pdf_section_numdist": "1. 数値変数 - 分布",
        "pdf_section_scatter": "2. 散布図 - 関係",
        "pdf_section_catbar": "3. カテゴリ変数 - 棒グラフ",
        "pdf_section_numfull": "4. 数値変数 - 詳細統計",
        "pdf_section_catfreq": "5. カテゴリ変数 - 度数表",
        "pdf_section_corr": "6. 相関分析",
        "pdf_section_text": "7. テキスト分析 - 上位語",
        "pdf_notext": "分析できるテキストデータがありません。",
    },
    "KR": {  # Korean
        "title": "📊 Digital Payment Usage & Financial Discipline Survey",
        "subtitle": "survey data analysis",
        "upload_subheader": "📁 설문 데이터 업로드",
        "upload_label": "여기에 파일을 드래그 앤 드롭하거나 클릭하여 선택하세요 (CSV, XLS, XLSX)",
        "data_preview": "데이터 미리보기 (최대 첫 1000행)",
        "text_processing_subheader": "📝 텍스트 전처리",
        "text_columns_detected": "감지된 텍스트 열:",
        "select_text_col": "전처리할 텍스트 열 선택",
        "no_text_columns": "텍스트 형식의 열이 없습니다.",
        "text_processing_note": "텍스트는 소문자로 변환되고, 구두점이 제거되며, 공백 기준으로 분할되고, 영어 불용어가 제거됩니다.",
        "sample_tokens": "전처리된 토큰 샘플",
        "top_words": "출현 빈도 상위 10개 단어",
        "stats_subheader": "📈 기술통계 및 분포",
        "select_numeric_col": "통계/그래프용 숫자 열 선택",
        "no_numeric_cols": "사용 가능한 숫자 열이 없습니다.",
        "desc_stats": "선택한 열의 기술통계",
        "freq_table_subheader": "📊 범주형 빈도표",
        "select_categorical_col": "빈도표를 만들 범주형 열 선택",
        "no_categorical_cols": "범주형 열이 없습니다.",
        "freq_count": "빈도",
        "freq_percent": "비율(%)",
        "visual_subheader": "📉 데이터 시각화",
        "histogram": "히스토그램",
        "boxplot": "박스플롯",
        "correlation_subheader": "🔗 상관관계 및 통계 검정",
        "pearson_header": "피어슨 상관",
        "spearman_header": "스피어만 순위 상관",
        "chi_header": "카이제곱 검정",
        "select_x_numeric": "X 변수(숫자)를 선택",
        "select_y_numeric": "Y 변수(숫자)를 선택",
        "not_enough_numeric": "이 분석에 필요한 숫자 열이 부족합니다.",
        "pearson_result": "피어슨 상관 결과",
        "spearman_result": "스피어만 상관 결과",
        "corr_coef": "상관계수 (r)",
        "p_value": "p-값",
        "interpretation": "해석",
        "select_x_cat": "X 변수(범주형)를 선택",
        "select_y_cat": "Y 변수(범주형)를 선택",
        "not_enough_categorical": "카이제곱 검정에 필요한 범주형 열이 부족합니다.",
        "chi_square_result": "카이제곱 검정 결과",
        "chi_square_stat": "카이제곱 통계량",
        "chi_square_df": "자유도 (df)",
        "chi_square_p": "p-값",
        "alpha_note": "유의수준 α = 0.05에서 검정합니다.",
        "significant_assoc": "두 변수 사이에 통계적으로 유의한 관계가 있습니다.",
        "no_significant_assoc": "두 변수 사이에 통계적으로 유의한 관계가 없습니다.",
        "corr_direction_positive": "양의 관계: X가 증가하면 Y도 증가하는 경향이 있습니다.",
        "corr_direction_negative": "음의 관계: X가 증가하면 Y는 감소하는 경향이 있습니다.",
        "corr_direction_zero": "명확한 관계 방향이 없습니다(거의 0).",
        "corr_strength_none": "거의 관계가 없습니다.",
        "corr_strength_weak": "약한 관계입니다.",
        "corr_strength_moderate": "보통 정도의 관계입니다.",
        "corr_strength_strong": "강한 관계입니다.",
        "warning_select_valid": "올바른 열 조합을 선택하세요.",
        "header_github": "GitHub에서 포크",
        "nav_desc": "기술통계",
        "nav_visual": "시각화",
        "nav_corr": "상관 및 검정",
        "nav_text": "텍스트 처리",
        "export_title": "보고서 내보내기",
        "export_desc": "기술통계, 정규성 검정, 히스토그램, 박스플롯, 상관분석, 텍스트 분석 요약을 포함한 전체 PDF 보고서를 생성합니다.",
        "export_button": "PDF 보고서 생성",
        "export_filename": "survey_full_report_kr.pdf",
        "pdf_title": "설문 데이터 전체 보고서",
        "pdf_section_numdist": "1. 수치 변수 - 분포",
        "pdf_section_scatter": "2. 산점도 - 관계",
        "pdf_section_catbar": "3. 범주형 변수 - 막대 그래프",
        "pdf_section_numfull": "4. 수치 변수 - 상세 통계",
        "pdf_section_catfreq": "5. 범주형 변수 - 도수표",
        "pdf_section_corr": "6. 상관 분석",
        "pdf_section_text": "7. 텍스트 분석 - 상위 단어",
        "pdf_notext": "분석할 텍스트 데이터가 없습니다.",
    },
    "CN": {  # Chinese (Simplified)
        "title": "📊 Digital Payment Usage & Financial Discipline Survey",
        "subtitle": "survey data analysis",
        "upload_subheader": "📁 上传问卷数据",
        "upload_label": "将文件拖放到此处或点击选择（CSV, XLS, XLSX）",
        "data_preview": "数据预览（前 1000 行）",
        "text_processing_subheader": "📝 文本预处理",
        "text_columns_detected": "检测到的文本列：",
        "select_text_col": "选择要处理的文本列",
        "no_text_columns": "未找到文本类型的列。",
        "text_processing_note": "文本将被转为小写，去除标点符号，以空格分词，并移除英文停用词。",
        "sample_tokens": "预处理后的词元示例",
        "top_words": "词频最高的 10 个词",
        "stats_subheader": "📈 描述性统计与分布",
        "select_numeric_col": "选择用于统计/绘图的数值列",
        "no_numeric_cols": "没有可用的数值列。",
        "desc_stats": "所选列的描述性统计",
        "freq_table_subheader": "📊 分类频数表",
        "select_categorical_col": "选择用于频数表的分类列",
        "no_categorical_cols": "没有分类列。",
        "freq_count": "频数",
        "freq_percent": "百分比（%）",
        "visual_subheader": "📉 数据可视化",
        "histogram": "直方图",
        "boxplot": "箱线图",
        "correlation_subheader": "🔗 相关性与统计检验",
        "pearson_header": "皮尔逊相关",
        "spearman_header": "斯皮尔曼等级相关",
        "chi_header": "卡方检验",
        "select_x_numeric": "选择 X 变量（数值）",
        "select_y_numeric": "选择 Y 变量（数值）",
        "not_enough_numeric": "可用于该分析的数值列不足。",
        "pearson_result": "皮尔逊相关结果",
        "spearman_result": "斯皮尔曼相关结果",
        "corr_coef": "相关系数 (r)",
        "p_value": "p 值",
        "interpretation": "解释",
        "select_x_cat": "选择 X 变量（分类）",
        "select_y_cat": "选择 Y 变量（分类）",
        "not_enough_categorical": "用于卡方检验的分类列不足。",
        "chi_square_result": "卡方检验结果",
        "chi_square_stat": "卡方统计量",
        "chi_square_df": "自由度 (df)",
        "chi_square_p": "p 值",
        "alpha_note": "在显著性水平 α = 0.05 下进行检验。",
        "significant_assoc": "两个变量之间存在统计上显著的关联。",
        "no_significant_assoc": "两个变量之间不存在统计上显著的关联。",
        "corr_direction_positive": "正相关：X 增加时，Y 通常也增加。",
        "corr_direction_negative": "负相关：X 增加时，Y 通常减少。",
        "corr_direction_zero": "没有明显的相关方向（接近 0）。",
        "corr_strength_none": "几乎没有相关关系。",
        "corr_strength_weak": "相关关系较弱。",
        "corr_strength_moderate": "相关关系中等。",
        "corr_strength_strong": "相关关系较强。",
        "warning_select_valid": "请选择有效的列组合。",
        "header_github": "在 GitHub 上 Fork",
        "nav_desc": "描述性统计",
        "nav_visual": "可视化",
        "nav_corr": "相关与检验",
        "nav_text": "文本处理",
        "export_title": "导出报告",
        "export_desc": "生成包含描述性统计、正态性检验、直方图、箱线图、相关分析和文本分析摘要的完整 PDF 报告。",
        "export_button": "生成 PDF 报告",
        "export_filename": "survey_full_report_cn.pdf",
        "pdf_title": "问卷数据完整报告",
        "pdf_section_numdist": "1. 数值变量 - 分布",
        "pdf_section_scatter": "2. 散点图 - 关系",
        "pdf_section_catbar": "3. 类别变量 - 条形图",
        "pdf_section_numfull": "4. 数值变量 - 详细统计",
        "pdf_section_catfreq": "5. 类别变量 - 频数表",
        "pdf_section_corr": "6. 相关分析",
        "pdf_section_text": "7. 文本分析 - 高频词",
        "pdf_notext": "没有可供分析的文本数据。",
    },
    "AR": {  # Arabic
        "title": "📊 Digital Payment Usage & Financial Discipline Survey",
        "subtitle": "survey data analysis",
        "upload_subheader": "📁 رفع بيانات الاستبيان",
        "upload_label": "اسحب وأفلت الملف هنا أو اضغط للاختيار (CSV, XLS, XLSX)",
        "data_preview": "معاينة البيانات (حتى أول 1000 صف)",
        "text_processing_subheader": "📝 معالجة النصوص",
        "text_columns_detected": "الأعمدة النصية المكتشفة:",
        "select_text_col": "اختر عمود النص للمعالجة",
        "no_text_columns": "لا توجد أعمدة من نوع نصي.",
        "text_processing_note": "سيتم تحويل النص إلى حروف صغيرة، وإزالة علامات الترقيم، وتقسيمه إلى كلمات، وحذف كلمات الوقف الإنجليزية.",
        "sample_tokens": "عينة من الرموز المعالجة",
        "top_words": "أكثر 10 كلمات تكراراً",
        "stats_subheader": "📈 الإحصاءات الوصفية والتوزيع",
        "select_numeric_col": "اختر عموداً رقمياً للإحصاءات والرسوم",
        "no_numeric_cols": "لا توجد أعمدة رقمية متاحة.",
        "desc_stats": "الإحصاءات الوصفية للعمود المحدد",
        "freq_table_subheader": "📊 جدول التكرار للفئات",
        "select_categorical_col": "اختر عموداً فئوياً لجدول التكرار",
        "no_categorical_cols": "لا توجد أعمدة فئوية.",
        "freq_count": "العدد",
        "freq_percent": "النسبة المئوية (%)",
        "visual_subheader": "📉 عرض البيانات بيانياً",
        "histogram": "مخطط التوزيع (Histogram)",
        "boxplot": "مخطط الصندوق (Boxplot)",
        "correlation_subheader": "🔗 الارتباط والاختبارات الإحصائية",
        "pearson_header": "معامل ارتباط بيرسون",
        "spearman_header": "معامل ارتباط سبيرمان",
        "chi_header": "اختبار كاي تربيع",
        "select_x_numeric": "اختر متغير X (رقمي)",
        "select_y_numeric": "اختر متغير Y (رقمي)",
        "not_enough_numeric": "لا يوجد عدد كافٍ من الأعمدة الرقمية لهذا التحليل.",
        "pearson_result": "نتيجة ارتباط بيرسون",
        "spearman_result": "نتيجة ارتباط سبيرمان",
        "corr_coef": "معامل الارتباط (r)",
        "p_value": "قيمة p",
        "interpretation": "التفسير",
        "select_x_cat": "اختر متغير X (فئوي)",
        "select_y_cat": "اختر متغير Y (فئوي)",
        "not_enough_categorical": "لا يوجد عدد كافٍ من الأعمدة الفئوية لاختبار كاي تربيع.",
        "chi_square_result": "نتيجة اختبار كاي تربيع",
        "chi_square_stat": "إحصائية كاي تربيع",
        "chi_square_df": "درجات الحرية (df)",
        "chi_square_p": "قيمة p",
        "alpha_note": "تم الاختبار عند مستوى دلالة α = 0.05.",
        "significant_assoc": "هناك علاقة ذات دلالة إحصائية بين المتغيرين.",
        "no_significant_assoc": "لا توجد علاقة ذات دلالة إحصائية بين المتغيرين.",
        "corr_direction_positive": "علاقة طردية: عند زيادة X يميل Y إلى الزيادة.",
        "corr_direction_negative": "علاقة عكسية: عند زيادة X يميل Y إلى النقصان.",
        "corr_direction_zero": "لا يوجد اتجاه واضح للعلاقة (قيمة الارتباط قريبة من الصفر).",
        "corr_strength_none": "لا توجد علاقة تقريباً.",
        "corr_strength_weak": "علاقة ضعيفة.",
        "corr_strength_moderate": "علاقة متوسطة.",
        "corr_strength_strong": "علاقة قوية.",
        "warning_select_valid": "يرجى اختيار مجموعة أعمدة صحيحة.",
        "header_github": "Fork على GitHub",
        "nav_desc": "إحصاءات وصفية",
        "nav_visual": "الرسوم البيانية",
        "nav_corr": "الارتباط والاختبارات",
        "nav_text": "معالجة النصوص",
        "export_title": "تصدير التقرير",
        "export_desc": "إنشاء تقرير PDF كامل يحتوي على الإحصاءات الوصفية، واختبار التوزيع الطبيعي، والرسوم البيانية، والارتباطات، وملخص تحليل النصوص.",
        "export_button": "إنشاء تقرير PDF",
        "export_filename": "survey_full_report_ar.pdf",
        "pdf_title": "تقرير كامل لبيانات الاستبيان",
        "pdf_section_numdist": "١. المتغيرات العددية - التوزيع",
        "pdf_section_scatter": "٢. مخططات الانتشار - العلاقات",
        "pdf_section_catbar": "٣. المتغيرات الفئوية - المخططات الشريطية",
        "pdf_section_numfull": "٤. المتغيرات العددية - الإحصاءات الكاملة",
        "pdf_section_catfreq": "٥. المتغيرات الفئوية - جداول التكرار",
        "pdf_section_corr": "٦. تحليل الارتباط",
        "pdf_section_text": "٧. تحليل النص - أهم الكلمات",
        "pdf_notext": "لا توجد بيانات نصية للتحليل.",
    },
}

def get_text(key: str) -> str:
    lang = st.session_state.get("language", "EN")
    lang_dict = TEXTS.get(lang, TEXTS.get("EN", {}))
    return lang_dict.get(key, key)

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
        tokens = text.split()
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

def visualize_data(df: pd.DataFrame, col: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        st.warning(get_text("warning_select_valid"))
        return
    with st.spinner("Generating visualizations..."):
        time.sleep(0.5)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(s, kde=True, ax=ax, color="#16a34a")
            ax.set_title(get_text("histogram"))
            st.pyplot(fig)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            sns.boxplot(x=s, ax=ax2, color="#22c55e")
            ax2.set_title(get_text("boxplot"))
            st.pyplot(fig2)

def interpret_strength(r: float) -> str:
    if r is None or np.isnan(r):
        return get_text("corr_strength_none")
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = get_text("corr_strength_none")
    elif abs_r < 0.3:
        strength = get_text("corr_strength_weak")
    elif abs_r < 0.5:
        strength = get_text("corr_strength_moderate")
    else:
        strength = get_text("corr_strength_strong")
    if r > 0:
        direction = get_text("corr_direction_positive")
    elif r < 0:
        direction = get_text("corr_direction_negative")
    else:
        direction = get_text("corr_direction_zero")
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

# --------------------------- PDF REPORT FULL ---------------------------
def build_survey_report_pdf(df, numeric_cols, cat_cols, text_cols):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    story = []

    styles = getSampleStyleSheet()
    GREEN = colors.HexColor("#10B981")

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=GREEN,
        alignment=1,
        spaceAfter=12,
        spaceBefore=6,
    )
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=GREEN,
        spaceBefore=10,
        spaceAfter=6,
    )
    h3_style = ParagraphStyle(
        "Heading3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=4,
    )
    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        spaceAfter=4,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=9.5,
        spaceAfter=2,
    )

    def make_table(data, col_widths=None, font_size=8, header_bg=GREEN):
        if not data:
            return None
        tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
        n_rows = len(data)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), font_size),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), font_size),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
        if n_rows > 2:
            for r in range(1, n_rows):
                if r % 2 == 1:
                    style_cmds.append(
                        ("BACKGROUND", (0, r), (-1, r), colors.Color(0.96, 0.98, 0.97))
                    )
        tbl.setStyle(TableStyle(style_cmds))
        return tbl

    def fig_to_image(fig, width=6.5, height=2.5):
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format="png", dpi=100, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close(fig)
        return RLImage(img_buffer, width=width * inch, height=height * inch)

    story.append(Paragraph(get_text("pdf_title"), title_style))
    meta_lines = [
        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}",
        f"Numeric columns: {len(numeric_cols)}, Categorical columns: {len(cat_cols)}, Text columns: {len(text_cols)}",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, normal_style))
    story.append(Spacer(1, 0.2 * inch))

    # 1. Numeric distributions
    if numeric_cols:
        story.append(Paragraph(get_text("pdf_section_numdist"), h2_style))
        story.append(Spacer(1, 0.1 * inch))
        for col in numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            stats_dict = {
                "Mean": f"{s.mean():.4f}",
                "Median": f"{s.median():.4f}",
                "Std": f"{s.std():.4f}",
                "Min": f"{s.min():.4f}",
                "Max": f"{s.max():.4f}",
            }
            story.append(Paragraph(f"<b>{col}</b>", h3_style))
            stats_table_data = [["Statistic", "Value"]] + [[k, v] for k, v in stats_dict.items()]
            stats_tbl = make_table(stats_table_data, col_widths=[2.2 * inch, 2.2 * inch], font_size=8)
            if stats_tbl:
                story.append(stats_tbl)
            story.append(Spacer(1, 0.15 * inch))

            fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))
            axes[0].hist(s, bins=20, color="#16a34a", edgecolor="black", alpha=0.7)
            axes[0].set_title(f"Histogram - {col}", fontsize=10, fontweight="bold")
            axes[0].set_xlabel("Value")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(alpha=0.3)

            axes[1].boxplot(s, vert=True)
            axes[1].set_title(f"Boxplot - {col}", fontsize=10, fontweight="bold")
            axes[1].set_ylabel("Value")
            axes[1].grid(alpha=0.3, axis="y")

            plt.tight_layout()
            img = fig_to_image(fig, width=6.5, height=2.2)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))

    # 2. Scatter plots
    if len(numeric_cols) > 1:
        story.append(PageBreak())
        story.append(Paragraph(get_text("pdf_section_scatter"), h2_style))
        story.append(Spacer(1, 0.1 * inch))
        pairs_to_plot = min(3, len(numeric_cols) - 1)
        for i in range(pairs_to_plot):
            x_col = numeric_cols[i]
            y_col = numeric_cols[i + 1]
            x = pd.to_numeric(df[x_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = x.notna() & y.notna()
            x_clean, y_clean = x[mask], y[mask]
            if len(x_clean) < 2:
                continue

            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.scatter(x_clean, y_clean, alpha=0.6, color="#10b981", s=40, edgecolors="black", linewidth=0.5)
            z = np.polyfit(x_clean, y_clean, 1)
            p_line = np.poly1d(z)
            ax.plot(x_clean, p_line(x_clean), "r--", alpha=0.8, linewidth=2, label="Trend")
            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            ax.set_title(f"Scatter {x_col} vs {y_col}", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend()
            plt.tight_layout()

            img = fig_to_image(fig, width=4.5, height=3)
            story.append(img)
            story.append(Spacer(1, 0.15 * inch))

    # 3. Categorical bar charts
    if cat_cols:
        story.append(PageBreak())
        story.append(Paragraph(get_text("pdf_section_catbar"), h2_style))
        story.append(Spacer(1, 0.1 * inch))
        for cat_col in cat_cols[:3]:
            freq = df[cat_col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(5, 2.5))
            freq.plot(kind="bar", ax=ax, color="#22c55e", edgecolor="black")
            ax.set_title(f"Bar Chart - {cat_col}", fontsize=10, fontweight="bold")
            ax.set_xlabel(cat_col)
            ax.set_ylabel("Frequency")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()

            img = fig_to_image(fig, width=5, height=2.5)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))

    # 4. Numeric full stats
    if numeric_cols:
        story.append(PageBreak())
        story.append(Paragraph(get_text("pdf_section_numfull"), h2_style))
        story.append(Spacer(1, 0.1 * inch))
        for col in numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            if not s.mode().empty:
                mode_val = f"{s.mode().iloc[0]:.6f}"
            else:
                mode_val = "N/A"
            stats_dict = {
                "Mean": f"{s.mean():.6f}",
                "Median": f"{s.median():.6f}",
                "Mode": mode_val,
                "Std Dev": f"{s.std():.6f}",
                "Variance": f"{s.var():.6f}",
                "Min": f"{s.min():.6f}",
                "Max": f"{s.max():.6f}",
                "Range": f"{(s.max() - s.min()):.6f}",
                "Q1 (25%)": f"{s.quantile(0.25):.6f}",
                "Q3 (75%)": f"{s.quantile(0.75):.6f}",
                "IQR": f"{(s.quantile(0.75) - s.quantile(0.25)):.6f}",
                "Skewness": f"{s.skew():.6f}",
                "Kurtosis": f"{s.kurtosis():.6f}",
            }
            story.append(Paragraph(f"<b>{col}</b>", h3_style))
            table_data = [["Statistic", "Value"]] + [[k, v] for k, v in stats_dict.items()]
            tbl = make_table(table_data, col_widths=[2.5 * inch, 2.5 * inch], font_size=7)
            if tbl:
                story.append(tbl)
            story.append(Spacer(1, 0.15 * inch))

    # 5. Categorical frequency tables
    if cat_cols:
        story.append(PageBreak())
        story.append(Paragraph(get_text("pdf_section_catfreq"), h2_style))
        story.append(Spacer(1, 0.1 * inch))
        for col in cat_cols:
            freq = df[col].value_counts(dropna=False).head(15)
            pct = (freq / len(df) * 100).round(2)
            story.append(Paragraph(f"<b>{col}</b> Top 15", h3_style))
            table_data = [["Category", "Count", "Percent"]] + [
                [str(idx), str(int(freq[idx])), f"{pct[idx]:.2f}"] for idx in freq.index
            ]
            tbl = make_table(table_data, col_widths=[2 * inch, 1.5 * inch, 1.5 * inch], font_size=7)
            if tbl:
                story.append(tbl)
            story.append(Spacer(1, 0.15 * inch))

    # 6. Correlation matrix
    if len(numeric_cols) > 1:
        story.append(PageBreak())
        story.append(Paragraph(get_text("pdf_section_corr"), h2_style))
        story.append(Spacer(1, 0.1 * inch))
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = numeric_df.corr()
        table_data = [["Variable"] + list(numeric_cols)]
        for var in numeric_cols:
            row = [var]
            for col in numeric_cols:
                r = corr_matrix.loc[var, col]
                row.append(f"{r:.3f}")
            table_data.append(row)
        col_width = 6.5 / (len(numeric_cols) + 1)
        tbl = make_table(
            table_data,
            col_widths=[col_width * inch for _ in range(len(numeric_cols) + 1)],
            font_size=7,
        )
        if tbl:
            story.append(tbl)
        story.append(Spacer(1, 0.2 * inch))

    # 7. Text analysis
    if text_cols:
        story.append(PageBreak())
        story.append(Paragraph(get_text("pdf_section_text"), h2_style))
        story.append(Spacer(1, 0.1 * inch))
        for col in text_cols[:2]:
            story.append(Paragraph(f"<b>{col}</b>", h3_style))
            tokens_series = preprocess_text_series(df[col])
            all_tokens = []
            for token_list in tokens_series:
                all_tokens.extend(token_list)
            if not all_tokens:
                story.append(Paragraph(get_text("pdf_notext"), small_style))
                story.append(Spacer(1, 0.1 * inch))
                continue
            word_freq = Counter(all_tokens).most_common(15)
            table_data = [["Word", "Frequency"]] + [[word, str(count)] for word, count in word_freq]
            tbl = make_table(table_data, col_widths=[3.5 * inch, 2 * inch], font_size=8)
            if tbl:
                story.append(tbl)
            story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_pdf_button(df, numeric_cols, cat_cols, text_cols):
    if st.button(get_text("export_button"), key="btn_export_pdf", type="primary"):
        with st.spinner(get_text("export_desc")):
            time.sleep(0.5)
            pdf_buffer = build_survey_report_pdf(df, numeric_cols, cat_cols, text_cols)
        st.download_button(
            label=get_text("export_button"),
            data=pdf_buffer.getvalue(),
            file_name=get_text("export_filename"),
            mime="application/pdf",
            key="dl_export_pdf",
        )
        st.success("PDF generated successfully!")

# --------------------------- HEADER + HERO ---------------------------
st.markdown(
    f"""
    <div style="
        width:100%;
        padding:0.40rem 0.9rem;
        display:flex;
        justify-content:center;
        background:rgba(240, 253, 250, 0.96);
        box-shadow:0 10px 25px rgba(15, 118, 110, 0.15);
        border:1px solid rgba(45, 212, 191, 0.55);
        margin-bottom:0.9rem;
    ">
      <div style="font-weight:650; color:#047857; font-size:1.1rem;">
        {get_text('title')}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

content_font_size = "0.95rem"
st.markdown(
    f"<p style='text-align:center; color:#065f46; font-size:{content_font_size};'>"
    f"{get_text('subtitle')}</p>",
    unsafe_allow_html=True,
)

group_members = [
    {"name": "ADITYA ANGGARA PAMUNGKAS", "sid": "04202400051", "role": "Leader"},
    {"name": "MAULA AQIEL NURI", "sid": "04202400023", "role": "Member"},
    {"name": "SYAFIQ NUR RAMADHAN", "sid": "04202400073", "role": "Member"},
    {"name": "RIFAT FITROTU SALMAN", "sid": "04202400106", "role": "Member"},
]

st.markdown(
    """
    <div class='hero-card' style="margin-top:0.6rem; margin-bottom:0.4rem;">
      <h4 style="margin-top:0; margin-bottom:0.4rem; color:#047857;">
        👥 Group 5 Class 2
      </h4>
      <ul style="margin:0; padding-left:1.1rem; font-size:0.9rem; color:#065f46;">
    """
    + "\n".join(
        [
            f"<li><b>{m['name']}</b> ({m['sid']}) – {m['role']}</li>"
            for m in group_members
        ]
    )
    + """
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='decorative-divider'></div>", unsafe_allow_html=True)

# --------------------------- UPLOAD & PREVIEW + FILTER ---------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class='section-card'>
      <p class='section-title'>{get_text("upload_subheader")}</p>
      <p class='section-subtitle'>{get_text("subtitle")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

u1, u2, u3 = st.columns([1, 2, 1])
with u2:
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-weight:600; margin-bottom:0.2rem;'>📤</p>"
        f"<p style='margin-bottom:0.1rem; font-size:{content_font_size};'>"
        f"{get_text('upload_label')}</p>"
        f"<p class='helper-text'>Limit 200MB • CSV, XLS, XLSX</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload survey file",
        type=["csv", "xls", "xlsx"],
        label_visibility="collapsed",
        accept_multiple_files=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)

df = load_data(uploaded)
if df is None:
    st.info(get_text("no_file"))
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

filter_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
filtered_df = df
if filter_cols:
    st.markdown("##### Filter data (optional)")
    fcol = st.selectbox("Filter column", ["(No filter)"] + filter_cols, index=0)
    if fcol != "(No filter)":
        unique_vals = df[fcol].dropna().unique().tolist()
        selected_vals = st.multiselect("Select values", options=unique_vals, default=unique_vals)
        if selected_vals:
            filtered_df = df[df[fcol].isin(selected_vals)]

st.markdown(f"#### {get_text('data_preview')}")
df_preview = filtered_df.head(1000)
st.dataframe(df_preview, height=400)

n_rows, n_cols = filtered_df.shape
n_numeric = filtered_df.select_dtypes(include=[np.number]).shape[1]
n_cat = filtered_df.select_dtypes(exclude=[np.number]).shape[1]
st.markdown(
    f"""
    <div class='section-card'>
      <p class='section-title'>{get_text("data_preview")}</p>
      <p class='section-subtitle'>{get_text("data_preview_subtitle")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = filtered_df.select_dtypes(exclude=[np.number]).columns.tolist()
text_cols = filtered_df.select_dtypes(include=["object", "string"]).columns.tolist()

# --------------------------- TABS ---------------------------
tab_desc, tab_vis, tab_corr, tab_text = st.tabs(
    [
        get_text("nav_desc"),
        get_text("nav_visual"),
        get_text("nav_corr"),
        get_text("nav_text"),
    ]
)

# Text processing
with tab_text:
    with st.expander(get_text("text_processing_subheader"), expanded=True):
        if not text_cols:
            st.warning(get_text("no_text_columns"))
        else:
            st.markdown(
                get_text("text_columns_detected")
                + f" `{', '.join(text_cols)}`"
            )
            text_col = st.selectbox(
                get_text("select_text_col"),
                options=text_cols,
                help="Select a column for text analysis",
            )
            st.markdown(
                f"<p class='helper-text'>{get_text('text_processing_note')}</p>",
                unsafe_allow_html=True,
            )
            processed = preprocess_text_series(filtered_df[text_col])
            st.markdown(f"**{get_text('sample_tokens')}**")
            st.write(processed.head(5).tolist())
            all_tokens = [t for row in processed for t in row]
            counter = Counter(all_tokens)
            top10 = counter.most_common(10)
            if top10:
                top_df = pd.DataFrame(top10, columns=["word", "count"])
                st.markdown(f"**{get_text('top_words')}**")
                st.table(top_df)

# Descriptive stats
with tab_desc:
    st.markdown(f"### {get_text('stats_subheader')}")
    if not numeric_cols:
        st.warning(get_text("no_numeric_cols"))
    else:
        tab_summ, tab_dist = st.tabs(["Summary & Normality", "Distribution"])
        with tab_summ:
            num_col = st.selectbox(
                get_text("select_numeric_col"),
                options=numeric_cols,
                help="Column for descriptive statistics",
                key="desc_num_col",
            )
            stats_df = descriptive_stats(filtered_df[num_col])
            st.markdown(f"**{get_text('desc_stats')}**")
            st.table(stats_df)
            s_norm = pd.to_numeric(filtered_df[num_col], errors="coerce").dropna()
            if len(s_norm) >= 8:
                stat, p_norm = normaltest(s_norm)
                st.markdown("**Normality test (D’Agostino-Pearson)**")
                st.write(f"Statistic: {stat:.4f}")
                st.write(f"p-value: {p_norm:.4f}")
                if p_norm < 0.05:
                    st.info("Data deviate significantly from normal distribution (reject H0 at α = 0.05).")
                else:
                    st.success("No significant deviation from normal distribution (fail to reject H0 at α = 0.05).")
            else:
                st.info("Not enough data points for normality test (need at least 8 non-missing values).")
        with tab_dist:
            num_col2 = st.selectbox(
                "Select column for distribution",
                options=numeric_cols,
                index=0,
                key="desc_num_dist",
            )
            visualize_data(filtered_df, num_col2)
    if not cat_cols:
        st.info(get_text("no_categorical_cols"))
    else:
        cat_col = st.selectbox(
            get_text("select_categorical_col"),
            options=cat_cols,
            help="Column for frequency table",
        )
        freq_df = frequency_tables(filtered_df[cat_col])
        freq_df.columns = [
            get_text("freq_count"),
            get_text("freq_percent"),
        ]
        st.markdown(f"### {get_text('freq_table_subheader')}")
        st.table(freq_df)

# Visualizations
with tab_vis:
    if not numeric_cols:
        st.warning(get_text("no_numeric_cols"))
    else:
        vis_tab1, vis_tab2 = st.tabs(["Histogram / Boxplot", "Scatter & Bar"])
        with vis_tab1:
            num_col = st.selectbox(
                get_text("select_numeric_col"),
                options=numeric_cols,
                help="Column for visualization",
                key="visual_num",
            )
            st.markdown(f"### {get_text('visual_subheader')}")
            visualize_data(filtered_df, num_col)
        with vis_tab2:
            if len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    x_sc = st.selectbox("X variable (numeric)", options=numeric_cols, key="scatter_x")
                with c2:
                    y_sc = st.selectbox("Y variable (numeric)", options=[c for c in numeric_cols if c != x_sc], key="scatter_y")
                s_x = pd.to_numeric(filtered_df[x_sc], errors="coerce")
                s_y = pd.to_numeric(filtered_df[y_sc], errors="coerce")
                mask = s_x.notna() & s_y.notna()
                if mask.sum() > 1:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.scatter(s_x[mask], s_y[mask], alpha=0.6, color="#0f766e")
                    ax.set_xlabel(x_sc)
                    ax.set_ylabel(y_sc)
                    ax.set_title("Scatter plot")
                    st.pyplot(fig)
                else:
                    st.info("Not enough valid data for scatter plot.")
            else:
                st.info("Need at least 2 numeric columns for scatter plot.")

            if cat_cols:
                cat_for_bar = st.selectbox(
                    "Categorical column for bar chart",
                    options=cat_cols,
                    key="bar_cat",
                )
                freq = filtered_df[cat_for_bar].value_counts().head(20)
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                sns.barplot(x=freq.values, y=freq.index, ax=ax2, color="#22c55e")
                ax2.set_xlabel("Count")
                ax2.set_ylabel(cat_for_bar)
                ax2.set_title("Bar chart (top 20)")
                st.pyplot(fig2)
            else:
                st.info("No categorical columns for bar chart.")

# Correlations & tests
with tab_corr:
    st.markdown(f"### {get_text('correlation_subheader')}")
    tab1, tab2, tab3 = st.tabs(
        [
            get_text("pearson_header"),
            get_text("spearman_header"),
            get_text("chi_header"),
        ]
    )
    with tab1:
        if len(numeric_cols) < 2:
            st.info(get_text("not_enough_numeric"))
        else:
            c1p, c2p = st.columns(2)
            with c1p:
                x_num = st.selectbox(
                    get_text("select_x_numeric"),
                    options=numeric_cols,
                    key="pearson_x",
                    help="Independent variable",
                )
            with c2p:
                y_num = st.selectbox(
                    get_text("select_y_numeric"),
                    options=[c for c in numeric_cols if c != x_num],
                    key="pearson_y",
                    help="Dependent variable",
                )
            if x_num and y_num:
                r, p = correlation_analysis(filtered_df, x_num, y_num, method="pearson")
                if np.isnan(r):
                    st.warning(get_text("warning_select_valid"))
                else:
                    st.markdown(f"**{get_text('pearson_result')}**")
                    out = pd.DataFrame(
                        {
                            get_text("corr_coef"): [r],
                            get_text("p_value"): [p],
                        }
                    )
                    st.table(out)
                    st.markdown(
                        f"**{get_text('interpretation')}:** "
                        f"{interpret_strength(r)}"
                    )

    with tab2:
        if len(numeric_cols) < 2:
            st.info(get_text("not_enough_numeric"))
        else:
            c1s, c2s = st.columns(2)
            with c1s:
                x_s = st.selectbox(
                    get_text("select_x_numeric"),
                    options=numeric_cols,
                    key="spearman_x",
                )
            with c2s:
                y_s = st.selectbox(
                    get_text("select_y_numeric"),
                    options=[c for c in numeric_cols if c != x_s],
                    key="spearman_y",
                )
            if x_s and y_s:
                r_s, p_s = correlation_analysis(filtered_df, x_s, y_s, method="spearman")
                if np.isnan(r_s):
                    st.warning(get_text("warning_select_valid"))
                else:
                    st.markdown(f"**{get_text('spearman_result')}**")
                    out_s = pd.DataFrame(
                        {
                            get_text("corr_coef"): [r_s],
                            get_text("p_value"): [p_s],
                        }
                    )
                    st.table(out_s)
                    st.markdown(
                        f"**{get_text('interpretation')}:** "
                        f"{interpret_strength(r_s)}"
                    )

    with tab3:
        chi_df = filtered_df.copy()
        chi_cat_candidates = [
            c for c in chi_df.columns
            if c.startswith("X") or c.startswith("Y") or c == "Responden"
        ]
        for c in chi_cat_candidates:
            chi_df[c] = chi_df[c].astype(str)
        cat_cols_chi = chi_cat_candidates
        if len(cat_cols_chi) < 2:
            st.info(get_text("not_enough_categorical"))
        else:
            c1c, c2c = st.columns(2)
            with c1c:
                x_cat = st.selectbox(
                    get_text("select_x_cat"),
                    options=cat_cols_chi,
                    key="chi_x",
                )
            with c2c:
                y_cat = st.selectbox(
                    get_text("select_y_cat"),
                    options=[c for c in cat_cols_chi if c != x_cat],
                    key="chi_y",
                )
            if x_cat and y_cat:
                table = pd.crosstab(chi_df[x_cat], chi_df[y_cat])
                if table.size == 0:
                    st.warning(get_text("warning_select_valid"))
                else:
                    chi2, p_val, dof_val, expected = chi2_contingency(table)
                    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
                    st.markdown(f"**{get_text('chi_square_result')}**")
                    out_c = pd.DataFrame(
                        {
                            get_text("chi_square_stat"): [chi2],
                            get_text("chi_square_df"): [dof_val],
                            get_text("chi_square_p"): [p_val],
                        }
                    )
                    st.table(out_c)
                    st.markdown("**Observed**")
                    st.dataframe(table, height=200)
                    st.markdown("**Expected**")
                    st.dataframe(expected_df, height=200)
                    st.markdown(f"_{get_text('alpha_note')}_")
                    if p_val < 0.05:
                        st.success(get_text("significant_assoc"))
                    else:
                        st.info(get_text("no_significant_assoc"))

# --------------------------- EXPORT PDF SECTION ---------------------------
st.markdown(f"### {get_text('export_title')}")
st.markdown(get_text("export_desc"))
generate_pdf_button(filtered_df, numeric_cols, cat_cols, text_cols)
