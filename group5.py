import streamlit as st
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, chi2_contingency, normaltest
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
from scipy import stats

# --------------------------- NLTK INIT ---------------------------
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
EN_STOPWORDS = set(stopwords.words("english"))
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

# --------------------------- MULTI-LANGUAGE TEXTS ---------------------------
TEXTS = {
    "EN": {
        "title": "📊 Digital Payment Usage & Financial Discipline Survey",
        "subtitle": "📈 Survey data analysis",
        "upload_subheader": "📁 Upload Survey Data",
        "upload_label": "📤 Drag & drop file here or click to browse (CSV, XLS, XLSX)",
        "data_preview": "👀 Data Preview (up to first 1000 rows)",
        "text_processing_subheader": "📝 Text Preprocessing",
        "text_columns_detected": "🔎 Detected text columns:",
        "select_text_col": "🧩 Select a text column to process",
        "no_text_columns": "⚠️ No text-type columns detected.",
        "text_processing_note": "ℹ️ Text will be lowercased, punctuation removed, tokenized (split by spaces), and English stopwords removed.",
        "sample_tokens": "🔤 Sample of processed tokens",
        "top_words": "🏆 Top 10 Words by Frequency",
        "stats_subheader": "📈 Descriptive Statistics & Distribution",
        "select_numeric_col": "🔢 Select a numeric column for statistics & plots",
        "no_numeric_cols": "⚠️ No numeric columns available.",
        "desc_stats": "📊 Descriptive statistics for the selected column",
        "freq_table_subheader": "📊 Categorical Frequency Table",
        "select_categorical_col": "🏷️ Select a categorical column for frequency table",
        "no_categorical_cols": "⚠️ No categorical columns available.",
        "freq_count": "🔢 Count",
        "freq_percent": "📏 Percent (%)",
        "visual_subheader": "📉 Data Visualizations",
        "histogram": "📊 Histogram",
        "boxplot": "📦 Boxplot",
        "correlation_subheader": "🔗 Correlation & Statistical Tests",
        "pearson_header": "📐 Pearson Correlation",
        "spearman_header": "📐 Spearman Rank Correlation",
        "chi_header": "🎲 Chi-square Test",
        "select_x_numeric": "📌 Select X variable (numeric)",
        "select_y_numeric": "🎯 Select Y variable (numeric)",
        "not_enough_numeric": "⚠️ Not enough numeric columns for this analysis.",
        "pearson_result": "📐 Pearson Correlation Result",
        "spearman_result": "📐 Spearman Rank Correlation Result",
        "corr_coef": "📊 Correlation coefficient (r)",
        "p_value": "📎 p-value",
        "interpretation": "🧠 Interpretation",
        "select_x_cat": "📌 Select X variable (categorical)",
        "select_y_cat": "🎯 Select Y variable (categorical)",
        "not_enough_categorical": "⚠️ Not enough categorical columns for Chi-square test.",
        "chi_square_result": "🎲 Chi-square Test Result",
        "chi_square_stat": "📊 Chi-square statistic",
        "chi_square_df": "📏 Degrees of freedom (df)",
        "chi_square_p": "📎 p-value",
        "alpha_note": "ℹ️ Significance tested at α = 0.05.",
        "significant_assoc": "✅ There is a statistically significant association between the two variables.",
        "no_significant_assoc": "❌ There is no statistically significant association between the two variables.",
        "corr_direction_positive": "⬆️ Positive relationship: as X increases, Y tends to increase.",
        "corr_direction_negative": "⬇️ Negative relationship: as X increases, Y tends to decrease.",
        "corr_direction_zero": "➖ No clear direction of relationship (near zero).",
        "corr_strength_none": "⚪ Virtually no relationship.",
        "corr_strength_weak": "🟡 Weak relationship.",
        "corr_strength_moderate": "🟠 Moderate relationship.",
        "corr_strength_strong": "🔴 Strong relationship.",
        "warning_select_valid": "⚠️ Please select a valid combination of columns.",
        "header_github": "🐙 Fork on GitHub",
        "nav_desc": "📊 Descriptive Stats",
        "nav_visual": "📉 Visualizations",
        "nav_corr": "🔗 Correlations & Tests",
        "nav_text": "📝 Text Processing",
        "export_title": "📄 Export Report",
        "export_desc": "🖨️ Generate a complete PDF with all descriptive stats, normality test, histograms, boxplots, correlations, and text analysis summary.",
        "export_button": "📥 Generate PDF report",
        "export_filename": "survey_full_report.pdf",
        "pdf_title": "📊 Digital Payment Usage & Financial Discipline",
        "pdf_section_numdist": "1️⃣ Numeric Variables - Distributions",
        "pdf_section_scatter": "2️⃣ Scatter Plots - Relationships",
        "pdf_section_catbar": "3️⃣ Categorical Variables - Bar Charts",
        "pdf_section_numfull": "4️⃣ Numeric Variables - Full Statistics",
        "pdf_section_catfreq": "5️⃣ Categorical Variables - Frequency Tables",
        "pdf_section_corr": "6️⃣ Correlation Analysis",
        "pdf_section_text": "7️⃣ Text Analysis - Top Words",
        "pdf_notext": "⚠️ No text data to analyze.",
        "filter_data_optional": "🔍 Filter data (optional)",
        "filter_column": "📌 Filter column",
        "no_filter": "🚫 (No filter)",
        "select_values": "✅ Select values",
        "summary_normality": "📊 Summary & Normality",
        "distribution": "📈 Distribution",
        "select_column_distribution": "📌 Select column for distribution",
        "normality_test": "🧪 Normality test (D’Agostino-Pearson)",
        "statistic": "📊 Statistic",
        "deviate_normal": "⚠️ Data deviate significantly from normal distribution (reject H0 at α = 0.05).",
        "no_deviate_normal": "✅ No significant deviation from normal distribution (fail to reject H0 at α = 0.05).",
        "not_enough_normality": "⚠️ Not enough data points for normality test (need at least 8 non-missing values).",
        "histogram_boxplot": "📊 Histogram / 📦 Boxplot",
        "scatter_bar": "📈 Scatter & 📊 Bar",
        "x_variable_numeric": "📌 X variable (numeric)",
        "y_variable_numeric": "🎯 Y variable (numeric)",
        "scatter_plot": "📈 Scatter plot",
        "not_enough_scatter": "⚠️ Not enough valid data for scatter plot.",
        "need_2_numeric": "⚠️ Need at least 2 numeric columns for scatter plot.",
        "categorical_bar": "🏷️ Categorical column for bar chart",
        "bar_chart": "📊 Bar chart (top 20)",
        "no_categorical_bar": "⚠️ No categorical columns for bar chart.",
        "independent_variable": "🎛️ Independent variable",
        "dependent_variable": "🎯 Dependent variable",
        "observed": "👁️ Observed",
        "expected": "📐 Expected",
        "no_file": "📂 Please upload a file to get started.",
        "data_preview_subtitle": "📈 survey data analysis",
        "leader": "👑 Leader",
        "member": "👥 Member",
        "upload_limit": "📦 Limit 200MB • CSV, XLS, XLSX",
        "statistic_label": "📊 Statistic",
        "p_value_label": "📎 p-value",
        "bar_chart_top20": "📊 Bar chart (top 20)",
        "pdf_meta_rows": "📏 Rows: {0}, Columns: {1}",
        "pdf_meta_cols": "🔢 Numeric columns: {0}, 🏷️ Categorical columns: {1}, 🔤 Text columns: {2}",
        "group_info": (
            "👥 Group 5 Class 2\n"
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 Leader\n"
            "MAULA AQIEL NURI (04202400023) – 👥 Member\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 Member\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 Member"
        ),
    },
    "ID": {
        "title": "📊 Penggunaan Pembayaran Digital & Disiplin Keuangan Survei",
        "subtitle": "📈 analisis data survei",
        "upload_subheader": "📁 Unggah Data Survei",
        "upload_label": "📤 Tarik & letakkan file di sini atau klik untuk memilih (CSV, XLS, XLSX)",
        "data_preview": "👀 Pratinjau Data (maksimal 1000 baris pertama)",
        "text_processing_subheader": "📝 Pemrosesan Teks",
        "text_columns_detected": "🔎 Kolom teks terdeteksi:",
        "select_text_col": "🧩 Pilih kolom teks untuk diproses",
        "no_text_columns": "⚠️ Tidak ada kolom bertipe teks.",
        "text_processing_note": "ℹ️ Teks akan di-lowercase, tanda baca dihapus, dipisah per kata, dan stopwords bahasa Inggris dihapus.",
        "sample_tokens": "🔤 Contoh token yang telah diproses",
        "top_words": "🏆 10 Kata Teratas berdasarkan Frekuensi",
        "stats_subheader": "📈 Statistik Deskriptif & Distribusi",
        "select_numeric_col": "🔢 Pilih kolom numerik untuk statistik & grafik",
        "no_numeric_cols": "⚠️ Tidak ada kolom numerik.",
        "desc_stats": "📊 Statistik deskriptif untuk kolom yang dipilih",
        "freq_table_subheader": "📊 Tabel Frekuensi Kategorikal",
        "select_categorical_col": "🏷️ Pilih kolom kategorikal untuk tabel frekuensi",
        "no_categorical_cols": "⚠️ Tidak ada kolom kategorikal.",
        "freq_count": "🔢 Frekuensi",
        "freq_percent": "📏 Persentase (%)",
        "visual_subheader": "📉 Visualisasi Data",
        "histogram": "📊 Histogram",
        "boxplot": "📦 Boxplot",
        "correlation_subheader": "🔗 Korelasi & Uji Statistik",
        "pearson_header": "📐 Korelasi Pearson",
        "spearman_header": "📐 Korelasi Spearman",
        "chi_header": "🎲 Uji Chi-square",
        "select_x_numeric": "📌 Pilih variabel X (numerik)",
        "select_y_numeric": "🎯 Pilih variabel Y (numerik)",
        "not_enough_numeric": "⚠️ Kolom numerik tidak mencukupi untuk analisis ini.",
        "pearson_result": "📐 Hasil Korelasi Pearson",
        "spearman_result": "📐 Hasil Korelasi Spearman",
        "corr_coef": "📊 Koefisien korelasi (r)",
        "p_value": "📎 p-value",
        "interpretation": "🧠 Interpretasi",
        "select_x_cat": "📌 Pilih variabel X (kategorikal)",
        "select_y_cat": "🎯 Pilih variabel Y (kategorikal)",
        "not_enough_categorical": "⚠️ Kolom kategorikal tidak mencukupi untuk uji Chi-square.",
        "chi_square_result": "🎲 Hasil Uji Chi-square",
        "chi_square_stat": "📊 Statistik Chi-square",
        "chi_square_df": "📏 Derajat bebas (df)",
        "chi_square_p": "📎 p-value",
        "alpha_note": "ℹ️ Signifikansi diuji pada α = 0,05.",
        "significant_assoc": "✅ Terdapat hubungan yang signifikan secara statistik antara kedua variabel.",
        "no_significant_assoc": "❌ Tidak terdapat hubungan yang signifikan secara statistik antara kedua variabel.",
        "corr_direction_positive": "⬆️ Hubungan positif: ketika X naik, Y cenderung naik.",
        "corr_direction_negative": "⬇️ Hubungan negatif: ketika X naik, Y cenderung turun.",
        "corr_direction_zero": "➖ Tidak ada arah hubungan yang jelas (mendekati nol).",
        "corr_strength_none": "⚪ Hampir tidak ada hubungan.",
        "corr_strength_weak": "🟡 Hubungan lemah.",
        "corr_strength_moderate": "🟠 Hubungan sedang.",
        "corr_strength_strong": "🔴 Hubungan kuat.",
        "warning_select_valid": "⚠️ Silakan pilih kombinasi kolom yang valid.",
        "header_github": "🐙 Fork di GitHub",
        "nav_desc": "📊 Statistik Deskriptif",
        "nav_visual": "📉 Visualisasi",
        "nav_corr": "🔗 Korelasi & Uji",
        "nav_text": "📝 Pemrosesan Teks",
        "export_title": "📄 Ekspor Laporan",
        "export_desc": "🖨️ Buat PDF lengkap berisi statistik deskriptif, uji normalitas, histogram, boxplot, korelasi, dan ringkasan analisis teks.",
        "export_button": "📥 Buat laporan PDF",
        "export_filename": "laporan_survei_lengkap.pdf",
        "pdf_title": "📊 Laporan Lengkap Data Survei",
        "pdf_section_numdist": "1️⃣ Variabel Numerik - Distribusi",
        "pdf_section_scatter": "2️⃣ Scatter Plot - Hubungan",
        "pdf_section_catbar": "3️⃣ Variabel Kategorikal - Diagram Batang",
        "pdf_section_numfull": "4️⃣ Variabel Numerik - Statistik Lengkap",
        "pdf_section_catfreq": "5️⃣ Variabel Kategorikal - Tabel Frekuensi",
        "pdf_section_corr": "6️⃣ Analisis Korelasi",
        "pdf_section_text": "7️⃣ Analisis Teks - Kata Teratas",
        "pdf_notext": "⚠️ Tidak ada data teks untuk dianalisis.",
        "filter_data_optional": "🔍 Filter data (opsional)",
        "filter_column": "📌 Kolom filter",
        "no_filter": "🚫 (Tidak ada filter)",
        "select_values": "✅ Pilih nilai",
        "summary_normality": "📊 Ringkasan & Normalitas",
        "distribution": "📈 Distribusi",
        "select_column_distribution": "📌 Pilih kolom untuk distribusi",
        "normality_test": "🧪 Uji normalitas (D’Agostino-Pearson)",
        "statistic": "📊 Statistik",
        "deviate_normal": "⚠️ Data menyimpang signifikan dari distribusi normal (tolak H0 pada α = 0,05).",
        "no_deviate_normal": "✅ Tidak ada penyimpangan signifikan dari distribusi normal (gagal tolak H0 pada α = 0,05).",
        "not_enough_normality": "⚠️ Data tidak cukup untuk uji normalitas (minimal 8 nilai tidak kosong).",
        "histogram_boxplot": "📊 Histogram / 📦 Boxplot",
        "scatter_bar": "📈 Scatter & 📊 Batang",
        "x_variable_numeric": "📌 Variabel X (numerik)",
        "y_variable_numeric": "🎯 Variabel Y (numerik)",
        "scatter_plot": "📈 Plot scatter",
        "not_enough_scatter": "⚠️ Tidak cukup data valid untuk plot scatter.",
        "need_2_numeric": "⚠️ Minimal perlu 2 kolom numerik untuk plot scatter.",
        "categorical_bar": "🏷️ Kolom kategorikal untuk diagram batang",
        "bar_chart": "📊 Diagram batang (top 20)",
        "no_categorical_bar": "⚠️ Tidak ada kolom kategorikal untuk diagram batang.",
        "independent_variable": "🎛️ Variabel independen",
        "dependent_variable": "🎯 Variabel dependen",
        "observed": "👁️ Teramati",
        "expected": "📐 Diharapkan",
        "no_file": "📂 Silakan unggah file untuk memulai.",
        "data_preview_subtitle": "📈 analisis data survei",
        "leader": "👑 Pemimpin",
        "member": "👥 Anggota",
        "upload_limit": "📦 Batas 200MB • CSV, XLS, XLSX",
        "statistic_label": "📊 Statistik",
        "p_value_label": "📎 p-value",
        "bar_chart_top20": "📊 Diagram batang (top 20)",
        "pdf_meta_rows": "📏 Baris: {0}, Kolom: {1}",
        "pdf_meta_cols": "🔢 Kolom numerik: {0}, 🏷️ Kolom kategorikal: {1}, 🔤 Kolom teks: {2}",
        "group_info": (
            "👥 Group 5 Class 2\n"
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 Pemimpin\n"
            "MAULA AQIEL NURI (04202400023) – 👥 Anggota\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 Anggota\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 Anggota"
        ),
    },
    "JP": {  # Japanese
        "title": "📊 デジタル決済の利用状況と財務規律に関する調査",
        "subtitle": "📈 調査データ分析",
        "upload_subheader": "📁 アンケートデータのアップロード",
        "upload_label": "📤 ここにファイルをドラッグ＆ドロップ、またはクリックして選択（CSV, XLS, XLSX）",
        "data_preview": "👀 データプレビュー（先頭1000行まで）",
        "text_processing_subheader": "📝 テキスト前処理",
        "text_columns_detected": "🔎 検出されたテキスト列：",
        "select_text_col": "🧩 前処理するテキスト列を選択",
        "no_text_columns": "⚠️ テキスト型の列が見つかりません。",
        "text_processing_note": "ℹ️ テキストは小文字化され、句読点が削除され、スペースで分割され、英語のストップワードが除去されます。",
        "sample_tokens": "🔤 前処理されたトークンのサンプル",
        "top_words": "🏆 出現頻度トップ10の単語",
        "stats_subheader": "📈 記述統計と分布",
        "select_numeric_col": "🔢 統計・グラフ用の数値列を選択",
        "no_numeric_cols": "⚠️ 利用可能な数値列がありません。",
        "desc_stats": "📊 選択された列の記述統計",
        "freq_table_subheader": "📊 カテゴリ頻度表",
        "select_categorical_col": "🏷️ 頻度表を作成するカテゴリ列を選択",
        "no_categorical_cols": "⚠️ カテゴリ列がありません。",
        "freq_count": "🔢 度数",
        "freq_percent": "📏 割合（％）",
        "visual_subheader": "📉 データの可視化",
        "histogram": "📊 ヒストグラム",
        "boxplot": "📦 箱ひげ図",
        "correlation_subheader": "🔗 相関と統計的検定",
        "pearson_header": "📐 ピアソンの相関",
        "spearman_header": "📐 スピアマンの順位相関",
        "chi_header": "🎲 カイ二乗検定",
        "select_x_numeric": "📌 X変数（数値）を選択",
        "select_y_numeric": "🎯 Y変数（数値）を選択",
        "not_enough_numeric": "⚠️ この分析に必要な数値列が不足しています。",
        "pearson_result": "📐 ピアソン相関の結果",
        "spearman_result": "📐 スピアマン相関の結果",
        "corr_coef": "📊 相関係数 (r)",
        "p_value": "📎 p値",
        "interpretation": "🧠 解釈",
        "select_x_cat": "📌 X変数（カテゴリ）を選択",
        "select_y_cat": "🎯 Y変数（カテゴリ）を選択",
        "not_enough_categorical": "⚠️ カイ二乗検定に必要なカテゴリ列が不足しています。",
        "chi_square_result": "🎲 カイ二乗検定の結果",
        "chi_square_stat": "📊 カイ二乗統計量",
        "chi_square_df": "📏 自由度 (df)",
        "chi_square_p": "📎 p値",
        "alpha_note": "ℹ️ 有意水準 α = 0.05 で検定しています。",
        "significant_assoc": "✅ 2つの変数の間に統計的に有意な関係があります。",
        "no_significant_assoc": "❌ 2つの変数の間に統計的に有意な関係はありません。",
        "corr_direction_positive": "⬆️ 正の関係：Xが増加するとYも増加する傾向があります。",
        "corr_direction_negative": "⬇️ 負の関係：Xが増加するとYは減少する傾向があります。",
        "corr_direction_zero": "➖ 明確な関係の方向がありません（ほぼ0）。",
        "corr_strength_none": "⚪ ほとんど関係がありません。",
        "corr_strength_weak": "🟡 弱い関係です。",
        "corr_strength_moderate": "🟠 中程度の関係です。",
        "corr_strength_strong": "🔴 強い関係です。",
        "warning_select_valid": "⚠️ 有効な列の組み合わせを選択してください。",
        "header_github": "🐙 GitHubでフォーク",
        "nav_desc": "📊 記述統計",
        "nav_visual": "📉 可視化",
        "nav_corr": "🔗 相関・検定",
        "nav_text": "📝 テキスト処理",
        "export_title": "📄 レポートのエクスポート",
        "export_desc": "🖨️ 記述統計・正規性検定・ヒストグラム・箱ひげ図・相関・テキスト分析サマリーを含むPDFレポートを生成します。",
        "export_button": "📥 PDFレポートを生成",
        "export_filename": "調査報告書全文",
        "pdf_title": "📊 アンケート完全レポート",
        "pdf_section_numdist": "1️⃣ 数値変数 - 分布",
        "pdf_section_scatter": "2️⃣ 散布図 - 関係",
        "pdf_section_catbar": "3️⃣ カテゴリ変数 - 棒グラフ",
        "pdf_section_numfull": "4️⃣ 数値変数 - 詳細統計",
        "pdf_section_catfreq": "5️⃣ カテゴリ変数 - 度数表",
        "pdf_section_corr": "6️⃣ 相関分析",
        "pdf_section_text": "7️⃣ テキスト分析 - 上位語",
        "pdf_notext": "⚠️ 分析できるテキストデータがありません。",
        "filter_data_optional": "🔍 データフィルター（オプション）",
        "filter_column": "📌 フィルター列",
        "no_filter": "🚫 （フィルターなし）",
        "select_values": "✅ 値を選択",
        "summary_normality": "📊 要約と正規性",
        "distribution": "📈 分布",
        "select_column_distribution": "📌 分布用の列を選択",
        "normality_test": "🧪 正規性検定（D’Agostino-Pearson）",
        "statistic": "📊 統計量",
        "deviate_normal": "⚠️ データは正規分布から有意に逸脱しています（α = 0.05 でH0棄却）。",
        "no_deviate_normal": "✅ 正規分布から有意な逸脱は見られません（α = 0.05 でH0棄却できず）。",
        "not_enough_normality": "⚠️ 正規性検定にはデータ点が不足しています（8個以上の欠損でない値が必要）。",
        "histogram_boxplot": "📊 ヒストグラム / 📦 箱ひげ図",
        "scatter_bar": "📈 散布図 & 📊 棒グラフ",
        "x_variable_numeric": "📌 X変数（数値）",
        "y_variable_numeric": "🎯 Y変数（数値）",
        "scatter_plot": "📈 散布図",
        "not_enough_scatter": "⚠️ 散布図を作成するのに十分な有効データがありません。",
        "need_2_numeric": "⚠️ 散布図には少なくとも2つの数値列が必要です。",
        "categorical_bar": "🏷️ 棒グラフ用のカテゴリ列",
        "bar_chart": "📊 棒グラフ（上位20）",
        "no_categorical_bar": "⚠️ 棒グラフ用のカテゴリ列がありません。",
        "independent_variable": "🎛️ 独立変数",
        "dependent_variable": "🎯 従属変数",
        "observed": "👁️ 観測値",
        "expected": "📐 期待値",
        "no_file": "📂 まずファイルをアップロードしてください。",
        "data_preview_subtitle": "📈 調査データ分析",
        "leader": "👑 リーダー",
        "member": "👥 メンバー",
        "upload_limit": "📦 上限 200MB ・ CSV, XLS, XLSX",
        "statistic_label": "📊 統計量",
        "p_value_label": "📎 p値",
        "bar_chart_top20": "📊 棒グラフ（上位20）",
        "pdf_meta_rows": "📏 行数: {0}, 列数: {1}",
        "pdf_meta_cols": "🔢 数値列: {0}, 🏷️ カテゴリ列: {1}, 🔤 テキスト列: {2}",
        "group_info": (
        "👥 Group 5 Class 2\n" 
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 リーダー\n"
            "MAULA AQIEL NURI (04202400023) – 👥 メンバー\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 メンバー\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 メンバー"
        ),
    },
    "KR": {  # Korean
        "title": "📊 디지털 결제 사용 및 재정적 절제력 설문조사",
        "subtitle": "📈 조사 데이터 분석",
        "upload_subheader": "📁 설문 데이터 업로드",
        "upload_label": "📤 여기에 파일을 드래그 앤 드롭하거나 클릭하여 선택하세요 (CSV, XLS, XLSX)",
        "data_preview": "👀 데이터 미리보기 (최대 첫 1000행)",
        "text_processing_subheader": "📝 텍스트 전처리",
        "text_columns_detected": "🔎 감지된 텍스트 열:",
        "select_text_col": "🧩 전처리할 텍스트 열 선택",
        "no_text_columns": "⚠️ 텍스트 형식의 열이 없습니다.",
        "text_processing_note": "ℹ️ 텍스트는 소문자로 변환되고, 구두점이 제거되며, 공백 기준으로 분할되고, 영어 불용어가 제거됩니다.",
        "sample_tokens": "🔤 전처리된 토큰 샘플",
        "top_words": "🏆 출현 빈도 상위 10개 단어",
        "stats_subheader": "📈 기술통계 및 분포",
        "select_numeric_col": "🔢 통계/그래프용 숫자 열 선택",
        "no_numeric_cols": "⚠️ 사용 가능한 숫자 열이 없습니다.",
        "desc_stats": "📊 선택한 열의 기술통계",
        "freq_table_subheader": "📊 범주형 빈도표",
        "select_categorical_col": "🏷️ 빈도표를 만들 범주형 열 선택",
        "no_categorical_cols": "⚠️ 범주형 열이 없습니다.",
        "freq_count": "🔢 빈도",
        "freq_percent": "📏 비율(%)",
        "visual_subheader": "📉 데이터 시각화",
        "histogram": "📊 히스토그램",
        "boxplot": "📦 박스플롯",
        "correlation_subheader": "🔗 상관관계 및 통계 검정",
        "pearson_header": "📐 피어슨 상관",
        "spearman_header": "📐 스피어만 순위 상관",
        "chi_header": "🎲 카이제곱 검정",
        "select_x_numeric": "📌 X 변수(숫자)를 선택",
        "select_y_numeric": "🎯 Y 변수(숫자)를 선택",
        "not_enough_numeric": "⚠️ 이 분석에 필요한 숫자 열이 부족합니다.",
        "pearson_result": "📐 피어슨 상관 결과",
        "spearman_result": "📐 스피어만 상관 결과",
        "corr_coef": "📊 상관계수 (r)",
        "p_value": "📎 p-값",
        "interpretation": "🧠 해석",
        "select_x_cat": "📌 X 변수(범주형)를 선택",
        "select_y_cat": "🎯 Y 변수(범주형)를 선택",
        "not_enough_categorical": "⚠️ 카이제곱 검정에 필요한 범주형 열이 부족합니다.",
        "chi_square_result": "🎲 카이제곱 검정 결과",
        "chi_square_stat": "📊 카이제곱 통계량",
        "chi_square_df": "📏 자유도 (df)",
        "chi_square_p": "📎 p-값",
        "alpha_note": "ℹ️ 유의수준 α = 0.05에서 검정합니다.",
        "significant_assoc": "✅ 두 변수 사이에 통계적으로 유의한 관계가 있습니다.",
        "no_significant_assoc": "❌ 두 변수 사이에 통계적으로 유의한 관계가 없습니다.",
        "corr_direction_positive": "⬆️ 양의 관계: X가 증가하면 Y도 증가하는 경향이 있습니다.",
        "corr_direction_negative": "⬇️ 음의 관계: X가 증가하면 Y는 감소하는 경향이 있습니다.",
        "corr_direction_zero": "➖ 명확한 관계 방향이 없습니다(거의 0).",
        "corr_strength_none": "⚪ 거의 관계가 없습니다.",
        "corr_strength_weak": "🟡 약한 관계입니다.",
        "corr_strength_moderate": "🟠 보통 정도의 관계입니다.",
        "corr_strength_strong": "🔴 강한 관계입니다.",
        "warning_select_valid": "⚠️ 올바른 열 조합을 선택하세요.",
        "header_github": "🐙 GitHub에서 포크",
        "nav_desc": "📊 기술통계",
        "nav_visual": "📉 시각화",
        "nav_corr": "🔗 상관 및 검정",
        "nav_text": "📝 텍스트 처리",
        "export_title": "📄 보고서 내보내기",
        "export_desc": "🖨️ 기술통계, 정규성 검정, 히스토그램, 박스플롯, 상관분석, 텍스트 분석 요약을 포함한 전체 PDF 보고서를 생성합니다.",
        "export_button": "📥 PDF 보고서 생성",
        "export_filename": "설문조사 전체 보고서",
        "pdf_title": "📊 설문 데이터 전체 보고서",
        "pdf_section_numdist": "1️⃣ 수치 변수 - 분포",
        "pdf_section_scatter": "2️⃣ 산점도 - 관계",
        "pdf_section_catbar": "3️⃣ 범주형 변수 - 막대 그래프",
        "pdf_section_numfull": "4️⃣ 수치 변수 - 상세 통계",
        "pdf_section_catfreq": "5️⃣ 범주형 변수 - 도수표",
        "pdf_section_corr": "6️⃣ 상관 분석",
        "pdf_section_text": "7️⃣ 텍스트 분석 - 상위 단어",
        "pdf_notext": "⚠️ 분석할 텍스트 데이터가 없습니다.",
        "filter_data_optional": "🔍 데이터 필터 (선택)",
        "filter_column": "📌 필터 열",
        "no_filter": "🚫 (필터 없음)",
        "select_values": "✅ 값 선택",
        "summary_normality": "📊 요약 및 정규성",
        "distribution": "📈 분포",
        "select_column_distribution": "📌 분포용 열 선택",
        "normality_test": "🧪 정규성 검정 (D’Agostino-Pearson)",
        "statistic": "📊 통계량",
        "deviate_normal": "⚠️ 데이터가 정규분포로부터 유의하게 벗어납니다 (α = 0.05에서 H0 기각).",
        "no_deviate_normal": "✅ 정규분포로부터 유의한 벗어남이 없습니다 (α = 0.05에서 H0 기각 실패).",
        "not_enough_normality": "⚠️ 정규성 검정을 위한 데이터가 부족합니다 (결측이 아닌 값이 최소 8개 필요).",
        "histogram_boxplot": "📊 히스토그램 / 📦 박스플롯",
        "scatter_bar": "📈 산점도 & 📊 막대 그래프",
        "x_variable_numeric": "📌 X 변수 (숫자형)",
        "y_variable_numeric": "🎯 Y 변수 (숫자형)",
        "scatter_plot": "📈 산점도",
        "not_enough_scatter": "⚠️ 산점도를 그리기 위한 유효한 데이터가 충분하지 않습니다.",
        "need_2_numeric": "⚠️ 산점도에는 최소 2개의 숫자형 열이 필요합니다.",
        "categorical_bar": "🏷️ 막대 그래프용 범주형 열",
        "bar_chart": "📊 막대 그래프 (상위 20)",
        "no_categorical_bar": "⚠️ 막대 그래프용 범주형 열이 없습니다.",
        "independent_variable": "🎛️ 독립 변수",
        "dependent_variable": "🎯 종속 변수",
        "observed": "👁️ 관측값",
        "expected": "📐 기대값",
        "no_file": "📂 먼저 파일을 업로드하세요.",
        "data_preview_subtitle": "📈 조사 데이터 분석",
        "leader": "👑 리더",
        "member": "👥 구성원",
        "upload_limit": "📦 최대 200MB • CSV, XLS, XLSX",
        "statistic_label": "📊 통계량",
        "p_value_label": "📎 p-값",
        "bar_chart_top20": "📊 막대 그래프 (상위 20)",
        "pdf_meta_rows": "📏 행: {0}, 열: {1}",
        "pdf_meta_cols": "🔢 숫자 열: {0}, 🏷️ 범주형 열: {1}, 🔤 텍스트 열: {2}",
        "group_info": (
            "👥 Group 5 Class 2\n"
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 리더\n"
            "MAULA AQIEL NURI (04202400023) – 👥 구성원\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 구성원\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 구성원"
        ),
    },
    "CN": {  # Chinese (Simplified)
        "title": "📊 数字支付使用与财务纪律调查",
        "subtitle": "📈 调查数据分析",
        "upload_subheader": "📁 上传问卷数据",
        "upload_label": "📤 将文件拖放到此处或点击选择（CSV, XLS, XLSX）",
        "data_preview": "👀 数据预览（前 1000 行）",
        "text_processing_subheader": "📝 文本预处理",
        "text_columns_detected": "🔎 检测到的文本列：",
        "select_text_col": "🧩 选择要处理的文本列",
        "no_text_columns": "⚠️ 未找到文本类型的列。",
        "text_processing_note": "ℹ️ 文本将被转为小写，去除标点符号，以空格分词，并移除英文停用词。",
        "sample_tokens": "🔤 预处理后的词元示例",
        "top_words": "🏆 词频最高的 10 个词",
        "stats_subheader": "📈 描述性统计与分布",
        "select_numeric_col": "🔢 选择用于统计/绘图的数值列",
        "no_numeric_cols": "⚠️ 没有可用的数值列。",
        "desc_stats": "📊 所选列的描述性统计",
        "freq_table_subheader": "📊 分类频数表",
        "select_categorical_col": "🏷️ 选择用于频数表的分类列",
        "no_categorical_cols": "⚠️ 没有分类列。",
        "freq_count": "🔢 频数",
        "freq_percent": "📏 百分比（%）",
        "visual_subheader": "📉 数据可视化",
        "histogram": "📊 直方图",
        "boxplot": "📦 箱线图",
        "correlation_subheader": "🔗 相关性与统计检验",
        "pearson_header": "📐 皮尔逊相关",
        "spearman_header": "📐 斯皮尔曼等级相关",
        "chi_header": "🎲 卡方检验",
        "select_x_numeric": "📌 选择 X 变量（数值）",
        "select_y_numeric": "🎯 选择 Y 变量（数值）",
        "not_enough_numeric": "⚠️ 可用于该分析的数值列不足。",
        "pearson_result": "📐 皮尔逊相关结果",
        "spearman_result": "📐 斯皮尔曼相关结果",
        "corr_coef": "📊 相关系数 (r)",
        "p_value": "📎 p 值",
        "interpretation": "🧠 解释",
        "select_x_cat": "📌 选择 X 变量（分类）",
        "select_y_cat": "🎯 选择 Y 变量（分类）",
        "not_enough_categorical": "⚠️ 用于卡方检验的分类列不足。",
        "chi_square_result": "🎲 卡方检验结果",
        "chi_square_stat": "📊 卡方统计量",
        "chi_square_df": "📏 自由度 (df)",
        "chi_square_p": "📎 p 值",
        "alpha_note": "ℹ️ 在显著性水平 α = 0.05 下进行检验。",
        "significant_assoc": "✅ 两个变量之间存在统计上显著的关联。",
        "no_significant_assoc": "❌ 两个变量之间不存在统计上显著的关联。",
        "corr_direction_positive": "⬆️ 正相关：X 增加时，Y 通常也增加。",
        "corr_direction_negative": "⬇️ 负相关：X 增加时，Y 通常减少。",
        "corr_direction_zero": "➖ 没有明显的相关方向（接近 0）。",
        "corr_strength_none": "⚪ 几乎没有相关关系。",
        "corr_strength_weak": "🟡 相关关系较弱。",
        "corr_strength_moderate": "🟠 相关关系中等。",
        "corr_strength_strong": "🔴 相关关系较强。",
        "warning_select_valid": "⚠️ 请选择有效的列组合。",
        "header_github": "🐙 在 GitHub 上 Fork",
        "nav_desc": "📊 描述性统计",
        "nav_visual": "📉 可视化",
        "nav_corr": "🔗 相关与检验",
        "nav_text": "📝 文本处理",
        "export_title": "📄 导出报告",
        "export_desc": "🖨️ 生成包含描述性统计、正态性检验、直方图、箱线图、相关分析和文本分析摘要的完整 PDF 报告。",
        "export_button": "📥 生成 PDF 报告",
        "export_filename": "调查完整报告",
        "pdf_title": "📊 问卷数据完整报告",
        "pdf_section_numdist": "1️⃣ 数值变量 - 分布",
        "pdf_section_scatter": "2️⃣ 散点图 - 关系",
        "pdf_section_catbar": "3️⃣ 类别变量 - 条形图",
        "pdf_section_numfull": "4️⃣ 数值变量 - 详细统计",
        "pdf_section_catfreq": "5️⃣ 类别变量 - 频数表",
        "pdf_section_corr": "6️⃣ 相关分析",
        "pdf_section_text": "7️⃣ 文本分析 - 高频词",
        "pdf_notext": "⚠️ 没有可供分析的文本数据。",
        "filter_data_optional": "🔍 数据筛选（可选）",
        "filter_column": "📌 筛选列",
        "no_filter": "🚫 （无筛选）",
        "select_values": "✅ 选择值",
        "summary_normality": "📊 概要与正态性",
        "distribution": "📈 分布",
        "select_column_distribution": "📌 选择用于分布的列",
        "normality_test": "🧪 正态性检验（D’Agostino-Pearson）",
        "statistic": "📊 统计量",
        "deviate_normal": "⚠️ 数据显著偏离正态分布（在 α = 0.05 下拒绝 H0）。",
        "no_deviate_normal": "✅ 数据未显著偏离正态分布（在 α = 0.05 下不能拒绝 H0）。",
        "not_enough_normality": "⚠️ 正态性检验的数据点不足（至少需要 8 个非缺失值）。",
        "histogram_boxplot": "📊 直方图 / 📦 箱线图",
        "scatter_bar": "📈 散点图 & 📊 条形图",
        "x_variable_numeric": "📌 X 变量（数值）",
        "y_variable_numeric": "🎯 Y 变量（数值）",
        "scatter_plot": "📈 散点图",
        "not_enough_scatter": "⚠️ 用于绘制散点图的有效数据不足。",
        "need_2_numeric": "⚠️ 散点图至少需要 2 列数值型数据。",
        "categorical_bar": "🏷️ 用于条形图的分类列",
        "bar_chart": "📊 条形图（前 20）",
        "no_categorical_bar": "⚠️ 没有用于条形图的分类列。",
        "independent_variable": "🎛️ 自变量",
        "dependent_variable": "🎯 因变量",
        "observed": "👁️ 观察值",
        "expected": "📐 期望值",
        "no_file": "📂 请先上传文件以开始。",
        "data_preview_subtitle": "📈 调查数据分析",
        "leader": "👑 组长",
        "member": "👥 成员",
        "upload_limit": "📦 限制 200MB • CSV, XLS, XLSX",
        "statistic_label": "📊 统计量",
        "p_value_label": "📎 p 值",
        "bar_chart_top20": "📊 条形图（前 20）",
        "pdf_meta_rows": "📏 行数: {0}, 列数: {1}",
        "pdf_meta_cols": "🔢 数值列: {0}, 🏷️ 分类列: {1}, 🔤 文本列: {2}",
        "group_info": (
            "👥 Group 5 Class 2\n"
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 组长\n"
            "MAULA AQIEL NURI (04202400023) – 👥 成员\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 成员\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 成员"
        ),
    },
    "AR": {  # Arabic
        "title": "📊استبيان حول استخدام الدفع الرقمي والانضباط المالي",
        "subtitle": "📈 تحليل بيانات الاستطلاع المجموعة 5",
        "upload_subheader": "📁 رفع بيانات الاستبيان",
        "upload_label": "📤 اسحب وأفلت الملف هنا أو اضغط للاختيار (CSV, XLS, XLSX)",
        "data_preview": "👀 معاينة البيانات (حتى أول 1000 صف)",
        "text_processing_subheader": "📝 معالجة النصوص",
        "text_columns_detected": "🔎 الأعمدة النصية المكتشفة:",
        "select_text_col": "🧩 اختر عمود النص للمعالجة",
        "no_text_columns": "⚠️ لا توجد أعمدة من نوع نصي.",
        "text_processing_note": "ℹ️ سيتم تحويل النص إلى حروف صغيرة، وإزالة علامات الترقيم، وتقسيمه إلى كلمات، وحذف كلمات الوقف الإنجليزية.",
        "sample_tokens": "🔤 عينة من الرموز المعالجة",
        "top_words": "🏆 أكثر 10 كلمات تكراراً",
        "stats_subheader": "📈 الإحصاءات الوصفية والتوزيع",
        "select_numeric_col": "🔢 اختر عموداً رقمياً للإحصاءات والرسوم",
        "no_numeric_cols": "⚠️ لا توجد أعمدة رقمية متاحة.",
        "desc_stats": "📊 الإحصاءات الوصفية للعمود المحدد",
        "freq_table_subheader": "📊 جدول التكرار للفئات",
        "select_categorical_col": "🏷️ اختر عموداً فئوياً لجدول التكرار",
        "no_categorical_cols": "⚠️ لا توجد أعمدة فئوية.",
        "freq_count": "🔢 العدد",
        "freq_percent": "📏 النسبة المئوية (%)",
        "visual_subheader": "📉 عرض البيانات بيانياً",
        "histogram": "📊 مخطط التوزيع (Histogram)",
        "boxplot": "📦 مخطط الصندوق (Boxplot)",
        "correlation_subheader": "🔗 الارتباط والاختبارات الإحصائية",
        "pearson_header": "📐 معامل ارتباط بيرسون",
        "spearman_header": "📐 معامل ارتباط سبيرمان",
        "chi_header": "🎲 اختبار كاي تربيع",
        "select_x_numeric": "📌 اختر متغير X (رقمي)",
        "select_y_numeric": "🎯 اختر متغير Y (رقمي)",
        "not_enough_numeric": "⚠️ لا يوجد عدد كافٍ من الأعمدة الرقمية لهذا التحليل.",
        "pearson_result": "📐 نتيجة ارتباط بيرسون",
        "spearman_result": "📐 نتيجة ارتباط سبيرمان",
        "corr_coef": "📊 معامل الارتباط (r)",
        "p_value": "📎 قيمة p",
        "interpretation": "🧠 التفسير",
        "select_x_cat": "📌 اختر متغير X (فئوي)",
        "select_y_cat": "🎯 اختر متغير Y (فئوي)",
        "not_enough_categorical": "⚠️ لا يوجد عدد كافٍ من الأعمدة الفئوية لاختبار كاي تربيع.",
        "chi_square_result": "🎲 نتيجة اختبار كاي تربيع",
        "chi_square_stat": "📊 إحصائية كاي تربيع",
        "chi_square_df": "📏 درجات الحرية (df)",
        "chi_square_p": "📎 قيمة p",
        "alpha_note": "ℹ️ تم الاختبار عند مستوى دلالة α = 0.05.",
        "significant_assoc": "✅ هناك علاقة ذات دلالة إحصائية بين المتغيرين.",
        "no_significant_assoc": "❌ لا توجد علاقة ذات دلالة إحصائية بين المتغيرين.",
        "corr_direction_positive": "⬆️ علاقة إيجابية: عندما يزيد X، يميل Y إلى الزيادة.",
        "corr_direction_negative": "⬇️ علاقة سلبية: عندما يزيد X، يميل Y إلى النقصان.",
        "corr_direction_zero": "➖ لا يوجد اتجاه علاقة واضح (قريب من الصفر).",
        "corr_strength_none": "⚪ لا توجد علاقة تقريباً.",
        "corr_strength_weak": "🟡 علاقة ضعيفة.",
        "corr_strength_moderate": "🟠 علاقة معتدلة.",
        "corr_strength_strong": "🔴 علاقة قوية.",
        "warning_select_valid": "⚠️ يرجى اختيار تركيبة أعمدة صالحة.",
        "header_github": "🐙 استنساخ على GitHub",
        "nav_desc": "📊 الإحصاءات الوصفية",
        "nav_visual": "📉 الرسوم البيانية",
        "nav_corr": "🔗 الارتباط والاختبارات",
        "nav_text": "📝 معالجة النصوص",
        "export_title": "📄 تصدير التقرير",
        "export_desc": "🖨️ إنشاء تقرير PDF كامل يتضمن الإحصاءات الوصفية، اختبار الطبيعة التوزيعية، المخططات التوزيعية، مخططات الصندوق، الارتباط، وملخص تحليل النصوص.",
        "export_button": "📥 إنشاء تقرير PDF",
        "export_filename": "تقرير الاستطلاع الكامل",
        "pdf_title": "📊 التقرير الكامل لبيانات الاستبيان",
        "pdf_section_numdist": "1️⃣ المتغيرات الرقمية - التوزيع",
        "pdf_section_scatter": "2️⃣ مخطط التبعثر - العلاقة",
        "pdf_section_catbar": "3️⃣ المتغيرات الفئوية - المخطط الشريطي",
        "pdf_section_numfull": "4️⃣ المتغيرات الرقمية - الإحصاءات التفصيلية",
        "pdf_section_catfreq": "5️⃣ المتغيرات الفئوية - جدول التكرار",
        "pdf_section_corr": "6️⃣ تحليل الارتباط",
        "pdf_section_text": "7️⃣ تحليل النصوص - الكلمات الأعلى تكراراً",
        "pdf_notext": "⚠️ لا توجد بيانات نصية للتحليل.",
        "filter_data_optional": "🔍 تصفية البيانات (اختياري)",
        "filter_column": "📌 عمود التصفية",
        "no_filter": "🚫 (بدون تصفية)",
        "select_values": "✅ اختر القيم",
        "summary_normality": "📊 الملخص والطبيعة التوزيعية",
        "distribution": "📈 التوزيع",
        "select_column_distribution": "📌 اختر عموداً للتوزيع",
        "normality_test": "🧪 اختبار الطبيعة التوزيعية (D’Agostino-Pearson)",
        "statistic": "📊 الإحصائية",
        "deviate_normal": "⚠️ البيانات تنحرف بشكل ملحوظ عن التوزيع الطبيعي (رفض H0 عند α = 0.05).",
        "no_deviate_normal": "✅ لا يوجد انحراف ملحوظ عن التوزيع الطبيعي (فشل رفض H0 عند α = 0.05).",
        "not_enough_normality": "⚠️ لا توجد بيانات كافية لاختبار الطبيعة التوزيعية (يلزم 8 قيم غير مفقودة على الأقل).",
        "histogram_boxplot": "📊 مخطط التوزيع / 📦 مخطط الصندوق",
        "scatter_bar": "📈 مخطط التبعثر & 📊 مخطط شريطي",
        "x_variable_numeric": "📌 المتغير X (رقمي)",
        "y_variable_numeric": "🎯 المتغير Y (رقمي)",
        "scatter_plot": "📈 مخطط التبعثر",
        "not_enough_scatter": "⚠️ لا توجد بيانات كافية لرسم مخطط التبعثر.",
        "need_2_numeric": "⚠️ يلزم عمودان رقميان على الأقل لرسم مخطط التبعثر.",
        "categorical_bar": "🏷️ عمود فئوي للمخطط الشريطي",
        "bar_chart": "📊 مخطط شريطي (أعلى 20)",
        "no_categorical_bar": "⚠️ لا توجد أعمدة فئوية للمخطط الشريطي.",
        "independent_variable": "🎛️ المتغير المستقل",
        "dependent_variable": "🎯 المتغير التابع",
        "observed": "👁️ القيم المرصودة",
        "expected": "📐 القيم المتوقعة",
        "no_file": "📂 يرجى رفع ملف للبدء.",
        "data_preview_subtitle": "📈 تحليل بيانات الاستطلاع",
        "leader": "👑 القائد",
        "member": "👥 عضو",
        "upload_limit": "📦 الحد 200MB • CSV, XLS, XLSX",
        "statistic_label": "📊 الإحصائية",
        "p_value_label": "📎 قيمة p",
        "bar_chart_top20": "📊 مخطط شريطي (أعلى 20)",
        "pdf_meta_rows": "📏 الصفوف: {0}، الأعمدة: {1}",
        "pdf_meta_cols": "🔢 الأعمدة الرقمية: {0}، 🏷️ الأعمدة الفئوية: {1}، 🔤 أعمدة النص: {2}",    
        "group_info": (
            "👥 المجموعة 5 الصف 2\n"
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 القائد\n"
            "MAULA AQIEL NURI (04202400023) – 👥 عضو\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 عضو\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 عضو"
        ),
    },
        "PT": {  # Portuguese
        "title": "📊 Uso de Pagamentos Digitais & Disciplina Financeira",
        "subtitle": "📈 análise de dados de pesquisa",
        "upload_subheader": "📁 Enviar Dados da Pesquisa",
        "upload_label": "📤 Arraste e solte o arquivo aqui ou clique para escolher (CSV, XLS, XLSX)",
        "data_preview": "👀 Pré-visualização dos dados (até as primeiras 1000 linhas)",
        "text_processing_subheader": "📝 Pré-processamento de Texto",
        "text_columns_detected": "🔎 Colunas de texto detectadas:",
        "select_text_col": "🧩 Selecione uma coluna de texto para processar",
        "no_text_columns": "⚠️ Nenhuma coluna do tipo texto foi detectada.",
        "text_processing_note": "ℹ️ O texto será convertido para minúsculas, sem pontuação, tokenizado (separado por espaços) e terá stopwords em inglês removidas.",
        "sample_tokens": "🔤 Amostra de tokens processados",
        "top_words": "🏆 Top 10 palavras por frequência",
        "stats_subheader": "📈 Estatísticas Descritivas & Distribuição",
        "select_numeric_col": "🔢 Selecione uma coluna numérica para estatísticas e gráficos",
        "no_numeric_cols": "⚠️ Nenhuma coluna numérica disponível.",
        "desc_stats": "📊 Estatísticas descritivas para a coluna selecionada",
        "freq_table_subheader": "📊 Tabela de Frequência Categórica",
        "select_categorical_col": "🏷️ Selecione uma coluna categórica para a tabela de frequência",
        "no_categorical_cols": "⚠️ Nenhuma coluna categórica disponível.",
        "freq_count": "🔢 Contagem",
        "freq_percent": "📏 Percentual (%)",
        "visual_subheader": "📉 Visualizações de Dados",
        "histogram": "📊 Histograma",
        "boxplot": "📦 Boxplot",
        "correlation_subheader": "🔗 Correlação & Testes Estatísticos",
        "pearson_header": "📐 Correlação de Pearson",
        "spearman_header": "📐 Correlação de Spearman",
        "chi_header": "🎲 Teste Qui-quadrado",
        "select_x_numeric": "📌 Selecione a variável X (numérica)",
        "select_y_numeric": "🎯 Selecione a variável Y (numérica)",
        "not_enough_numeric": "⚠️ Colunas numéricas insuficientes para esta análise.",
        "pearson_result": "📐 Resultado da Correlação de Pearson",
        "spearman_result": "📐 Resultado da Correlação de Spearman",
        "corr_coef": "📊 Coeficiente de correlação (r)",
        "p_value": "📎 p-valor",
        "interpretation": "🧠 Interpretação",
        "select_x_cat": "📌 Selecione a variável X (categórica)",
        "select_y_cat": "🎯 Selecione a variável Y (categórica)",
        "not_enough_categorical": "⚠️ Colunas categóricas insuficientes para o teste Qui-quadrado.",
        "chi_square_result": "🎲 Resultado do Teste Qui-quadrado",
        "chi_square_stat": "📊 Estatística Qui-quadrado",
        "chi_square_df": "📏 Graus de liberdade (df)",
        "chi_square_p": "📎 p-valor",
        "alpha_note": "ℹ️ Significância testada em α = 0,05.",
        "significant_assoc": "✅ Há uma associação estatisticamente significativa entre as duas variáveis.",
        "no_significant_assoc": "❌ Não há associação estatisticamente significativa entre as duas variáveis.",
        "corr_direction_positive": "⬆️ Relação positiva: conforme X aumenta, Y tende a aumentar.",
        "corr_direction_negative": "⬇️ Relação negativa: conforme X aumenta, Y tende a diminuir.",
        "corr_direction_zero": "➖ Nenhuma direção clara de relação (próximo de zero).",
        "corr_strength_none": "⚪ Praticamente nenhuma relação.",
        "corr_strength_weak": "🟡 Relação fraca.",
        "corr_strength_moderate": "🟠 Relação moderada.",
        "corr_strength_strong": "🔴 Relação forte.",
        "warning_select_valid": "⚠️ Selecione uma combinação válida de colunas.",
        "header_github": "🐙 Fork no GitHub",
        "nav_desc": "📊 Estatísticas Descritivas",
        "nav_visual": "📉 Visualizações",
        "nav_corr": "🔗 Correlações & Testes",
        "nav_text": "📝 Processamento de Texto",
        "export_title": "📄 Exportar Relatório",
        "export_desc": "🖨️ Gerar um PDF completo com todas as estatísticas descritivas, teste de normalidade, histogramas, boxplots, correlações e resumo da análise de texto.",
        "export_button": "📥 Gerar relatório em PDF",
        "export_filename": "relatorio_pesquisa_completo.pdf",
        "pdf_title": "📊 Relatório Completo de Dados da Pesquisa",
        "pdf_section_numdist": "1️⃣ Variáveis Numéricas - Distribuições",
        "pdf_section_scatter": "2️⃣ Gráficos de Dispersão - Relações",
        "pdf_section_catbar": "3️⃣ Variáveis Categóricas - Gráficos de Barras",
        "pdf_section_numfull": "4️⃣ Variáveis Numéricas - Estatísticas Completas",
        "pdf_section_catfreq": "5️⃣ Variáveis Categóricas - Tabelas de Frequência",
        "pdf_section_corr": "6️⃣ Análise de Correlação",
        "pdf_section_text": "7️⃣ Análise de Texto - Palavras Principais",
        "pdf_notext": "⚠️ Não há dados de texto para analisar.",
        "filter_data_optional": "🔍 Filtrar dados (opcional)",
        "filter_column": "📌 Coluna de filtro",
        "no_filter": "🚫 (Sem filtro)",
        "select_values": "✅ Selecionar valores",
        "summary_normality": "📊 Resumo & Normalidade",
        "distribution": "📈 Distribuição",
        "select_column_distribution": "📌 Selecione a coluna para distribuição",
        "normality_test": "🧪 Teste de normalidade (D’Agostino-Pearson)",
        "statistic": "📊 Estatística",
        "deviate_normal": "⚠️ Os dados desviam-se significativamente da distribuição normal (rejeita H0 em α = 0,05).",
        "no_deviate_normal": "✅ Nenhum desvio significativo da distribuição normal (falha em rejeitar H0 em α = 0,05).",
        "not_enough_normality": "⚠️ Dados insuficientes para o teste de normalidade (necessário pelo menos 8 valores não nulos).",
        "histogram_boxplot": "📊 Histograma / 📦 Boxplot",
        "scatter_bar": "📈 Dispersão & 📊 Barras",
        "x_variable_numeric": "📌 Variável X (numérica)",
        "y_variable_numeric": "🎯 Variável Y (numérica)",
        "scatter_plot": "📈 Gráfico de dispersão",
        "not_enough_scatter": "⚠️ Dados válidos insuficientes para o gráfico de dispersão.",
        "need_2_numeric": "⚠️ São necessárias pelo menos 2 colunas numéricas para o gráfico de dispersão.",
        "categorical_bar": "🏷️ Coluna categórica para gráfico de barras",
        "bar_chart": "📊 Gráfico de barras (top 20)",
        "no_categorical_bar": "⚠️ Nenhuma coluna categórica para gráfico de barras.",
        "independent_variable": "🎛️ Variável independente",
        "dependent_variable": "🎯 Variável dependente",
        "observed": "👁️ Observado",
        "expected": "📐 Esperado",
        "no_file": "📂 Envie um arquivo para começar.",
        "data_preview_subtitle": "📈 análise de dados de pesquisa",
        "leader": "👑 Líder",
        "member": "👥 Membro",
        "upload_limit": "📦 Limite 200MB • CSV, XLS, XLSX",
        "statistic_label": "📊 Estatística",
        "p_value_label": "📎 p-valor",
        "bar_chart_top20": "📊 Gráfico de barras (top 20)",
        "pdf_meta_rows": "📏 Linhas: {0}, Colunas: {1}",
        "pdf_meta_cols": "🔢 Colunas numéricas: {0}, 🏷️ Colunas categóricas: {1}, 🔤 Colunas de texto: {2}",
        "group_info": (
            "👥 Grupo 5 Turma 2\n"
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 Líder\n"
            "MAULA AQIEL NURI (04202400023) – 👥 Membro\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 Membro\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 Membro"
        ),
    },
    "FR": {  # French
        "title": "📊 Utilisation des paiements numériques & discipline financière",
        "subtitle": "📈 analyse des données d’enquête",
        "upload_subheader": "📁 Importer les données de l’enquête",
        "upload_label": "📤 Glissez-déposez le fichier ici ou cliquez pour parcourir (CSV, XLS, XLSX)",
        "data_preview": "👀 Aperçu des données (jusqu’aux 1000 premières lignes)",
        "text_processing_subheader": "📝 Prétraitement du texte",
        "text_columns_detected": "🔎 Colonnes de texte détectées :",
        "select_text_col": "🧩 Sélectionnez une colonne de texte à traiter",
        "no_text_columns": "⚠️ Aucune colonne de type texte détectée.",
        "text_processing_note": "ℹ️ Le texte sera mis en minuscules, la ponctuation sera supprimée, tokenisé (séparé par des espaces) et les stopwords anglais seront retirés.",
        "sample_tokens": "🔤 Exemple de tokens traités",
        "top_words": "🏆 Top 10 des mots par fréquence",
        "stats_subheader": "📈 Statistiques descriptives & distribution",
        "select_numeric_col": "🔢 Sélectionnez une colonne numérique pour les statistiques et graphiques",
        "no_numeric_cols": "⚠️ Aucune colonne numérique disponible.",
        "desc_stats": "📊 Statistiques descriptives pour la colonne sélectionnée",
        "freq_table_subheader": "📊 Tableau de fréquence catégorielle",
        "select_categorical_col": "🏷️ Sélectionnez une colonne catégorielle pour le tableau de fréquence",
        "no_categorical_cols": "⚠️ Aucune colonne catégorielle disponible.",
        "freq_count": "🔢 Effectif",
        "freq_percent": "📏 Pourcentage (%)",
        "visual_subheader": "📉 Visualisations des données",
        "histogram": "📊 Histogramme",
        "boxplot": "📦 Boîte à moustaches (boxplot)",
        "correlation_subheader": "🔗 Corrélation & tests statistiques",
        "pearson_header": "📐 Corrélation de Pearson",
        "spearman_header": "📐 Corrélation de Spearman",
        "chi_header": "🎲 Test du Chi-deux",
        "select_x_numeric": "📌 Sélectionnez la variable X (numérique)",
        "select_y_numeric": "🎯 Sélectionnez la variable Y (numérique)",
        "not_enough_numeric": "⚠️ Colonnes numériques insuffisantes pour cette analyse.",
        "pearson_result": "📐 Résultat de la corrélation de Pearson",
        "spearman_result": "📐 Résultat de la corrélation de Spearman",
        "corr_coef": "📊 Coefficient de corrélation (r)",
        "p_value": "📎 p-valeur",
        "interpretation": "🧠 Interprétation",
        "select_x_cat": "📌 Sélectionnez la variable X (catégorielle)",
        "select_y_cat": "🎯 Sélectionnez la variable Y (catégorielle)",
        "not_enough_categorical": "⚠️ Colonnes catégorielles insuffisantes pour le test du Chi-deux.",
        "chi_square_result": "🎲 Résultat du test du Chi-deux",
        "chi_square_stat": "📊 Statistique du Chi-deux",
        "chi_square_df": "📏 Degrés de liberté (df)",
        "chi_square_p": "📎 p-valeur",
        "alpha_note": "ℹ️ Significativité testée à α = 0,05.",
        "significant_assoc": "✅ Il existe une association statistiquement significative entre les deux variables.",
        "no_significant_assoc": "❌ Il n’existe pas d’association statistiquement significative entre les deux variables.",
        "corr_direction_positive": "⬆️ Relation positive : lorsque X augmente, Y a tendance à augmenter.",
        "corr_direction_negative": "⬇️ Relation négative : lorsque X augmente, Y a tendance à diminuer.",
        "corr_direction_zero": "➖ Aucune direction claire de la relation (proche de zéro).",
        "corr_strength_none": "⚪ Pratiquement aucune relation.",
        "corr_strength_weak": "🟡 Relation faible.",
        "corr_strength_moderate": "🟠 Relation modérée.",
        "corr_strength_strong": "🔴 Relation forte.",
        "warning_select_valid": "⚠️ Veuillez sélectionner une combinaison valide de colonnes.",
        "header_github": "🐙 Fork sur GitHub",
        "nav_desc": "📊 Statistiques descriptives",
        "nav_visual": "📉 Visualisations",
        "nav_corr": "🔗 Corrélations & tests",
        "nav_text": "📝 Traitement de texte",
        "export_title": "📄 Exporter le rapport",
        "export_desc": "🖨️ Générer un PDF complet avec toutes les statistiques descriptives, test de normalité, histogrammes, boxplots, corrélations et résumé de l’analyse de texte.",
        "export_button": "📥 Générer le rapport PDF",
        "export_filename": "rapport_enquete_complet.pdf",
        "pdf_title": "📊 Rapport complet des données d’enquête",
        "pdf_section_numdist": "1️⃣ Variables numériques - Distributions",
        "pdf_section_scatter": "2️⃣ Nuages de points - Relations",
        "pdf_section_catbar": "3️⃣ Variables catégorielles - Diagrammes en barres",
        "pdf_section_numfull": "4️⃣ Variables numériques - Statistiques complètes",
        "pdf_section_catfreq": "5️⃣ Variables catégorielles - Tableaux de fréquence",
        "pdf_section_corr": "6️⃣ Analyse de corrélation",
        "pdf_section_text": "7️⃣ Analyse de texte - Mots principaux",
        "pdf_notext": "⚠️ Aucun texte à analyser.",
        "filter_data_optional": "🔍 Filtrer les données (optionnel)",
        "filter_column": "📌 Colonne de filtre",
        "no_filter": "🚫 (Aucun filtre)",
        "select_values": "✅ Sélectionner les valeurs",
        "summary_normality": "📊 Résumé & normalité",
        "distribution": "📈 Distribution",
        "select_column_distribution": "📌 Sélectionnez la colonne pour la distribution",
        "normality_test": "🧪 Test de normalité (D’Agostino-Pearson)",
        "statistic": "📊 Statistique",
        "deviate_normal": "⚠️ Les données s’écartent significativement de la distribution normale (rejet de H0 à α = 0,05).",
        "no_deviate_normal": "✅ Pas d’écart significatif par rapport à la distribution normale (H0 non rejetée à α = 0,05).",
        "not_enough_normality": "⚠️ Données insuffisantes pour le test de normalité (au moins 8 valeurs non manquantes nécessaires).",
        "histogram_boxplot": "📊 Histogramme / 📦 Boxplot",
        "scatter_bar": "📈 Nuage de points & 📊 Barres",
        "x_variable_numeric": "📌 Variable X (numérique)",
        "y_variable_numeric": "🎯 Variable Y (numérique)",
        "scatter_plot": "📈 Nuage de points",
        "not_enough_scatter": "⚠️ Données valides insuffisantes pour le nuage de points.",
        "need_2_numeric": "⚠️ Au moins 2 colonnes numériques sont nécessaires pour le nuage de points.",
        "categorical_bar": "🏷️ Colonne catégorielle pour le diagramme en barres",
        "bar_chart": "📊 Diagramme en barres (top 20)",
        "no_categorical_bar": "⚠️ Aucune colonne catégorielle pour le diagramme en barres.",
        "independent_variable": "🎛️ Variable indépendante",
        "dependent_variable": "🎯 Variable dépendante",
        "observed": "👁️ Observé",
        "expected": "📐 Attendu",
        "no_file": "📂 Veuillez importer un fichier pour commencer.",
        "data_preview_subtitle": "📈 analyse des données d’enquête",
        "leader": "👑 Chef de groupe",
        "member": "👥 Membre",
        "upload_limit": "📦 Limite 200MB • CSV, XLS, XLSX",
        "statistic_label": "📊 Statistique",
        "p_value_label": "📎 p-valeur",
        "bar_chart_top20": "📊 Diagramme en barres (top 20)",
        "pdf_meta_rows": "📏 Lignes : {0}, Colonnes : {1}",
        "pdf_meta_cols": "🔢 Colonnes numériques : {0}, 🏷️ Colonnes catégorielles : {1}, 🔤 Colonnes de texte : {2}",
        "group_info": (
            "👥 Groupe 5 Classe 2\n"
            "ADITYA ANGGARA PAMUNGKAS (04202400051) – 👑 Chef de groupe\n"
            "MAULA AQIEL NURI (04202400023) – 👥 Membre\n"
            "SYAFIQ NUR RAMADHAN (04202400073) – 👥 Membre\n"
            "RIFAT FITROTU SALMAN (04202400106) – 👥 Membre"
        ),
    },
}

# --------------------------- SESSION DEFAULTS ---------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "language" not in st.session_state:
    st.session_state["language"] = "EN"
if "aurora_mode" not in st.session_state:
    st.session_state["aurora_mode"] = True
if "sound_mode" not in st.session_state:
    st.session_state["sound_mode"] = False
if "theme" not in st.session_state:
    st.session_state["theme"] = "Default"
if "pdf_buffer" not in st.session_state:
    st.session_state["pdf_buffer"] = None

# --------------------------- I18N HELPER ---------------------------
def get_text(key: str) -> str:
    """Retrieve the text for the current language from session state."""
    lang = st.session_state.get("language", "EN")
    return TEXTS.get(lang, TEXTS["EN"]).get(key, key)

# --------------------------- CALLBACK FUNCTIONS ---------------------------
def update_dark_mode():
    st.session_state["dark_mode"] = st.session_state.get("dark_mode_toggle", False)

def update_aurora_mode():
    st.session_state["aurora_mode"] = st.session_state.get("aurora_mode_toggle", True)

def update_language():
    st.session_state["language"] = st.session_state.get("language_radio", "EN")

# =========================== AURORA & GLOBAL CSS ===========================
CUSTOM_CSS = """
<style>
body {
    margin: 0;
    padding: 0;
    background: #020617;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background: transparent !important;
}
.aurora-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    z-index: -999;
    pointer-events: none;
}
.aurora-layer {
    position: absolute;
    width: 180%;
    height: 180%;
    top: -40%;
    left: -40%;
    background: radial-gradient(ellipse at 20% 20%, rgba(56, 189, 248, 0.32) 0%, transparent 55%);
    mix-blend-mode: screen;
    filter: blur(10px);
    opacity: 0.7;
    animation: aurora-flow 40s infinite alternate ease-in-out;
}

.aurora-layer:nth-child(2) {
    background: radial-gradient(ellipse at 80% 30%, rgba(45, 212, 191, 0.28) 0%, transparent 55%);
    animation-duration: 55s;
    animation-delay: -8s;
}
.aurora-layer:nth-child(3) {
    background: radial-gradient(ellipse at 30% 80%, rgba(244, 114, 182, 0.26) 0%, transparent 55%);
    animation-duration: 70s;
    animation-delay: -16s;
}
.aurora-layer:nth-child(4) {
    background: radial-gradient(ellipse at 70% 80%, rgba(129, 140, 248, 0.28) 0%, transparent 55%);
    animation-duration: 90s;
    animation-delay: -24s;
}
.aurora-layer:nth-child(5) {
    background: radial-gradient(ellipse at 50% 50%, rgba(52, 211, 153, 0.30) 0%, transparent 55%);
    animation-duration: 110s;
    animation-delay: -32s;
}
@keyframes aurora-flow {
    0% {
        transform: translate3d(-10%, -5%, 0) scale(1) rotate(0deg);
        opacity: 0.5;
    }
    25% {
        transform: translate3d(5%, -20%, 0) scale(1.1) rotate(8deg);
        opacity: 0.85;
    }
    50% {
        transform: translate3d(20%, 0%, 0) scale(1.2) rotate(-6deg);
        opacity: 1;
    }
    75% {
        transform: translate3d(-5%, 18%, 0) scale(1.1) rotate(4deg);
        opacity: 0.8;
    }
    100% {
        transform: translate3d(-18%, 0%, 0) scale(1.0) rotate(-10deg);
        opacity: 0.6;
    }
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
    padding: 1.6rem 2.2rem 2.6rem;   /* padding bawah diperbesar */
    border: 2px dashed #22c55e;
    text-align: center;
    box-shadow: 0 12px 30px rgba(34, 197, 94, 0.35);
    margin-bottom: 1.4rem;
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

.top-bar {
    width: 100%;
    padding: 0.5rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(240, 253, 250, 0.96);
    box-shadow: 0 10px 25px rgba(15, 118, 110, 0.15);
    border: 1px solid rgba(45, 212, 191, 0.55);
    margin-bottom: 0.9rem;
    border-radius: 0 0 18px 18px;
}
.stFileUploader > div:first-child {
    padding: 0;
    background: transparent;
    border: none;
}
.stFileUploader label {
    display: none;
}
</style>
"""


# --------------------------- PAGE CONFIG & GLOBAL CSS ---------------------------
st.set_page_config(
    page_title="Digital Payment Usage & Financial Discipline Survey",
    layout="wide",
)

# Aurora background container
if st.session_state["aurora_mode"]:
    st.markdown(
        """
        <div class="aurora-container">
            <div class="aurora-layer"></div>
            <div class="aurora-layer"></div>
            <div class="aurora-layer"></div>
            <div class="aurora-layer"></div>
            <div class="aurora-layer"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Apply CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------- TOP BAR ---------------------------
st.markdown('<div class="top-bar">', unsafe_allow_html=True)
col_title, col_dm, col_am, col_lang = st.columns([4, 1, 1, 2])

with col_title:
    st.markdown(
        f"<div style='font-weight:650; color:#047857; font-size:1.2rem;'>{get_text('title')}</div>",
        unsafe_allow_html=True,
    )

with col_dm:
    st.toggle(
        "🌙 Dark mode",
        value=st.session_state["dark_mode"],
        key="dark_mode_toggle",
        label_visibility="collapsed",
        on_change=update_dark_mode,
    )

with col_am:
    st.toggle(
        "🌌 Aurora",
        value=st.session_state["aurora_mode"],
        key="aurora_mode_toggle",
        label_visibility="collapsed",
        on_change=update_aurora_mode,
    )

with col_lang:
    st.radio(
        "🌏 Language",
        options=["EN", "ID", "JP", "KR", "CN", "AR", "PT", "FR"],
        horizontal=True,
        index=["EN", "ID", "JP", "KR", "CN", "AR", "PT", "FR"].index(
            st.session_state["language"]
        ),
        key="language_radio",
        label_visibility="collapsed",
        on_change=update_language,
    )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- DARK MODE OVERRIDES ---------------------------
if st.session_state["dark_mode"]:
    st.markdown(
        """
        <style>
        .main-card, .hero-card, .upload-card, .section-card {
            background-color: rgba(15, 23, 42, 0.96) !important;
            color: #e5e7eb !important;
        }
        .helper-text {
            color: #a7f3d0 !important;
        }
        .top-bar {
            background: rgba(15, 23, 42, 0.96) !important;
            border-color: rgba(45, 212, 191, 0.55) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------- GLOBAL UI SETTINGS ---------------------------
content_font_size = "0.95rem"  # font-size untuk teks upload dan helper

# --------------------------- GROUP MEMBERS SECTION ---------------------------
st.markdown(
    f"""
    <div class='section-card' style="margin-top:0.4rem; margin-bottom:0.4rem;">
      <p style="margin:0.1rem 0; color:#065f46; font-size:0.9rem; white-space:pre-line;">
        {get_text("group_info")}
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- UPLOAD & PREVIEW + FILTER ---------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Open section-card + upload-card
st.markdown(
    f"""
    <div class='section-card'>
      <p class='section-title'>{get_text("upload_subheader")}</p>
      <p class='section-subtitle'>{get_text("subtitle")}</p>
      <div class='upload-card' style="margin-top:0.6rem;">
        <p style='font-weight:600; margin-bottom:0.2rem;'>📤</p>
        <p style='margin-bottom:0.1rem; font-size:{content_font_size};'>
          {get_text('upload_label')}
        </p>
        <p class='helper-text'>{get_text("upload_limit")}</p>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload survey file",              # hanya label internal
    type=["csv", "xls", "xlsx"],
    label_visibility="collapsed",
    accept_multiple_files=False,
    key="upload_box_internal",
)

# Close after upload-card and section-card
st.markdown(
    """
      </div>  <!-- end .upload-card -->
    </div>    <!-- end .section-card -->
    """,
    unsafe_allow_html=True,
)

# ================== LOAD & FILTER DATA ==================
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


df = load_data(uploaded)
if df is None:
    st.info(get_text("no_file"))
    st.markdown("</div>", unsafe_allow_html=True)  # tutup main-card
    st.stop()

filter_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
filtered_df = df
if filter_cols:
    st.markdown(f"##### {get_text('filter_data_optional')}")
    fcol = st.selectbox(
        get_text("filter_column"),
        [get_text("no_filter")] + filter_cols,
        index=0,
        key="filter_column",
    )
    if fcol != get_text("no_filter"):
        unique_vals = df[fcol].dropna().unique().tolist()
        selected_vals = st.multiselect(
            get_text("select_values"),
            options=unique_vals,
            default=unique_vals,
        )
        if selected_vals:
            filtered_df = df[df[fcol].isin(selected_vals)]

st.markdown(f"#### {get_text('data_preview')}")
df_preview = filtered_df.head(1000)
st.dataframe(df_preview, height=400)

n_rows, n_cols = filtered_df.shape
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = filtered_df.select_dtypes(exclude=[np.number]).columns.tolist()
text_cols = filtered_df.select_dtypes(include=["object", "string"]).columns.tolist()

st.markdown(
    f"""
    <div class='section-card'>
      <p class='section-title'>{get_text("data_preview")}</p>
      <p class='section-subtitle'>{get_text("data_preview_subtitle")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- HELPER FUNCTIONS ---------------------------
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

    # TITLE + META
    story.append(Paragraph(get_text("pdf_title"), title_style))
    meta_lines = [
        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}",
        f"Numeric columns: {len(numeric_cols)}, Categorical columns: {len(cat_cols)}, Text columns: {len(text_cols)}",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, normal_style))
    story.append(Spacer(1, 0.2 * inch))

    # SUMMARY
    story.append(Paragraph(get_text("pdf_section_summary"), h2_style))
    story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph(get_text("pdf_summary_overall"), h3_style))
    overall_text = (
        f"Total responses: {df.shape[0]} | "
        f"Numeric columns: {len(numeric_cols)} | "
        f"Categorical columns: {len(cat_cols)} | "
        f"Text columns: {len(text_cols)}"
    )
    story.append(Paragraph(overall_text, normal_style))

    missing_info = df.isna().sum()
    mv_rows = [["Column", "Missing", "Percent"]]
    for col in df.columns:
        miss = int(missing_info[col])
        pct = (miss / len(df) * 100) if len(df) > 0 else 0
        mv_rows.append([col, str(miss), f"{pct:.2f}%"])
    mv_tbl = make_table(mv_rows, col_widths=[2.5 * inch, 1.2 * inch, 1.2 * inch], font_size=7)
    if mv_tbl:
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph(get_text("pdf_summary_missing"), h3_style))
        story.append(mv_tbl)

    story.append(Spacer(1, 0.2 * inch))

    # 1. DESCRIPTIVE NUMERIC
    if numeric_cols:
        story.append(Paragraph(get_text("pdf_section_numdesc"), h2_style))
        story.append(Spacer(1, 0.05 * inch))

        desc = df[numeric_cols].apply(pd.to_numeric, errors="coerce").describe().T
        desc_rows = [["Column", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]]
        for col in desc.index:
            row = desc.loc[col]
            desc_rows.append([
                col,
                f"{row['count']:.0f}",
                f"{row['mean']:.3f}",
                f"{row['std']:.3f}",
                f"{row['min']:.3f}",
                f"{row['25%']:.3f}",
                f"{row['50%']:.3f}",
                f"{row['75%']:.3f}",
                f"{row['max']:.3f}",
            ])
        desc_tbl = make_table(desc_rows, font_size=6.5)
        if desc_tbl:
            story.append(desc_tbl)
            story.append(Spacer(1, 0.2 * inch))

    # 1b. NUMERIC DISTRIBUTIONS
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

    # 2. SCATTER PLOTS
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

    # 3. CATEGORICAL BAR CHARTS
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

    # 4. NUMERIC FULL STATS
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

    # 5. CATEGORICAL FREQUENCY
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

    # 5b. CATEGORICAL DETAIL (CROSSTAB + CHI-SQUARE)
    last_ctab = None
    if len(cat_cols) >= 2:
        story.append(PageBreak())
        story.append(Paragraph(get_text("pdf_section_catdetail"), h2_style))
        story.append(Spacer(1, 0.1 * inch))

        max_pairs = min(3, len(cat_cols) - 1)
        for i in range(max_pairs):
            col_a = cat_cols[i]
            col_b = cat_cols[i + 1]
            story.append(Paragraph(f"<b>{col_a}</b> x <b>{col_b}</b>", h3_style))

            ctab = pd.crosstab(df[col_a], df[col_b])
            if ctab.empty:
                story.append(Paragraph(get_text("pdf_catdetail_nodata"), small_style))
                story.append(Spacer(1, 0.1 * inch))
                continue

            last_ctab = ctab.copy()

            ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100

            rows = [[""] + list(ctab.columns)]
            for idx in ctab.index[:10]:
                row = [str(idx)]
                for c in ctab.columns:
                    row.append(f"{ctab.loc[idx, c]} ({ctab_pct.loc[idx, c]:.1f}%)")
                rows.append(row)
            tbl = make_table(rows, font_size=6.5)
            if tbl:
                story.append(tbl)

            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            ctab_pct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
            ax.set_title(f"{col_a} vs {col_b} (%)", fontsize=10, fontweight="bold")
            ax.set_xlabel(col_a)
            ax.set_ylabel("Percent")
            ax.legend(fontsize=6)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            img = fig_to_image(fig, width=5.5, height=2.8)
            story.append(Spacer(1, 0.05 * inch))
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))

            # chi-square untuk pasangan ini
            if ctab.shape[0] > 1 and ctab.shape[1] > 1:
                chi2, p, dof, _ = stats.chi2_contingency(ctab)
                chi_text = f"Chi-square: {chi2:.3f}, df={dof}, p-value={p:.4f}"
                story.append(Paragraph(chi_text, small_style))
                story.append(Spacer(1, 0.1 * inch))

    # 6. CORRELATION MATRIX + DETAIL
    corr_pairs = []
    top_pairs = []
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

        # detail korelasi
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(get_text("pdf_section_corrdetail"), h3_style))

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                a, b = numeric_cols[i], numeric_cols[j]
                r = corr_matrix.loc[a, b]
                corr_pairs.append((abs(r), a, b, r))
        corr_pairs.sort(reverse=True)
        top_pairs = corr_pairs[:5]

        corr_rows = [["Var A", "Var B", "r", "p-value", "N"]]
        for _, a, b, r in top_pairs:
            x = pd.to_numeric(df[a], errors="coerce")
            y = pd.to_numeric(df[b], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 3:
                r_val, p_val = stats.pearsonr(x[mask], y[mask])
                corr_rows.append([a, b, f"{r_val:.3f}", f"{p_val:.4f}", str(mask.sum())])
        corr_tbl = make_table(corr_rows, font_size=7)
        if corr_tbl:
            story.append(corr_tbl)
            story.append(Spacer(1, 0.2 * inch))

    # 7. TEXT ANALYSIS
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

            lengths = df[col].dropna().astype(str).str.len()
            if not lengths.empty:
                len_stats = {
                    "Min length": lengths.min(),
                    "Max length": lengths.max(),
                    "Mean length": lengths.mean(),
                    "Median length": lengths.median(),
                }
                len_rows = [["Metric", "Value"]] + [
                    [k, f"{v:.1f}" if isinstance(v, float) else str(v)]
                    for k, v in len_stats.items()
                ]
                len_tbl = make_table(len_rows, col_widths=[2.5 * inch, 2 * inch], font_size=8)
                if len_tbl:
                    story.append(Spacer(1, 0.05 * inch))
                    story.append(len_tbl)

            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(get_text("pdf_text_samples"), small_style))
            examples = df[col].dropna().astype(str).head(5).tolist()
            for idx, ex in enumerate(examples, 1):
                story.append(Paragraph(f"{idx}. {ex}", small_style))
            story.append(Spacer(1, 0.2 * inch))

    # 8. INSIGHTS & HIGHLIGHTS
    story.append(PageBreak())
    story.append(Paragraph(get_text("pdf_section_insights"), h2_style))
    story.append(Spacer(1, 0.1 * inch))

    bullets = []

    if numeric_cols:
        for col in numeric_cols[:3]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                bullets.append(
                    f"{col}: mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}, range=({s.min():.2f}–{s.max():.2f})"
                )

    for col in cat_cols[:3]:
        top = df[col].value_counts(normalize=True).head(3)
        if not top.empty:
            parts = [f"{idx} ({pct*100:.1f}%)" for idx, pct in top.items()]
            bullets.append(f"{col}: top categories → " + ", ".join(parts))

    if len(numeric_cols) > 1 and top_pairs:
        for _, a, b, r in top_pairs[:3]:
            bullets.append(f"Strong correlation between {a} and {b}: r={r:.3f}")

    if not bullets:
        bullets.append(get_text("pdf_insight_none"))

    for b in bullets:
        story.append(Paragraph(f"• {b}", normal_style))

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

# --------------------------- DATA OVERVIEW ---------------------------
st.markdown(
    f"""
    <div class='section-card'>
      <p class='section-title'>Data Overview</p>
      <p class='section-subtitle'>Key metrics and summary</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div class='summary-badge'>
          <span class='summary-dot'></span> {n_rows} Rows
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class='summary-badge'>
          <span class='summary-dot'></span> {len(cat_cols)} Categorical/Text
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class='summary-badge'>
          <span class='summary-dot'></span> {n_cols} Columns
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class='summary-badge'>
          <span class='summary-dot'></span> {len(numeric_cols)} Numeric
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("**Summary:**")
st.markdown(f"- Total rows: {n_rows}")
st.markdown(f"- Total columns: {n_cols}")
st.markdown(f"- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}")
st.markdown(f"- Categorical/Text columns: {', '.join(cat_cols) if cat_cols else 'None'}")

# --------------------------- DESCRIPTIVE STATISTICS ---------------------------
st.markdown(f"### {get_text('stats_subheader')}")
with st.container():
    if not numeric_cols:
        st.warning(get_text("no_numeric_cols"))
    else:
        with st.expander(get_text("summary_normality"), expanded=True):
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
                st.markdown(f"**{get_text('normality_test')}**")
                st.write(f"{get_text('statistic_label')}: {stat:.4f}")
                st.write(f"{get_text('p_value_label')}: {p_norm:.4f}")
                if p_norm < 0.05:
                    st.info(get_text("deviate_normal"))
                else:
                    st.success(get_text("no_deviate_normal"))
            else:
                st.info(get_text("not_enough_normality"))

        with st.expander(get_text("distribution"), expanded=False):
            num_col2 = st.selectbox(
                get_text("select_column_distribution"),
                options=numeric_cols,
                index=0,
                key="desc_num_dist",
            )
            visualize_data(filtered_df, num_col2)

    if not cat_cols:
        st.info(get_text("no_categorical_cols"))
    else:
        with st.expander(get_text("freq_table_subheader"), expanded=False):
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
            st.table(freq_df)

# --------------------------- VISUALIZATIONS ---------------------------
st.markdown(f"### {get_text('visual_subheader')}")
with st.container():
    if not numeric_cols:
        st.warning(get_text("no_numeric_cols"))
    else:
        with st.expander("Histogram / Boxplot", expanded=True):
            num_col = st.selectbox(
                get_text("select_numeric_col"),
                options=numeric_cols,
                help="Column for visualization",
                key="visual_num",
            )
            st.markdown(f"### {get_text('visual_subheader')}")
            visualize_data(filtered_df, num_col)

        with st.expander("Scatter & Bar", expanded=False):
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

# --------------------------- CORRELATIONS & TESTS ---------------------------
st.markdown(f"### {get_text('correlation_subheader')}")
with st.container():
    with st.expander(get_text("pearson_header"), expanded=True):
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

    with st.expander(get_text("spearman_header"), expanded=False):
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

    with st.expander(get_text("chi_header"), expanded=False):
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

# --------------------------- TEXT PROCESSING ---------------------------
st.markdown("### Text Processing")
with st.container():
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
            all_tokens = [t for row in processed for t in row]
            total_words = len(all_tokens)
            unique_words = len(set(all_tokens))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Words", total_words)
            with col2:
                st.metric("Unique Words", unique_words)
            word_freq = Counter(all_tokens)
            top10 = word_freq.most_common(10)
            if top10:
                top_df = pd.DataFrame(top10, columns=["word", "count"])
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x="count", y="word", data=top_df, ax=ax, color="#22c55e")
                ax.set_title("Top 10 Words by Frequency")
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Word")
                st.pyplot(fig)
            with st.expander("Advanced", expanded=False):
                st.markdown(f"**{get_text('sample_tokens')}**")
                st.write(processed.head(5).tolist())

# --------------------------- INSIGHTS & HIGHLIGHTS ---------------------------
st.markdown(
    f"""
    <div class='section-card'>
      <p class='section-title'>Insights & Highlights</p>
      <p class='section-subtitle'>Key findings from the analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

insights = []
insights.append(f"- Total records analyzed: {n_rows}")
insights.append(f"- Total variables: {n_cols}")
if numeric_cols:
    insights.append(f"- Numeric variables: {len(numeric_cols)} ({', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''})")
if cat_cols:
    insights.append(f"- Categorical/Text variables: {len(cat_cols)} ({', '.join(cat_cols[:3])}{'...' if len(cat_cols) > 3 else ''})")
if text_cols:
    insights.append(f"- Text columns available for analysis: {len(text_cols)}")
insights.append("- Data processed locally for privacy")

for insight in insights:
    st.markdown(insight)

# --------------------------- EXPORT PDF SECTION ---------------------------
st.markdown(f"### {get_text('export_title')}")
st.markdown(get_text("export_desc"))
generate_pdf_button(filtered_df, numeric_cols, cat_cols, text_cols)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- FOOTER ---------------------------
st.markdown(
    """
    <div style='text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(240, 253, 250, 0.94); border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.35);'>
      <p style='margin: 0; color: #047857; font-weight: 600;'>👥 Group 5 Class 2</p>
      <p style='margin: 0; color: #047857;'>Version 1.0</p>
      <p style='margin: 0; color: #065f46; font-size: 0.9rem;'>Privacy: Data is processed locally and not stored on servers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
