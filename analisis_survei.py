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
import time  # For loading spinner
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# Initialize NLTK resources (only stopwords, without punkt)
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
st.set_page_config(page_title="Survey Data Analyzer", layout="wide")

# Top bar: Dark mode + language selector
top_col1, top_col2 = st.columns([3, 3])
with top_col1:
    dm = st.toggle("🌙 Dark mode", value=st.session_state["dark_mode"])
    st.session_state["dark_mode"] = dm
with top_col2:
    lang = st.radio(
        "Language",
        options=["EN", "ID", "JP", "KR", "CN"],
        horizontal=True,
        index=["EN", "ID", "JP", "KR", "CN"].index(st.session_state["language"]),
    )
    st.session_state["language"] = lang

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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Extra CSS for dark mode
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
    },
    "ID": {
        "title": "📊 Analisis Data Survei",
        "subtitle": "Unggah file survei (CSV/Excel) dan jelajahi statistik deskriptif, visualisasi, serta uji korelasi secara interaktif.",
        "upload_subheader": "📁 Unggah Data Survei",
        "upload_label": "Tarik & letakkan file di sini atau klik untuk memilih (CSV, XLS, XLSX)",
        "no_file": "Belum ada file yang diunggah. Silakan unggah file untuk mulai analisis.",
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
    },
    "JP": {
        "title": "📊 アンケートデータ分析",
        "subtitle": "アンケートファイル（CSV/Excel）をアップロードして、記述統計・可視化・相関分析をインタラクティブに行います。",
        "upload_subheader": "📁 アンケートデータのアップロード",
        "upload_label": "ここにドラッグ＆ドロップするかクリックしてファイルを選択（CSV, XLS, XLSX）",
        "no_file": "まだファイルがアップロードされていません。分析を開始するにはファイルをアップロードしてください。",
        "data_preview": "データプレビュー（先頭 1000 行まで）",
        "text_processing_subheader": "📝 テキスト前処理",
        "text_columns_detected": "テキスト列が検出されました:",
        "select_text_col": "処理するテキスト列を選択してください",
        "no_text_columns": "テキスト型の列は検出されませんでした。",
        "text_processing_note": "テキストは小文字化され、句読点が削除され、スペースで分割され、英語のストップワードが除去されます。",
        "sample_tokens": "前処理されたトークンのサンプル",
        "top_words": "出現頻度トップ10の単語",
        "stats_subheader": "📈 記述統計と分布",
        "select_numeric_col": "統計とグラフ用の数値列を選択してください",
        "no_numeric_cols": "利用できる数値列がありません。",
        "desc_stats": "選択した列の記述統計",
        "freq_table_subheader": "📊 カテゴリ別度数表",
        "select_categorical_col": "度数表を作成するカテゴリ列を選択してください",
        "no_categorical_cols": "利用できるカテゴリ列がありません。",
        "freq_count": "件数",
        "freq_percent": "割合 (%)",
        "visual_subheader": "📉 データの可視化",
        "histogram": "ヒストグラム",
        "boxplot": "ボックスプロット",
        "correlation_subheader": "🔗 相関と統計的検定",
        "pearson_header": "ピアソン相関",
        "spearman_header": "スピアマン相関",
        "chi_header": "カイ二乗検定",
        "select_x_numeric": "X 変数（数値）を選択",
        "select_y_numeric": "Y 変数（数値）を選択",
        "not_enough_numeric": "この分析を行うのに十分な数値列がありません。",
        "pearson_result": "ピアソン相関の結果",
        "spearman_result": "スピアマン相関の結果",
        "corr_coef": "相関係数 (r)",
        "p_value": "p 値",
        "interpretation": "解釈",
        "select_x_cat": "X 変数（カテゴリ）を選択",
        "select_y_cat": "Y 変数（カテゴリ）を選択",
        "not_enough_categorical": "カイ二乗検定を行うのに十分なカテゴリ列がありません。",
        "chi_square_result": "カイ二乗検定の結果",
        "chi_square_stat": "カイ二乗統計量",
        "chi_square_df": "自由度 (df)",
        "chi_square_p": "p 値",
        "alpha_note": "有意水準 α = 0.05 で検定しています。",
        "significant_assoc": "2 つの変数の間には統計的に有意な関連があります。",
        "no_significant_assoc": "2 つの変数の間に統計的に有意な関連は認められません。",
        "corr_direction_positive": "正の関係：X が増加すると Y も増加する傾向があります。",
        "corr_direction_negative": "負の関係：X が増加すると Y は減少する傾向があります。",
        "corr_direction_zero": "明確な関係は見られません（相関係数は 0 に近い）。",
        "corr_strength_none": "ほとんど関係がありません。",
        "corr_strength_weak": "弱い関係です。",
        "corr_strength_moderate": "中程度の関係です。",
        "corr_strength_strong": "強い関係です。",
        "warning_select_valid": "有効な列の組み合わせを選択してください。",
        "header_github": "GitHub でフォーク",
        "nav_desc": "記述統計",
        "nav_visual": "可視化",
        "nav_corr": "相関・検定",
        "nav_text": "テキスト処理",
        "export_title": "レポートのエクスポート",
        "export_desc": "記述統計、正規性検定、ヒストグラム、ボックスプロット、相関、テキスト分析サマリーを含む完全な PDF レポートを生成します。",
        "export_button": "PDF レポートを作成",
        "export_filename": "survey_full_report_ja.pdf",
    },
    "KR": {
        "title": "📊 설문 데이터 분석",
        "subtitle": "설문 파일(CSV/Excel)을 업로드하고 기술통계, 시각화, 상관분석을 인터랙티브하게 확인하세요.",
        "upload_subheader": "📁 설문 데이터 업로드",
        "upload_label": "여기에 드래그하여 놓거나 클릭해서 파일 선택 (CSV, XLS, XLSX)",
        "no_file": "아직 업로드된 파일이 없습니다. 분석을 시작하려면 파일을 업로드하세요.",
        "data_preview": "데이터 미리보기 (최대 상위 1000행)",
        "text_processing_subheader": "📝 텍스트 전처리",
        "text_columns_detected": "감지된 텍스트 열:",
        "select_text_col": "전처리할 텍스트 열을 선택하세요",
        "no_text_columns": "텍스트 형식의 열이 없습니다.",
        "text_processing_note": "텍스트를 소문자로 변환하고, 구두점을 제거하며, 공백 기준으로 토큰화한 뒤 영어 불용어를 제거합니다.",
        "sample_tokens": "전처리된 토큰 예시",
        "top_words": "출현 빈도 상위 10개 단어",
        "stats_subheader": "📈 기술통계 및 분포",
        "select_numeric_col": "통계 및 그래프용 숫자 열을 선택하세요",
        "no_numeric_cols": "사용 가능한 숫자 열이 없습니다.",
        "desc_stats": "선택한 열의 기술통계",
        "freq_table_subheader": "📊 범주형 빈도표",
        "select_categorical_col": "빈도표를 생성할 범주형 열을 선택하세요",
        "no_categorical_cols": "사용 가능한 범주형 열이 없습니다.",
        "freq_count": "빈도",
        "freq_percent": "비율 (%)",
        "visual_subheader": "📉 데이터 시각화",
        "histogram": "히스토그램",
        "boxplot": "박스플롯",
        "correlation_subheader": "🔗 상관관계 및 통계 검정",
        "pearson_header": "피어슨 상관계수",
        "spearman_header": "스피어만 상관계수",
        "chi_header": "카이제곱 검정",
        "select_x_numeric": "X 변수(숫자)를 선택",
        "select_y_numeric": "Y 변수(숫자)를 선택",
        "not_enough_numeric": "이 분석을 수행하기에 숫자 열이 충분하지 않습니다.",
        "pearson_result": "피어슨 상관분석 결과",
        "spearman_result": "스피어만 상관분석 결과",
        "corr_coef": "상관계수 (r)",
        "p_value": "p-값",
        "interpretation": "해석",
        "select_x_cat": "X 변수(범주형)를 선택",
        "select_y_cat": "Y 변수(범주형)를 선택",
        "not_enough_categorical": "카이제곱 검정을 수행하기에 범주형 열이 충분하지 않습니다.",
        "chi_square_result": "카이제곱 검정 결과",
        "chi_square_stat": "카이제곱 통계량",
        "chi_square_df": "자유도 (df)",
        "chi_square_p": "p-값",
        "alpha_note": "유의수준 α = 0.05에서 검정합니다.",
        "significant_assoc": "두 변수 사이에 통계적으로 유의한 관계가 있습니다.",
        "no_significant_assoc": "두 변수 사이에 통계적으로 유의한 관계가 없습니다.",
        "corr_direction_positive": "양의 관계: X가 증가할수록 Y도 증가하는 경향이 있습니다.",
        "corr_direction_negative": "음의 관계: X가 증가할수록 Y는 감소하는 경향이 있습니다.",
        "corr_direction_zero": "뚜렷한 관계가 보이지 않습니다 (상관계수가 0에 가까움).",
        "corr_strength_none": "거의 관계가 없습니다.",
        "corr_strength_weak": "약한 관계입니다.",
        "corr_strength_moderate": "중간 정도의 관계입니다.",
        "corr_strength_strong": "강한 관계입니다.",
        "warning_select_valid": "유효한 열 조합을 선택하세요.",
        "header_github": "GitHub에서 포크",
        "nav_desc": "기술통계",
        "nav_visual": "시각화",
        "nav_corr": "상관/검정",
        "nav_text": "텍스트 전처리",
        "export_title": "보고서 내보내기",
        "export_desc": "기술통계, 정규성 검정, 히스토그램, 박스플롯, 상관관계, 텍스트 분석 요약을 포함한 전체 PDF 보고서를 생성합니다.",
        "export_button": "PDF 보고서 생성",
        "export_filename": "survey_full_report_ko.pdf",
    },
    "CN": {
        "title": "📊 问卷数据分析",
        "subtitle": "上传问卷文件（CSV/Excel），交互式地查看描述性统计、可视化图表和相关性检验结果。",
        "upload_subheader": "📁 上传问卷数据",
        "upload_label": "将文件拖到此处或点击选择文件（CSV、XLS、XLSX）",
        "no_file": "尚未上传文件。请先上传文件以开始分析。",
        "data_preview": "数据预览（最多前 1000 行）",
        "text_processing_subheader": "📝 文本预处理",
        "text_columns_detected": "检测到的文本列：",
        "select_text_col": "请选择要处理的文本列",
        "no_text_columns": "未检测到文本类型的列。",
        "text_processing_note": "文本将转换为小写，移除标点符号，按空格分词，并删除英文停用词。",
        "sample_tokens": "预处理后词元示例",
        "top_words": "词频前 10 的单词",
        "stats_subheader": "📈 描述性统计与分布",
        "select_numeric_col": "请选择用于统计与绘图的数值列",
        "no_numeric_cols": "没有可用的数值列。",
        "desc_stats": "所选列的描述性统计",
        "freq_table_subheader": "📊 分类频数表",
        "select_categorical_col": "请选择要生成频数表的分类列",
        "no_categorical_cols": "没有可用的分类列。",
        "freq_count": "频数",
        "freq_percent": "百分比 (%)",
        "visual_subheader": "📉 数据可视化",
        "histogram": "直方图",
        "boxplot": "箱线图",
        "correlation_subheader": "🔗 相关性与统计检验",
        "pearson_header": "皮尔逊相关",
        "spearman_header": "斯皮尔曼相关",
        "chi_header": "卡方检验",
        "select_x_numeric": "选择 X 变量（数值型）",
        "select_y_numeric": "选择 Y 变量（数值型）",
        "not_enough_numeric": "可用于该分析的数值列数量不足。",
        "pearson_result": "皮尔逊相关结果",
        "spearman_result": "斯皮尔曼相关结果",
        "corr_coef": "相关系数 (r)",
        "p_value": "p 值",
        "interpretation": "解读",
        "select_x_cat": "选择 X 变量（分类变量）",
        "select_y_cat": "选择 Y 变量（分类变量）",
        "not_enough_categorical": "可用于卡方检验的分类列数量不足。",
        "chi_square_result": "卡方检验结果",
        "chi_square_stat": "卡方统计量",
        "chi_square_df": "自由度 (df)",
        "chi_square_p": "p 值",
        "alpha_note": "在显著性水平 α = 0.05 下进行检验。",
        "significant_assoc": "两个变量之间存在统计学上显著的关联。",
        "no_significant_assoc": "两个变量之间不存在统计学上显著的关联。",
        "corr_direction_positive": "正相关：X 增加时，Y 一般也随之增加。",
        "corr_direction_negative": "负相关：X 增加时，Y 一般随之减少。",
        "corr_direction_zero": "未观察到明显关系（相关系数接近 0）。",
        "corr_strength_none": "几乎没有关系。",
        "corr_strength_weak": "相关关系较弱。",
        "corr_strength_moderate": "相关关系中等。",
        "corr_strength_strong": "相关关系较强。",
        "warning_select_valid": "请选择有效的列组合。",
        "header_github": "在 GitHub 上 Fork",
        "nav_desc": "描述性统计",
        "nav_visual": "可视化",
        "nav_corr": "相关/检验",
        "nav_text": "文本处理",
        "export_title": "导出报告",
        "export_desc": "生成完整 PDF 报告，包含描述性统计、正态性检验、直方图、箱线图、相关分析及文本分析摘要。",
        "export_button": "生成 PDF 报告",
        "export_filename": "survey_full_report_zh.pdf",
    },
}

def get_text(key: str) -> str:
    lang = st.session_state.get("language", "EN")
    base = TEXTS.get(lang, TEXTS["EN"])
    return base.get(key, TEXTS["EN"].get(key, key))

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
    with st.spinner('Generating visualizations...'):
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
    if r > 0.05:
        direction = get_text("corr_direction_positive")
    elif r < -0.05:
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

# ----------- PDF REPORT FULL EXPORT -----------
def build_survey_report_pdf(df, numeric_cols, cat_cols, text_cols):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from io import BytesIO
    import matplotlib.pyplot as plt
    import seaborn as sns

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 36
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, y, "Survey Data Full Report")
    y -= 30

    # Metadata
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    y -= 18
    c.drawString(margin, y, f"Numeric columns: {len(numeric_cols)} | Categorical columns: {len(cat_cols)} | Text columns: {len(text_cols)}")
    y -= 25

    # --- Descriptive Stats, Normality & Histogram ---
    for col in numeric_cols:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, f"[NUMERIC] {col}")
        y -= 15
        stats_strs = []
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        stats_strs += [f"Mean: {s.mean():.4f}", f"Median: {s.median():.4f}", f"Std: {s.std():.4f}"]
        stats_strs += [f"Min: {s.min():.4f}", f"Max: {s.max():.4f}", f"Mode: {s.mode().iloc[0] if not s.mode().empty else 'NA'}"]
        for stt in stats_strs:
            c.setFont("Helvetica", 10)
            c.drawString(margin+12, y, stt)
            y -= 13
        # Normality test
        if len(s) >= 8:
            stat, p_norm = normaltest(s)
            c.setFont("Helvetica", 10)
            c.drawString(margin+12, y, f"Normality (D’Agostino): stat={stat:.3f}, p-value={p_norm:.3f}, {'NORMAL' if p_norm>0.05 else 'NOT normal'}")
            y -= 14
        else:
            c.setFont("Helvetica", 10)
            c.drawString(margin+12, y, "Normality: Not enough data (min 8 values needed)")
            y -= 14

        # Histogram
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        sns.histplot(s, kde=True, ax=ax, color="#16a34a")
        ax.set_title(f"{col} Histogram")
        img_hist = BytesIO()
        plt.tight_layout()
        plt.savefig(img_hist, format='png')
        plt.close(fig)
        img_hist.seek(0)

        if y < 130: c.showPage(); y = height - margin
        c.drawImage(ImageReader(img_hist), margin+5, y-100, width=210, height=95)
        y -= 110

        # Boxplot
        fig, ax = plt.subplots(figsize=(2.6,1.2))
        sns.boxplot(x=s, ax=ax, color="#22c55e")
        ax.set_title(f"{col} Boxplot", fontsize=9)
        img_box = BytesIO()
        plt.tight_layout()
        plt.savefig(img_box, format='png')
        plt.close(fig)
        img_box.seek(0)

        if y < 100: c.showPage(); y = height - margin
        c.drawImage(ImageReader(img_box), margin+260, y-48, width=120, height=36)
        y -= 25

        y -= 6

    # --- Visualizations: Scatter plot for all numeric pairs ---
    if len(numeric_cols) >= 2:
        import itertools
        pairs = list(itertools.combinations(numeric_cols, 2))
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Scatterplots (Numeric VS Numeric)")
        y -= 18
        for xcol, ycol in pairs:
            if y < 170: c.showPage(); y = height - margin
            s_x = pd.to_numeric(df[xcol], errors="coerce")
            s_y = pd.to_numeric(df[ycol], errors="coerce")
            mask = s_x.notna() & s_y.notna()
            if mask.sum() < 10:
                continue
            fig, ax = plt.subplots(figsize=(3.1, 2.1))
            ax.scatter(s_x[mask], s_y[mask], alpha=0.5, color="#0f766e", s=10)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.setTitle = f"{xcol} vs {ycol}"
            plt.tight_layout()
            img_sc = BytesIO()
            plt.savefig(img_sc, format='png')
            plt.close(fig)
            img_sc.seek(0)
            c.setFont("Helvetica", 9)
            c.drawString(margin+6, y, f"{xcol} ~ {ycol}")
            c.drawImage(ImageReader(img_sc), margin+65, y-65, width=130, height=65)
            y -= 70
        y -= 4

    # --- Correlation matrix ---
    if len(numeric_cols) >= 2:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Correlation Matrix (Pearson)")
        y -= 13
        corrm = df[numeric_cols].corr(method="pearson").round(3)
        colw = 60
        c.setFont("Helvetica", 9)
        c.setFillGray(0.92, 1)
        c.rect(margin, y-15, colw * (1+len(corrm.columns)), 14+14*len(corrm), fill=1, stroke=0)
        c.setFillGray(0,1)
        c.setFont("Helvetica-Bold", 9)
        for i,col in enumerate(corrm.columns):
            c.drawString(margin + colw + i*colw, y, f"{col[:6]}")
        for i,row in enumerate(corrm.itertuples()):
            c.setFont("Helvetica-Bold", 9)
            c.drawString(margin, y-12-14*i, str(corrm.index[i]))
            c.setFont("Helvetica", 9)
            for j,val in enumerate(row[1:]):
                c.drawString(margin + colw + j*colw, y-12-14*i, str(val))
        y -= (18 + 14*len(corrm))
    else:
        y -= 14

    # --- Categorical freq tables ---
    for catcol in cat_cols:
        c.setFont("Helvetica-Bold", 12)
        if y < 120: c.showPage(); y = height - margin
        c.drawString(margin, y, f"[CATEGORY] {catcol} - Top 10")
        y -= 14
        vc = df[catcol].value_counts(dropna=False).head(10)
        c.setFont("Helvetica", 10)
        for idx, (val, cnt) in enumerate(vc.items()):
            displ = str(val)[:25]
            c.drawString(margin+10, y, f"{displ:>10} : {cnt}")
            y -= 12
        y -= 6

    # --- Text processing summary ---
    if text_cols:
        for textcol in text_cols:
            txts = df[textcol].dropna().astype(str)
            eng_stop = set(stopwords.words("english"))
            punct_table = str.maketrans("", "", string.punctuation)
            tokens = []
            for text in txts:
                txt = text.lower().translate(punct_table)
                tokens += [t for t in txt.split() if t.isalpha() and t not in eng_stop]
            counter = Counter(tokens)
            top10 = counter.most_common(10)
            c.setFont("Helvetica-Bold", 11)
            if y < 80: c.showPage(); y = height - margin
            c.drawString(margin, y, f"Text Summary [{textcol}] Top Words")
            y -= 15
            c.setFont("Helvetica", 10)
            for w,cnt in top10:
                c.drawString(margin+9, y, f"{w:>12} : {cnt}")
                y -= 12
            y -= 8

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# --------------------------- HEADER + HERO + GROUP CARD ---------------------------
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
    {"name": "ADITYA ANGGARA PAMUNGKAS", "sid": "4202400051", "role": "Leader"},
    {"name": "MAULA AQIEL NURI",        "sid": "4202400023", "role": "Member"},
    {"name": "SYAFIQ NUR RAMADHAN",     "sid": "4202400073", "role": "Member"},
    {"name": "RIFAT FITROTU SALMAN",    "sid": "4202400106", "role": "Member"},
]

st.markdown(
    """
    <div class='hero-card' style="margin-top:0.6rem; margin-bottom:0.4rem;">
      <h4 style="margin-top:0; margin-bottom:0.4rem; color:#047857;">
        👥 Group 5: Digital Payment & Financial Discipline
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
st.markdown(f"### {get_text('upload_subheader')}")

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
        label="",
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

# Filter by categorical column (no rerun)
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
max_rows_preview = 1000
df_preview = filtered_df.head(max_rows_preview)
st.dataframe(df_preview, height=400, use_container_width=True)

n_rows, n_cols = filtered_df.shape
n_numeric = filtered_df.select_dtypes(include=[np.number]).shape[1]
n_cat = filtered_df.select_dtypes(exclude=[np.number]).shape[1]
st.markdown(
    f"""
    <div style="margin:0.6rem 0 1.0rem 0;">
      <span class="summary-badge">
        <span class="summary-dot"></span>
        {n_rows} rows
      </span>
      <span class="summary-badge">
        <span class="summary-dot"></span>
        {n_cols} columns
      </span>
      <span class="summary-badge">
        <span class="summary-dot"></span>
        {n_numeric} numeric • {n_cat} non-numeric
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = filtered_df.select_dtypes(exclude=[np.number]).columns.tolist()
text_cols = filtered_df.select_dtypes(include=["object", "string"]).columns.tolist()

# --------------------------- TABS FOR FEATURES ---------------------------
tab_desc, tab_vis, tab_corr, tab_text = st.tabs(
    [
        get_text("nav_desc"),
        get_text("nav_visual"),
        get_text("nav_corr"),
        get_text("nav_text"),
    ]
)

# Text processing tab
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

# Descriptive stats tab
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
            # Descriptive stats
            stats_df = descriptive_stats(filtered_df[num_col])
            st.markdown(f"**{get_text('desc_stats')}**")
            st.table(stats_df)

            # Normality test (D’Agostino-Pearson)
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

# Visualization tab
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
            # Scatter plot X vs Y
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

            # Bar chart for categorical column
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

# Correlation & tests tab
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
                try:
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
                except Exception:
                    st.warning(get_text("warning_select_valid"))

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
                try:
                    r_s, p_s = correlation_analysis(filtered_df, x_s, y_s, method="spearman")
                    if np.isnan(r_s):
                        st.warning(get_text("warning_select_valid"))
                    else:
                        st.markdown(
                            f"**{get_text('spearman_result')}**"
                        )
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
                except Exception:
                    st.warning(get_text("warning_select_valid"))

    with tab3:
        if len(cat_cols) < 2:
            st.info(get_text("not_enough_categorical"))
        else:
            c1c, c2c = st.columns(2)
            with c1c:
                x_cat = st.selectbox(
                    get_text("select_x_cat"),
                    options=cat_cols,
                    key="chi_x",
                )
            with c2c:
                y_cat = st.selectbox(
                    get_text("select_y_cat"),
                    options=[c for c in cat_cols if c != x_cat],
                    key="chi_y",
                )
            if x_cat and y_cat:
                try:
                    chi2, p_val, dof_val, expected_df = chi_square_test(
                        filtered_df, x_cat, y_cat
                    )
                    if chi2 is None:
                        st.warning(get_text("warning_select_valid"))
                    else:
                        st.markdown(
                            f"**{get_text('chi_square_result')}**"
                        )
                        out_c = pd.DataFrame(
                            {
                                get_text("chi_square_stat"): [chi2],
                                get_text("chi_square_df"): [dof_val],
                                get_text("chi_square_p"): [p_val],
                            }
                        )
                        st.table(out_c)

                        st.markdown("**Observed**")
                        observed_table = pd.crosstab(filtered_df[x_cat], filtered_df[y_cat])
                        st.dataframe(
                            observed_table, height=200, use_container_width=True
                        )

                        st.markdown("**Expected**")
                        st.dataframe(expected_df, height=200, use_container_width=True)

                        st.markdown(f"_{get_text('alpha_note')}_")
                        if p_val < 0.05:
                            st.success(get_text("significant_assoc"))
                        else:
                            st.info(get_text("no_significant_assoc"))
                except Exception:
                    st.warning(get_text("warning_select_valid"))

    st.markdown("#### Automatic correlation summary (numeric variables)")
    if len(numeric_cols) >= 2:
        corr_matrix = filtered_df[numeric_cols].corr(method="pearson")
        st.dataframe(corr_matrix, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute full correlation matrix.")

# --------------------------- EXPORT REPORT TO PDF ---------------------------
st.markdown(f"### {get_text('export_title')}")
st.write(get_text("export_desc"))

if st.button(get_text("export_button")):
    pdf_bytes = build_survey_report_pdf(filtered_df, numeric_cols, cat_cols, text_cols)
    st.download_button(
        label=get_text("export_button"),
        data=pdf_bytes,
        file_name=get_text("export_filename"),
        mime="application/pdf",
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown(f"[🐙 {get_text('header_github')}]()")
