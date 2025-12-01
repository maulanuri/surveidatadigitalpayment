import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats  # make sure scipy is installed: pip install scipy

st.set_page_config(page_title="Survey Analysis X and Y", layout="wide")

st.title("Digital Payment Survey Analysis: X and Y")

# =======================
# 1. Upload / read data
# =======================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload survei.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Please upload survei.csv from the sidebar.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head())

st.write("Available columns:")
st.write(list(df.columns))

# =======================
# 2. Select X and Y items
# =======================

st.markdown("### Select items for X and Y (Likert scale)")

likert_cols = [
    c for c in df.columns
    if "10." in c or "11." in c or "12." in c
    or "13." in c or "14." in c or "15." in c or "16." in c
    or "17." in c or "18." in c or "19." in c or "20." in c
    or "22." in c or "23." in c
]

# If the list above is empty, try to detect columns that contain " = "
if not likert_cols:
    likert_cols = [c for c in df.columns if "=" in str(df[c].iloc[0])]

cols_x = st.multiselect(
    "Select items for variable X (e.g., financial discipline statements)",
    options=likert_cols
)

cols_y = st.multiselect(
    "Select items for variable Y (e.g., digital payment behavior/consumption statements)",
    options=likert_cols
)

st.markdown("Note: Make sure the selected items are Likert-scale questions that can be converted to numeric values (1–5).")

# Helper function: convert text like "5 = Strongly agree" -> 5
def likert_to_num(df_sub):
    out = df_sub.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).str.extract(r"(\\d+)").astype(float)
    return out

# Create composite scores if columns are selected
if cols_x:
    x_numeric = likert_to_num(df[cols_x])
    df["X_total"] = x_numeric.sum(axis=1, min_count=1)
else:
    x_numeric = None

if cols_y:
    y_numeric = likert_to_num(df[cols_y])
    df["Y_total"] = y_numeric.sum(axis=1, min_count=1)
else:
    y_numeric = None

st.markdown("---")

# =======================
# 3. Descriptive statistics
# =======================

st.header("A. Descriptive Statistics")

# Select numeric columns for analysis (including X_total and Y_total if available)
numeric_candidates = ["2. Age (numeric)"]
if "X_total" in df.columns:
    numeric_candidates.append("X_total")
if "Y_total" in df.columns:
    numeric_candidates.append("Y_total")

numeric_cols = st.multiselect(
    "Select numeric columns for descriptive statistics",
    options=list(df.columns),
    default=[c for c in numeric_candidates if c in df.columns]
)

cat_cols_default = [
    "1. Gender",
    "3. Education Level",
    "4. Employment Status",
    "5. Average Monthly Income",
    "6. How often did you use digital payment methods (e-wallet, mobile banking, QRIS, etc.) in the past week?",
    "8. What do you primarily use digital payments for?",
]
cat_cols = st.multiselect(
    "Select categorical columns for frequency tables",
    options=list(df.columns),
    default=[c for c in cat_cols_default if c in df.columns]
)

# 3.1 Statistics for each numeric variable
st.subheader("A.1 Statistics for each item / numeric variable")

if numeric_cols:
    for col in numeric_cols:
        seri = pd.to_numeric(df[col], errors="coerce").dropna()
        if seri.empty:
            st.warning(f"Column {col} has no valid numeric data.")
            continue

        mean_val = seri.mean()
        median_val = seri.median()
        mode_val = seri.mode()
        min_val = seri.min()
        max_val = seri.max()
        std_val = seri.std()

        st.markdown(f"#### Statistics for: {col}")
        stats_rows = [
            ("Mean", mean_val),
            ("Median", median_val),
            ("Minimum", min_val),
            ("Maximum", max_val),
            ("Std Dev", std_val),
        ] + [(f"Mode {i+1}", v) for i, v in enumerate(mode_val.values)]

        stats_df = pd.DataFrame(stats_rows, columns=["Statistic", "Value"])
        st.dataframe(stats_df)

        # Bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Statistic", y="Value", data=stats_df, ax=ax, palette="viridis")
        ax.set_title(f"Statistics for {col}")
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        plt.xticks(rotation=30)
        st.pyplot(fig)
else:
    st.info("Select at least one numeric column to view statistics.")

# 3.2 Frequency and percentage tables
st.subheader("A.2 Frequency & Percentage Tables")

if cat_cols:
    for col in cat_cols:
        st.markdown(f"#### Frequency table: {col}")
        freq = df[col].value_counts(dropna=False)
        percent = df[col].value_counts(normalize=True, dropna=False) * 100

        freq_table = pd.DataFrame({
            "Category": freq.index.astype(str),
            "Frequency": freq.values,
            "Percentage": percent.values.round(2)
        })

        st.dataframe(freq_table)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.barplot(x="Category", y="Frequency", data=freq_table, ax=ax2, palette="magma")
        ax2.set_title(f"Frequency of {col}")
        ax2.set_xlabel("")
        ax2.set_ylabel("Frequency")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig2)
else:
    st.info("Select at least one categorical column to create frequency tables.")

# 3.3 Histogram and boxplot
st.subheader("A.3 Histogram & Boxplot (Optional)")

if numeric_cols:
    pilihan_plot_col = st.selectbox(
        "Select one numeric column for histogram and boxplot",
        options=numeric_cols
    )

    seri_plot = pd.to_numeric(df[pilihan_plot_col], errors="coerce").dropna()

    if not seri_plot.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Histogram – {pilihan_plot_col}**")
            fig_h, ax_h = plt.subplots(figsize=(5, 4))
            sns.histplot(seri_plot, kde=True, bins=10, ax=ax_h, color="skyblue")
            ax_h.set_xlabel(pilihan_plot_col)
            ax_h.set_ylabel("Frequency")
            st.pyplot(fig_h)

        with col2:
            st.markdown(f"**Boxplot – {pilihan_plot_col}**")
            fig_b, ax_b = plt.subplots(figsize=(3, 4))
            sns.boxplot(y=seri_plot, ax=ax_b, color="orange")
            ax_b.set_ylabel(pilihan_plot_col)
            st.pyplot(fig_b)
    else:
        st.warning(f"Column {pilihan_plot_col} has no valid numeric data.")
else:
    st.info("Select at least one numeric column for histogram & boxplot.")

st.markdown("---")

# =======================
# 4. Association Analysis X and Y
# =======================

st.header("B. Association Analysis between X_total and Y_total")

if ("X_total" in df.columns) and ("Y_total" in df.columns):
    st.write("X_total and Y_total have been computed from the items you selected above.")

    method = st.radio(
        "Select association method:",
        ("Pearson Correlation (numeric)", "Spearman Rank Correlation (numeric)", "Chi-square Test (categorical)")
    )

    valid = df[["X_total", "Y_total"]].dropna()

    if valid.empty:
        st.warning("No complete data available for X_total and Y_total.")
    else:
        if "Pearson" in method:
            st.markdown(
                "- Use when variables are numeric (e.g., Likert-scale totals such as X_total and Y_total).\n"
                "- Reports: correlation coefficient (r), p-value, and interpretation (positive/negative, weak/moderate/strong)."
            )

            r, p = stats.pearsonr(valid["X_total"], valid["Y_total"])
            st.subheader("Pearson Correlation")
            st.write(f"r = {r:.3f}")
            st.write(f"p-value = {p:.4f}")

            if r > 0:
                direction = "positive"
            elif r < 0:
                direction = "negative"
            else:
                direction = "no correlation"

            if abs(r) < 0.3:
                strength = "weak"
            elif abs(r) < 0.5:
                strength = "moderate"
            else:
                strength = "strong"

            st.write(f"Interpretation: {direction} correlation with {strength} strength.")

            fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
            sns.regplot(x="X_total", y="Y_total", data=valid, ax=ax_scatter, scatter_kws={"alpha": 0.7})
            ax_scatter.set_title("Scatter Plot X_total vs Y_total")
            st.pyplot(fig_scatter)

        elif "Spearman" in method:
            st.markdown(
                "- Use when data are not normally distributed or are ordinal/ranked.\n"
                "- Reports: Spearman correlation coefficient (rho), p-value, and interpretation (positive/negative, weak/moderate/strong)."
            )

            r, p = stats.spearmanr(valid["X_total"], valid["Y_total"])
            st.subheader("Spearman Rank Correlation")
            st.write(f"rho = {r:.3f}")
            st.write(f"p-value = {p:.4f}")

            if r > 0:
                direction = "positive"
            elif r < 0:
                direction = "negative"
            else:
                direction = "no correlation"

            if abs(r) < 0.3:
                strength = "weak"
            elif abs(r) < 0.5:
                strength = "moderate"
            else:
                strength = "strong"

            st.write(f"Interpretation: {direction} correlation with {strength} strength.")

        else:  # Chi-square
            st.markdown(
                "- Use when X and Y are categorical variables.\n"
                "- Here, X_total and Y_total are first grouped into categories (e.g., low/medium/high).\n"
                "- Reports: chi-square value, degrees of freedom (df), p-value, and interpretation of association significance."
            )

            st.subheader("Chi-square Test")

            bins = st.slider(
                "Number of categories (bins) to convert X_total and Y_total into categorical variables",
                2, 5, 3
            )

            valid["X_cat"] = pd.qcut(valid["X_total"], q=bins, duplicates="drop")
            valid["Y_cat"] = pd.qcut(valid["Y_total"], q=bins, duplicates="drop")

            ctab = pd.crosstab(valid["X_cat"], valid["Y_cat"])
            st.write("Crosstab:")
            st.dataframe(ctab)

            chi2, p, dof, expected = stats.chi2_contingency(ctab)
            st.write(f"Chi-square = {chi2:.3f}")
            st.write(f"df = {dof}")
            st.write(f"p-value = {p:.4f}")

            st.write("Interpretation: if p-value < 0.05, there is a statistically significant association between X_cat and Y_cat.")
else:
    st.info("For association analysis, first select items for X and Y so that X_total and Y_total can be computed.")
