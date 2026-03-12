"""
Exploratory Data Analysis (EDA) module for Telecom Customer Churn.

This module automates the generation of an EDA Jupyter Notebook and
executes data visualizations and statistical analysis (Correlation, VIF)
for the raw Telco customer churn dataset. It follows Agentic Architecture
standards with strict type hinting and modular design.
"""

import os

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
import seaborn as sns
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from statsmodels.stats.outliers_influence import variance_inflation_factor


def generate_eda_notebook(data_path: str, nb_path: str) -> None:
    """
    Generates a Jupyter Notebook for interactive EDA.

    Args:
        data_path: Path to the raw CSV data.
        nb_path: Output path for the generated Jupyter Notebook.
    """
    nb = new_notebook()
    nb.cells.extend(
        [
            new_markdown_cell(
                "# Exploratory Data Analysis: Telecom Customer Churn\n\nThis notebook analyzes the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset, forming our baseline 'Hard Signals' before we use an Agentic AI to generate 'Soft Signals' (synthetic ticket notes)."
            ),
            new_code_cell(
                "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\nfrom statsmodels.stats.outliers_influence import variance_inflation_factor\n\n# Configure visual settings\nsns.set_theme(style='whitegrid')\n\n# Data loading\ndata_path = r'../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'\ndf = pd.read_csv(data_path)\ndisplay(df.head())"
            ),
            new_markdown_cell(
                "## 1. Data Cleaning\nWe have a quantitative column (`TotalCharges`) that is parsed as an `object` due to empty spaces. We need to coerce this into `numeric`."
            ),
            new_code_cell(
                "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\nprint(f'Missing values before drop: {df[\"TotalCharges\"].isnull().sum()}')\ndf.dropna(inplace=True)\nprint(f'Shape after cleaning: {df.shape}')"
            ),
            new_markdown_cell(
                "## 2. Target Variable Analysis: Churn\nLet's evaluate the class imbalance. This will inform whether we need to use scale_pos_weight in XGBoost or oversampling techniques like SMOTE."
            ),
            new_code_cell(
                "plt.figure(figsize=(6,4))\nsns.countplot(data=df, x='Churn', hue='Churn', palette='Set2', legend=False)\nplt.title('Churn Distribution')\nplt.show()"
            ),
            new_markdown_cell(
                "## 3. Quantitative Variables vs Churn\nThe core 'Hard Signals' include billing logic (`MonthlyCharges`, `TotalCharges`) and customer loyalty (`tenure`)."
            ),
            new_code_cell(
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\nsns.boxplot(data=df, x='Churn', y='tenure', ax=axes[0], hue='Churn', palette='Set2')\naxes[0].set_title('Tenure by Churn')\n\nsns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1], hue='Churn', palette='Set2')\naxes[1].set_title('Monthly Charges by Churn')\n\nsns.boxplot(data=df, x='Churn', y='TotalCharges', ax=axes[2], hue='Churn', palette='Set2')\naxes[2].set_title('Total Charges by Churn')\nplt.show()"
            ),
            new_markdown_cell(
                "## 4. Categorical Variables vs Churn\nWe investigate features that represent the quality of service (e.g., `InternetService`, `TechSupport`) and customer commitments (`Contract`). These will be highly important for generating Synthetic Ticket Notes."
            ),
            new_code_cell(
                "features = ['InternetService', 'Contract', 'PaymentMethod', 'TechSupport']\nfig, axes = plt.subplots(2, 2, figsize=(14, 10))\nfor i, feature in enumerate(features):\n    ax = axes[i//2, i%2]\n    sns.countplot(data=df, x=feature, hue='Churn', ax=ax, palette='Set2')\n    ax.set_title(f'Churn by {feature}')\n    ax.tick_params(axis='x', rotation=45)\nplt.tight_layout()\nplt.show()"
            ),
            new_markdown_cell(
                "## 5. Correlation & VIF Analysis\nWe drop redundant columns and encode categorical variables to check for multicollinearity."
            ),
            new_code_cell(
                "# Collapse Redundant Columns\ndf['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})\nreplace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']\nfor col in replace_cols:\n    df[col] = df[col].replace({'No internet service': 'No'})\n\n# Map Binary\nbin_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']\nfor col in bin_cols:\n    df[col] = df[col].map({'Yes': 1, 'No': 0})\ndf['gender'] = df['gender'].map({'Female': 1, 'Male': 0})\n\n# Get Dummies\ndf_encoded = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True).astype(float)\n\n# Correlation Map\nplt.figure(figsize=(12, 10))\nsns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm', fmt='.2f')\nplt.title('Feature Correlation Map')\nplt.tight_layout()\nplt.show()\n\n# VIF Calculation\nX = df_encoded.drop('Churn', axis=1)\nvif_data = pd.DataFrame()\nvif_data['Feature'] = X.columns\nvif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]\ndisplay(vif_data.sort_values(by='VIF', ascending=False))"
            ),
        ]
    )

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook generated at {nb_path}")


def run_eda_visualizations(data_path: str, fig_dir: str) -> None:
    """
    Executes the analytical logic and exports the required visualizations to disk.
    Also calculates and persists VIF metadata to a local CSV for documentation.

    Args:
        data_path: Path to the raw CSV data.
        fig_dir: Output directory for saving generated figures.
    """
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    sns.set_theme(style="whitegrid")

    # 1. Save qualitative variables
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Churn Distribution")
    plt.savefig(os.path.join(fig_dir, "churn_distribution.png"))
    plt.close()

    # 2. Save quantitative variables
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[0], hue="Churn", palette="Set2")
    axes[0].set_title("Tenure by Churn")
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1], hue="Churn", palette="Set2")
    axes[1].set_title("Monthly Charges by Churn")
    sns.boxplot(data=df, x="Churn", y="TotalCharges", ax=axes[2], hue="Churn", palette="Set2")
    axes[2].set_title("Total Charges by Churn")
    plt.savefig(os.path.join(fig_dir, "quantitative_vs_churn.png"))
    plt.close()

    # 3. Save qualitative features
    features = ["InternetService", "Contract", "PaymentMethod", "TechSupport"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        sns.countplot(data=df, x=feature, hue="Churn", ax=ax, palette="Set2")
        ax.set_title(f"Churn by {feature}")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "qualitative_vs_churn.png"))
    plt.close()

    # 4. Collapse Redundant Categorical Columns
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})
    replace_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in replace_cols:
        df[col] = df[col].replace({"No internet service": "No"})

    # Map Binary
    bin_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "Churn",
    ]
    for col in bin_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    # 5. Encoding
    df_encoded = pd.get_dummies(df.drop("customerID", axis=1), drop_first=True).astype(float)

    # 6. Correlation Map
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_encoded.corr(), annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Map")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_map.png"))
    plt.close()

    # 7. VIF
    X = df_encoded.drop("Churn", axis=1)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    # Save VIF to CSV so the LLM/User can read it
    vif_path = os.path.join(fig_dir, "vif_results.csv")
    vif_data.sort_values(by="VIF", ascending=False).to_csv(vif_path, index=False)

    print("EDA visual generation complete. Missing values, collinearity, and correlation handled.")
    print(f"VIF summary saved to {vif_path}")


def main() -> None:
    """
    Entry point for the EDA module. Orchestrates paths and triggers execution.
    """
    base_dir = r"c:\Users\sebas\Desktop\ai-ml-telecom-customer-churn"
    data_path = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    nb_path = os.path.join(base_dir, "notebooks", "01_eda_telco_churn.ipynb")
    fig_dir = os.path.join(base_dir, "reports", "figures", "eda")

    os.makedirs(fig_dir, exist_ok=True)

    generate_eda_notebook(data_path=data_path, nb_path=nb_path)
    run_eda_visualizations(data_path=data_path, fig_dir=fig_dir)


if __name__ == "__main__":
    main()
