import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)

from src.config import get_config, get_path


# ==========================================
# CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = get_path("paths", "reports")
DATA_PATH = get_path("data", "raw_path")
RESULT_DIR = get_path("paths", "progress")

FIGURE_DIR = get_path("paths", "figures")

DOCS_DIR = get_path("paths", "docs")

FIGURE_CATALOG = DOCS_DIR / "images" / "figure_catalog.csv"

EXPERIMENT_LOG = DOCS_DIR / "experiment_logs.csv"


# Behavior analysis configuration
PROFILE_A_CONFIG = get_config(
    "behavior_analysis",
    default={}
)

# Final model configuration
PROFILE_B_CONFIG = get_config(
    "final_model",
    default={}
)


# Report metadata
REPORT_ID = datetime.now().strftime(
    "REPORT_%Y%m%d_%H%M%S"
)

GENERATED_AT = datetime.now().astimezone().isoformat(
    timespec="seconds"
)

MD_PATH = REPORT_DIR / f"report_{REPORT_ID}.md"

HTML_PATH = REPORT_DIR / f"report_{REPORT_ID}.html"

PDF_PATH = REPORT_DIR / f"report_{REPORT_ID}.pdf"


# ==========================================
# CONFIGURATION HELPERS
# ==========================================

def build_profile_path(config, config_name):

    feature_space = config.get("feature_space")
    scaler = config.get("scaler")
    n_clusters = config.get("n_clusters")

    required_values = {
        "feature_space": feature_space,
        "scaler": scaler,
        "n_clusters": n_clusters
    }

    missing = [
        key
        for key, value in required_values.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            f"Missing configuration for {config_name}: {missing}"
        )

    scaler_mapping = {
        "StandardScaler": "standard",
        "MinMaxScaler": "minmax"
    }

    if scaler not in scaler_mapping:
        raise ValueError(
            f"Unsupported scaler in {config_name}: {scaler}"
        )

    scaler_name = scaler_mapping[scaler]

    filename = (
        f"{feature_space}_"
        f"{scaler_name}_"
        f"k{n_clusters}_profile.csv"
    )

    return RESULT_DIR / filename


def describe_configuration(config):

    feature_space = config.get(
        "feature_space",
        "unknown"
    )

    scaler = config.get(
        "scaler",
        "unknown"
    )

    n_clusters = config.get(
        "n_clusters",
        "unknown"
    )

    if feature_space == "with_gender":
        feature_label = "With Gender"

    elif feature_space == "without_gender":
        feature_label = "Without Gender"

    else:
        feature_label = feature_space.replace(
            "_",
            " "
        ).title()

    return (
        f"{feature_label}, "
        f"{scaler}, "
        f"K={n_clusters}"
    )


PROFILE_A_PATH = build_profile_path(
    PROFILE_A_CONFIG,
    "behavior_analysis"
)

PROFILE_B_PATH = build_profile_path(
    PROFILE_B_CONFIG,
    "final_model"
)


# ==========================================
# 1. LOAD PROJECT DATA
# ==========================================

def load_csv(path):

    if not path.exists():

        print(
            "Warning: File not found:",
            path
        )

        return None

    return pd.read_csv(path)


def load_project_data():

    print("\n========== LOADING PROJECT DATA ==========")

    data = {

        "raw": load_csv(DATA_PATH),

        "elbow": load_csv(
            RESULT_DIR / "kmeans_elbow_results.csv"
        ),

        "evaluation": load_csv(
            RESULT_DIR / "clustering_metrics.csv"
        ),

        "experiments": load_csv(
            RESULT_DIR / "feature_scaling_experiments.csv"
        ),

        "profile_a": load_csv(PROFILE_A_PATH),

        "profile_b": load_csv(PROFILE_B_PATH),

        "figure_catalog": load_csv(
            FIGURE_CATALOG
        ),

        "tracking": load_csv(
            EXPERIMENT_LOG
        )

    }

    for name, df in data.items():

        if df is not None:

            print(
                f"{name}: {df.shape}"
            )

    return data


# ==========================================
# 2. DATASET SUMMARY
# ==========================================

def dataset_summary(df):

    if df is None:

        return "Dataset information is unavailable."

    lines = [

        f"- Number of customers: {len(df)}",

        f"- Number of columns: {len(df.columns)}",

        f"- Missing values: {int(df.isnull().sum().sum())}",

        f"- Duplicate records: {int(df.duplicated().sum())}",

        "- Features: " + ", ".join(
            map(str, df.columns)
        )

    ]

    return "\n".join(lines)


# ==========================================
# 3. FORMAT DATAFRAME AS MARKDOWN
# ==========================================

def dataframe_to_markdown(df, max_rows=20):

    if df is None or df.empty:

        return "No data available.\n"

    display_df = df.head(max_rows).copy()

    for col in display_df.columns:

        if pd.api.types.is_numeric_dtype(display_df[col]):

            display_df[col] = display_df[col].map(
                lambda x: (
                    f"{x:.4f}"
                    if pd.notna(x)
                    and isinstance(x, (float, int))
                    else x
                )
            )

    headers = list(display_df.columns)

    lines = [

        "| " + " | ".join(headers) + " |",

        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |"

    ]

    for _, row in display_df.iterrows():

        values = [
            str(value) if pd.notna(value) else ""
            for value in row
        ]

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    if len(df) > max_rows:

        lines.append(
            f"\nShowing {max_rows} of {len(df)} records."
        )

    return "\n".join(lines)


# ==========================================
# 4. EXPERIMENT COMPARISON
# ==========================================

def summarize_experiments(df):

    if df is None or df.empty:

        return None

    required = [

        "Feature_Space",
        "Scaler",
        "K",
        "Silhouette_Score",
        "Davies_Bouldin_Index",
        "Calinski_Harabasz_Score"

    ]

    if not all(col in df.columns for col in required):

        return None

    records = []

    grouped = df.groupby(
        ["Feature_Space", "Scaler"]
    )

    for (feature, scaler), group in grouped:

        best_silhouette = group.loc[
            group["Silhouette_Score"].idxmax()
        ]

        best_dbi = group.loc[
            group["Davies_Bouldin_Index"].idxmin()
        ]

        best_ch = group.loc[
            group["Calinski_Harabasz_Score"].idxmax()
        ]

        records.append({

            "Feature_Space": feature,

            "Scaler": scaler,

            "Best_K_Silhouette": int(
                best_silhouette["K"]
            ),

            "Best_Silhouette": best_silhouette[
                "Silhouette_Score"
            ],

            "Best_K_DBI": int(
                best_dbi["K"]
            ),

            "Best_DBI": best_dbi[
                "Davies_Bouldin_Index"
            ],

            "Best_K_CH": int(
                best_ch["K"]
            ),

            "Best_CH": best_ch[
                "Calinski_Harabasz_Score"
            ]

        })

    return pd.DataFrame(records)


# ==========================================
# 5. BUILD MARKDOWN REPORT
# ==========================================

def build_markdown(data):

    raw = data["raw"]

    report = []

    report.append("# MALL CUSTOMER SEGMENTATION REPORT\n")

    report.append(f"**Report ID:** {REPORT_ID}\n")

    report.append(f"**Generated at:** {GENERATED_AT}\n")

    report.append("---\n")

    # Dataset
    report.append("## 1. Dataset Overview\n")

    report.append(
        dataset_summary(raw)
    )

    # EDA
    report.append("\n## 2. Exploratory Data Analysis\n")

    if raw is not None:

        numeric = raw.select_dtypes(
            include="number"
        )

        report.append(
            "### Descriptive Statistics\n"
        )

        report.append(
            dataframe_to_markdown(
                numeric.describe().round(4).reset_index()
            )
        )

    # Elbow
    report.append("\n## 3. Elbow Method\n")

    report.append(
        dataframe_to_markdown(data["elbow"])
    )

    # Evaluation
    report.append("\n## 4. Clustering Evaluation\n")

    report.append(
        dataframe_to_markdown(data["evaluation"])
    )

    # Experiments
    report.append(
        "\n## 5. Feature Space and Scaling Experiments\n"
    )

    experiment_summary = summarize_experiments(
        data["experiments"]
    )

    report.append(
        dataframe_to_markdown(experiment_summary)
    )

    # Cluster profiles
    report.append("\n## 6. Customer Cluster Profiles\n")

    report.append(
        f"### Configuration A: "
        f"{describe_configuration(PROFILE_A_CONFIG)}\n"
    )

    report.append(
        dataframe_to_markdown(data["profile_a"])
    )

    report.append(
        f"\n### Configuration B: "
        f"{describe_configuration(PROFILE_B_CONFIG)}\n"
    )

    report.append(
        dataframe_to_markdown(data["profile_b"])
    )

    # Tracking
    report.append("\n## 7. Experiment Tracking\n")

    report.append(
        dataframe_to_markdown(
            data["tracking"],
            max_rows=20
        )
    )

    # Figures
    report.append("\n## 8. Visualization Catalog\n")

    catalog = data["figure_catalog"]

    if catalog is not None:

        for _, figure in catalog.iterrows():

            report.append(
                f"- {figure['Figure_ID']}: "
                f"{figure['Filename']} "
                f"({figure['Category']})"
            )

    # Conclusion
    report.append("\n## 9. Conclusion\n")

    report.append(
        "This report summarizes the available data exploration, "
        "preprocessing-related experiments, K-Means training, "
        "clustering evaluation, customer profiles and visualizations."
    )

    report.append(
        "\nThe final clustering configuration should be selected "
        "by considering numerical metrics, cluster sizes, "
        "customer behavior and business interpretability."
    )

    report.append("\n---\n")

    report.append(
        "*Generated automatically by the Mall Customer "
        "Segmentation project.*"
    )

    return "\n".join(report)


# ==========================================
# 6. CONVERT MARKDOWN TO HTML
# ==========================================

def inline_format(text):

    text = html.escape(str(text))

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"<strong>\1</strong>",
        text
    )

    text = re.sub(
        r"`(.*?)`",
        r"<code>\1</code>",
        text
    )

    return text


def markdown_to_html(markdown_text):

    lines = markdown_text.splitlines()

    output = []

    inside_list = False

    index = 0

    while index < len(lines):

        line = lines[index].strip()

        if not line:

            if inside_list:

                output.append("</ul>")
                inside_list = False

            index += 1

            continue

        # Markdown tables
        if line.startswith("|"):

            table_lines = []

            while (
                index < len(lines)
                and lines[index].strip().startswith("|")
            ):

                table_lines.append(
                    lines[index].strip()
                )

                index += 1

            output.append("<table>")

            for row_index, row in enumerate(table_lines):

                cells = [

                    cell.strip()
                    for cell in row.strip("|").split("|")

                ]

                if all(
                    set(cell) <= {"-", ":", " "}
                    for cell in cells
                ):

                    continue

                tag = "th" if row_index == 0 else "td"

                output.append("<tr>")

                for cell in cells:

                    output.append(
                        f"<{tag}>{inline_format(cell)}</{tag}>"
                    )

                output.append("</tr>")

            output.append("</table>")

            continue

        # Headings
        if line.startswith("#"):

            if inside_list:

                output.append("</ul>")
                inside_list = False

            level = len(line) - len(line.lstrip("#"))

            level = min(level, 6)

            title = line[level:].strip()

            output.append(
                f"<h{level}>{inline_format(title)}</h{level}>"
            )

        # Lists
        elif line.startswith("- "):

            if not inside_list:

                output.append("<ul>")
                inside_list = True

            output.append(
                f"<li>{inline_format(line[2:])}</li>"
            )

        # Horizontal line
        elif line == "---":

            output.append("<hr>")

        else:

            if inside_list:

                output.append("</ul>")
                inside_list = False

            output.append(
                f"<p>{inline_format(line)}</p>"
            )

        index += 1

    if inside_list:

        output.append("</ul>")

    body = "\n".join(output)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mall Customer Segmentation Report</title>

<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 20px;
    line-height: 1.6;
    color: #253047;
}}

h1, h2, h3 {{
    color: #174a73;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 13px;
}}

th, td {{
    border: 1px solid #d5dce5;
    padding: 9px;
    text-align: left;
}}

th {{
    background: #eaf1f8;
}}

tr:nth-child(even) {{
    background: #f7f9fc;
}}

code {{
    background: #eef1f5;
    padding: 3px;
}}

hr {{
    border: 0;
    border-top: 1px solid #ddd;
    margin: 25px 0;
}}
</style>

</head>
<body>
{body}
</body>
</html>
"""


# ==========================================
# 7. BUILD PDF
# ==========================================

def build_pdf(markdown_text, data):

    print("\n========== GENERATING PDF ==========")

    document = SimpleDocTemplate(

        str(PDF_PATH),

        pagesize=A4,

        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,

        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm

    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=26,
            spaceAfter=18
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#174A73"),
            spaceBefore=14,
            spaceAfter=8
        )
    )

    cell_style = ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9
    )

    story = []

    # Report title
    story.append(
        Paragraph(
            "MALL CUSTOMER SEGMENTATION REPORT",
            styles["ReportTitle"]
        )
    )

    story.append(
        Paragraph(
            f"Report ID: {REPORT_ID}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Generated at: {GENERATED_AT}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------
    # PDF helper functions
    # --------------------------------------

    def add_heading(title):

        story.append(
            Paragraph(
                title,
                styles["SectionHeading"]
            )
        )

    def add_table(title, df, max_rows=20):

        if df is None or df.empty:

            return

        add_heading(title)

        display_df = df.head(max_rows).copy()

        table_data = [

            [

                Paragraph(
                    html.escape(str(col)),
                    cell_style
                )

                for col in display_df.columns

            ]

        ]

        for _, row in display_df.iterrows():

            table_data.append(

                [

                    Paragraph(
                        html.escape(str(value)),
                        cell_style
                    )

                    for value in row

                ]

            )

        column_width = 18 * cm / len(display_df.columns)

        table = Table(

            table_data,

            colWidths=[
                column_width
            ] * len(display_df.columns),

            repeatRows=1,

            hAlign="LEFT"

        )

        table.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#DCE8F2")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#173A59")
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )

            ])
        )

        story.append(table)

        story.append(
            Spacer(1, 10)
        )

    # --------------------------------------
    # 1. Dataset
    # --------------------------------------

    add_heading("1. Dataset Overview")

    if data["raw"] is not None:

        df = data["raw"]

        info = [

            ["Property", "Value"],

            ["Customers", str(len(df))],

            ["Columns", str(len(df.columns))],

            [
                "Missing values",
                str(int(df.isnull().sum().sum()))
            ],

            [
                "Duplicates",
                str(int(df.duplicated().sum()))
            ]

        ]

        story.append(
            Table(info)
        )

    # --------------------------------------
    # 2. Elbow
    # --------------------------------------

    add_table(
        "2. Elbow Method",
        data["elbow"]
    )

    # --------------------------------------
    # 3. Evaluation
    # --------------------------------------

    add_table(
        "3. Clustering Evaluation",
        data["evaluation"]
    )

    # --------------------------------------
    # 4. Experiments
    # --------------------------------------

    add_table(
        "4. Feature and Scaling Experiments",
        summarize_experiments(data["experiments"])
    )

    # --------------------------------------
    # 5. Profiles
    # --------------------------------------

    add_table(
        "5. Configuration A - Cluster Profiles",
        data["profile_a"]
    )

    add_table(
        "6. Configuration B - Cluster Profiles",
        data["profile_b"]
    )

    # --------------------------------------
    # 7. Tracking
    # --------------------------------------

    add_table(
        "7. Experiment Tracking",
        data["tracking"]
    )

    # --------------------------------------
    # 8. Figures
    # --------------------------------------

    add_heading("8. Visualization Gallery")

    catalog = data["figure_catalog"]

    if catalog is not None:

        for _, figure in catalog.iterrows():

            relative_path = figure.get(
                "Relative_Path",
                ""
            )

            if not isinstance(relative_path, str):
                continue

            image_path = ROOT_DIR / relative_path

            if (
                image_path.exists()
                and image_path.suffix.lower() == ".png"
            ):

                story.append(
                    Paragraph(
                        html.escape(
                            str(figure["Filename"])
                        ),
                        styles["Heading3"]
                    )
                )

                image = Image(
                    str(image_path)
                )

                ratio = min(
                    (17 * cm) / image.imageWidth,
                    (8 * cm) / image.imageHeight
                )

                image.drawWidth = image.imageWidth * ratio

                image.drawHeight = image.imageHeight * ratio

                story.append(image)

                story.append(
                    Spacer(1, 12)
                )

    document.build(story)

    print("PDF saved:", PDF_PATH)


# ==========================================
# 8. EXPORT REPORTS
# ==========================================

def export_reports(data):

    print("\n========== EXPORTING REPORTS ==========")

    markdown_text = build_markdown(data)

    MD_PATH.write_text(
        markdown_text,
        encoding="utf-8"
    )

    print("Markdown saved:", MD_PATH)

    html_text = markdown_to_html(
        markdown_text
    )

    HTML_PATH.write_text(
        html_text,
        encoding="utf-8"
    )

    print("HTML saved:", HTML_PATH)

    build_pdf(
        markdown_text,
        data
    )


# ==========================================
# 9. MAIN
# ==========================================

def main():

    print("\n========================================")
    print("MALL CUSTOMER REPORT GENERATOR")
    print("========================================")

    print("Report directory:", REPORT_DIR)

    print("Dataset:", DATA_PATH)

    print("Results directory:", RESULT_DIR)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data = load_project_data()

    if data["raw"] is None:

        raise FileNotFoundError(
            "Raw dataset is required."
        )

    export_reports(data)

    print("\n========================================")
    print("REPORT GENERATION COMPLETED SUCCESSFULLY!")
    print("========================================")


if __name__ == "__main__":
    main()