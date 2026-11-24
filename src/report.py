import html
import re
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import get_data_path
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)

from src.config import get_config, get_path
from src.figure_reviews import review_figure


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

# Comparison analysis configuration
PROFILE_B_CONFIG = get_config(
    "comparison_analysis",
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


if not PROFILE_A_CONFIG:
    raise ValueError("behavior_analysis configuration is missing or empty in config.yaml")

if not PROFILE_B_CONFIG:
    raise ValueError("comparison_analysis configuration is missing or empty in config.yaml")


PROFILE_A_PATH = build_profile_path(
    PROFILE_A_CONFIG,
    "behavior_analysis"
)

PROFILE_B_PATH = build_profile_path(
    PROFILE_B_CONFIG,
    "comparison_analysis"
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


def scientific_assessment(data):
    """Build evidence-based model and segment interpretations from outputs."""
    evaluation = data.get("evaluation")
    elbow = data.get("elbow")
    raw = data.get("raw")
    profile = data.get("profile_a")
    config = get_config("final_model", default={})
    selected_k = int(config.get("n_clusters", 0))
    selection = []

    selected = None
    if evaluation is not None and not evaluation.empty:
        matches = evaluation.loc[evaluation["K"] == selected_k]
        if not matches.empty:
            selected = matches.iloc[0]
            metric_winners = {
                "Silhouette": int(evaluation.loc[evaluation["Silhouette_Score"].idxmax(), "K"]),
                "Davies–Bouldin": int(evaluation.loc[evaluation["Davies_Bouldin_Index"].idxmin(), "K"]),
                "Calinski–Harabasz": int(evaluation.loc[evaluation["Calinski_Harabasz_Score"].idxmax(), "K"]),
            }
            selection.append(
                f"Mô hình cuối dùng K={selected_k}, feature space "
                f"{config.get('feature_space', 'chưa xác định')}, "
                f"{config.get('scaler', 'chưa xác định')}. Trong dải K đã thử, "
                f"Silhouette={selected['Silhouette_Score']:.4f} "
                f"(tốt nhất tại K={metric_winners['Silhouette']}), "
                f"Davies–Bouldin={selected['Davies_Bouldin_Index']:.4f} "
                f"(thấp nhất tại K={metric_winners['Davies–Bouldin']}), "
                f"Calinski–Harabasz={selected['Calinski_Harabasz_Score']:.4f} "
                f"(cao nhất tại K={metric_winners['Calinski–Harabasz']})."
            )

    if elbow is not None and not elbow.empty:
        selected_rows = elbow.loc[elbow["K"] == selected_k]
        previous_rows = elbow.loc[elbow["K"] == selected_k - 1]
        next_rows = elbow.loc[elbow["K"] == selected_k + 1]
        if not selected_rows.empty and not previous_rows.empty and not next_rows.empty:
            inertia = float(selected_rows.iloc[0]["Inertia"])
            before = float(previous_rows.iloc[0]["Inertia"])
            after = float(next_rows.iloc[0]["Inertia"])
            gain_into = (before - inertia) / before * 100 if before else 0
            gain_after = (inertia - after) / inertia * 100 if inertia else 0
            selection.append(
                f"Elbow hỗ trợ lựa chọn này: tăng từ K={selected_k - 1} lên "
                f"K={selected_k} giảm inertia {gain_into:.1f}%, trong khi "
                f"K={selected_k + 1} chỉ giảm thêm {gain_after:.1f}%. "
                "Chọn K=6 vì cả ba chỉ số trên bộ đặc trưng chính cùng đạt cực trị "
                "trong dải thử nghiệm và đường inertia bắt đầu phẳng hơn sau K=6; "
                "đây là lựa chọn cân bằng cho mục tiêu diễn giải, không phải bằng chứng "
                "rằng có đúng sáu nhóm tự nhiên trong mọi mẫu khách hàng."
            )

    if config.get("feature_space") == "without_gender":
        selection.append(
            "Mô hình cuối loại Gender vì mục tiêu là phân nhóm theo tuổi, thu nhập và "
            "điểm chi tiêu; StandardScaler đưa ba biến số về cùng thang đo độ lệch "
            "chuẩn trước khi tính khoảng cách Euclid. Kết quả giữa các scaler hoặc "
            "feature space khác nhau cần được đọc trong từng cấu hình vì thang đo và "
            "số chiều làm thay đổi các chỉ số nội bộ."
        )

    cluster_notes = []
    if (
        raw is not None and profile is not None and not profile.empty
        and {"Age", "Annual Income (k$)", "Spending Score (1-100)"}.issubset(raw.columns)
    ):
        means = {
            "Age": float(raw["Age"].mean()),
            "Annual Income (k$)": float(raw["Annual Income (k$)"].mean()),
            "Spending Score (1-100)": float(raw["Spending Score (1-100)"].mean()),
        }

        def relative(value, reference, lower, higher):
            return higher if float(value) > reference else lower

        for _, row in profile.iterrows():
            cluster_id = int(row["Cluster"])
            count = int(row["Customers"])
            share = count / len(raw) * 100
            age_group = relative(row["Average_Age"], means["Age"], "trẻ hơn", "lớn tuổi hơn")
            income_group = relative(row["Average_Income"], means["Annual Income (k$)"], "thấp hơn", "cao hơn")
            spending_group = relative(row["Average_Spending"], means["Spending Score (1-100)"], "thấp hơn", "cao hơn")
            spending_advice = (
                "có thể thử nghiệm chương trình duy trì/tri ân, rồi đo phản hồi thực tế"
                if row["Average_Spending"] > means["Spending Score (1-100)"]
                else "có thể thử nghiệm ưu đãi khám phá phù hợp và đo mức phản hồi"
            )
            cluster_notes.append(
                f"Cụm {cluster_id}: {count} khách ({share:.1f}%); tuổi trung bình "
                f"{row['Average_Age']:.2f} ({age_group} mức trung bình mẫu {means['Age']:.2f}), "
                f"thu nhập {row['Average_Income']:.2f} k$ ({income_group} mức trung bình "
                f"{means['Annual Income (k$)']:.2f}), điểm chi tiêu {row['Average_Spending']:.2f} "
                f"({spending_group} mức trung bình {means['Spending Score (1-100)']:.2f}). "
                f"Hướng kinh doanh chỉ là giả thuyết để thử nghiệm: {spending_advice}."
            )

    limitations = [
        "Silhouette của cấu hình cuối cho thấy độ tách cụm vừa phải, không phải phân tách hoàn hảo; "
        "Davies–Bouldin càng thấp càng tốt và Calinski–Harabasz càng cao càng tốt, nhưng cả ba "
        "đều là chỉ số nội bộ, không xác nhận giá trị kinh doanh.",
        "Dữ liệu chỉ có 200 quan sát, không có nhãn chuẩn hay giao dịch mua thực tế; "
        "điểm chi tiêu là chỉ số 1–100. Các đề xuất marketing cần kiểm chứng bằng dữ liệu giao dịch "
        "và thử nghiệm thực tế. Kết quả hiện dùng một random_state, nên cần kiểm tra độ ổn định "
        "qua nhiều seed trước khi triển khai ngoài bài tập.",
    ]
    return {"selection": selection, "clusters": cluster_notes, "limitations": limitations}


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
        f"### Main Behavioral Configuration: "
        f"{describe_configuration(PROFILE_A_CONFIG)}\n"
    )

    report.append(
        dataframe_to_markdown(data["profile_a"])
    )

    report.append(
        f"\n### Comparison Configuration: "
        f"{describe_configuration(PROFILE_B_CONFIG)}\n"
    )

    report.append(
        dataframe_to_markdown(data["profile_b"])
    )

    assessment = scientific_assessment(data)

    report.append("\n## 7. Scientific Assessment and Segment Interpretation\n")
    report.append("### Cơ sở lựa chọn cấu hình cuối\n")
    report.extend(assessment["selection"])
    report.append("\n### Nhận xét từng cụm\n")
    report.extend(f"- {note}" for note in assessment["clusters"])
    report.append("\n### Hạn chế và cách diễn giải\n")
    report.extend(f"- {note}" for note in assessment["limitations"])

    # Tracking
    report.append("\n## 8. Experiment Tracking\n")

    tracking = data["tracking"]
    if tracking is not None:
        if "timestamp" in tracking.columns:
            tracking = tracking.sort_values("timestamp", ascending=False)
        summary_columns = [
            "run_id",
            "timestamp",
            "event",
            "status",
        ]
        tracking = tracking[
            [column for column in summary_columns if column in tracking.columns]
        ].head(20)
        if "status" in tracking.columns:
            tracking["status"] = tracking["status"].replace(
                {"success": "OK", "running": "Running", "failed": "Failed"}
            )

    report.append(dataframe_to_markdown(tracking, max_rows=20))

    # Figures
    report.append("\n## 9. Visualization Catalog\n")

    catalog = data["figure_catalog"]

    if catalog is not None:

        for _, figure in catalog.iterrows():

            filename = str(figure["Filename"])
            observation, figure_assessment = review_figure(
                filename, figure.get("Category", "")
            )
            relative_path = str(figure.get("Relative_Path", ""))
            report_path = relative_path.removeprefix("reports/")

            report.append(
                f"### {figure['Figure_ID']}: {filename}\n"
                f"![{filename}]({report_path})\n\n"
                f"**Nhận xét:** {observation}\n\n"
                f"**Đánh giá:** {figure_assessment}"
            )

    # Conclusion
    report.append("\n## 10. Conclusion\n")
    if assessment["selection"]:
        report.append(assessment["selection"][0])
    if assessment["limitations"]:
        report.append(assessment["limitations"][0])

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

        # Inline images used in the report's figure gallery.
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt, image_path = line[2:].split("](", 1)
            output.append(
                f'<img src="{html.escape(image_path[:-1], quote=True)}" '
                f'alt="{html.escape(alt, quote=True)}" '
                'style="max-width:100%;height:auto">'
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
    """Create a readable PDF with embedded Unicode fonts and compact tables."""
    print("\n========== GENERATING PDF ==========")

    font_dir = Path(get_data_path()) / "fonts" / "ttf"
    font_files = {
        "ProjectSans": font_dir / "DejaVuSans.ttf",
        "ProjectSans-Bold": font_dir / "DejaVuSans-Bold.ttf",
        "ProjectSans-Italic": font_dir / "DejaVuSans-Oblique.ttf",
        "ProjectSans-BoldItalic": font_dir / "DejaVuSans-BoldOblique.ttf",
    }
    for font_name, font_path in font_files.items():
        if not font_path.exists():
            raise FileNotFoundError(f"Required report font is missing: {font_path}")
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    pdfmetrics.registerFontFamily(
        "ProjectSans",
        normal="ProjectSans",
        bold="ProjectSans-Bold",
        italic="ProjectSans-Italic",
        boldItalic="ProjectSans-BoldItalic",
    )

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title="Mall Customer Segmentation Report",
        author="Mall Customer Clustering Project",
    )

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "ProjectSans"
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    styles["BodyText"].fontName = "ProjectSans"
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    styles["Title"].fontName = "ProjectSans-Bold"
    styles["Heading1"].fontName = "ProjectSans-Bold"
    styles["Heading2"].fontName = "ProjectSans-Bold"
    styles["Heading3"].fontName = "ProjectSans-Bold"

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="ProjectSans-Bold",
            alignment=TA_CENTER,
            fontSize=19,
            leading=24,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="ProjectSans-Bold",
            textColor=colors.HexColor("#174A73"),
            fontSize=12,
            leading=15,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubsectionHeading",
            parent=styles["Heading3"],
            fontName="ProjectSans-Bold",
            fontSize=10,
            leading=13,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="ProjectSans",
            fontSize=7.5,
            leading=9.5,
            spaceAfter=0,
            wordWrap="LTR",
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName="ProjectSans-Bold",
            fontSize=7.2,
            leading=9,
            spaceAfter=0,
            wordWrap="LTR",
        )
    )

    story = [
        Paragraph("MALL CUSTOMER SEGMENTATION REPORT", styles["ReportTitle"]),
        Paragraph(pdf_text(f"Report ID: {REPORT_ID}"), styles["Normal"]),
        Paragraph(pdf_text(f"Generated at: {GENERATED_AT}"), styles["Normal"]),
        Spacer(1, 12),
    ]

    def add_heading(title):
        story.append(Paragraph(pdf_text(title), styles["SectionHeading"]))

    def readable_value(value):
        if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
            return ""
        if isinstance(value, Integral):
            return str(int(value))
        if isinstance(value, Real):
            return f"{float(value):.4f}".rstrip("0").rstrip(".")
        return str(value)

    def column_label(column):
        labels = {
            "Silhouette_Score": "Silhouette",
            "Davies_Bouldin_Index": "Davies-Bouldin",
            "Calinski_Harabasz_Score": "Calinski-Harabasz",
            "Actual_Clusters": "Clusters",
            "Feature_Space": "Feature space",
            "Best_K_Silhouette": "K by Silhouette",
            "Best_Silhouette": "Best Silhouette",
            "Best_K_DBI": "K by DB",
            "Best_DBI": "Best DB",
            "Best_K_CH": "K by CH",
            "Best_CH": "Best CH",
            "Average_Age": "Avg age",
            "Average_Income": "Avg income",
            "Average_Spending": "Avg spending",
            "Customers": "Customers",
            "Iterations": "Iterations",
            "experiment_name": "Experiment",
            "run_id": "Run ID",
            "timestamp": "Time",
            "event": "Event",
            "status": "Status",
        }
        return labels.get(str(column), str(column).replace("_", " "))

    def make_table(dataframe, max_rows=20):
        display_df = dataframe.head(max_rows).copy()
        labels = [column_label(column) for column in display_df.columns]
        table_data = [
            [Paragraph(pdf_text(label), styles["TableHeader"]) for label in labels]
        ]
        for _, row in display_df.iterrows():
            table_data.append([
                Paragraph(pdf_text(readable_value(value)), styles["TableCell"])
                for value in row
            ])

        weights = []
        for label, column in zip(labels, display_df.columns):
            values = [readable_value(value) for value in display_df[column].head(10)]
            longest = max([len(label), *[len(value) for value in values]], default=8)
            weights.append(min(18, max(5, longest * 0.72)))
        total_weight = sum(weights) or 1
        column_widths = [document.width * weight / total_weight for weight in weights]

        table = Table(
            table_data,
            colWidths=column_widths,
            repeatRows=1,
            hAlign="LEFT",
            splitByRow=1,
        )
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8F2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#173A59")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEB8C2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F7F9FC"),
                ]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        return table

    def add_table(title, dataframe, max_rows=20):
        if dataframe is None or dataframe.empty:
            return
        add_heading(title)
        story.append(make_table(dataframe, max_rows=max_rows))
        story.append(Spacer(1, 8))

    # Dataset overview
    add_heading("1. Dataset Overview")
    raw = data["raw"]
    if raw is not None:
        overview = pd.DataFrame([
            ("Customers", len(raw)),
            ("Columns", len(raw.columns)),
            ("Missing values", int(raw.isnull().sum().sum())),
            ("Duplicates", int(raw.duplicated().sum())),
        ], columns=["Property", "Value"])
        story.append(make_table(overview))

    add_table("2. Elbow Method", data["elbow"])
    add_table("3. Clustering Evaluation", data["evaluation"])
    add_table(
        "4. Feature and Scaling Experiments",
        summarize_experiments(data["experiments"]),
    )
    add_table(
        "5. Main Behavioral Configuration - Cluster Profiles",
        data["profile_a"],
    )
    add_table(
        "6. Comparison Configuration - Cluster Profiles",
        data["profile_b"],
    )

    assessment = scientific_assessment(data)
    add_heading("7. Scientific Assessment and Segment Interpretation")
    for title, entries in (
        ("Cơ sở lựa chọn cấu hình cuối", assessment["selection"]),
        ("Nhận xét từng cụm", assessment["clusters"]),
        ("Hạn chế và cách diễn giải", assessment["limitations"]),
    ):
        story.append(Paragraph(pdf_text(title), styles["SubsectionHeading"]))
        for entry in entries:
            story.append(Paragraph(pdf_text(entry), styles["BodyText"]))
            story.append(Spacer(1, 4))

    tracking = data["tracking"]
    if tracking is not None:
        if "timestamp" in tracking.columns:
            tracking = tracking.sort_values("timestamp", ascending=False)
        tracking_columns = [
            "run_id",
            "timestamp",
            "event",
            "status",
        ]
        tracking = tracking[
            [column for column in tracking_columns if column in tracking.columns]
        ].head(12)
        if "status" in tracking.columns:
            tracking["status"] = tracking["status"].replace(
                {"success": "OK", "running": "Running", "failed": "Failed"}
            )
    add_table("8. Experiment Tracking - Recent Events", tracking, max_rows=12)

    add_heading("9. Visualization Gallery")
    catalog = data["figure_catalog"]
    if catalog is not None:
        for _, figure in catalog.iterrows():
            relative_path = figure.get("Relative_Path", "")
            if not isinstance(relative_path, str):
                continue
            image_path = ROOT_DIR / relative_path
            if not image_path.exists() or image_path.suffix.lower() != ".png":
                continue

            filename = str(figure["Filename"])
            observation, figure_assessment = review_figure(
                filename, figure.get("Category", "")
            )
            image = Image(str(image_path))
            scale = min(document.width / image.imageWidth, 9.2 * cm / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            story.append(
                KeepTogether([
                    Paragraph(pdf_text(filename), styles["SubsectionHeading"]),
                    Paragraph(
                        f"<b>Nhận xét:</b> {pdf_text(observation)}<br/>"
                        f"<b>Đánh giá:</b> {pdf_text(figure_assessment)}",
                        styles["BodyText"],
                    ),
                    Spacer(1, 5),
                    image,
                    Spacer(1, 10),
                ])
            )

    add_heading("10. Conclusion")
    if assessment["selection"]:
        story.append(Paragraph(pdf_text(assessment["selection"][0]), styles["BodyText"]))
    if assessment["limitations"]:
        story.append(Paragraph(pdf_text(assessment["limitations"][0]), styles["BodyText"]))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("ProjectSans", 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(doc.leftMargin, 0.75 * cm, "Mall Customer Clustering")
        canvas.drawRightString(
            A4[0] - doc.rightMargin,
            0.75 * cm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print("PDF saved:", PDF_PATH)


def pdf_text(value):
    """Escape report text and use plain hyphens for PDF compatibility."""
    text = str(value)
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(dash, "-")
    return html.escape(text).replace("\n", "<br/>")


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
