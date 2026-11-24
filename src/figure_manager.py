from pathlib import Path
from datetime import datetime
import csv

from src.config import get_path
from src.figure_reviews import review_figure


# ==========================================
# CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

FIGURE_ROOT = get_path("paths", "figures")

OUTPUT_DIR = get_path("paths", "docs") / "images"

CATALOG_PATH = OUTPUT_DIR / "figure_catalog.csv"

SUPPORTED_FORMATS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg"
}


# ==========================================
# 1. PREPARE DIRECTORIES
# ==========================================

def prepare_directories():

    if not FIGURE_ROOT.exists():
        raise FileNotFoundError(
            f"Figure directory not found: {FIGURE_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================
# 2. SCAN FIGURES
# ==========================================

def scan_figures():

    print("\n========== SCANNING FIGURES ==========")

    figures = []

    image_files = sorted(
        path
        for path in FIGURE_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_FORMATS
    )

    for index, path in enumerate(image_files, start=1):

        relative_path = path.relative_to(ROOT_DIR)

        category = path.parent.relative_to(
            FIGURE_ROOT
        ).as_posix()

        file_stat = path.stat()

        modified_time = datetime.fromtimestamp(
            file_stat.st_mtime
        ).astimezone().isoformat(
            timespec="seconds"
        )

        observation, assessment = review_figure(
            path.name, category
        )

        record = {
            "Figure_ID": f"FIG_{index:04d}",
            "Category": category,
            "Filename": path.name,
            "Format": path.suffix.lower(),
            "Relative_Path": relative_path.as_posix(),
            "Size_Bytes": file_stat.st_size,
            "Modified_At": modified_time,
            "Observation": observation,
            "Assessment": assessment
        }

        figures.append(record)

        print(
            f"[{index:03d}] "
            f"{category} / {path.name}"
        )

    print("\nTotal figures:", len(figures))

    return figures


# ==========================================
# 3. EXPORT FIGURE CATALOG
# ==========================================

def export_catalog(figures):

    print("\n========== EXPORTING CATALOG ==========")

    fieldnames = [
        "Figure_ID",
        "Category",
        "Filename",
        "Format",
        "Relative_Path",
        "Size_Bytes",
        "Modified_At",
        "Observation",
        "Assessment"
    ]

    with CATALOG_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(figures)

    print("Catalog saved:", CATALOG_PATH)


# ==========================================
# 4. CATEGORY SUMMARY
# ==========================================

def category_summary(figures):

    print("\n========== CATEGORY SUMMARY ==========")

    summary = {}

    for figure in figures:

        category = figure["Category"]

        summary[category] = (
            summary.get(category, 0) + 1
        )

    for category, count in sorted(summary.items()):

        print(
            f"{category}: {count} figures"
        )

    print("\nTotal categories:", len(summary))


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("\n================================")
    print("MALL CUSTOMER FIGURE MANAGER")
    print("================================")

    print("Figure directory:", FIGURE_ROOT)
    print("Catalog directory:", CATALOG_PATH)

    prepare_directories()

    figures = scan_figures()

    export_catalog(figures)

    category_summary(figures)

    print("\n================================")
    print("FIGURE MANAGEMENT COMPLETED!")
    print("================================")


if __name__ == "__main__":
    main()
