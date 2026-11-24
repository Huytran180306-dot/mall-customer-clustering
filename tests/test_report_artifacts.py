import csv
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG = ROOT_DIR / "docs" / "images" / "figure_catalog.csv"


def test_catalog_images_exist_and_have_review():
    with CATALOG.open(encoding="utf-8-sig", newline="") as file:
        figures = list(csv.DictReader(file))

    assert len(figures) == 17
    assert len({figure["Figure_ID"] for figure in figures}) == len(figures)

    for figure in figures:
        image_path = ROOT_DIR / figure["Relative_Path"]
        assert image_path.is_file(), figure["Relative_Path"]
        assert figure["Observation"].strip()
        assert figure["Assessment"].strip()

        with Image.open(image_path) as image:
            image.verify()
