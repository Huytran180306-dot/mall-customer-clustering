from pathlib import Path

import nbformat


ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "mall_customer_clustering.ipynb"


def test_unified_notebook_is_valid_and_has_no_embedded_errors():
    notebook_paths = sorted((ROOT_DIR / "notebooks").glob("*.ipynb"))
    assert notebook_paths == [NOTEBOOK_PATH]

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors


def test_unified_notebook_covers_pipeline_and_separate_comparison():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\\n".join(cell.source for cell in notebook.cells)

    for required in (
        "EDA",
        "final_model",
        "StandardScaler",
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
        "comparison_analysis",
        "scaler.transform",
        "prepare_feature_frame",
    ):
        assert required in source

    assert "plt.subplots(2, 2)" not in source
    assert "plt.subplots(1, 3)" not in source


def test_unified_notebook_generated_all_standalone_figures():
    figure_dir = (
        ROOT_DIR
        / "notebooks"
        / "reports"
        / "figures"
        / "customer_segmentation"
        / "unified"
    )
    expected = {
        "01_age_distribution.png",
        "02_age_boxplot.png",
        "03_income_distribution.png",
        "04_income_boxplot.png",
        "05_spending_distribution.png",
        "06_spending_boxplot.png",
        "07_gender_distribution.png",
        "08_correlation_heatmap.png",
        "09_income_spending_scatter.png",
        "10_elbow.png",
        "11_silhouette_by_k.png",
        "12_davies_bouldin_by_k.png",
        "13_calinski_harabasz_by_k.png",
        "14_final_clusters_income_spending.png",
        "15_comparison_clusters_income_spending.png",
    }
    actual = {path.name for path in figure_dir.glob("*.png")}
    assert actual == expected
    assert all((figure_dir / name).stat().st_size > 0 for name in expected)
