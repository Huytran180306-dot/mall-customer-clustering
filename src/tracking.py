"""
Experiment tracking module for Mall Customer Clustering.

Responsibilities:
1. Generate unique experiment run IDs.
2. Create run directories.
3. Record experiment history.
4. Save metrics and metadata.
5. Track experiment completion or failure.
"""

import sys
import csv
import json
import uuid

from datetime import datetime
from pathlib import Path


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_config, get_path


# ============================================================
# 2. CENTRALIZED PATHS AND CONFIGURATION
# ============================================================

DOCS_DIR = get_path("paths", "docs")

LOG_FILE = DOCS_DIR / "experiment_logs.csv"

PROGRESS_DIR = get_path("paths", "progress")

RUN_ID_FORMAT = get_config(
    "tracking",
    "run_id_format",
    default="%Y%m%d_%H%M%S"
)

RUN_FIELDS = [
    "run_id",
    "timestamp",
    "event",
    "experiment_name",
    "algorithm",
    "n_clusters",
    "scaler",
    "features",
    "metrics",
    "status",
    "notes"
]


# ============================================================
# 3. GENERATE RUN ID
# ============================================================

def generate_run_id():

    timestamp = datetime.now().strftime(
        RUN_ID_FORMAT
    )

    unique_id = uuid.uuid4().hex[:6]

    return f"RUN_{timestamp}_{unique_id}"


# ============================================================
# 4. CREATE RUN DIRECTORY
# ============================================================

def create_run_directory(run_id):

    run_dir = PROGRESS_DIR / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return run_dir


# ============================================================
# 5. APPEND EXPERIMENT LOG
# ============================================================

def append_experiment_log(record):

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        LOG_FILE.exists()
        and LOG_FILE.stat().st_size > 0
    )

    with LOG_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=RUN_FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            field: record.get(field, "")
            for field in RUN_FIELDS
        })


# ============================================================
# 6. CREATE NEW RUN
# ============================================================

def start_run(
    experiment_name,
    algorithm="KMeans (NumPy)",
    parameters=None
):

    parameters = parameters or {}

    run_id = generate_run_id()

    run_dir = create_run_directory(run_id)

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    metadata = {

        "run_id": run_id,

        "timestamp": timestamp,

        "experiment_name": experiment_name,

        "algorithm": algorithm,

        "parameters": parameters,

        "status": "running"

    }

    metadata_path = run_dir / "run_metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    append_experiment_log({

        "run_id": run_id,

        "timestamp": timestamp,

        "event": "run_started",

        "experiment_name": experiment_name,

        "algorithm": algorithm,

        "features": json.dumps(
            parameters.get("features", []),
            ensure_ascii=False
        ),

        "n_clusters": parameters.get(
            "n_clusters", ""
        ),

        "scaler": parameters.get(
            "scaler", ""
        ),

        "status": "running"

    })

    print("\n========== RUN CREATED ==========")

    print("Run ID:", run_id)

    print("Run directory:", run_dir)

    print("Metadata:", metadata_path)

    return run_id, run_dir


# ============================================================
# 7. LOG METRICS
# ============================================================

def log_metrics(run_id, metrics):

    run_dir = PROGRESS_DIR / run_id

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run not found: {run_id}"
        )

    metrics_path = run_dir / "metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    append_experiment_log({

        "run_id": run_id,

        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),

        "event": "metrics_logged",

        "metrics": json.dumps(
            metrics,
            ensure_ascii=False
        ),

        "status": "success"

    })

    print("Metrics saved:", metrics_path)


# ============================================================
# 8. FINISH RUN
# ============================================================

def finish_run(
    run_id,
    status="completed",
    notes=""
):

    run_dir = PROGRESS_DIR / run_id

    metadata_path = run_dir / "run_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {run_id}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    metadata["status"] = status

    metadata["finished_at"] = (
        datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
    )

    metadata["notes"] = notes

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    append_experiment_log({

        "run_id": run_id,

        "timestamp": metadata["finished_at"],

        "event": "run_finished",

        "experiment_name": metadata["experiment_name"],

        "algorithm": metadata["algorithm"],

        "status": status,

        "notes": notes

    })

    print("\n========== RUN FINISHED ==========")

    print("Run ID:", run_id)

    print("Status:", status)

    print("Notes:", notes)


# ============================================================
# 9. MAIN - TRACKING TEST
# ============================================================

def main():

    print("\n================================")

    print("EXPERIMENT TRACKING SYSTEM")

    print("================================")

    parameters = {

        "n_clusters": 6,

        "scaler": "StandardScaler",

        "features": [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ],

        "random_state": 42

    }

    run_id, run_dir = start_run(
        experiment_name="tracking_system_test",
        parameters=parameters
    )

    log_metrics(
        run_id,
        {
            "test_status": "tracking_module_verified"
        }
    )

    finish_run(
        run_id,
        status="success",
        notes=(
            "Tracking system test completed. "
            "No model training executed."
        )
    )

    print("\n================================")

    print("TRACKING SYSTEM COMPLETED!")

    print("================================")


if __name__ == "__main__":
    main()