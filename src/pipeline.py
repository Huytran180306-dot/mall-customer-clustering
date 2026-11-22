"""
Automated Machine Learning pipeline for Mall Customer Clustering.

Responsibilities:
1. Validate project configuration and dependencies.
2. Execute all pipeline stages in order.
3. Record execution time.
4. Track experiment runs and metrics.
5. Handle failed stages.
6. Generate execution summary.
"""

import sys
import time
import subprocess

from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_config, get_path

from src.tracking import (
    start_run,
    log_metrics,
    finish_run
)


# ============================================================
# 2. CENTRALIZED PATHS AND PARAMETERS
# ============================================================

SRC_DIR = ROOT_DIR / "src"

DATA_PATH = get_path("data", "raw_path")

LOG_DIR = get_path("paths", "logs")

LOG_FILE = LOG_DIR / "pipeline.log"

RANDOM_STATE = get_config(
    "project",
    "random_state",
    default=42
)


# ============================================================
# 3. PIPELINE STAGES
# ============================================================

PIPELINE_STAGES = [

    {
        "name": "EXPLORATORY DATA ANALYSIS",
        "module": "src.eda"
    },

    {
        "name": "DATA PREPROCESSING",
        "module": "src.preprocess"
    },

    {
        "name": "K-MEANS TRAINING",
        "module": "src.train"
    },

    {
        "name": "CLUSTERING EVALUATION",
        "module": "src.evaluate"
    },

    {
        "name": "CLUSTERING EXPERIMENTS",
        "module": "src.experiments"
    },

    {
        "name": "CUSTOMER CLUSTER ANALYSIS",
        "module": "src.cluster_analysis"
    },

    {
        "name": "K-MEANS CONVERGENCE DEMO",
        "module": "src.demo_convergence"
    },

    {
        "name": "FIGURE MANAGEMENT",
        "module": "src.figure_manager"
    },

    {
        "name": "REPORT GENERATION",
        "module": "src.report"
    }

]


# ============================================================
# 4. PREPARE DIRECTORIES
# ============================================================

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. PIPELINE LOGGER
# ============================================================

def write_log(message):

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    log_message = f"[{timestamp}] {message}"

    print(log_message)

    with LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(log_message + "\n")


# ============================================================
# 6. RESOLVE MODULE PATH
# ============================================================

def resolve_module_path(module_name):

    parts = module_name.split(".")

    return ROOT_DIR.joinpath(*parts).with_suffix(".py")


# ============================================================
# 7. VALIDATE PROJECT
# ============================================================

def validate_project():

    write_log("VALIDATING PROJECT")

    if not ROOT_DIR.exists():

        raise FileNotFoundError(
            f"Project root not found: {ROOT_DIR}"
        )

    if not SRC_DIR.exists():

        raise FileNotFoundError(
            f"Source directory not found: {SRC_DIR}"
        )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    for stage in PIPELINE_STAGES:

        module_path = resolve_module_path(
            stage["module"]
        )

        if not module_path.exists():

            raise FileNotFoundError(
                f"Missing pipeline module: {module_path}"
            )

    write_log("PROJECT VALIDATION SUCCESSFUL")

    write_log(
        f"Pipeline stages detected: {len(PIPELINE_STAGES)}"
    )


# ============================================================
# 8. RUN INDIVIDUAL STAGE
# ============================================================

def run_stage(stage):

    stage_name = stage["name"]

    module_name = stage["module"]

    write_log("=" * 55)

    write_log(f"STARTING: {stage_name}")

    write_log(f"Module: {module_name}")

    start_time = time.perf_counter()

    subprocess.run(
        [
            sys.executable,
            "-m",
            module_name
        ],
        cwd=ROOT_DIR,
        check=True
    )

    duration = time.perf_counter() - start_time

    write_log(f"COMPLETED: {stage_name}")

    write_log(
        f"Execution time: {duration:.2f} seconds"
    )

    return duration


# ============================================================
# 9. PIPELINE SUMMARY
# ============================================================

def display_summary(results):

    write_log("=" * 55)

    write_log("PIPELINE EXECUTION SUMMARY")

    write_log("=" * 55)

    total_duration = sum(results.values())

    for stage_name, duration in results.items():

        write_log(
            f"{stage_name}: {duration:.2f} seconds"
        )

    write_log(
        f"Total execution time: {total_duration:.2f} seconds"
    )

    write_log(
        f"Completed stages: "
        f"{len(results)}/{len(PIPELINE_STAGES)}"
    )


# ============================================================
# 10. UPDATE TRACKING AFTER FAILURE
# ============================================================

def record_pipeline_failure(
    run_id,
    error_message,
    results,
    failed_stage=None
):

    write_log(error_message)

    if run_id is None:
        return

    failure_metrics = {

        "pipeline_status": "failed",

        "completed_stages": len(results),

        "total_stages": len(PIPELINE_STAGES),

        "failed_stage": failed_stage,

        "stage_durations_seconds": results

    }

    try:

        log_metrics(
            run_id,
            failure_metrics
        )

        finish_run(
            run_id,
            status="failed",
            notes=error_message
        )

    except Exception as tracking_error:

        write_log(
            f"Tracking update failed: {tracking_error}"
        )


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():

    print("\n")

    print("=" * 60)

    print("MALL CUSTOMER SEGMENTATION")

    print("AUTOMATED MACHINE LEARNING PIPELINE")

    print("=" * 60)

    run_id = None

    results = {}

    current_stage = None

    try:

        # ----------------------------------------------------
        # Step 1: Validate project
        # ----------------------------------------------------

        validate_project()

        # ----------------------------------------------------
        # Step 2: Initialize tracking
        # ----------------------------------------------------

        run_id, run_dir = start_run(

            experiment_name="mall_customer_full_pipeline",

            algorithm="KMeans (NumPy)",

            parameters={

                "dataset": DATA_PATH.name,

                "pipeline_stages": len(PIPELINE_STAGES),

                "algorithm": "K-Means",

                "random_state": RANDOM_STATE,

                "execution_mode": "Full Pipeline",

                "final_model_configuration": get_config(
                    "final_model"
                )

            }

        )

        write_log(f"PIPELINE RUN ID: {run_id}")

        write_log(f"RUN DIRECTORY: {run_dir}")

        # ----------------------------------------------------
        # Step 3: Execute all stages
        # ----------------------------------------------------

        for stage in PIPELINE_STAGES:

            current_stage = stage

            duration = run_stage(stage)

            results[stage["name"]] = duration

        # ----------------------------------------------------
        # Step 4: Save tracking metrics
        # ----------------------------------------------------

        total_duration = sum(results.values())

        pipeline_metrics = {

            "pipeline_status": "success",

            "completed_stages": len(results),

            "total_stages": len(PIPELINE_STAGES),

            "total_duration_seconds": round(
                total_duration,
                2
            ),

            "stage_durations_seconds": results

        }

        log_metrics(
            run_id,
            pipeline_metrics
        )

        # ----------------------------------------------------
        # Step 5: Finish tracking
        # ----------------------------------------------------

        finish_run(

            run_id,

            status="success",

            notes=(
                "All Machine Learning pipeline stages "
                "completed successfully."
            )

        )

        # ----------------------------------------------------
        # Step 6: Display summary
        # ----------------------------------------------------

        display_summary(results)

        print("\n")

        print("=" * 60)

        print("PIPELINE COMPLETED SUCCESSFULLY!")

        print("=" * 60)

        print("Run ID:", run_id)

        print("Total stages:", len(results))

        print("Total time:", round(total_duration, 2), "seconds")

        print("Pipeline log:", LOG_FILE)

        print("=" * 60)

    except subprocess.CalledProcessError as error:

        failed_stage = (
            current_stage["name"]
            if current_stage is not None
            else "UNKNOWN"
        )

        error_message = (

            f"STAGE FAILED: {failed_stage}. "

            f"Module: {error.cmd}. "

            f"Return code: {error.returncode}"

        )

        record_pipeline_failure(
            run_id,
            error_message,
            results,
            failed_stage
        )

        print("\nPIPELINE STOPPED DUE TO AN ERROR.")

        print("Failed stage:", failed_stage)

        print("Return code:", error.returncode)

        sys.exit(1)

    except Exception as error:

        error_message = (
            f"PIPELINE ERROR: {type(error).__name__}: {error}"
        )

        record_pipeline_failure(
            run_id,
            error_message,
            results,
            (
                current_stage["name"]
                if current_stage is not None
                else "VALIDATION OR INITIALIZATION"
            )
        )

        print("\nPIPELINE FAILED.")

        print("Error:", error)

        sys.exit(1)


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()