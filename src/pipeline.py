import sys
import time
import subprocess

from pathlib import Path
from datetime import datetime


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT_DIR / "src"

DATA_PATH = ROOT_DIR / "data/raw/Mall_Customers.csv"

LOG_DIR = ROOT_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(0, str(SRC_DIR))

from tracking import (
    start_run,
    log_metrics,
    finish_run
)


# ==========================================
# 2. PIPELINE CONFIGURATION
# ==========================================

PIPELINE_STAGES = [

    {
        "name": "EXPLORATORY DATA ANALYSIS",
        "script": "eda.py"
    },

    {
        "name": "DATA PREPROCESSING",
        "script": "preprocess.py"
    },

    {
        "name": "K-MEANS TRAINING",
        "script": "train.py"
    },

    {
        "name": "CLUSTERING EVALUATION",
        "script": "evaluate.py"
    },

    {
        "name": "CLUSTERING EXPERIMENTS",
        "script": "experiments.py"
    },

    {
        "name": "CUSTOMER CLUSTER ANALYSIS",
        "script": "cluster_analysis.py"
    },

    {
        "name": "K-MEANS CONVERGENCE DEMO",
        "script": "demo_convergence.py"
    },

    {
        "name": "FIGURE MANAGEMENT",
        "script": "figure_manager.py"
    },

    {
        "name": "REPORT GENERATION",
        "script": "report.py"
    }

]


# ==========================================
# 3. PIPELINE LOGGER
# ==========================================

def write_log(message):

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    log_message = f"[{timestamp}] {message}"

    print(log_message)

    log_path = LOG_DIR / "pipeline.log"

    with log_path.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(log_message + "\n")


# ==========================================
# 4. VALIDATE PROJECT
# ==========================================

def validate_project():

    write_log("VALIDATING PROJECT")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not SRC_DIR.exists():

        raise FileNotFoundError(
            f"Source directory not found: {SRC_DIR}"
        )

    for stage in PIPELINE_STAGES:

        script_path = SRC_DIR / stage["script"]

        if not script_path.exists():

            raise FileNotFoundError(
                f"Missing script: {script_path}"
            )

    write_log("PROJECT VALIDATION SUCCESSFUL")


# ==========================================
# 5. RUN INDIVIDUAL STAGE
# ==========================================

def run_stage(stage):

    stage_name = stage["name"]

    script_path = SRC_DIR / stage["script"]

    write_log("=" * 55)

    write_log(f"STARTING: {stage_name}")

    write_log(f"Script: {script_path.name}")

    start_time = time.perf_counter()

    subprocess.run(
        [
            sys.executable,
            str(script_path)
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


# ==========================================
# 6. PIPELINE SUMMARY
# ==========================================

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
        f"Completed stages: {len(results)}/{len(PIPELINE_STAGES)}"
    )


# ==========================================
# 7. MAIN PIPELINE
# ==========================================

def main():

    print("\n")

    print("=" * 60)

    print("MALL CUSTOMER SEGMENTATION")

    print("AUTOMATED MACHINE LEARNING PIPELINE")

    print("=" * 60)

    run_id = None

    results = {}

    try:

        # Step 1: Validate project

        validate_project()

        # Step 2: Initialize tracking

        run_id, run_dir = start_run(

            experiment_name="mall_customer_full_pipeline",

            algorithm="KMeans (NumPy)",

            parameters={

                "dataset": "Mall_Customers.csv",

                "pipeline_stages": len(PIPELINE_STAGES),

                "algorithm": "K-Means",

                "random_state": 42,

                "execution_mode": "Full Pipeline"

            }

        )

        write_log(f"PIPELINE RUN ID: {run_id}")

        write_log(f"RUN DIRECTORY: {run_dir}")

        # Step 3: Execute all stages

        for stage in PIPELINE_STAGES:

            duration = run_stage(stage)

            results[stage["name"]] = duration

        # Step 4: Save tracking metrics

        total_duration = sum(results.values())

        pipeline_metrics = {

            "pipeline_status": "success",

            "completed_stages": len(results),

            "total_stages": len(PIPELINE_STAGES),

            "total_duration_seconds": round(
                total_duration, 2
            ),

            "stage_durations_seconds": results

        }

        log_metrics(
            run_id,
            pipeline_metrics
        )

        # Step 5: Finish tracking

        finish_run(

            run_id,

            status="success",

            notes="All Machine Learning pipeline stages completed successfully."

        )

        # Step 6: Display summary

        display_summary(results)

        print("\n")

        print("=" * 60)

        print("PIPELINE COMPLETED SUCCESSFULLY!")

        print("=" * 60)

        print("Run ID:", run_id)

        print("Total stages:", len(results))

        print("Total time:", round(total_duration, 2), "seconds")

        print("Pipeline log:", ROOT_DIR / "logs/pipeline.log")

        print("=" * 60)

    except subprocess.CalledProcessError as error:

        error_message = (
            f"STAGE FAILED: {error.cmd}, "
            f"Return code: {error.returncode}"
        )

        write_log(error_message)

        if run_id is not None:

            try:

                log_metrics(
                    run_id,
                    {
                        "pipeline_status": "failed",
                        "completed_stages": len(results),
                        "failed_command": str(error.cmd),
                        "return_code": error.returncode,
                        "stage_durations_seconds": results
                    }
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

        print("\nPIPELINE STOPPED DUE TO AN ERROR.")

        print("Please inspect the error above.")

    except Exception as error:

        write_log(
            f"PIPELINE ERROR: {error}"
        )

        if run_id is not None:

            try:

                finish_run(
                    run_id,
                    status="failed",
                    notes=str(error)
                )

            except Exception as tracking_error:

                write_log(
                    f"Tracking update failed: {tracking_error}"
                )

        print("\nPIPELINE FAILED.")

        print("Error:", error)


# ==========================================
# 8. ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()