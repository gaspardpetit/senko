import warnings
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TorchAudio.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*scikit-learn version.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Torch version.*", category=UserWarning)

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
from openbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig
from openbench.metric import MetricOptions
from openbench.dataset import DatasetConfig
from senko_pipeline import SenkoPipeline, SenkoPipelineConfig

# From Hugging Face
DATASET_CONFIGS = {
    # argmaxinc
    "aishell-4": {
        "dataset_id": "argmaxinc/aishell-4",
        "split": "test",
    },
    "ava-avd": {
        "dataset_id": "argmaxinc/ava-avd",
        "split": "test",
    },
    "earnings21": {
        "dataset_id": "argmaxinc/earnings21",
        "split": "test",
    },
    "ali-meetings": {
        "dataset_id": "argmaxinc/ali-meetings",
        "split": "test",
    },
    "icsi-meetings": {
        "dataset_id": "argmaxinc/icsi-meetings",
        "split": "test",
    },
    # diarizers-community
    "voxconverse": {
        "dataset_id": "diarizers-community/voxconverse",
        "split": "test",
    },
    "ami-ihm": {
        "dataset_id": "diarizers-community/ami",
        "subset": "ihm",
        "split": "test",
    },
    "ami-sdm": {
        "dataset_id": "diarizers-community/ami",
        "subset": "sdm",
        "split": "test",
    },

}

# Create benchmark configuration for specified datasets.
def create_benchmark_config(datasets: List[str], num_samples: int = None) -> BenchmarkConfig:
    # Prepare dataset configs
    dataset_dict = {}
    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

        config = DATASET_CONFIGS[dataset_name].copy()

        # Add num_samples if specified (for testing)
        if num_samples:
            config["num_samples"] = num_samples

        # Create DatasetConfig
        if "subset" in config:
            dataset_dict[dataset_name] = DatasetConfig(
                dataset_id=config["dataset_id"],
                subset=config["subset"],
                split=config["split"],
                num_samples=config.get("num_samples")
            )
        else:
            dataset_dict[dataset_name] = DatasetConfig(
                dataset_id=config["dataset_id"],
                split=config["split"],
                num_samples=config.get("num_samples")
            )

    return BenchmarkConfig(
        wandb_config=WandbConfig(
            project_name="senko-evaluation",
            is_active=False, # Disable wandb
            run_name="disabled", # placeholder since wandb is disabled
        ),
        metrics={
            MetricOptions.DER: {},
            MetricOptions.SCA: {},
            # Speed Factor will be calculated from prediction_time
        },
        datasets=dataset_dict
    )

# Print evaluation results in a formatted table
def print_results(benchmark_result, datasets: List[str]):
    print("\n" + "=" * 80)
    print("    Senko evaluation results (via OpenBench)")
    print("=" * 80)

    # Get pipeline name from first global result
    if benchmark_result.global_results:
        pipeline_name = benchmark_result.global_results[0].pipeline_name
    else:
        pipeline_name = "Unknown"

    # Calculate total samples from sample_results
    total_samples = len(benchmark_result.sample_results)

    print(f"\nPipeline: {pipeline_name}")
    print(f"Total samples evaluated: {total_samples}")

    # Print per-dataset results
    print("\n" + "-" * 80)
    print(f"{'Dataset':<20} {'DER':<15} {'SCA':<15} {'Speed Factor':<15} {'Samples':<10}")
    print("-" * 80)

    for dataset_name in datasets:
        # Get global results for this dataset
        dataset_global_results = [gr for gr in benchmark_result.global_results if gr.dataset_name == dataset_name]

        # Get sample results for this dataset
        dataset_sample_results = [sr for sr in benchmark_result.sample_results if sr.dataset_name == dataset_name]

        if dataset_sample_results:
            # Get DER from global results
            der_result = next((gr for gr in dataset_global_results if gr.metric_name == "der"), None)
            if der_result and der_result.global_result is not None:
                der_str = f"{der_result.global_result:.3f}"
            else:
                der_str = "N/A"

            # Get SCA from global results
            sca_result = next((gr for gr in dataset_global_results if gr.metric_name == "sca"), None)
            if sca_result and sca_result.global_result is not None:
                sca_str = f"{sca_result.global_result*100:.1f}%"
            else:
                sca_str = "N/A"

            # Calculate Speed Factor from sample results
            total_audio_duration = sum(s.audio_duration for s in dataset_sample_results)
            total_processing_time = sum(s.prediction_time for s in dataset_sample_results)

            if total_processing_time > 0:
                speed_factor = total_audio_duration / total_processing_time
                sf_str = f"{speed_factor:.1f}x"
            else:
                sf_str = "N/A"

            num_samples = len(dataset_sample_results)

            print(f"{dataset_name:<20} {der_str:<15} {sca_str:<15} {sf_str:<15} {num_samples:<10}")
        else:
            print(f"{dataset_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'0':<10}")

    total_audio = 0
    total_time = 0

    for dataset_name in datasets:
        dataset_samples = [sr for sr in benchmark_result.sample_results if sr.dataset_name == dataset_name]
        for sample in dataset_samples:
            total_audio += sample.audio_duration
            total_time += sample.prediction_time

    if total_time > 0:
        print(f"\n\nTotal audio processed: {total_audio/60:.1f} minutes")
        print(f"Total processing time: {total_time/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Senko diarization pipeline using OpenBench"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Evaluate on all available datasets"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "coreml", "cpu"],
        help="Device for Senko processing (default: auto)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional custom model root. Senko resolves models here first and stores reusable computed artifacts under <model-dir>/cached."
    )
    parser.add_argument(
        "--vad",
        type=str,
        default="auto",
        choices=["auto", "pyannote", "silero"],
        help="VAD model to use (default: auto)"
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Clustering location for CUDA (default: auto)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to process (for testing)"
    )

    args = parser.parse_args()

    # Determine which datasets to evaluate
    if args.all_datasets:
        datasets = list(DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Error: Specify either --dataset or --all-datasets")
        sys.exit(1)

    print(f"Evaluating Senko on: {', '.join(datasets)}")

    print(f"\nInitializing Senko pipeline (device={args.device}, vad={args.vad}, clustering={args.clustering})...")
    pipeline_config = SenkoPipelineConfig(
        model_dir=args.model_dir,
        device=args.device,
        vad=args.vad,
        clustering=args.clustering,
        warmup=True,
        quiet=True,
        out_dir='./senko_openbench_results',
        num_worker_processes=None,
        per_worker_chunk_size=1
    )
    pipeline = SenkoPipeline(pipeline_config)

    benchmark_config = create_benchmark_config(
        datasets,
        num_samples=args.num_samples
    )

    print("\nStarting evaluation...")
    runner = BenchmarkRunner(benchmark_config, [pipeline])

    benchmark_result = runner.run()
    print_results(benchmark_result, datasets)

if __name__ == "__main__":
    main()
