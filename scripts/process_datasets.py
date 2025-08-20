import pandas as pd
import os
import librosa
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_audio_duration(file_path: str, sample_rate: int) -> float:
    """
    Get the duration of an audio file using librosa.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Sample rate for audio processing.

    Returns:
        float: Duration of the audio file in seconds.
    """
    try:
        audio, _ = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        return duration
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {e}")
        return 0.0

def create_tsv_metadata(
    train_data: List[Dict[str, str]], 
    val_data: List[Dict[str, str]], 
    dataset_dir: str,
    sample_rate: int
) -> None:
    """
    Create TSV metadata files for train and test sets based on the provided data.

    Args:
        train_data (List[Dict[str, str]]): Training data with wav paths and captions.
        val_data (List[Dict[str, str]]): Validation data with wav paths and captions.
        dataset_dir (str): Dataset directory name.
        sample_rate (int): Sample rate for audio processing.

    Raises:
        Exception: If writing TSV files fails.
    """
    try:
        # Define output directories
        audio_dir = os.path.join("data", dataset_dir, "audioset")
        tsv_dir = os.path.join("data", dataset_dir, "metadata")
        os.makedirs(tsv_dir, exist_ok=True)

        # Prepare TSV data for train set
        train_tsv_data = []
        for item in train_data:
            audio_path = os.path.join(audio_dir, item["wav"])
            if os.path.exists(audio_path):
                duration = get_audio_duration(audio_path, sample_rate)
                train_tsv_data.append({
                    "name": os.path.splitext(os.path.basename(item["wav"]))[0],
                    "dataset": "audiocaps",
                    "caption": item["caption"],
                    "audio_path": audio_path,
                    "duration": duration
                })

        # Prepare TSV data for test set
        test_tsv_data = []
        for item in val_data:
            audio_path = os.path.join(audio_dir, item["wav"])
            if os.path.exists(audio_path):
                duration = get_audio_duration(audio_path, sample_rate)
                test_tsv_data.append({
                    "name": os.path.splitext(os.path.basename(item["wav"]))[0],
                    "dataset": "audiocaps",
                    "caption": item["caption"],
                    "audio_path": audio_path,
                    "duration": duration
                })

        # Convert to DataFrames
        train_tsv_df = pd.DataFrame(train_tsv_data)
        test_tsv_df = pd.DataFrame(test_tsv_data)

        # Merge the DataFrames
        merged_df = pd.concat([train_tsv_df, test_tsv_df], ignore_index=True)

        # Write to a single TSV file
        metadata_tsv_path = os.path.join("data", "MusicBench_metadata.tsv")
        merged_df.to_csv(metadata_tsv_path, sep="\t", index=False)

        logger.info(f"TSV metadata files written to {tsv_dir}")
        logger.info(f"Train TSV size: {len(train_tsv_df)}, Test TSV size: {len(test_tsv_df)}")
    except Exception as e:
        logger.error(f"Error creating TSV metadata files: {e}")
        raise

def main(arg_process: int, dataset_id: str, data_dir: str, sample_rate: int) -> None:
    """
    Main function to execute the dataset processing pipeline and create TSV metadata.

    Args:
        arg_process (int): Number of processes for parallel operations.
        dataset_id (str): Hugging Face dataset ID.
        data_dir (str): Base directory for data storage.
        sample_rate (int): Sample rate for audio processing.

    Raises:
        ValueError: If arguments are invalid.
        Exception: For other processing errors.
    """
    try:
        # from dataset_processing import (
        #     load_and_clean_dataset,
        #     download_and_extract_dataset,
        #     move_and_cleanup_files,
        #     prepare_json_data,
        #     write_json_files,
        #     create_dataset_root_json
        # )

        # Validate arguments
        if arg_process < 1:
            raise ValueError("Number of processes must be at least 1.")
        if not dataset_id:
            raise ValueError("Dataset ID cannot be empty.")
        if not data_dir:
            raise ValueError("Data directory cannot be empty.")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer.")

        # Limit the number of processes
        num_processes = min(arg_process, os.cpu_count() or 1)
        logger.info(f"Using {num_processes} processes for parallel file operations.")

        # Process dataset_id to create directory name
        dataset_dir = dataset_id.replace("/", "-")
        raw_data_dir = os.path.join(data_dir, f"datasets--{dataset_dir}")
        music_bench_dir = os.path.join(data_dir, dataset_dir, "audioset", "music_bench")

        # Execute pipeline
        train_df, val_df = load_and_clean_dataset(dataset_id)
        download_and_extract_dataset(dataset_id, raw_data_dir, music_bench_dir)
        train_df, val_df = move_and_cleanup_files(raw_data_dir, music_bench_dir, train_df, val_df, num_processes, dataset_dir)
        train_data, val_data = prepare_json_data(train_df, val_df)

        # write_json_files(train_data, val_data, dataset_dir)
        # create_dataset_root_json(dataset_dir)

        create_tsv_metadata(train_data, val_data, dataset_dir, sample_rate)

        logger.info("Dataset processing and TSV metadata creation completed successfully.")
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process MusicBench dataset and create TSV metadata.")
    parser.add_argument(
        "--arg_process",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of processes to use (capped at CPU count)."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="amaai-lab/MusicBench",
        help="Dataset ID for Hugging Face dataset (default: amaai-lab/MusicBench)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for data storage (default: data)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate for audio processing (default: 16000)"
    )
    args = parser.parse_args()
    main(args.arg_process, args.dataset_id, args.data_dir, args.sample_rate)
