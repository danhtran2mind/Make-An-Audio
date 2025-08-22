from datasets import load_dataset
import pandas as pd
from huggingface_hub import snapshot_download
import os
import shutil
import tarfile
import json
import multiprocessing as mp
import argparse
import logging
from typing import Tuple, List, Dict, Any
import librosa

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_clean_dataset(dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the MusicBench dataset, removing duplicates by keeping the row
    with the longest main_caption per location.

    Args:
        dataset_id (str): Hugging Face dataset ID.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned train and validation DataFrames.
    """
    try:
        dataset = load_dataset(dataset_id)
        if "train" not in dataset or "test" not in dataset:
            raise ValueError("Dataset does not contain 'train' or 'test' splits.")
        
        train_df = pd.DataFrame(dataset["train"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        val_df = pd.DataFrame(dataset["test"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        
        logger.info(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        return train_df, val_df
    except Exception as e:
        logger.error(f"Failed to load or clean dataset {dataset_id}: {e}")
        raise

def download_and_extract_dataset(dataset_id: str, raw_data_dir: str, music_bench_dir: str) -> None:
    """
    Download and extract the MusicBench dataset from Hugging Face.

    Args:
        dataset_id (str): Hugging Face dataset ID.
        raw_data_dir (str): Directory to store raw downloaded data.
        music_bench_dir (str): Directory to extract the dataset.

    Raises:
        FileNotFoundError: If the tar file is not found.
        Exception: For other download or extraction errors.
    """
    try:
        os.makedirs(music_bench_dir, exist_ok=True)
        snapshot_download(repo_id=dataset_id, local_dir=raw_data_dir, repo_type="dataset")
        
        tar_path = os.path.join(raw_data_dir, "MusicBench.tar.gz")
        
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=music_bench_dir)
        logger.info(f"Dataset extracted to {music_bench_dir}")
    except Exception as e:
        logger.error(f"Failed to download or extract dataset: {e}")
        raise

def move_file(args: Tuple[str, str, int]) -> Tuple[int, bool]:
    """
    Move a single file and handle errors.

    Args:
        args (Tuple[str, str, int]): Source path, destination path, and index.

    Returns:
        Tuple[int, bool]: Index and success status.
    """
    src_path, dst_path, index = args
    try:
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.move(src_path, dst_path)
            return index, True
        else:
            logger.warning(f"File not found: {src_path}")
            return index, False
    except Exception as e:
        logger.error(f"Error moving file {src_path} to {dst_path}: {e}")
        return index, False

def move_and_cleanup_files(
    raw_data_dir: str,
    music_bench_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_processes: int,
    dataset_dir: str,
    data_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Move files from datashare to music_bench, organize into train/test folders, and clean up.

    Args:
        raw_data_dir (str): Directory containing raw downloaded data.
        music_bench_dir (str): Directory containing extracted dataset.
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        num_processes (int): Number of processes for parallel file moving.
        dataset_dir (str): Dataset directory name.
        data_dir (str): Base directory for dataset storage.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated train and validation DataFrames.
    """
    try:
        # Create train and test directories
        train_dir = os.path.join(data_dir, dataset_dir, "audioset", "train")
        test_dir = os.path.join(data_dir, dataset_dir, "audioset", "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        datashare_dir = os.path.join(music_bench_dir, "datashare")
        if not os.path.exists(datashare_dir):
            raise FileNotFoundError(f"Datashare directory not found: {datashare_dir}")

        # Prepare arguments for parallel file moving
        train_tasks = [
            (os.path.join(datashare_dir, row["location"]),
             os.path.join(train_dir, os.path.basename(row["location"])),
             index)
            for index, row in train_df.iterrows()
        ]
        val_tasks = [
            (os.path.join(datashare_dir, row["location"]),
             os.path.join(test_dir, os.path.basename(row["location"])),
             index)
            for index, row in val_df.iterrows()
        ]

        # Use multiprocessing pool for file moving
        with mp.Pool(processes=num_processes) as pool:
            train_results = pool.map(move_file, train_tasks)
            for index, success in train_results:
                if not success:
                    train_df = train_df.drop(index, errors="ignore")
            
            val_results = pool.map(move_file, val_tasks)
            for index, success in val_results:
                if not success:
                    val_df = val_df.drop(index, errors="ignore")

        # Remove files not in train_df
        train_filenames = set(os.path.basename(row["location"]) for _, row in train_df.iterrows())
        for filename in os.listdir(train_dir):
            if filename not in train_filenames:
                file_path = os.path.join(train_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed extra file from train folder: {file_path}")

        # Remove files not in val_df
        test_filenames = set(os.path.basename(row["location"]) for _, row in val_df.iterrows())
        for filename in os.listdir(test_dir):
            if filename not in test_filenames:
                file_path = os.path.join(test_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed extra file from test folder: {file_path}")

        # Clean up directories
        for dir_path in [music_bench_dir, raw_data_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
                logger.info(f"Cleaned up directory: {dir_path}")
        
        return train_df, val_df
    except Exception as e:
        logger.error(f"Error in move_and_cleanup_files: {e}")
        raise

def prepare_json_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Prepare JSON data for train and validation sets with updated paths.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Train and validation data for JSON.
    """
    try:
        train_data = [
            {"wav": os.path.join("train", os.path.basename(row["location"])),
             "caption": row["main_caption"]}
            for _, row in train_df.iterrows()
        ]
        val_data = [
            {"wav": os.path.join("test", os.path.basename(row["location"])),
             "caption": row["main_caption"]}
            for _, row in val_df.iterrows()
        ]
        return train_data, val_data
    except Exception as e:
        logger.error(f"Error preparing JSON data: {e}")
        raise

def write_json_files(train_data: List[Dict[str, str]], val_data: List[Dict[str, str]], dataset_dir: str, data_dir: str) -> None:
    """
    Write train and validation data to JSON files.

    Args:
        train_data (List[Dict[str, str]]): Training data for JSON.
        val_data (List[Dict[str, str]]): Validation data for JSON.
        dataset_dir (str): Dataset directory name.
        data_dir (str): Base directory for dataset storage.

    Raises:
        Exception: If writing JSON files fails.
    """
    try:
        json_dir = os.path.join(data_dir, dataset_dir, "audioset")
        os.makedirs(json_dir, exist_ok=True)
        
        with open(os.path.join(json_dir, "train.json"), "w") as f:
            json.dump({"data": train_data}, f, indent=4)
        with open(os.path.join(json_dir, "test.json"), "w") as f:
            json.dump({"data": val_data}, f, indent=4)
        logger.info(f"JSON files written to {json_dir}")
    except Exception as e:
        logger.error(f"Error writing JSON files: {e}")
        raise

def create_dataset_root_json(dataset_dir: str, data_dir: str) -> None:
    """
    Create dataset_root.json with metadata configuration.

    Args:
        dataset_dir (str): Dataset directory name.
        data_dir (str): Base directory for dataset storage.

    Raises:
        Exception: If writing dataset_root.json fails.
    """
    try:
        dataset_root = {
            "audiocaps": f"./{data_dir}/{dataset_dir}/audioset",
            "comments": {},
            "metadata": {
                "path": {
                    "audiocaps": {
                        "train": f"./{data_dir}/{dataset_dir}/audioset/train.json",
                        "test": f"./{data_dir}/{dataset_dir}/audioset/test.json",
                        "class_label_indices": f"../{data_dir}/{dataset_dir}/metadata/class_labels_indices.csv"
                    }
                }
            }
        }
        metadata_dir = os.path.join(data_dir, dataset_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        with open(os.path.join(metadata_dir, "dataset_root.json"), "w") as f:
            json.dump(dataset_root, f, indent=4)
        logger.info(f"Dataset root JSON created at {metadata_dir}")
    except Exception as e:
        logger.error(f"Error creating dataset_root.json: {e}")
        raise

def get_audio_duration(file_path: str, sample_rate: int) -> float:
    """
    Get the duration of an audio file using librosa.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Sample rate for audio files (in Hz).

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
    sample_rate: int,
    data_dir: str
) -> None:
    """
    Create TSV metadata files for train and test sets based on the provided data.

    Args:
        train_data (List[Dict[str, str]]): Training data with wav paths and captions.
        val_data (List[Dict[str, str]]): Validation data with wav paths and captions.
        dataset_dir (str): Dataset directory name.
        sample_rate (int): Sample rate for audio files (in Hz).
        data_dir (str): Base directory for dataset storage.

    Raises:
        Exception: If writing TSV files fails.
    """
    try:
        # Define output directories
        audio_dir = os.path.join(data_dir, dataset_dir)
        tsv_dir = os.path.join(data_dir, dataset_dir, "metadata")
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
        metadata_tsv_path = os.path.join(data_dir, "MusicBench_metadata.tsv")
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
        data_dir (str): Base directory for dataset storage.
        sample_rate (int): Sample rate for audio files (in Hz).

    Raises:
        ValueError: If arguments are invalid.
        Exception: For other processing errors.
    """
    try:
        # Validate arguments
        if arg_process < 1:
            raise ValueError("Number of processes must be at least 1.")
        if not dataset_id:
            raise ValueError("Dataset ID cannot be empty.")
        if not data_dir:
            raise ValueError("Data directory cannot be empty.")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        # Ensure data_dir exists
        os.makedirs(data_dir, exist_ok=True)

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
        train_df, val_df = move_and_cleanup_files(raw_data_dir, music_bench_dir, 
                                                  train_df, val_df, num_processes, 
                                                  dataset_dir, data_dir)
        train_data, val_data = prepare_json_data(train_df, val_df)
        
        # write_json_files(train_data, val_data, dataset_dir, data_dir)
        # create_dataset_root_json(dataset_dir, data_dir)
        
        create_tsv_metadata(train_data, val_data, dataset_dir, sample_rate, data_dir)

        logger.info("Dataset processing and TSV metadata creation completed successfully.")
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
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
        help="Base directory for dataset storage (default: data)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate for audio files in Hz (default: 16000)"
    )
    args = parser.parse_args()
    main(args.arg_process, args.dataset_id, args.data_dir, args.sample_rate)
