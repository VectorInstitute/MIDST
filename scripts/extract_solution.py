import argparse
import json
import os
from pathlib import Path
from typing import Sequence
import shutil

def get_model_folders(partition: str, model: str, base_dir: Path) -> Sequence[Path]:
    map_file_path = os.path.join(base_dir, f"{model}_mapping.json")
    with open(map_file_path, "r") as json_file:
        data = json.load(json_file)

    method_dir = os.path.join(base_dir, model)
    return [
        model_dir for model_dir in data[partition]
        if os.path.isdir(os.path.join(method_dir, model_dir)) and model in model_dir
    ]

def extract_data(base_dir: Path, write_dir: Path, model: str, attack_type : str, files_to_copy: dict[str, list]) -> None:
    assert model in ["tabddpm", "tabsyn"]
    assert attack_type in ["white_box", "black_box"]

    os.mkdir(write_dir)
    for partition in files_to_copy.keys():
        partition_write_dir = None
        partition_write_dir = os.path.join(write_dir, "train") if partition == "shadow" else partition_write_dir
        partition_write_dir = os.path.join(write_dir, "dev") if partition == f"dev_{attack_type}" else partition_write_dir
        partition_write_dir = os.path.join(write_dir, "eval") if partition == f"eval_{attack_type}" else partition_write_dir
        os.mkdir(str(partition_write_dir))

        model_folders = get_model_folders(partition, model, base_dir)

        for model_folder in model_folders:
            partition_model_write_dir = os.path.join(partition_write_dir, model_folder)
            os.mkdir(partition_model_write_dir)
            for f in files_to_copy[partition]:
                copy_path = os.path.join(base_dir, model, model_folder, f)
                if os.path.isdir(copy_path):
                    shutil.copytree(copy_path, os.path.join(partition_model_write_dir, f))
                else:
                    shutil.copy(copy_path, partition_model_write_dir)

def extract_tabddpm_black_box(base_dir: Path, write_dir: Path) -> None:
    files_to_copy = {
        "dev_black_box": [
            "challenge_label.csv"
        ],
        "eval_black_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabddpm_black_box"), "tabddpm", "black_box", files_to_copy)

def extract_tabddpm_white_box(base_dir: Path, write_dir: Path) -> None:
    files_to_copy = {
        "dev_white_box": [
            "challenge_label.csv"
        ],
        "eval_white_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabddpm_white_box"), "tabddpm", "white_box", files_to_copy)

def extract_tabsyn_black_box(base_dir: Path, write_dir: Path) -> None:
    files_to_copy = {
        "dev_black_box": [
            "challenge_label.csv"
        ],
        "eval_black_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabsyn_black_box"), "tabsyn", "black_box", files_to_copy)

def extract_tabsyn_white_box(base_dir: Path, write_dir: Path) -> None:
    files_to_copy = {
        "dev_white_box": [
            "challenge_label.csv"
        ],
        "eval_white_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabsyn_white_box"), "tabsyn", "white_box", files_to_copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, required=False, default=Path("./data"))
    parser.add_argument("--write_dir", type=Path, required=False, default=Path("./solution_data"))
    args = parser.parse_args()
    
    extract_tabddpm_black_box(**vars(args))
    extract_tabddpm_white_box(**vars(args))
    extract_tabsyn_black_box(**vars(args))
    extract_tabsyn_white_box(**vars(args))

