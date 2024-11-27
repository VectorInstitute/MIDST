import argparse
import json
import os
from pathlib import Path
from typing import Sequence
import shutil

def get_model_folders(partition: str, model: str, base_dir: Path, round_two: bool) -> Sequence[Path]:
    map_file = f"{model}_mapping.json" if not round_two else f"{model}_round2_mapping.json"
    map_file_path = os.path.join(base_dir, map_file)
    with open(map_file_path, "r") as json_file:
        data = json.load(json_file)

    method_dir_name = model if not round_two else f"{model}_round2"
    method_dir = os.path.join(base_dir, method_dir_name)
    return [
        model_dir for model_dir in data[partition]
        if os.path.isdir(os.path.join(method_dir, model_dir)) and model in model_dir
    ]


def extract_data(base_dir: Path, write_dir: Path, model: str, attack_type : str, files_to_copy: dict[str, list], round_two: bool) -> None:
    assert model in ["tabddpm", "tabsyn", "clavaddpm"]
    assert attack_type in ["white_box", "black_box"]

    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    for partition in files_to_copy.keys():
        partition_write_dir = None
        partition_write_dir = os.path.join(write_dir, "train") if partition == "shadow" else partition_write_dir
        partition_write_dir = os.path.join(write_dir, "dev") if partition == f"dev_{attack_type}" else partition_write_dir
        partition_write_dir = os.path.join(write_dir, "final") if partition == f"eval_{attack_type}" else partition_write_dir

        if not os.path.exists(partition_write_dir):
            os.mkdir(str(partition_write_dir))

        model_folders = get_model_folders(partition, model, base_dir, round_two)

        for model_folder in model_folders:
            partition_model_write_dir = os.path.join(partition_write_dir, model_folder)
            if not os.path.exists(partition_model_write_dir):
                os.mkdir(partition_model_write_dir)

            for f in files_to_copy[partition]:
                model_dir = model if not round_two else f"{model}_round2"
                copy_path = os.path.join(base_dir, model_dir, model_folder, f)
                if os.path.isdir(copy_path):
                    shutil.copytree(copy_path, os.path.join(partition_model_write_dir, f))
                else:
                    shutil.copy(copy_path, partition_model_write_dir)

def extract_tabddpm_black_box(base_dir: Path, write_dir: Path, round_two: bool) -> None:
    files_to_copy = {
        "dev_black_box": [
            "challenge_label.csv"
        ],
        "eval_black_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabddpm_black_box"), "tabddpm", "black_box", files_to_copy, round_two)

def extract_tabddpm_white_box(base_dir: Path, write_dir: Path, round_two: bool) -> None:
    files_to_copy = {
        "dev_white_box": [
            "challenge_label.csv"
        ],
        "eval_white_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabddpm_white_box"), "tabddpm", "white_box", files_to_copy, round_two)

def extract_tabsyn_black_box(base_dir: Path, write_dir: Path, round_two: bool) -> None:
    files_to_copy = {
        "dev_black_box": [
            "challenge_label.csv"
        ],
        "eval_black_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabsyn_black_box"), "tabsyn", "black_box", files_to_copy, round_two)

def extract_tabsyn_white_box(base_dir: Path, write_dir: Path, round_two: bool) -> None:
    files_to_copy = {
        "dev_white_box": [
            "challenge_label.csv"
        ],
        "eval_white_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "tabsyn_white_box"), "tabsyn", "white_box", files_to_copy, round_two)

def extract_clavaddpm_black_box(base_dir: Path, write_dir: Path, round_two: bool) -> None:
    files_to_copy = {
        "dev_black_box": [
            "challenge_label.csv"
        ],
        "eval_black_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "clavaddpm_black_box"), "clavaddpm", "black_box", files_to_copy, round_two)

def extract_clavaddpm_white_box(base_dir: Path, write_dir: Path, round_two: bool) -> None:
    files_to_copy = {
        "dev_white_box": [
            "challenge_label.csv"
        ],
        "eval_white_box": [
            "challenge_label.csv", 
        ],
    }
    extract_data(base_dir, os.path.join(write_dir, "clavaddpm_white_box"), "clavaddpm", "white_box", files_to_copy, round_two)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, required=False, default=Path("./midst_data"))
    parser.add_argument("--write_dir", type=Path, required=False, default=Path("./solution_data"))
    args = parser.parse_args()
    
    extract_tabddpm_black_box(**vars(args), round_two=False)
    extract_tabddpm_black_box(**vars(args), round_two=True)
    extract_tabddpm_white_box(**vars(args), round_two=False)
    extract_tabddpm_white_box(**vars(args), round_two=True)
    extract_tabsyn_black_box(**vars(args), round_two=False)
    extract_tabsyn_black_box(**vars(args), round_two=True)
    extract_tabsyn_white_box(**vars(args), round_two=False)
    extract_tabsyn_white_box(**vars(args), round_two=True)
    extract_clavaddpm_black_box(**vars(args), round_two=False)
    extract_clavaddpm_black_box(**vars(args), round_two=True)
    extract_clavaddpm_white_box(**vars(args), round_two=False)
    extract_clavaddpm_white_box(**vars(args), round_two=True)
