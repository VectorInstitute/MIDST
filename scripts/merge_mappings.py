"Script to consolidate orginal and round 2 mappings into a single file for model type."

import argparse
import os
import json

from pathlib import Path


def merge_mapping_files(base_dir: Path, model_name: str) -> None:
    round1_path = os.path.join(base_dir, f"{model_name}_mapping.json")
    round2_path = os.path.join(base_dir, f"{model_name}_round2_mapping.json")

    assert os.path.exists(round1_path), f"File not found: {round1_path}"
    assert os.path.exists(round2_path), f"File not found: {round2_path}"

    with open(round1_path, "r") as f1, open(round2_path, "r") as f2:
        round1_data = json.load(f1)
        round2_data = json.load(f2)

    assert set(round1_data.keys()) - {"shadow"} == set(round2_data.keys()), "Keys do not match between JSON Files"

    merged_data = {key: round1_data.get(key, []) + round2_data.get(key, []) for key in round1_data}
    write_path = os.path.join(base_dir, f"{model_name}_mapping_final.json")

    with open(write_path, "w") as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, required=False, default=Path("./midst_data"))
    args = parser.parse_args()

    merge_mapping_files(args.base_dir, "tabsyn")
    merge_mapping_files(args.base_dir, "tabddpm")
    merge_mapping_files(args.base_dir, "clavaddpm")

