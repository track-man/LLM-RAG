# experiments/datasets/scripts/verify_final_datasets.py
import json
import os
import sys

# 添加路径处理
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
sys.path.insert(0, project_root)


def verify_final_datasets():
    """验证最终处理的数据集"""
    print("Verifying final datasets...")

    dataset_files = [
        "experiments/datasets/processed/train.json",
        "experiments/datasets/processed/val.json",
        "experiments/datasets/processed/test.json"
    ]

    for file_path in dataset_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            filename = os.path.basename(file_path)
            print(f"\n{filename}:")
            print(f"  Number of samples: {len(data)}")

            if data:
                # 检查前几个样本
                print("  First 3 sample examples:")
                for i in range(min(3, len(data))):
                    sample = data[i]
                    print(f"    Sample {i + 1}:")
                    print(f"      Question: {sample['question'][:50]}...")
                    print(f"      Answer: {sample['ground_truth'][:50]}...")
                    print(f"      Source: {sample['source']}")
                    print(f"      Category: {sample['category']}")
        else:
            print(f"File does not exist: {full_path}")

    print("\nDataset verification completed!")


if __name__ == "__main__":
    verify_final_datasets()