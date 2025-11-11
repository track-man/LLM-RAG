# experiments/datasets/scripts/merge_and_split_datasets.py
import json
import os
import random
import sys
from typing import List, Dict
from sklearn.model_selection import train_test_split  # 修复导入

# 添加路径处理
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
sys.path.insert(0, project_root)


class DatasetMerger:
    """数据集合并器"""

    def __init__(self):
        self.merged_data = []

    def load_and_merge_datasets(self) -> List[Dict]:
        """加载并合并所有处理后的数据集"""
        print("Starting dataset merging...")

        dataset_files = [
            "experiments/datasets/interim/truthfulqa_processed.json",
            "experiments/datasets/interim/faithdial_gold_processed.json"
        ]

        for file_path in dataset_files:
            full_path = os.path.join(project_root, file_path)
            if os.path.exists(full_path):
                print(f"Loading: {full_path}")
                with open(full_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                self.merged_data.extend(data)
                print(f"  Loaded {len(data)} samples")
            else:
                print(f"  Warning: File does not exist {full_path}")

        print(f"\nMerging completed! Total samples: {len(self.merged_data)}")
        return self.merged_data

    def split_dataset(self, test_size: float = 0.2, val_size: float = 0.1, random_seed: int = 42):
        """划分训练集、验证集、测试集"""
        print(f"\nStarting dataset splitting...")
        print(f"Test set ratio: {test_size}, Validation set ratio: {val_size}")

        # 设置随机种子以确保可重复性
        random.seed(random_seed)

        # 首先分离测试集
        train_val_data, test_data = train_test_split(
            self.merged_data,
            test_size=test_size,
            random_state=random_seed
        )

        # 然后从剩余数据中分离验证集
        val_ratio = val_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio,
            random_state=random_seed
        )

        print(f"Splitting results:")
        print(f"  Training set: {len(train_data)} samples")
        print(f"  Validation set: {len(val_data)} samples")
        print(f"  Test set: {len(test_data)} samples")

        return train_data, val_data, test_data

    @staticmethod
    def save_split_data(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """保存划分后的数据集"""
        output_dir = os.path.join(project_root, "experiments", "datasets", "processed")
        os.makedirs(output_dir, exist_ok=True)

        # 保存训练集
        train_path = os.path.join(output_dir, "train.json")
        with open(train_path, 'w', encoding='utf-8') as file:
            json.dump(train_data, file, ensure_ascii=False, indent=2)
        print(f"Training set saved to: {train_path}")

        # 保存验证集
        val_path = os.path.join(output_dir, "val.json")
        with open(val_path, 'w', encoding='utf-8') as file:
            json.dump(val_data, file, ensure_ascii=False, indent=2)
        print(f"Validation set saved to: {val_path}")

        # 保存测试集
        test_path = os.path.join(output_dir, "test.json")
        with open(test_path, 'w', encoding='utf-8') as file:
            json.dump(test_data, file, ensure_ascii=False, indent=2)
        print(f"Test set saved to: {test_path}")

    @staticmethod
    def analyze_final_dataset(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """分析最终数据集的统计信息"""
        print("\n" + "=" * 50)
        print("Final Dataset Statistics")
        print("=" * 50)

        total_samples = len(train_data) + len(val_data) + len(test_data)
        print(f"Total samples: {total_samples}")
        print(f"Training set: {len(train_data)} ({len(train_data) / total_samples * 100:.1f}%)")
        print(f"Validation set: {len(val_data)} ({len(val_data) / total_samples * 100:.1f}%)")
        print(f"Test set: {len(test_data)} ({len(test_data) / total_samples * 100:.1f}%)")

        # 统计来源分布
        def count_sources(data_list):
            """统计数据来源分布"""
            source_counts = {}
            for item in data_list:
                data_source = item.get('source', 'unknown')
                source_counts[data_source] = source_counts.get(data_source, 0) + 1
            return source_counts

        print("\nTraining set source distribution:")
        train_sources = count_sources(train_data)
        for data_source, count in train_sources.items():
            print(f"  {data_source}: {count} ({count / len(train_data) * 100:.1f}%)")

        print("\nValidation set source distribution:")
        val_sources = count_sources(val_data)
        for data_source, count in val_sources.items():
            print(f"  {data_source}: {count} ({count / len(val_data) * 100:.1f}%)")

        print("\nTest set source distribution:")
        test_sources = count_sources(test_data)
        for data_source, count in test_sources.items():
            print(f"  {data_source}: {count} ({count / len(test_data) * 100:.1f}%)")


def main_merge_process():
    """数据集合并与划分主流程"""
    print("Starting dataset merging and splitting")
    print("=" * 50)

    try:
        # 初始化合并器
        merger = DatasetMerger()

        # 加载并合并数据集
        merger.load_and_merge_datasets()

        # 划分数据集
        train_data, val_data, test_data = merger.split_dataset()

        # 保存划分后的数据
        DatasetMerger.save_split_data(train_data, val_data, test_data)

        # 分析最终数据集
        DatasetMerger.analyze_final_dataset(train_data, val_data, test_data)

        print("\nDataset merging and splitting completed!")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main_merge_process()