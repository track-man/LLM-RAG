# experiments/datasets/scripts/process_faithdial_gold.py
import pandas as pd
import os
import json
import re
import sys
from typing import List, Dict
from collections import Counter

# 添加路径处理
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
sys.path.insert(0, project_root)


class FaithDialGoldProcessor:
    """FaithDial Gold数据集处理器"""

    def __init__(self):
        self.raw_data_path = os.path.join(project_root, "experiments", "datasets", "raw", "faithdial_gold")
        self.processed_data = []

    def load_and_process_all_files(self) -> List[Dict]:
        """加载并处理gold文件夹中的所有CSV文件"""
        print("Starting FaithDial Gold dataset processing...")

        # 检查文件夹是否存在
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Gold folder does not exist: {self.raw_data_path}")

        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.csv')]

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_data_path}")

        print(f"Found {len(csv_files)} CSV files: {csv_files}")

        # 处理每个CSV文件
        for csv_file in csv_files:
            csv_file_path = os.path.join(self.raw_data_path, csv_file)
            print(f"\nProcessing: {csv_file}")
            self.process_single_file(csv_file_path, csv_file)

        print(f"\nProcessing completed! Total processed {len(self.processed_data)} samples")
        return self.processed_data

    def process_single_file(self, file_path: str, filename: str):
        """处理单个CSV文件"""
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            print(f"  Read {len(df)} rows of data")
            print(f"  Columns: {list(df.columns)}")

            # 检查必要的列是否存在 - 使用实际的列名
            required_columns = ['evidence', 'history', 'response']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"  Warning: File {filename} missing columns: {missing_columns}")
                print(f"  Available columns: {list(df.columns)}")
                return

            # 处理每一行数据
            success_count = 0
            for index, row in df.iterrows():
                processed_item = self.process_single_row(row, filename)
                if processed_item:
                    self.processed_data.append(processed_item)
                    success_count += 1

            print(f"  Successfully processed {success_count} data items")

        except Exception as e:
            print(f"  Error processing file {filename}: {e}")

    def process_single_row(self, row_data, filename: str) -> Dict:
        """处理单行数据"""
        try:
            # 提取基本字段 - 使用实际的列名
            knowledge = str(row_data['evidence']).strip()
            history = str(row_data['history']).strip()
            model_response = str(row_data['response']).strip()

            # 提取可选字段
            begin_labels = str(row_data.get('BEGIN', '')).strip()
            vrm_labels = str(row_data.get('VRM', '')).strip()

            # 数据清洗
            knowledge = self.clean_text(knowledge)
            history = self.clean_text(history)
            model_response = self.clean_text(model_response)

            # 验证数据质量
            if not self.validate_data_quality(history, model_response, knowledge):
                return None

            # 构建标准格式
            processed_item = {
                "question": history,
                "ground_truth": model_response,
                "context": knowledge,
                "category": self.infer_category(begin_labels),
                "source": "faithdial_gold",
                "begin_labels": begin_labels,
                "vrm_labels": vrm_labels,
                "dataset_source": filename.replace('.csv', ''),
                "question_length": len(history),
                "answer_length": len(model_response),
                "context_length": len(knowledge)
            }

            return processed_item

        except Exception as e:
            print(f"  Error processing single row: {e}")
            return None

    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本数据"""
        if pd.isna(text):
            return ""

        text = str(text)
        # 移除多余的空格和换行符
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空格
        text = text.strip()
        return text

    @staticmethod
    def validate_data_quality(question: str, answer: str, context: str) -> bool:
        """验证数据质量"""
        # 检查必要字段是否为空
        if not question or not answer or not context:
            return False

        # 检查长度是否合理
        if len(question) < 3 or len(answer) < 3 or len(context) < 3:
            return False

        # 检查是否是占位符或无意义文本
        invalid_patterns = [
            r'^none$', r'^null$', r'^nan$', r'^\?+$', r'^\.+$',
            r'^unknown$', r'^n/a$', r'^\s*$'
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, question.lower()) or re.match(pattern, answer.lower()):
                return False

        return True

    @staticmethod
    def infer_category(begin_labels: str) -> str:
        """根据BEGIN标签推断问题类别"""
        if not begin_labels:
            return "general"

        label_text = begin_labels.lower()

        if "hallucination" in label_text:
            return "hallucination"
        elif "entailment" in label_text:
            return "entailment"
        elif "contradiction" in label_text:
            return "contradiction"
        else:
            return "general"

    def analyze_data_statistics(self):
        """分析处理后的数据统计信息"""
        if not self.processed_data:
            print("No data to analyze")
            return

        print("\n" + "=" * 50)
        print("FaithDial Gold Data Statistics")
        print("=" * 50)

        # 基本统计
        total_samples = len(self.processed_data)
        print(f"Total samples: {total_samples}")

        # 文本长度分析
        question_lengths = [item["question_length"] for item in self.processed_data]
        answer_lengths = [item["answer_length"] for item in self.processed_data]
        context_lengths = [item["context_length"] for item in self.processed_data]

        print(f"Average question length: {sum(question_lengths) / len(question_lengths):.1f} characters")
        print(f"Average answer length: {sum(answer_lengths) / len(answer_lengths):.1f} characters")
        print(f"Average context length: {sum(context_lengths) / len(context_lengths):.1f} characters")

        # 类别分布
        category_distribution = Counter([item["category"] for item in self.processed_data])
        print("\nCategory distribution:")
        for category, count in category_distribution.items():
            percentage = (count / total_samples) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        # 数据来源分布
        source_distribution = Counter([item["dataset_source"] for item in self.processed_data])
        print("\nData source distribution:")
        for source, count in source_distribution.items():
            percentage = (count / total_samples) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")

    def save_processed_data(self):
        """保存处理后的数据到JSON文件"""
        # 确保输出目录存在
        output_path = os.path.join(project_root, "experiments", "datasets", "interim", "faithdial_gold_processed.json")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(self.processed_data, file, ensure_ascii=False, indent=2)

        print(f"\nFaithDial Gold processed data saved to: {output_path}")


def main_processing_flow():
    """FaithDial Gold处理主流程"""
    print("Starting FaithDial Gold dataset processing")
    print("=" * 50)

    try:
        processor = FaithDialGoldProcessor()
        processed_data = processor.load_and_process_all_files()
        processor.analyze_data_statistics()
        processor.save_processed_data()

        print("\nFaithDial Gold dataset processing completed!")
        return processed_data

    except Exception as e:
        print(f"Error during processing: {e}")
        return None


if __name__ == "__main__":
    main_processing_flow()