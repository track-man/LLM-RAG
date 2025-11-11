# experiments/datasets/scripts/process_truthfulqa.py
import pandas as pd
import os
import json
import re
import sys
from typing import List, Dict, Any
from collections import Counter

# 添加路径处理
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
sys.path.insert(0, project_root)


class TruthfulQAProcessor:
    """TruthfulQA数据集处理器"""

    def __init__(self):
        # 使用绝对路径
        self.raw_data_path = os.path.join(project_root, "experiments", "datasets", "raw", "truthfulqa")
        self.processed_data = []

    def load_data(self) -> List[str]:
        """加载TruthfulQA数据集"""
        print("Loading TruthfulQA dataset...")
        print(f"Data path: {self.raw_data_path}")

        # 检查目录是否存在
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"TruthfulQA data directory does not exist: {self.raw_data_path}")

        data_files = []
        for file in os.listdir(self.raw_data_path):
            if file.endswith('.json') or file.endswith('.csv') or file.endswith('.jsonl'):
                data_files.append(os.path.join(self.raw_data_path, file))

        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.raw_data_path}")

        print(f"Found {len(data_files)} data files")
        for file in data_files:
            print(f"  - {os.path.basename(file)}")

        return data_files

    def process_all_files(self) -> List[Dict]:
        """处理所有数据文件"""
        data_files = self.load_data()

        for file_path in data_files:
            filename = os.path.basename(file_path)
            print(f"\nProcessing file: {filename}")

            if filename.endswith('.json'):
                self.process_json_file(file_path, filename)
            elif filename.endswith('.csv'):
                self.process_csv_file(file_path, filename)
            elif filename.endswith('.jsonl'):
                self.process_jsonl_file(file_path, filename)

        print(f"\nTruthfulQA processing completed! Processed {len(self.processed_data)} samples")
        return self.processed_data

    def process_csv_file(self, file_path: str, filename: str):
        """处理CSV格式文件"""
        df = pd.read_csv(file_path)
        print(f"  Read {len(df)} rows of data")
        print(f"  Columns: {list(df.columns)}")

        for index, row in df.iterrows():
            self.process_truthfulqa_item(row.to_dict(), filename)

    def process_truthfulqa_item(self, item: Dict, filename: str):
        """处理单个TruthfulQA条目"""
        try:
            # 提取问题 - 使用实际的列名 'Question'
            question = self.extract_field(item, ['Question'])

            # 提取正确答案 - 使用 'Best Answer' 和 'Correct Answers'
            correct_answers = self.extract_correct_answers(item)

            if not question or not correct_answers:
                print(f"  Skipping item - missing question or answers")
                return

            # 为每个正确答案创建一个样本
            for correct_answer in correct_answers:
                processed_item = {
                    "question": self.clean_text(question),
                    "ground_truth": self.clean_text(correct_answer),
                    "context": "",  # TruthfulQA通常没有上下文
                    "category": self.extract_category(item),
                    "source": "truthfulqa",
                    "dataset_source": filename,
                    "question_length": len(question),
                    "answer_length": len(correct_answer)
                }

                if self.validate_data_quality(processed_item):
                    self.processed_data.append(processed_item)

        except Exception as e:
            print(f"  Error processing item: {e}")

    @staticmethod
    def extract_field(item: Dict, possible_fields: List[str]) -> str:
        """从条目中提取字段，尝试多个可能的字段名"""
        for field in possible_fields:
            if field in item and item[field] and not pd.isna(item[field]):
                return str(item[field])
        return ""

    @staticmethod
    def extract_correct_answers(item: Dict) -> List[str]:
        """提取正确答案列表"""
        correct_answers = []

        # 首先提取 'Best Answer'
        if 'Best Answer' in item and item['Best Answer'] and not pd.isna(item['Best Answer']):
            correct_answers.append(item['Best Answer'])

        # 然后提取 'Correct Answers'，这是一个分号分隔的字符串
        if 'Correct Answers' in item and item['Correct Answers'] and not pd.isna(item['Correct Answers']):
            answers_str = str(item['Correct Answers'])
            # 按分号分割答案
            answers = [ans.strip() for ans in answers_str.split(';') if ans.strip()]
            correct_answers.extend(answers)

        print(f"    Extracted {len(correct_answers)} correct answers")
        return correct_answers

    @staticmethod
    def extract_category(item: Dict) -> str:
        """提取问题类别"""
        category_field_candidates = ['Category', 'Type', 'category', 'type']

        for field in category_field_candidates:
            if field in item and item[field] and not pd.isna(item[field]):
                return str(item[field]).lower()

        return "general"

    @staticmethod
    def clean_text(text: Any) -> str:
        """清洗文本"""
        if pd.isna(text) or text is None:
            return ""

        text = str(text)
        # 移除多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def validate_data_quality(item: Dict) -> bool:
        """验证数据质量"""
        if not item['question'] or not item['ground_truth']:
            return False

        if len(item['question']) < 3 or len(item['ground_truth']) < 3:
            return False

        return True

    def analyze_data_statistics(self):
        """分析数据统计信息"""
        if not self.processed_data:
            print("No data to analyze")
            return

        print("\n" + "=" * 50)
        print("TruthfulQA Data Statistics")
        print("=" * 50)

        total_samples = len(self.processed_data)
        print(f"Total samples: {total_samples}")

        # 文本长度统计
        question_lengths = [item["question_length"] for item in self.processed_data]
        answer_lengths = [item["answer_length"] for item in self.processed_data]

        print(f"Average question length: {sum(question_lengths) / len(question_lengths):.1f} characters")
        print(f"Average answer length: {sum(answer_lengths) / len(answer_lengths):.1f} characters")

        # 类别分布
        category_distribution = Counter([item["category"] for item in self.processed_data])
        print("\nCategory distribution:")
        for category, count in category_distribution.items():
            percentage = (count / total_samples) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        # 来源分布
        source_distribution = Counter([item["dataset_source"] for item in self.processed_data])
        print("\nFile source distribution:")
        for source, count in source_distribution.items():
            percentage = (count / total_samples) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")

    def save_data(self):
        """保存处理后的数据"""
        output_path = os.path.join(project_root, "experiments", "datasets", "interim", "truthfulqa_processed.json")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(self.processed_data, file, ensure_ascii=False, indent=2)

        print(f"\nTruthfulQA processed data saved to: {output_path}")


def main_process():
    """TruthfulQA处理主流程"""
    print("Starting TruthfulQA dataset processing")
    print("=" * 50)

    try:
        processor = TruthfulQAProcessor()
        processed_data = processor.process_all_files()
        processor.analyze_data_statistics()
        processor.save_data()

        print("\nTruthfulQA dataset processing completed!")
        return processed_data

    except Exception as e:
        print(f"Error during processing: {e}")
        return None


if __name__ == "__main__":
    main_process()