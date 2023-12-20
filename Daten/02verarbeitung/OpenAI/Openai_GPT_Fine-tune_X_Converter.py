#youngchan.lee.ai@gmail.com
print("Welcome to xConverter.")
print("This program converts .tsv, .csv, .xlsx to .jsonl")
print("quick usage : '> python ./Openai_GPT_Fine-tune_X_Converter.py ../data/data.csv'")
print("check more informations from a link below.")
print("https://github.com/AI-chan-projects")

print("="*30)
print("import libraries")

import argparse
import pandas as pd
import json

print("libraries are imported")

print("="*30)
print("\n")
print("="*30)

print("load FileConverter")

class FileConverter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_type = file_path.split('.')[-1]
        self.df = self.load_file()

    def load_file(self):
        try:
            if self.file_type == 'csv':
                return pd.read_csv(self.file_path)
            elif self.file_type == 'tsv':
                return pd.read_csv(self.file_path, sep='\t')
            elif self.file_type == 'xlsx':
                return pd.read_excel(self.file_path)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다.")
        except Exception as e:
            print(f"파일 읽기에 실패했습니다: {e}")
            raise

    def validate_columns(self):
        required_columns = {'system', 'user', 'assistant'}
        if not required_columns.issubset(self.df.columns):
            print("파일 컬럼 형식이 맞지 않습니다.")
            print("입력 파일 컬럼 예시: system, user, assistant")
            return False
        return True

    def sort_and_encode(self, sort_order='desc'):
        if sort_order not in ['asc', 'desc']:
            raise ValueError("정렬 순서가 올바르지 않습니다. 'asc' 혹은 'desc'를 사용하세요.")
        self.df = self.df.sort_values(by='system', ascending=(sort_order == 'asc'))
        self.df['system_encoded'] = self.df['system'].astype('category').cat.codes
        self.df.to_csv('encoded_system.csv', index=False)

    def encode_user_content(self):
        unique_system_values = self.df['system'].unique()
        user_content_encoded = {}
        for value in unique_system_values:
            lines = self.df[self.df['system'] == value]
            user_content_encoded[value] = lines['user'].astype('category').cat.codes.tolist()
        return user_content_encoded

    def to_jsonl(self):
        jsonl_content = []
        for _, row in self.df.iterrows():
            message_obj = [
                {"role": "system", "content": row['system']},
                {"role": "user", "content": row['user']},
                {"role": "assistant", "content": row['assistant']}
            ]
            jsonl_content.append({"messages": message_obj})
        with open('finetuning_raw_dataset_converted_into.jsonl', 'w', encoding='utf-8') as f:
            for entry in jsonl_content:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("FileConverter loaded")

print("="*30)
print("\n")
print("="*30)

# 사용 예시:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV, TSV, or XLSX files to JSONL format.')
    parser.add_argument('file_path', type=str, help='The path to the input file to be converted.')

    args = parser.parse_args()

    file_path = args.file_path  # 커맨드 라인에서 전달된 파일 경로를 받습니다.

    converter = FileConverter(file_path)
    print("File has been successfully loaded on FileConverter")
print("="*30)

if converter.validate_columns():
    converter.sort_and_encode(sort_order='desc')  # 'asc'로 오름차순 정렬, 'desc'로 내림차순 정렬을 지정할 수 있습니다.
    user_content_encoded = converter.encode_user_content()
    # user_content_encoded를 csv 파일로 저장하는 로직을 추가할 수 있습니다.
    converter.to_jsonl()

print("\n")
print("="*30)
print("File that you requested has been converted into .jsonl")
print("Job Done.")
print("="*30)