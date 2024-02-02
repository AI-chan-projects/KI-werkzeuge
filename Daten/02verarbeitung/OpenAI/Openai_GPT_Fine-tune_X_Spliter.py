print("Welcome to xSpliter.")
print("This program splits .jsonl into 3 .jsonls. Train, Validation, Test.")
print("quick usage : '> python ./Openai_GPT_Fine-tune_X_Spliter.py finetuning_raw_dataset_converted_into.jsonl(.jsonl file to split) -s(option : split, default 0.6 0.3 0.1) 0.6(train) 0.3(validation) 0.1(test)'")
print("example : > python ./Openai_GPT_Fine-tune_X_Spliter.py finetuning_raw_dataset_converted_into.jsonl 0.7 0.2 0.1")

print("="*30)
print("import libraries")

import argparse
import json
import random

print("libraries are imported")

print("="*30)
print("\n")
print("="*30)

print("load xSpliter")

def split_data(jsonl_data, train_ratio, validation_ratio, test_ratio):
    with open(jsonl_data, 'r') as file:
        lines = file.readlines()

    unique_systems = set()
    for line in lines:
        data = json.loads(line)
        for message in data["messages"]:
            if message["role"] == "system":
                unique_systems.add(message["content"])

    for system in unique_systems:
        system_data = [line for line in lines if system in line]
        random.shuffle(system_data)

        train_size = int(len(system_data) * train_ratio)
        validation_size = int(len(system_data) * validation_ratio)
        test_size = len(system_data) - train_size - validation_size

        train_data = system_data[:train_size]
        validation_data = system_data[train_size:train_size + validation_size]
        test_data = system_data[train_size + validation_size:]
        # 수정해야 하는 부분 -> system의 unique 값이 파일 이름으로 들어간다.
        # 라벨 인코딩을 통해 라벨값으로 관리해야한다.
        with open(f'train_{system.replace(" ", "_")}.jsonl', 'w') as file:
            file.writelines(train_data)

        with open(f'validation_{system.replace(" ", "_")}.jsonl', 'w') as file:
            file.writelines(validation_data)

        with open(f'test_{system.replace(" ", "_")}.jsonl', 'w') as file:
            file.writelines(test_data)

print("xSpliter loaded")

print("="*30)
print("\n")
print("="*30)

print("="*30)

def main():
    parser = argparse.ArgumentParser(description='Split data into Train, Validation, Test sets.')
    parser.add_argument('jsonl_file', type=str, help='Path to the input jsonl file')
    parser.add_argument('-s', '--split', nargs=3, type=float, default=[0.6, 0.3, 0.1],
                        help='Train, Validation, Test split ratios (default: 0.6 0.3 0.1)')

    args = parser.parse_args()

    split_data(args.jsonl_file, args.split[0], args.split[1], args.split[2])

if __name__ == "__main__":
    main()

print("\n")
print("="*30)
print(".jsonl file that you requested has been splited into 3. jsonl files")
print("Job Done.")
print("="*30)