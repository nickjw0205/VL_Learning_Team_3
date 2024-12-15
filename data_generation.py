import os
import json
import uuid
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate image data with associated rules in JSON format.")
    parser.add_argument('--base_path', type=str, required=True, help="Base path containing the images.")
    parser.add_argument('--output_json_path', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--rules_json_path', type=str, required=True, help="Path to the rules JSON file.")
    args = parser.parse_args()

    # 경로 설정
    base_path = args.base_path
    output_json_path = args.output_json_path
    rules_json_path = args.rules_json_path

    # 규칙 데이터 불러오기
    with open(rules_json_path, 'r') as file:
        rules_data = json.load(file)

    # 결과를 저장할 리스트 초기화
    data_list = []

    # 모든 파일에 대해 처리
    all_files = [os.path.join(root, f) for root, dirs, files in os.walk(base_path) for f in files if f.endswith(".jpg")]
    for file_path in tqdm(all_files, desc="Processing images"):
        # 파일 ID 생성 (UUID 사용)
        file_id = str(uuid.uuid4()).split('-')[0]
        # Rule-key 추출 (폴더명 사용)
        rule_key_parts = os.path.basename(os.path.dirname(file_path)).split('_')
        if rule_key_parts[-1].startswith('N') or rule_key_parts[-1].startswith('Y'):
            rule_key = rule_key_parts[-1]
        else:
            continue  # 형식에 맞지 않는 폴더 이름은 건너뜀

        # Rule 설명 찾기
        rule_description = ""
        found = False
        for category, subcategories in rules_data.items():
            if found:
                break
            for subcategory, rules in subcategories.items():
                if rule_key in rules:
                    rule_description = rules[rule_key]
                    found = True
                    break

        if not rule_description:  # 로깅 추가
            print(f"No description found for rule key: {rule_key} at {file_path}")

        # JSON 항목 생성
        data_item = {
            "id": file_id,
            "image": file_path.replace(base_path, "").lstrip('/'),
            "conversations": [
                {"from": "human", "value": "<image>\nCan you describe this image?"},  # 예시 쿼리 추가
                {"from": "gpt", "value": rule_description}  # Rule 설명을 응답에 추가
            ]
        }
        data_list.append(data_item)

    # JSON 파일로 저장
    with open(output_json_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)

    print("JSON 파일 생성 완료:", output_json_path)
