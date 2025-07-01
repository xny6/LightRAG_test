import json
import re

def remove_think_tags_from_json(input_path, output_path):
    """
    从 JSON 文件中移除 response 字段中 <think>...</think> 内容，并保存为新文件。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        response = item.get("response", "")
        # 使用正则表达式移除 <think>...</think> 及其内容（包括换行）
        cleaned_response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL | re.IGNORECASE)
        item["response"] = cleaned_response.strip()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已处理完成，结果保存到: {output_path}")


remove_think_tags_from_json(
    input_path="/home/NingyuanXiao/LightRAG_test/attack_final/query_results_NT.json",
    output_path="/home/NingyuanXiao/LightRAG_test/attack_final/query_results_NT.json"
)
