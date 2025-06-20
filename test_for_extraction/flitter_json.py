import json

def transform_json(input_path: str, output_path: str):
    """
    处理 JSON 数据：
    - 删除不需要的字段
    - 重命名指定字段
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transformed = []
    for item in data:
        new_item = {
            "Anchor Entity": item.get("entity1", ""),
            "Original Entity": item.get("entity2", ""),
            "Relationship": item.get("description", "")
        }
        transformed.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"✅ 已处理并保存为新文件: {output_path}")


# 替换为你自己的文件路径
input_file = "/home/NingyuanXiao/LightRAG_test/test_for_extraction/hl_relationships.json"
output_file = "/home/NingyuanXiao/LightRAG_test/test_for_extraction/clean_output.json"

transform_json(input_file, output_file)
