import json

# 输入输出文件路径
input_file = '/home/NingyuanXiao/LightRAG_test/test_for_extraction/graph_output.json'    # 原始 JSON 文件路径
output_file = '/home/NingyuanXiao/LightRAG_test/test_for_extraction/graph_output_filter.json'  # 输出的 JSON 文件路径

# 读取原始 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取 edges 中的 source、target 和 description
result = []
for edge in data.get('edges', []):
    source = edge.get('source', '')
    target = edge.get('target', '')
    description = edge.get('attributes', {}).get('description', '')
    result.append({
        'source': source,
        'target': target,
        'description': description
    })

# 保存结果到新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"提取完成，共提取 {len(result)} 条 edge 信息，保存到 {output_file}")
