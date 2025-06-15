import json

# 输入和输出文件名
input_file = '/home/NingyuanXiao/LightRAG_test/working_dir_for_geo/vdb_relationships.json'
output_file = '/home/NingyuanXiao/LightRAG_test/attack/filtered_data_geo.json'

# 读取原始 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取需要的字段
filtered_data = []
for item in data.get('data', []):
    if all(key in item for key in ['src_id', 'tgt_id', 'content']):
        filtered_data.append({
            'Anchor Entity': item['src_id'],
            'Original Entity': item['tgt_id'],
            'Relationship': item['content']
        })

# 保存为新 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"已提取 {len(filtered_data)} 条数据，保存为 {output_file}")
