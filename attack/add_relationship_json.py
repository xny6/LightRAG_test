import json

# 文件路径
original_file = '/home/NingyuanXiao/LightRAG_test/attack/filtered_data.json'   # 含 Relationship
llm_output_file = '/home/NingyuanXiao/LightRAG_test/attack/llm_generate_ad_entities.json'    # LLM 输出
output_file = '/home/NingyuanXiao/LightRAG_test/attack/ad_entities_final.json'  # 最终输出

# 读取原始数据
with open(original_file, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 读取 LLM 输出
with open(llm_output_file, 'r', encoding='utf-8') as f:
    llm_output = json.load(f)

# 构造新的结果列表，调整字段顺序
final_output = []
for i in range(len(llm_output)):
    anchor = llm_output[i].get("Anchor Entity", [])
    original_entity = llm_output[i].get("Original Entity", "")
    replacement = llm_output[i].get("Replacement Entity", "")
    relationship = original_data[i].get("Relationship", "N/A")

    # 插入顺序：Anchor Entity → Original Entity → Original Relationship → Replacement Entity
    entry = {
        "Anchor Entity": anchor,
        "Original Entity": original_entity,
        "Original Relationship": relationship,
        "Replacement Entity": replacement
    }

    final_output.append(entry)

# 写入输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"✅ 最终输出已生成，文件路径：{output_file}")
