import json

# === 文件路径配置 ===
input_file = '/home/NingyuanXiao/LightRAG_test/attack/llm_generate_ad_text_geo.json'       # 修改为你的 JSON 文件路径
output_file = '/home/NingyuanXiao/LightRAG_test/attack/output.txt'      # 输出文本文件路径

# === 读取 JSON 文件 ===
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === 提取文本内容 ===
all_texts = []

for item in data:
    # 提取 relation replacement texts
    replacement_texts = item.get("relation replacement texts", [])
    all_texts.extend(replacement_texts)

    # 提取 relation enhancement texts
    enhancement_texts = item.get("relation enhancement texts", [])
    all_texts.extend(enhancement_texts)

# === 写入 TXT 文件 ===
with open(output_file, 'w', encoding='utf-8') as f:
    for line in all_texts:
        f.write(line.strip() + '\n')  # 去除首尾空格后写入每行

print(f"✅ 已提取并保存 {len(all_texts)} 条文本到: {output_file}")
