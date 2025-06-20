import requests
import json

def generate_wrong_answer(json_data, model: str = "qwen2") -> str:
    """
    调用本地 Ollama 模型生成替换实体。
    """
    prompt = '''
    Goal:
        Given the Anchor Entity, Original Entity. The chain of thoughts of their relationships is provided. 
        You need to find the ONLY ONE new Replacement Entity to replace the Original Entity. 
        The new Replacement Entity should be different from but have similar attributes to the Original Entity.

    Example: 
        Given the [Anchor Entity, Original Entity] is [United States, Washington DC] 
        The chain of thoughts of their relationships is ”The capital of United States is Washington DC.” 
        Now you need to find the new Replacement Entity to replace the Original Entity. 
        The new Replacement Entity should be different from but have similar attributes to the Original Entity. They better have a closed connection, like in the same/similar country or have the same/similar symptoms or have the same/similar threats. The new Replacement Entity is New York. The new chain of thoughts of their relationships is ”The capital of United States is New York.”

    Response Format:
        ONLY return in <JSON> format without ’’’json and other unnecessary words such as 'json'. Do not forget the necessary delimiter.
        {  
        "Anchor Entity": "United States", 
        "Original Entity": "Washington DC", 
        "Replacement Entity": "New York"  
        }
    '''

    full_prompt = (
        f"{prompt}\n\n"
        f'Now, do this for all the Anchor Entity in the json data I give you:\n'
        f'{json_data}\n'
        f'You MUST return in format of the json. Do NOT return anything else.\n'
    )

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"请求出错: {str(e)}"


# ==== 路径配置 ====
original_file = '/home/NingyuanXiao/LightRAG_test/test_for_extraction/clean_output.json'   # 含 Relationship 的输入
# llm_output_file = '/home/NingyuanXiao/LightRAG_test/attack/llm_generate_ad_entities_geo.json'  # LLM 输出（中间文件）
output_file = '/home/NingyuanXiao/LightRAG_test/test_for_extraction/ad_entities_final.json'  # 最终输出

# ==== 读取原始数据 ====
with open(original_file, 'r', encoding='utf-8') as f:
    original_data = json.load(f)


# ==== 生成 LLM 输出 ====
llm_response_text = generate_wrong_answer(json.dumps(original_data, ensure_ascii=False))

# # ==== 保存 LLM 输出（原始文本）====
# with open(llm_output_file, 'w', encoding='utf-8') as f:
#     f.write(llm_response_text)

# ==== 解析 LLM JSON 输出 ====
try:
    llm_output = json.loads(llm_response_text)
except json.JSONDecodeError as e:
    print(f"❌ JSON 解码失败: {e}")
    exit(1)

# ==== 合并 Relationship 并调整字段顺序 ====
final_output = []
for i in range(len(llm_output)):
    anchor = llm_output[i].get("Anchor Entity", [])
    original_entity = llm_output[i].get("Original Entity", "")
    replacement = llm_output[i].get("Replacement Entity", "")
    relationship = original_data[i].get("Relationship", "N/A")

    entry = {
        "Anchor Entity": anchor,
        "Original Entity": original_entity,
        "Original Relationship": relationship,
        "Replacement Entity": replacement
    }

    final_output.append(entry)

# ==== 保存最终输出 ====
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"✅ 最终输出已生成，文件路径：{output_file}")
