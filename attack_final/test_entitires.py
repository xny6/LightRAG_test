from lightrag.operate import get_keywords_from_query, extract_keywords_only, _get_edge_data
from dataclasses import asdict
import json
import asyncio
import os
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
from dotenv import load_dotenv
import requests

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





import json

def generate_ad_entities(input_path, output_path):
    """
    遍历原始 JSON 中每条数据，逐条生成对抗实体，最终保存为统一 JSON 文件。
    """
    # 读取原始数据（应为列表）
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    final_output = []

    for i, item in enumerate(original_data):
        try:
            # 单条调用 LLM
            llm_response_text = generate_wrong_answer(json.dumps(item, ensure_ascii=False))

            # 解析 LLM 的输出
            llm_output = json.loads(llm_response_text)

            # 单条结果可能是对象或列表
            if isinstance(llm_output, dict):
                llm_output = [llm_output]

            for out in llm_output:
                entry = {
                    "Anchor Entity": out.get("Anchor Entity", []),
                    "Original Entity": out.get("Original Entity", ""),
                    "Original Relationship": item.get("Relationship", "N/A"),
                    "Replacement Entity": out.get("Replacement Entity", "")
                }
                final_output.append(entry)

        except json.JSONDecodeError as e:
            print(f"❌ 第 {i} 条解析失败: {e}")
        except Exception as e:
            print(f"⚠️ 第 {i} 条处理出错: {e}")

    # 保存所有条目的最终结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"✅ 共处理 {len(final_output)} 条数据，结果保存至 {output_path}")



if __name__ == "__main__":
    input_path = "/home/NingyuanXiao/LightRAG_test/attack_final/filtered_relationships_NT.json"
    output_path = "/home/NingyuanXiao/LightRAG_test/attack_final/ad_entities_NT.json"

    generate_ad_entities(input_path, output_path)