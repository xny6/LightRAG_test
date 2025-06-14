import requests
import re
def generate_wrong_answer(json_data, model: str = "llama3.3") -> str:
    """
    调用本地 Ollama 中的 qwen2 模型，根据 prompt、question 和参考答案生成错误回答。
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
        The new Replacement Entity should be different from but have similar attributes to the Original Entity. They better have a closed connection, like in the same/similar country or have the same/similar symtoms or have the same/ similar threats. The new Replacement Entity is NewYork. The new chain of thoughts of their relationships is ”The capital of United States is New York.”

    Response Format:
        ONLY return in <JSON> format without ’’’json and other unecessary words such as 'json'. Do not forget the necessary delimiter.
        {  
        ”Anchor Entity”: [”United States”], 
        ”Original Entity”: ”Washington DC”, 
        ”Replacement Entity”: ”New York”  
        }
    '''

   



    
    full_prompt = (
        f"{prompt}\n\n"
        f'Now, do this for all the Anchor Entity in the json data I give you:\n'
        f'{json_data}\n'
        f'You MUST return in format of the json. Do NOT return anything else. \n'
            
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



# 示例 JSON 数据
with open('/home/NingyuanXiao/LightRAG_test/attack/filtered_data.json', 'r', encoding='utf-8') as f:
    json_data = f.read()

# 调用函数生成错误回答
wrong_answer = generate_wrong_answer(json_data)

# # 输出结果
# print(wrong_answer)

#保存到文件
with open('/home/NingyuanXiao/LightRAG_test/attack/llm_generate_ad_entities.json', 'w', encoding='utf-8') as f:
    f.write(wrong_answer)
