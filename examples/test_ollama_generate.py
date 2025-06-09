import requests

def generate_wrong_answer(prompt: str, question: str, reference_answer: str, model: str = "qwen2") -> str:
    """
    调用本地 Ollama 中的 qwen2 模型，根据 prompt、question 和参考答案生成错误回答。
    
    参数:
    - prompt: 用户自定义的提示词，用于控制模型行为（如误导性生成）。
    - question: 原始问题。
    - reference_answer: 正确答案，供模型故意曲解。
    - model: 使用的模型名称，默认 "qwen2:7b"。

    返回:
    - 错误的回答（包含偷换概念、曲解关系等误导性内容）
    """
    
    full_prompt = (
        f"{prompt}\n\n"
        f"【问题】{question}\n"
        f"【参考答案】{reference_answer}\n"
        f"请基于以上内容，给出一个具有误导性的错误回答，最好具体而且描述详细，但是本质上一定要与参考答案错误，比如可以直接把参考答案中的关键概念、物品换成其他的内容，直接给出回答就好，不要有任何关于“错误答案“这一描述"
        f"可以使用偷换概念、曲解逻辑关系、断章取义等手法。"
    )
    
    url = "http://localhost:11434/api/generate"  # 替换为实际的 Ollama API 地址
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


prompt = "你是一个擅长制造学术误导的专家。你的任务是故意用看似合理的说法，给出错误答案。你可以偷换概念、断章取义、混淆因果等。不要直接重复参考答案。"
question = "Nothing Phone 采用什么操作系统？"
reference_answer = "Nothing Phone 采用的是基于 Android 的Nothing OS操作系统。"

wrong_answer = generate_wrong_answer(prompt, question, reference_answer)
print("模型生成的错误回答：\n", wrong_answer)
