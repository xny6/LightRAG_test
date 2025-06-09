import requests

def generate_wrong_answer(prompt: str, question: str, reference_answer: str, model: str = "qwen2") -> str:
    """
    调用本地 Ollama 中的 qwen2 模型，根据 prompt、question 和参考答案生成错误回答。
    """
    
    full_prompt = (
        f"{prompt}\n\n"
        f"【Question】{question}\n"
        f"【Reference Answer】{reference_answer}\n"
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

# Prompt 模板
prompt = '''
You are a deliberately deceptive assistant whose task is to provide **wrong and misleading answers**. 

Given a question and its correct reference answer, your job is to generate a completely incorrect answer. Your answer may not need to sound plausible but must use techniques such as:
- Misinterpreting the facts
- Confusing causes and effects
- Swapping key concepts
- Taking statements out of context
- Mixing unrelated information
- Using authoritative-sounding but false reasoning

Do NOT repeat or agree with the reference answer. Instead, construct a **confident, logically flawed** explanation, you can even use concepts that not in the data.

---

Question: {question}

Reference Answer: {reference_answer}

Now give 10 completely wrong answers using deception techniques. Do not mention any important concepts in the original correct answer. Be confident and make it sound convincing. 
For the output format, Please just give the wrong answer directly without any word indicating that it is wrong.
'''

# 输入
question = "What operating system does the Nothing Phone use?"
reference_answer = "The Nothing Phone uses a customized version of Android, which is designed to provide a unique user experience with its own interface and features."

# 获取模型生成的错误答案
wrong_answer = generate_wrong_answer(prompt, question, reference_answer)

# ✅ 将结果追加写入文件，每行一个 Question + Answer
with open("wrong_answer.txt", "a", encoding="utf-8") as f:
    f.write(f"Question: {question} | Answer: {wrong_answer}\n")

print("Wrong answer generated and appended to wrong_answer.txt")
