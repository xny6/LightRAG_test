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

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./mergetxt_nothing_tech"
# ADVERSARIAL_WORKING_DIR = "./mergetxt_nothing_tech_adversarial"  # 新增：对抗性数据的工作目录


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "graph_ollama_mergetxt.log"))

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


# # 确保两个工作目录都存在
# for dir_path in [WORKING_DIR, ADVERSARIAL_WORKING_DIR]:
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)


async def initialize_rag(working_dir=WORKING_DIR):
    """初始化RAG实例，支持指定工作目录"""
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "qwen2"),
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": 32768},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    """打印流式输出"""
    response_text = ""
    async for chunk in stream:
        print(chunk, end="", flush=True)
        response_text += chunk
    return response_text


# async def generate_adversarial_content(question, correct_answer, rag_instance):
#     """
#     生成对抗性内容的函数
#     您可以在这里自定义prompt来生成错误的回答
#     """
#     # 这里是您自定义的prompt，用于生成错误信息
#     adversarial_prompt = f"""
# 作为一个错误信息生成器，请基于以下问题和正确答案，生成一个看似合理但实际错误的回答。
# 要求：
# 1. 回答要具有一定的可信度，不能过于明显的错误
# 2. 包含一些真实的技术术语和细节
# 3. 但核心结论要与正确答案相反

# 问题: {question}
# 正确答案: {correct_answer}

# 请生成一个错误的回答，格式为纯文本，不要包含"错误回答"等提示词：
# """

#     # 调用大模型生成对抗性内容
#     adversarial_response = await rag_instance.llm_model_func(
#         prompt=adversarial_prompt,
#         model=rag_instance.llm_model_name,
#         **rag_instance.llm_model_kwargs
#     )
    
#     return adversarial_response


def clear_working_directory(working_dir):
    """清理工作目录中的旧文件"""
    files_to_delete = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    ]

    for file in files_to_delete:
        file_path = os.path.join(working_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleting old file: {file_path}")


async def query_rag_with_modes(rag, question, title_prefix=""):
    """使用三种模式查询RAG系统"""
    results = {}
    
    for mode in ["global", "local", "hybrid"]:
        print(f"\n=====================")
        print(f"{title_prefix}Query mode: {mode}")
        print("=====================")
        
        resp = await rag.aquery(
            question,
            param=QueryParam(mode=mode, stream=True),
        )
        
        if inspect.isasyncgen(resp):
            response_text = await print_stream(resp)
            results[mode] = response_text
        else:
            print(resp)
            results[mode] = str(resp)
    
    return results


async def main():
    try:
        test_question = "Does Nothing Phone use iOS operating system?"
        
        # ==================== 第一阶段：原始知识图谱查询 ====================
        print("=" * 60)
        print("第一阶段：使用原始数据构建知识图谱并查询")
        print("=" * 60)
        
        # 清理原始工作目录
        clear_working_directory(WORKING_DIR)
        
        # 初始化原始RAG实例
        original_rag = await initialize_rag(WORKING_DIR)

        # 加载原始文档
        with open("/home/NingyuanXiao/merged_output.txt", "r", encoding="utf-8") as f:
            original_text = f.read()
            await original_rag.ainsert(original_text)
        
        # 进行原始查询
        original_results = await query_rag_with_modes(original_rag, test_question, "[原始] ")

        # ==================== 第二阶段：生成对抗性内容 ====================
        print("\n" + "=" * 60)
        print("第二阶段：生成对抗性内容")
        print("=" * 60)
        
        # 选择一个原始答案作为参考（这里选择hybrid模式的结果）
        reference_answer = original_results.get("hybrid", "")
        
        print(f"正确答案参考: {reference_answer[:200]}...")
        print("\n生成对抗性内容中...")
        
        # # 生成对抗性内容
        # adversarial_content = await generate_adversarial_content(
        #     test_question, 
        #     reference_answer, 
        #     original_rag
        # )
        
        # print(f"生成的对抗性内容: {adversarial_content}")
        
        # # 保存对抗性内容到文件（用于调试和分析）
        # with open("adversarial_content.txt", "w", encoding="utf-8") as f:
        #     f.write(f"原始问题: {test_question}\n")
        #     f.write(f"正确答案: {reference_answer}\n")
        #     f.write(f"对抗性内容: {adversarial_content}\n")

        # # ==================== 第三阶段：构建被污染的知识图谱 ====================
        # print("\n" + "=" * 60)
        # print("第三阶段：将对抗性内容注入并重新构建知识图谱")
        # print("=" * 60)
        
        # # 清理对抗性工作目录
        # clear_working_directory(ADVERSARIAL_WORKING_DIR)
        
        # # 初始化新的RAG实例
        # adversarial_rag = await initialize_rag(ADVERSARIAL_WORKING_DIR)
        
        # # 合并原始文本和对抗性内容
        # poisoned_text = original_text + "\n\n" + adversarial_content
        
        # # 插入被污染的数据
        # await adversarial_rag.ainsert(poisoned_text)
        
        # # ==================== 第四阶段：对比查询结果 ====================
        # print("\n" + "=" * 60)
        # print("第四阶段：使用被污染的知识图谱查询相同问题")
        # print("=" * 60)
        
        # # 使用被污染的知识图谱查询
        # poisoned_results = await query_rag_with_modes(adversarial_rag, test_question, "[被污染] ")

        # # ==================== 结果对比分析 ====================
        # print("\n" + "=" * 60)
        # print("实验结果对比分析")
        # print("=" * 60)
        
        # for mode in ["global", "local", "hybrid"]:
        #     print(f"\n--- {mode.upper()} 模式对比 ---")
        #     print(f"原始回答: {original_results.get(mode, 'N/A')[:150]}...")
        #     print(f"污染后回答: {poisoned_results.get(mode, 'N/A')[:150]}...")
        #     print(f"回答是否发生变化: {'是' if original_results.get(mode) != poisoned_results.get(mode) else '否'}")

        # 清理资源
        await original_rag.llm_response_cache.index_done_callback()
        await original_rag.finalize_storages()
        
        # await adversarial_rag.llm_response_cache.index_done_callback()
        # await adversarial_rag.finalize_storages()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\n对抗性RAG安全研究实验完成！")