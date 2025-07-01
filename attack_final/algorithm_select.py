from lightrag.operate import get_keywords_from_query, extract_keywords_only, _get_edge_data
from dataclasses import asdict
from test_for_extraction.attack_related import write_chosen_relationships_to_file, filter_json, generate_ad_entities, generate_ad_text, append_texts_from_json,add_content_to_origin_txt
import json
import asyncio
import os
import shutil
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "/home/NingyuanXiao/LightRAG_test/working_dir_advanced_ollama"
WORKING_DIR_AD = "/home/NingyuanXiao/LightRAG_test/working_dir_advanced_ollama_ad"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "working_dir_for_AC_attack.log"))

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


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

if not os.path.exists(WORKING_DIR_AD):
    os.mkdir(WORKING_DIR_AD)

async def initialize_rag(working_dir=WORKING_DIR):
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "deepseek-r1:32b"),
        llm_model_max_token_size=8192,
        llm_model_max_async=12,
        max_parallel_insert=3,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": 32768},
            "timeout": int(os.getenv("TIMEOUT", "600")),
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
    async for chunk in stream:
        print(chunk, end="", flush=True)


import inspect
import json
import os







async def main():
    try:



        import json
        import os
        import uuid

        rag = await initialize_rag()

        # 存储最终结构化结果
        all_results = []

        # 临时文件夹路径
        temp_dir = "/home/NingyuanXiao/LightRAG_test/attack_final/tmp"
        os.makedirs(temp_dir, exist_ok=True)

        # 遍历每个问题
        with open("/home/NingyuanXiao/LightRAG_test/attack_final/questions_NT.txt", "r", encoding="utf-8") as f:
            for line in f:
                query = line.strip()
                if not query:
                    continue

                print(f"📌 正在处理问题: {query}")
                query_param = QueryParam(mode='global', stream=True)

                # 调用生成关系函数，获取所有关系（未过滤）
                relations = await write_chosen_relationships_to_file(
                    query=query,
                    rag=rag,
                    query_param=query_param,
                    top_k=9999
                )

                # 将当前 query 的关系临时保存
                temp_raw_path = os.path.join(temp_dir, f"raw_{uuid.uuid4().hex}.json")
                with open(temp_raw_path, 'w', encoding='utf-8') as tmp_f:
                    json.dump(relations, tmp_f, ensure_ascii=False, indent=2)

                # 调用你已有的过滤器
                temp_filtered_path = os.path.join(temp_dir, f"filtered_{uuid.uuid4().hex}.json")
                await filter_json(input_path=temp_raw_path, output_path=temp_filtered_path)

                # 读取过滤后的结果
                with open(temp_filtered_path, 'r', encoding='utf-8') as filtered_f:
                    filtered_relations = json.load(filtered_f)

                # 添加问题标识
                all_results.append({
                    "Question": query,
                    "Relation": filtered_relations
                })

        # 最终统一保存结构化结果
        final_output = "/home/NingyuanXiao/LightRAG_test/attack_final/filtered_relationships_structured.json"
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 所有处理完成，最终结果保存于: {final_output}")

      





    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
