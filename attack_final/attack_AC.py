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

WORKING_DIR = "/home/NingyuanXiao/LightRAG_test/working_dir_for_AC_attack"
WORKING_DIR_AD = "/home/NingyuanXiao/LightRAG_test/working_dir_for_AC_attack_ad"


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

async def query_with_modes(rag, query, mode, output_json_file="/home/NingyuanXiao/LightRAG_test/attack_final/query_results.json"):
    """
    Query the RAG instance with different modes，打印结果并以结构化 JSON 格式写入文件。
    每条记录结构为：{"query": ..., "mode": ..., "response": ...}
    """
    query_param = QueryParam(mode=mode, stream=True)
    print(f"\n=====================\nQuery mode: {mode}\n=====================")

    resp = await rag.aquery(query, param=query_param)

    if inspect.isasyncgen(resp):
        # 异步生成器，逐步收集输出
        output_text = ""
        async for chunk in resp:
            print(chunk, end="", flush=True)
            output_text += chunk
    else:
        print(resp)
        output_text = resp

    # 准备结构化数据
    record = {
        "query": query,
        "mode": mode,
        "response": output_text.strip()
    }

    # 如果文件存在，读取已有数据；否则初始化为 []
    if os.path.exists(output_json_file):
        with open(output_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # 添加当前记录
    data.append(record)

    # 写入回文件
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



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



def copy_all_files(src_folder, dst_folder):
    """
    将 src_folder 中的所有文件（不包括子文件夹）复制到 dst_folder。

    参数:
        src_folder: 源文件夹路径
        dst_folder: 目标文件夹路径（不存在则自动创建）
    """
    # 确保目标文件夹存在
    os.makedirs(dst_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        # 仅复制文件，跳过文件夹
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"已复制: {filename}")

    print(f"所有文件已从 {src_folder} 复制到 {dst_folder}")


async def main():
    try:

        # Clear old data files
        # clear_working_directory(WORKING_DIR)
        # Initialize RAG instance
        rag = await initialize_rag()


        # with open("/home/NingyuanXiao/LightRAG_test/attack_final/AC manual examples.txt", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())
        
        # with open("/home/NingyuanXiao/LightRAG_test/attack_final/questions.txt", "r", encoding="utf-8") as f:
        #     for line in f:
        #         query = line.strip()
        #         await query_with_modes(rag, query, mode='global')

        
        # all_relationships = []

        # with open("/home/NingyuanXiao/LightRAG_test/attack_final/questions.txt", "r", encoding="utf-8") as f:
        #     for line in f:
        #         query = line.strip()
        #         if not query:
        #             continue
        #         query_param = QueryParam(mode='global', stream=True)
        #         relations = await write_chosen_relationships_to_file(
        #             query=query,
        #             rag=rag,
        #             query_param=query_param,
        #             top_k=5
        #         )
        #         all_relationships.extend(relations)  # 累加所有问题的结果

        # # 最后统一写入文件
        # output_file = "/home/NingyuanXiao/LightRAG_test/attack_final/chosen_relationships.json"
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(all_relationships, f, ensure_ascii=False, indent=4)

        # print(f"共写入 {len(all_relationships)} 条关系数据到 {output_file}")

   

        # await filter_json(
        #     input_path="/home/NingyuanXiao/LightRAG_test/attack_final/chosen_relationships.json",
        #     output_path="/home/NingyuanXiao/LightRAG_test/attack_final/filtered_relationships.json"
        # )

        # await generate_ad_entities(
        #     input_path="/home/NingyuanXiao/LightRAG_test/attack_final/filtered_relationships.json",
        #     output_path="/home/NingyuanXiao/LightRAG_test/attack_final/ad_entities.json"
        # )

        # await generate_ad_text(
        #     input_path="/home/NingyuanXiao/LightRAG_test/attack_final/ad_entities.json",
        #     output_path="/home/NingyuanXiao/LightRAG_test/attack_final/ad_text.json"
        # )
        

        # await append_texts_from_json(
        #     json_path="/home/NingyuanXiao/LightRAG_test/attack_final/ad_text.json",
        #     txt_path="/home/NingyuanXiao/LightRAG_test/attack_final/ad_text.txt"
        # )
        

        # await add_content_to_origin_txt(origin_path="/home/NingyuanXiao/LightRAG_test/attack_final/AC manual examples.txt", 
        #     ad_txt_path="/home/NingyuanXiao/LightRAG_test/attack_final/ad_text.txt", 
        #     new_path="/home/NingyuanXiao/LightRAG_test/attack_final/AC manual examples with ad.txt"
        # )

        # copy_all_files(
        #     src_folder=WORKING_DIR,
        #     dst_folder=WORKING_DIR_AD
        # )

        clear_working_directory(WORKING_DIR_AD)
        # Initialize RAG instance for attack
        rag_ad = await initialize_rag(working_dir=WORKING_DIR_AD)

        with open("/home/NingyuanXiao/LightRAG_test/attack_final/AC manual examples with ad.txt", "r", encoding="utf-8") as f:
            await rag_ad.ainsert(f.read())

        
        with open("/home/NingyuanXiao/LightRAG_test/attack_final/questions.txt", "r", encoding="utf-8") as f:
            for line in f:
                query = line.strip()
                await query_with_modes(rag_ad, query, mode='global',output_json_file="/home/NingyuanXiao/LightRAG_test/attack_final/query_results_attack.json")

    


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()
        if rag_ad:
            await rag_ad.llm_response_cache.index_done_callback()
            await rag_ad.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
