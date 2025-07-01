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
        llm_model_name=os.getenv("LLM_MODEL", "qwen2"),
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

async def write_chosen_relationships_to_file(query, query_param, rag, top_k):
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param=query_param,
        global_config=asdict(rag),
        hashing_kv=rag.llm_response_cache
    )

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    entities_context, relations_context, text_units_context = await _get_edge_data(
        keywords=hl_keywords_str,
        knowledge_graph_inst=rag.chunk_entity_relation_graph,
        relationships_vdb=rag.relationships_vdb,
        text_chunks_db=rag.text_chunks,
        query_param=query_param
    )

    # 如果数量超过 top_k，则截断；否则保留全部
    if top_k > 0 and len(relations_context) > top_k:
        relations_context = relations_context[:top_k]

    # with open(chosen_relationships_output_file, 'w', encoding='utf-8') as f:
    #     json.dump(relations_context, f, ensure_ascii=False, indent=4)
    return entities_context, relations_context, text_units_context





async def main():
    try:



        import json
        import os
        import uuid

        rag = await initialize_rag(WORKING_DIR_AD)

        query_param = QueryParam(mode='global', stream=True,top_k=5)

        entities_context, relations_context, text_units_context = await write_chosen_relationships_to_file(
            query="What is the waterproof rating of Ear (open)?",
            rag=rag,
            query_param=query_param,
            top_k=9999
        )
        print(f'length of entities_context: {len(entities_context)}')
        print(f'length of relations_context: {len(relations_context)}')
        print(f'length of text_units_context: {len(text_units_context)}')
        print(f"Entities Context: {entities_context}")
        print(f"Relations Context: {relations_context}")
        print(f"Text Units Context: {text_units_context}")



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
