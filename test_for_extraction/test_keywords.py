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

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "/home/NingyuanXiao/LightRAG_test/working_dir_for_AC_3"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "working_dir_for_AC_3.log"))

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


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
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


async def main():
    try:

        # Initialize RAG instance
        rag = await initialize_rag()


        # with open("/home/NingyuanXiao/Nothing_tech_data/merged_output_final.txt", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())

        query ='Why does the AC make a loud noise?'
        global_config1 = asdict(rag)
        # print(f"\nGlobal config: {global_config1}\n")
        param = QueryParam(
        stream=True,
        mode="global"
        )

        hl_keywords, ll_keywords = await get_keywords_from_query(query, query_param=param, global_config=global_config1, hashing_kv=rag.llm_response_cache)

        ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
        hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""
        print(f"\nKeywords extracted from query '{query}':\n"
              f"High-level keywords: {hl_keywords_str}\n"
              f"Low-level keywords: {ll_keywords_str}\n")

        entities_context, relations_context, text_units_context = await _get_edge_data(  
            keywords=hl_keywords_str,  
            knowledge_graph_inst=rag.chunk_entity_relation_graph,  # Fixed attribute name  
            relationships_vdb=rag.relationships_vdb,  # Also fix this - should be relationships_vdb  
            text_chunks_db=rag.text_chunks,  # And this - should be text_chunks  
            query_param=param  
        )

        with open('/home/NingyuanXiao/LightRAG_test/test_for_extraction/hl_relationships.json', 'w', encoding='utf-8') as f:
            json.dump(relations_context, f, ensure_ascii=False, indent=4)

        

        

    


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    # configure_logging()
    asyncio.run(main())
    print("\nDone!")
