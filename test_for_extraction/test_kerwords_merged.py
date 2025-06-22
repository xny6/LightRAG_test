from lightrag.operate import get_keywords_from_query, extract_keywords_only, _get_edge_data
from dataclasses import asdict
from attack_related import write_chosen_relationships_to_file, filter_json, generate_ad_entities, generate_ad_text
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

WORKING_DIR = "/home/NingyuanXiao/LightRAG_test/working_dir_for_geo"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "working_dir_for_geo.log"))

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
        llm_model_name=os.getenv("LLM_MODEL", "qwen2"),
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

        query ='What is the capital of Italy?'
        query_param = QueryParam(mode='global',stream=True)


        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        resp = await rag.aquery(
            "What is the capital of Italy?",
            param=QueryParam(mode="global", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)



        # await write_chosen_relationships_to_file(
        #     query=query,
        #     rag=rag,
        #     query_param=query_param,
        #     chosen_relationships_output_file="/home/NingyuanXiao/LightRAG_test/test_for_extraction/geo_chosen_relationships.json"
        # )

        # await filter_json(
        #     input_path="/home/NingyuanXiao/LightRAG_test/test_for_extraction/geo_chosen_relationships.json",
        #     output_path="/home/NingyuanXiao/LightRAG_test/test_for_extraction/geo_filtered_relationships.json"
        # )

        # await generate_ad_entities(
        #     input_path="/home/NingyuanXiao/LightRAG_test/test_for_extraction/geo_filtered_relationships.json",
        #     output_path="/home/NingyuanXiao/LightRAG_test/test_for_extraction/geo_ad_entities.json"
        # )

        # await generate_ad_text(
        #     input_path="/home/NingyuanXiao/LightRAG_test/test_for_extraction/geo_ad_entities.json",
        #     output_path="/home/NingyuanXiao/LightRAG_test/test_for_extraction/geo_ad_text.json"
        # )
        
        

        

    


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
