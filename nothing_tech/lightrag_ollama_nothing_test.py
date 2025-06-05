import asyncio
import os
import sys
import inspect
import logging
import logging.config
import pandas as pd
import csv
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

def get_working_dir():
    """获取工作目录，支持命令行参数和环境变量"""
    if len(sys.argv) > 1:
        return sys.argv[1]  # 命令行参数
    return os.getenv("RAG_WORKING_DIR", "./default_rag")

WORKING_DIR = get_working_dir()

def get_input_file():
    """获取输入文件路径"""
    if len(sys.argv) > 2:
        return sys.argv[2]  # 命令行参数
    return os.getenv("INPUT_FILE", "./data.csv")

def get_csv_columns():
    """获取要处理的CSV列，支持多列"""
    columns = os.getenv("CSV_COLUMNS", "")
    if columns:
        return [col.strip() for col in columns.split(",")]
    return None  # 如果未指定，使用所有文本列

def detect_text_columns(df):
    """自动检测CSV中的文本列"""
    text_columns = []
    for col in df.columns:
        # 检查列是否包含文本数据
        sample_values = df[col].dropna().head(10)
        if len(sample_values) > 0:
            # 检查是否为字符串类型，且平均长度大于10（过滤掉短标识符）
            if sample_values.dtype == 'object':
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 10:
                    text_columns.append(col)
    return text_columns

def process_csv_file(file_path, columns=None):
    """处理CSV文件，将其转换为文本格式"""
    try:
        # 读取CSV文件
        print(f"📊 Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        # 确定要处理的列
        if columns is None:
            columns = detect_text_columns(df)
            print(f"🔍 Auto-detected text columns: {columns}")
        else:
            # 验证指定的列是否存在
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️  Warning: Columns not found: {missing_cols}")
                columns = [col for col in columns if col in df.columns]
        
        if not columns:
            print("❌ No suitable text columns found!")
            return None
        
        # 将CSV数据转换为文本格式
        text_content = []
        
        # 添加数据集描述
        text_content.append(f"Dataset Information:")
        text_content.append(f"Total records: {len(df)}")
        text_content.append(f"Columns being processed: {', '.join(columns)}")
        text_content.append(f"Source file: {os.path.basename(file_path)}")
        text_content.append("\n" + "="*50 + "\n")
        
        # 处理每一行数据
        for idx, row in df.iterrows():
            record_text = f"Record {idx + 1}:\n"
            
            # 添加每个指定列的内容
            for col in columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    record_text += f"{col}: {str(value).strip()}\n"
            
            # 只添加非空记录
            if len(record_text.split('\n')) > 2:  # 超过标题行
                text_content.append(record_text)
                text_content.append("-" * 30)  # 分隔符
        
        final_text = "\n".join(text_content)
        print(f"✅ Processed {len(df)} records into text format")
        print(f"   Total text length: {len(final_text)} characters")
        
        return final_text
        
    except Exception as e:
        print(f"❌ Error processing CSV file: {e}")
        return None

def is_csv_file(file_path):
    """检查文件是否为CSV格式"""
    return file_path.lower().endswith('.csv')

async def initialize_rag():
    """初始化RAG系统"""
    print(f"📁 Working Directory: {WORKING_DIR}")
    
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        print(f"✅ Created working directory: {WORKING_DIR}")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
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

async def clean_old_data():
    """清理旧数据（可选）"""
    files_to_delete = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    ]

    print("🧹 Cleaning old data files...")
    for file in files_to_delete:
        file_path = os.path.join(WORKING_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   Deleted: {file}")

async def main():
    try:
        # 询问是否清理旧数据
        if os.path.exists(WORKING_DIR):
            response = input(f"⚠️  Working directory '{WORKING_DIR}' exists. Clean old data? (y/N): ")
            if response.lower() in ['y', 'yes']:
                await clean_old_data()
        
        # 初始化RAG
        rag = await initialize_rag()
        
        # 获取输入文件
        input_file = get_input_file()
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return
        
        print(f"📄 Processing file: {input_file}")
        
        # 根据文件类型处理内容
        if is_csv_file(input_file):
            # 处理CSV文件
            specified_columns = get_csv_columns()
            if specified_columns:
                print(f"🎯 Using specified columns: {specified_columns}")
            
            content = process_csv_file(input_file, specified_columns)
            if content is None:
                print("❌ Failed to process CSV file")
                return
        else:
            # 处理普通文本文件
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()
        
        # 插入内容到RAG系统
        print("🔄 Inserting content into RAG system...")
        await rag.ainsert(content)
        
        print("✅ Document processed successfully!")
        
        # 交互式查询
        print("\n" + "="*50)
        print("🤖 Interactive Query Mode")
        print("   Type 'quit' or 'exit' to stop")
        print("   Type 'mode:naive', 'mode:local' to switch query modes")
        print("="*50)
        
        query_mode = "local"  # 默认模式
        
        while True:
            query = input(f"\n[{query_mode}] Your question: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            
            if query.startswith('mode:'):
                new_mode = query.split(':', 1)[1].strip()
                if new_mode in ['naive', 'local', 'global', 'hybrid']:
                    query_mode = new_mode
                    print(f"✅ Switched to {query_mode} mode")
                else:
                    print("❌ Invalid mode. Available: naive, local, global, hybrid")
                continue
            
            if not query:
                continue
            
            print(f"\n🔍 [{query_mode}] Searching...")
            resp = await rag.aquery(
                query,
                param=QueryParam(mode=query_mode, stream=True),
            )
            
            if inspect.isasyncgen(resp):
                async for chunk in resp:
                    print(chunk, end="", flush=True)
            else:
                print(resp)
            print("\n")
    
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if 'rag' in locals():
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()

if __name__ == "__main__":
    print("🚀 LightRAG Knowledge Graph Builder (CSV Support)")
    print(f"📁 Usage: python {sys.argv[0]} [working_dir] [input_file.csv]")
    print(f"📁 Current working dir: {get_working_dir()}")
    print(f"📄 Current input file: {get_input_file()}")
    print()
    print("🔧 CSV Configuration:")
    print(f"   Set CSV_COLUMNS env var to specify columns (comma-separated)")
    print(f"   Example: export CSV_COLUMNS='title,description,content'")
    print(f"   If not set, will auto-detect text columns")
    print()
    
    asyncio.run(main())
    print("\n✅ Done!")