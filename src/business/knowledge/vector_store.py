from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
EMBEDDING_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "bge-small-zh-v1.5"
COLLECTION_NAME = "maintenance_knowledge"


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """加载项目目录中的本地中文 Embedding 模型。"""

    if not EMBEDDING_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"未找到本地 Embedding 模型目录：{EMBEDDING_MODEL_DIR}"
        )

    return HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_DIR),
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store() -> Chroma:
    """获取已持久化的维修知识向量库。"""

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=get_embeddings(),
    )
