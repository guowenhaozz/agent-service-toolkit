import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from business.knowledge.vector_store import (  # noqa: E402
    COLLECTION_NAME,
    VECTOR_STORE_DIR,
    get_embeddings,
    get_vector_store,
)
from langchain_chroma import Chroma  # noqa: E402


KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"


def load_markdown_documents() -> list[Document]:
    """读取知识目录内的全部Markdown文档。"""

    documents = []

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                },
            )
        )

    return documents


def main() -> None:
    documents = load_markdown_documents()

    if not documents:
        raise RuntimeError(
            f"未在 {KNOWLEDGE_DIR} 找到Markdown知识文档。"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=[
            "\n## ",
            "\n# ",
            "\n\n",
            "\n",
            "。",
            "，",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    print(f"已读取文档：{len(documents)}份")
    print(f"切分后的知识块：{len(chunks)}个")

    # 只重建maintenance_knowledge这一份集合，不影响业务SQLite数据。
    old_store = get_vector_store()

    try:
        old_store.delete_collection()
        print("已清理旧的维修知识向量集合。")
    except Exception:
        print("未发现旧集合，开始首次构建。")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
    )

    print(f"向量库构建完成：{VECTOR_STORE_DIR}")
    print("集合名称：maintenance_knowledge")


if __name__ == "__main__":
    main()