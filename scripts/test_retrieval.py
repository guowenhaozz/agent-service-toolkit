import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from business.knowledge.vector_store import get_vector_store  # noqa: E402


def main() -> None:
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "DVC-002出现深度相机读取超时，应该如何排查和处理？"
    )

    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=3)

    print(f"\n查询问题：{query}")
    print(f"返回知识块数量：{len(results)}\n")

    for index, document in enumerate(results, start=1):
        print(f"--- 结果 {index} ---")
        print(f"来源：{document.metadata.get('source')}")
        print(document.page_content)
        print()


if __name__ == "__main__":
    main()