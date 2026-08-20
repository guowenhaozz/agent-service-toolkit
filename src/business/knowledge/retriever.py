from business.knowledge.vector_store import get_vector_store

MAX_DISTANCE = 1.2


def retrieve_maintenance_knowledge(
    query: str,
    k: int = 3,
) -> dict:
    """从维修知识库检索与问题相关的知识片段。"""

    normalized_query = query.strip()

    if not normalized_query:
        return {
            "success": False,
            "found": False,
            "message": "检索问题不能为空。",
            "documents": [],
        }

    try:
        vector_store = get_vector_store()

        results = vector_store.similarity_search_with_score(
            normalized_query,
            k=k,
        )
    except Exception:
        return {
            "success": False,
            "found": False,
            "message": "维修知识库暂时不可用。",
            "documents": [],
        }

    documents = []

    for document, distance in results:
        if distance > MAX_DISTANCE:
            continue

        documents.append(
            {
                "source": document.metadata.get("source", "未知来源"),
                "content": document.page_content,
                "distance": round(float(distance), 4),
            }
        )

    if not documents:
        return {
            "success": True,
            "found": False,
            "message": "知识库中没有足够相关的维修资料。",
            "documents": [],
        }

    return {
        "success": True,
        "found": True,
        "message": "已检索到相关维修资料。",
        "documents": documents,
    }
