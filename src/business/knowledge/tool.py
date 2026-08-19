import json

from langchain_core.tools import BaseTool, tool

from business.knowledge.retriever import retrieve_maintenance_knowledge


def search_maintenance_knowledge_func(query: str) -> str:
    """Search internal maintenance manuals and historical repair cases.

    Use this tool when the user asks about fault diagnosis,
    troubleshooting steps, repair suggestions, maintenance procedures,
    fault causes, or alarm handling.

    Do not use this tool to query real-time device status,
    alarms, or work orders.

    Args:
        query: A detailed maintenance or diagnostic question.
    """

    result = retrieve_maintenance_knowledge(query)
    return json.dumps(result, ensure_ascii=False)


search_maintenance_knowledge: BaseTool = tool(
    search_maintenance_knowledge_func
)
search_maintenance_knowledge.name = "SearchMaintenanceKnowledge"
