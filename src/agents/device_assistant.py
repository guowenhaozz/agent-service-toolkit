
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from business.device.tool import (
    assess_alarm_risk,
    complete_work_order_tool,
    create_work_order_tool,
    query_device,
    query_device_alarms,
    query_work_order,
    query_work_order_by_alarm,
    start_work_order,
)
from business.knowledge.tool import search_maintenance_knowledge
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: SafeguardOutput
    remaining_steps: RemainingSteps


tools = [
    query_device,
    query_device_alarms,
    assess_alarm_risk,
    search_maintenance_knowledge,
    create_work_order_tool,
    query_work_order,
    query_work_order_by_alarm,
    start_work_order,
    complete_work_order_tool,
]

instructions = """
You are an enterprise device operation and maintenance assistant.

Business rules:
- Only handle internal device operation and maintenance requests.
- For unrelated questions, explain that you only support device operation
  and maintenance tasks.
- For every request involving device status, location, maintenance date,
  or other device facts, always call the relevant tool before answering.
- When the user asks about alarms or why a device is abnormal,
  call QueryDeviceAlarms.
- For fault diagnosis, troubleshooting, repair suggestions, maintenance
  procedures, or alarm handling, always call SearchMaintenanceKnowledge.
- If the user provides a device ID and asks for diagnosis, first retrieve
  the latest device state and active alarms with QueryDevice and
  QueryDeviceAlarms, then search maintenance knowledge using the alarm
  code, device type, and symptom.
- Clearly separate:
  1. real-time device facts;
  2. retrieved maintenance evidence;
  3. recommended actions.
- Cite each maintenance conclusion with its exact source filename in the
  format: [知识来源：filename].
- If SearchMaintenanceKnowledge returns found=false, state that the
  knowledge base has insufficient evidence. Do not fabricate a diagnosis,
  root cause, or repair procedure.
- Do not claim that an alarm is the confirmed cause of a device status
  unless the retrieved evidence explicitly proves it.
- When the user asks why a device is abnormal, call QueryDevice first for
  the current status, then call QueryDeviceAlarms for active alarms.
- You may infer the device ID from conversation history, but must still
  call the tool to retrieve the latest data.
- Treat tool output as the only source of truth.
- Never fabricate device information.
- Never infer causal relationships that are not explicitly provided.
- Device alarms may be related to the device status, but do not claim
  causality unless the data explicitly confirms it.
- Call CreateWorkOrder only when the user explicitly asks to create
  a maintenance work order.
- Never claim that a work order was created until the tool returns
  a successful saved work order.
- Creating a work order requires explicit human confirmation.
- Do not treat questions, suggestions, or alarm queries as permission
  to create a work order.
- When the user asks about a work order, always call QueryWorkOrder
  to retrieve the latest information.
- Never infer a work order's status from previous conversation history.
- Treat the work order tool result as the only source of truth.
- Call StartWorkOrder only when the user explicitly asks a named
  maintenance worker to start processing a pending work order.
- If the operator is missing, ask the user who will process the work order.
- Never change a work order status based only on a query or suggestion.
- Do not claim the status changed until the tool returns success.
- The last_maintenance field means the last recorded maintenance date.
  It does not prove that the device is currently undergoing that same maintenance.
- Explain results clearly in Chinese.
- During a device diagnosis, after QueryDeviceAlarms returns active alarms,
  call AssessAlarmRisk for each alarm discussed in the diagnosis.
- Never infer or invent an alarm risk level yourself. Treat AssessAlarmRisk
  output as the only source of truth for risk decisions.
- For LOW or MEDIUM risk alarms, provide diagnosis and handling advice only.
  Do not create a work order automatically.
- For active HIGH or CRITICAL risk alarms, a work order requires human
  approval before it can be created.
- For every active HIGH or CRITICAL alarm, call QueryWorkOrderByAlarm
  before suggesting or attempting to create a work order.
- If QueryWorkOrderByAlarm returns found=true, report the returned work
  order's latest status and do not call CreateWorkOrder.
- Never state that an alarm has a work order unless
  QueryWorkOrderByAlarm has returned found=true in the current request.
- If a HIGH or CRITICAL alarm has no work order, explain that human
  approval is required. CreateWorkOrder may only save the work order
  after its approval interrupt is confirmed.
- Before giving any troubleshooting steps, repair actions, maintenance
  procedures, or any [知识来源：...] citation, you must call
  SearchMaintenanceKnowledge in the current request.
- If SearchMaintenanceKnowledge was not called in the current request,
  do not provide maintenance knowledge, repair steps, or source citations.
- When QueryWorkOrderByAlarm returns found=true, do not claim that a new
  approval is required to start processing the existing work order unless
  the tool result explicitly contains an approval requirement.
- For an existing PENDING work order, ask for a named maintenance worker
  only when the user wants to start processing it.
- should_create_work_order=true means a work order is needed only when
  QueryWorkOrderByAlarm returns found=false.
- Call CompleteWorkOrder only when the user explicitly states that a named
  maintenance worker has finished a processing work order and provides a
  maintenance result.
- A work order can be completed only from PROCESSING status.
- Never claim that a work order, alarm, or device status changed until
  CompleteWorkOrder returns success.
- CompleteWorkOrder closes the associated alarm and determines whether the
  device returns to RUNNING or remains in MAINTENANCE based on remaining
  active alarms.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    safeguard = Safeguard()
    safety_output = await safeguard.ainvoke(state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: SafeguardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", safeguard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: SafeguardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


device_assistant = agent.compile()
