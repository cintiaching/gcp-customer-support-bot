from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.constants import START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import AnyMessage, add_messages

from src.bot.retriever import lookup_policy
from src.bot.tools import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket
from src.bot.utils import create_tool_node_with_fallback


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

part_2_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
]


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


class Agent:
    def __init__(self, llm):
        self.llm = llm
        self.assistant_runnable = assistant_prompt | self.llm.bind_tools(part_2_tools)

    def build_graph(self):
        builder = StateGraph(State)
        # NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
        # having to take an action
        builder.add_node("fetch_user_info", user_info)
        builder.add_edge(START, "fetch_user_info")
        builder.add_node("assistant", Assistant(self.assistant_runnable))
        builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
        builder.add_edge("fetch_user_info", "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        memory = MemorySaver()
        graph = builder.compile(
            checkpointer=memory,
            # NEW: The graph will always halt before executing the "tools" node.
            # The user can approve or reject (or even alter the request) before
            # the assistant continues
            interrupt_before=["tools"],
        )
        return graph
