import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # messages: Sequence[BaseMessage]
    # The 'next' field indicates where to route to next
    next: str


workflow = StateGraph(AgentState)
