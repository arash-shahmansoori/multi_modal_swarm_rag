import functools
from typing import Dict

from langchain_core.messages import HumanMessage

from agent_tools import retrieve_pdf, retrieve_pdf_with_img_table
from type_extensions import T


def agent_retrieve_with_img_table(state: Dict[str, T], name: str) -> Dict[str, T]:
    """Retriever agent with image and table

    Args:
        state (Dict[str, T]): Input state
        name (str): Agent name

    Returns:
        Dict[str, T]: Agent response
    """

    summary, status = retrieve_pdf_with_img_table(state["messages"][0].content)

    return {
        "messages": [
            HumanMessage(
                content=f"summary: {summary}" + f" status: {status}", name=name
            )
        ]
    }


def agent_retrieve_with_pdf(state: Dict[str, T], name: str) -> Dict[str, T]:
    """Retriever agent with pdf

    Args:
        state (Dict[str, T]): Input state
        name (str): Agent name

    Returns:
        Dict[str, T]: Agent response
    """

    summary, status = retrieve_pdf(state["messages"][0].content)

    return {
        "messages": [
            HumanMessage(
                content=f"summary: {summary}" + f" status: {status}", name=name
            )
        ]
    }


# Retriever
retriever_with_pdf = functools.partial(agent_retrieve_with_pdf, name="Retriever-PDF")
retriever_with_img_table = functools.partial(
    agent_retrieve_with_img_table, name="Retriever-Image-Table"
)


retrievers = [retriever_with_pdf, retriever_with_img_table]
