from typing import Dict

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import ChatOpenAI

from configs import parse_kwargs
from models import openai_chat_llm
from type_extensions import T


def create_sup_agent(
    sup_llm: ChatOpenAI,
    sup_prompt: ChatPromptTemplate,
    function_def: Dict[str, T],
    sup_fn_name: str,
) -> RunnableSerializable:
    """Create superviser agent

    Args:
        sup_llm (ChatOpenAI): Chat OpenAI LLM
        sup_prompt (ChatPromptTemplate): Input prompt
        function_def (Dict[str, T]): function definition schema
        sup_fn_name (str): function name

    Returns:
        RunnableSerializable: Runnable-Serializable superviser agent
    """
    # Superviser node
    supervisor = (
        sup_prompt
        | sup_llm.bind_functions(functions=[function_def], function_call=sup_fn_name)
        | JsonOutputFunctionsParser()
    )

    return supervisor


kwargs = parse_kwargs()
superviser = create_sup_agent(openai_chat_llm, *kwargs["sup_args"])
