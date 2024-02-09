import os
from typing import Tuple

from langchain.prompts import PromptTemplate
from langchain_core.tools import tool

from configs import parse_args, parse_kwargs
from data_ingestion import vec_db_persist
from models import openai_chat_llm, openai_embedding
from prompts import create_user_prompt_summarize
from utils import parser


@tool("retrieve_pdf", return_direct=False)
def retrieve_pdf(query: str) -> Tuple[str]:
    """Retrieve the information based on query and relevant information
    from the pdf file.

    Args:
        query (str): Query for retrieving information

    Returns:
        Tuple[str]: Multi-modal LLM response formatted as (summary, status)
    """

    args = parse_args()
    kwargs = parse_kwargs()

    persist_dir = args.persist_dir_pdf

    if os.path.isdir(persist_dir):
        # Persisted vector store
        vectordb = kwargs["vector_store"](
            persist_directory=persist_dir, embedding_function=openai_embedding
        )
    else:
        vector_store = kwargs["vector_store"]
        source_loader = kwargs["source_loader"]
        splitter = kwargs["splitter"]

        embedding = openai_embedding

        _ = vec_db_persist(
            args.name,
            args.persist_dir_pdf,
            source_loader,
            splitter,
            args.chunk_size,
            args.chunk_overlap,
            vector_store,
            embedding,
        )

        # Persisted vector store
        vectordb = kwargs["vector_store"](
            persist_directory=persist_dir, embedding_function=openai_embedding
        )

    # Retrieve contents for a givent question from the source
    resources = vectordb.similarity_search(query)
    retrieved_texts = [resource.page_content for resource in resources]

    summary_prompt = create_user_prompt_summarize(retrieved_texts)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{summary_prompt}\n{query}\n",
        input_variables=["summary_prompt", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | openai_chat_llm | parser

    results = chain.invoke({"summary_prompt": summary_prompt, "query": query})

    summary, status = results.summary, results.status

    return summary, status
