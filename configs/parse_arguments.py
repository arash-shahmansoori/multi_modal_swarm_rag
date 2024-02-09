import argparse
from argparse import Namespace
from typing import Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent_func_schema import create_schema_superviser
from prompts import system_prompt_supervisor
from type_extensions import T


def parse_args() -> Namespace:
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Arguments for Multi-Modal Swarm RAG")
    ################################ RBHSR RAG parameters ###########################
    parser.add_argument("--sim_top_k", default=2, type=int)
    parser.add_argument("--chunk_size", default=500, type=int)
    parser.add_argument("--chunk_overlap", default=0, type=int)
    parser.add_argument("--data_dir_path", default="./llama2/", type=str)
    parser.add_argument("--persist_dir", default="qdrant_index", type=str)
    parser.add_argument("--persist_dir_pdf", default="chroma_index", type=str)
    parser.add_argument("--table_dir", default="./table_images/", type=str)
    parser.add_argument("--name", default="./llama2.pdf", type=str)
    parser.add_argument("--file_name", default="llama2.pdf", type=str)
    parser.add_argument("--results_name", default="results.txt", type=str)
    parser.add_argument("--results_dir", default="./results/", type=str)

    args = parser.parse_args()
    return args


def parse_kwargs() -> Dict[str, T]:

    subject = "Directly compare the performance of llamma1 and llamma2 in STEM?"
    additional_info = (
        "Use the tool to retrieve relevant information for answering the question."
    )

    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    # members = ["Retriever-Image-Table", "Retriever-Image", "Retriever-PDF"]
    members = ["Retriever-PDF", "Retriever-Image-Table"]

    member = "Retriever-PDF"

    options = members + ["FINISH"]

    sup_fn_name = "route"

    # Using openai function calling can make output parsing easier for us
    function_def = create_schema_superviser(sup_fn_name, options)

    sup_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_supervisor),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    sup_args = [sup_prompt, function_def, sup_fn_name]

    vector_store = Chroma
    source_loader = PyMuPDFLoader
    splitter = RecursiveCharacterTextSplitter

    return {
        "subject": subject,
        "additional_info": additional_info,
        "members": members,
        "sup_args": sup_args,
        "vector_store": vector_store,
        "source_loader": source_loader,
        "splitter": splitter,
        "member": member,
    }
