import os
from typing import Tuple

import torch
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from llama_index import SimpleDirectoryReader
from llama_index.response.notebook_utils import display_source_node
from llama_index.schema import ImageNode
from torchvision import transforms
from transformers import AutoModelForObjectDetection

from configs import parse_args
from data_ingestion import ingest_mm_data
from data_loader import load_mm_data
from models import openai_chat_llm, openai_mm_llm
from prompts import create_user_prompt_summarize
from utils import MaxResize, convert_pdf_to_img, detect_and_crop_save_table, parser


@tool("retrieve_pdf_with_img_table", return_direct=False)
def retrieve_pdf_with_img_table(query: str) -> Tuple[str]:
    """Retrieve the information based on query and relevant information
    from the pdf file including images and tables.

    Args:
        query (str): Query for retrieving information

    Returns:
        Tuple[str]: Multi-modal LLM response formatted as (summary, status)
    """

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir_path = args.data_dir_path
    persist_dir = args.persist_dir
    file_name = args.file_name
    img_sim_top_k = args.sim_top_k

    detection_transform = transforms.Compose(
        [
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # load table detection model
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    ).to(device)

    if not os.path.isdir(data_dir_path):
        convert_pdf_to_img(file_name)

    if os.path.isdir(persist_dir):
        retriever_engine = load_mm_data(persist_dir, img_sim_top_k)
    else:
        retriever_engine = ingest_mm_data(data_dir_path, persist_dir, img_sim_top_k)

    # retrieve for the query using text to image retrieval
    retrieval_results = retriever_engine.text_to_image_retrieve(query)

    retrieved_images = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_images.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)

    # print(retrieved_images)

    for file_path in retrieved_images:
        detect_and_crop_save_table(
            model, detection_transform, file_path, device, args.table_dir
        )

    # Read the cropped tables
    image_documents = SimpleDirectoryReader(args.table_dir).load_data()

    info = openai_mm_llm.complete(
        prompt=query,
        image_documents=image_documents,
    )

    user_prompt = create_user_prompt_summarize(info)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | openai_chat_llm | parser

    results = chain.invoke({"query": user_prompt})

    summary, status = results.summary, results.status

    return summary, status


tool_retrieve = [retrieve_pdf_with_img_table]
