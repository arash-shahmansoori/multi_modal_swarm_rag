import os

import dotenv
from langchain_openai import ChatOpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from openai import OpenAI

dotenv.load_dotenv()

llm_hp = {"model": "gpt-4-turbo-preview", "temperature": 0, "max_tokens": 2000}
llm_hp_v = {"model": "gpt-4-vision-preview", "temperature": 0.1, "max_tokens": 1500}

openai_chat_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), **llm_hp)
openai_mm_llm = OpenAIMultiModal(api_key=os.getenv("OPENAI_API_KEY"), **llm_hp_v)


def create_client():
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client
