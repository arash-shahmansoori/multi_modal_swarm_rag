import os

import dotenv
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()


openai_embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
