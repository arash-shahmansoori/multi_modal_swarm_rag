from models import create_client

client = create_client()

query = "Short Tweeter post for this groundbraking approach to increase my chances to land a job and attract attentions from companies and recruiters, the tweet should meet the character limitations for free X or tweets"
additional_requirement = ""  # Use for Title
# additional_requirement = "The abstract should be maximum 200 words. DO NOT INCLUDE TITLE ONLY OUTPUT THE ABSTRACT."  # Use for abstract
# additional_requirement = "The introduction should be appropriate for a journal paper. You need to specifically mention the main contributions of the blog/research paper including: (1) Blind information retrieval, i.e., prior to retrieving information related to a specific query the proposed system has no clue about the type of retriver that is appropriate to answer the query. (2) The proposed system is the first of its kind to showcase the power of stateful multi-agent systems for retrieving information through a stateful hierarchical swarm of agents including a superviser agent and retriever agents. (3) The proposed methos is highly robust ans scalable across different modalities for information retrieval, e.g., text, image, audio, and video, thanks to its agentic, stateful, and modular architecture."
# additional_requirement = "The proposed method should include a pictorial viewpoint of the proposed approach to clearly describing the proposed method. In particular, the pictorial viewpoint of the proposed method should describe clearly and easily all the steps in the proposed method using a descriptive and easy to read block diagram using Mermaid Markdown DIAGRAM. Subsequently, you describe the proposed algorithm based on the block diagram in details and easy to follow way. DO NOT INCLUDE TITLE, ABSTRACT, AND INTRODUCTION ONLY OUTPUT THE PROPOSED METHOD. Finally, revise the proposed method section and make sure that it follows the mermaid markdown workflow for the proposed method and all the above description provided regarding the proposed method. Make sure the diagram and the scientific workflow of the proposed method are coherent and match each other conceptually."  # Use for the main method
# additional_requirement = """The results section should include the following subsections. First, the setup including the types of models used for each agent. In particular, all the agents in the swarm use: "gpt-4-turbo-preview" as the main LLM model for textual data and "gpt-4-vision-preview" for image and tabular data. For the purpose of simulations, we use two different types of retriever one using ChromadDB as the vector database for retrieving the textual data from pdf document: retriever_with_pdf, and another using multi-modal Qdrant vector store for image and tabular data from pdf, additionally using Microsoft `table-transformer-detection' to extract table data from image, namely:retriever_with_img_table. In partucular, we are focusing on asking the tabular specific question about the LLAMMA2 paper and comparing its performance with LLAMMA 1 in terms of STEM related topic. This question only can be answered using the information from the tabular data within the LLAMMA2 paper; however the system initially has no clue about this and the user do not provide any additional information, so the system adaptively and blindly is able to explore the space of appropriate retriever to select and address the user query using the proper retriever, in this case: retriever_with_img_table."""


# system_prompt = """you are the best artist and philosopher and painter in the world. You create the best prompts based on philosophical concepts to generate images. The user provide you the concept that you are supposed to use to generate the prompt to be used for text to image generation, your task is to provide the best prompt considering all the nuances for text to image generation based on the philosophical concept. You create the prompt such that the potential generated image is unique, outstanding art form with deep philosophical concepts."""


system_prompt = """You are the best scientific artificial (general) intelligence researcher and blogger in the world. You provide the best suggestions for writing scientific blogs and papers related to artificial (general) intelligence, machine learning, and deep learning. You will be asked questions by the user for generating scientific suggestions for writing blogs and scientific papers. The user asks your suggestions for writing different parts of a scientific blog and papers, i.e., abstract, title, introduction, main methods, results, and conclusions. You only respond to what the user asks for and do not try to provide additional responses that are not related to the original user query."""

user_prompt = f"""
I have designed a RAG including a superviser agent and retriever agents. The superviser agent routs the user query to a retriever and iterates a few time with a given retriever each time, if it fails it switches to another retriever. The superviser agent continues back and forth iterations for each given retriever each time so that it finally retrieves the desired information related to user query and finishes the iteration and return the relevant retrieved information to user. Each retriever in this framework has special skill for retrieving information related to different parts or types of documents. For instance, one retriever is great at extracting relevant information from tables, another retriever is great at textual data and so on. So, if one retriever after few iterations with the superviser agent fails to retrieve the relevant information, the superviser agent automatically switches to another retriever until it receives the desired information related to the original query by a given retriever. The proposed system of agents including a superviser agent, and retriever agents form a stateful graph with the nodes denoting the agents and edges representing the communication between retrievers and the superviser agent. The aformentioned stateful graph is formes as follows: The superviser agent is the entry point of the graph and strats sending the query to a given retriever agent, the retriever agent tries to retriever the relevant information and provides a summary report of retrieved information together with retrieved status of success or failure. In any case, the retriever agent always send back the summary report of retrieved information and retrieved status to the superviser agent. The superviser agent then iterates a few times with the retriever if it receives the statisfied status it finishes and returns the retrieved information related to the original query; else it switches to another retriver and repeats the same process untils it receives the relevant retrieved information; in any case and if after iterating with all different retrievers available in the graph it is not able to receieves the relevant information it responds with I do not know the answer to this query and ends the execution.


According to the above information, provide the best possible {query} for this blog/research paper. The {query} should be smart, unique, concise, and capture all the key novel aspects of the aforementioned process above as much as possible. It should emphasize on the adaptive and blind retrival capabilities of the proposed workflow meaning prior to retrieving relevant information the system has no idea abouth the type of retriver appropriate for a specific query to retrieve the most relevant information and yet is able to blindly and adaptively rout between different types of retrievers. Finally, it should emphasize on the stateful nature of the proposed approach, the adaptive routing aspects, and the fact that it is a hirarchical swarm of agents proposing a very strong capable and flexible RAG systems. 

{additional_requirement}

Use the following diagram in your description:

```mermaid
graph LR
    SA[Supervisory Agent] -->|Dispatches Query| RA1[Retriever Agent 1]
    SA -->|Dispatches Query| RA2[Retriever Agent 2]
    SA -->|Dispatches Query| RAn[Retriever Agent n]
    RA1 -->|Returns Report & Status| SA
    RA2 -->|Returns Report & Status| SA
    RAn -->|Returns Report & Status| SA
    SA -->|Evaluates Status| Decision{'Decision Making'}
    Decision -->|Satisfied| Output[Return Results to User]
    Decision -->|Not Satisfied| Iteration[Iterate with Current or Switch Agent]
    Iteration --> SA
    Decision -->|All Agents Tried| Fail[Respond: Information Not Found]
```
"""


response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ],
    stream=True,
)


for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
