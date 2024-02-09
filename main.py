import os

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from build_agent import retrievers, superviser
from configs import parse_args, parse_kwargs
from prompts import create_user_prompt
from state_graph import workflow


def main():
    args = parse_args()
    kwargs = parse_kwargs()

    # Define the file path to save the results
    directory = "results"
    filename = "output.txt"
    results_path = os.path.join(directory, filename)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory does not exist, create it
        os.makedirs(directory)

    # Query to send to the compiled graph at entrypoint, i.e., Superviser node
    query = create_user_prompt(kwargs["subject"], kwargs["additional_info"])

    # Add nodes
    workflow.add_node("Supervisor", superviser)

    for member, retriever in zip(kwargs["members"], retrievers):
        workflow.add_node(member, retriever)

    # Add edges
    for member in kwargs["members"]:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "Supervisor")  # add one edge for each of the agents

    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in kwargs["members"]}

    # conditional_map = {kwargs["member"]: kwargs["member"]}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)

    # Finally, add entrypoint
    workflow.set_entry_point("Supervisor")

    # Compile the workflow
    graph = workflow.compile()

    results = ""
    for s in graph.stream({"messages": [HumanMessage(content=query)]}):
        if "__end__" not in s:
            print(s)
            print("----")

            results += f"{s}\n"
            with open(results_path, "w") as file:
                file.write("\n")
                file.write(f"{results}")
                file.write("----\n")


if __name__ == "__main__":
    main()
