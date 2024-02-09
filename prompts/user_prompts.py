def create_user_prompt(subject: str, additional_info: str) -> str:
    prompt = f"{subject}"
    info = f" {additional_info}"
    return prompt + info


def create_user_prompt_summarize(info: str) -> str:
    prompt = "Summarize the following information retrieved bt the Retriever."
    retrieved_info = f" Retrieved information:\n{info}"

    return prompt + retrieved_info
