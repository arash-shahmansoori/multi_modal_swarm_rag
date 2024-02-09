from typing import Callable, Dict, TypeVar

from llama_index.core.llms.types import CompletionResponse

T = TypeVar("T")
Retrieve = Callable[..., CompletionResponse]
Retriever = Callable[..., Dict[str, T]]
