from typing import Dict, List

from type_extensions import T


def create_schema_superviser(name: str, options: List[str]) -> Dict[str, T]:

    function_def = {
        "name": name,
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    return function_def
