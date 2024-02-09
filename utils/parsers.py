from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator


class Analysis(BaseModel):
    summary: str = Field(description="summary of the retrieved information")
    status: str = Field(description="status of the analysis, i.e., satisfied or failed")

    # You can add custom validation logic easily with Pydantic.
    @validator("summary")
    def summary_ends_with_period(cls, field):
        if field[-1] != ".":
            raise ValueError("Badly formed summary!")
        return field


parser = PydanticOutputParser(pydantic_object=Analysis)
