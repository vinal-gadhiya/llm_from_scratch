from pydantic import BaseModel

class UserInput(BaseModel):
    user_input: str

class ModelOutput(BaseModel):
    model_output: str