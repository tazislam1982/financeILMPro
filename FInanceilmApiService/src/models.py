from pydantic import BaseModel

class Context(BaseModel):
    text: str
    link_extracted: dict

# v1 API
class StreamChunk(BaseModel):
    text: str | None
    usage: dict | None

class Message(BaseModel):
    role: str
    content: str

class QuestionInput(BaseModel):
    messages: list[Message] = [] # message history
    stream: bool = False
    referrer: str = "site"

class UserFacingException(Exception):
    def __init__(self, message: str = "Sorry there was an issue. Please refresh and try again."):
        self.message = message