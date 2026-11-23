
from typing import Literal

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    filename: str
    status: Literal["STORED", "SKIPPED"]
    chunks: int
    message: str


class DocumentItem(BaseModel):
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total: int
