"""Files 模块 Schema。"""

from pydantic import BaseModel, Field


class FileItem(BaseModel):
    key: str = Field(..., description="对象在 bucket 中的 key")
    size: int = Field(..., description="文件大小（字节）")
    content_type: str = Field(default="", description="MIME 类型")
    last_modified: str = Field(default="", description="最后修改时间 ISO 格式")


class FileListResponse(BaseModel):
    bucket: str = Field(..., description="bucket 名称")
    count: int = Field(..., description="对象总数")
    items: list[FileItem] = Field(default_factory=list, description="对象列表")
