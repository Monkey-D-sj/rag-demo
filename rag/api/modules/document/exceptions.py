from rag.common.exception import AppError


class UnsupportedFileType(AppError):
    """不支持的文件类型"""

    status_code = 400


class FileTooLarge(AppError):
    """文件超过大小上限"""

    status_code = 400


class DocumentNotFound(AppError):
    """文档不存在"""

    status_code = 404
