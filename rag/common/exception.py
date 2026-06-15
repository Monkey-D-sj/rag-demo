class LLMRequestException(Exception):
    """LLM 请求异常"""

    def __init__(self, message: str, model: str = "", attempts: int = 0):
        """
        初始化 LLM 请求异常
        
        :param message: 异常消息
        :param model: 模型名称
        :param attempts: 尝试次数
        """
        super().__init__(message)
        self.model = model
        self.attempts = attempts
