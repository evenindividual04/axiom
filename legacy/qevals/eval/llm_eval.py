from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai.chat_models import AzureChatOpenAI


class AzureOpenAI(DeepEvalBaseLLM):
    """Minimal Deepeval model adapter for AzureChatOpenAI."""

    def __init__(self, model: AzureChatOpenAI):
        self.model = model

    def load_model(self) -> AzureChatOpenAI:
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return getattr(response, "content", str(response))

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return getattr(response, "content", str(response))

    def get_model_name(self) -> str:
        return "azure-chat-openai"
