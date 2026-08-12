"""Model construction in one place, so swapping providers touches one file."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from mesh.models.config import Settings

_MAX_RETRIES = 3


def build_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        openai_api_key=settings.openai_api_key,
        max_retries=_MAX_RETRIES,
    )


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embed_model,
        openai_api_key=settings.openai_api_key,
        max_retries=_MAX_RETRIES,
    )
