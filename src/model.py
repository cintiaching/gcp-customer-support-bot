from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings


def build_llm():
    llm = ChatVertexAI(
        model="gemini-2.0-flash-001",
        temperature=0,
        max_retries=3,
    )
    return llm


def build_embedding_model():
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    return embeddings