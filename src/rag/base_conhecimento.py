"""
Base de conhecimento para RAG com LangChain:
Documents -> chunks -> embeddings -> FAISS -> retriever.

Espelha a seção 13 do notebook.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MODELO_EMBEDDINGS,
    RETRIEVER_K,
)


def criar_documentos(protocolos, faqs):
    """Converte protocolos e FAQs para o formato Document (seção 13)."""
    documentos = []

    for protocolo in protocolos:
        documentos.append(
            Document(
                page_content=protocolo["conteudo"],
                metadata={
                    "id": protocolo["id"],
                    "tipo": "protocolo",
                    "fonte": protocolo["titulo"],
                },
            )
        )

    for faq in faqs:
        documentos.append(
            Document(
                page_content=(
                    f"Pergunta: {faq['pergunta']}\n\n"
                    f"Resposta: {faq['resposta']}"
                ),
                metadata={
                    "id": faq["id"],
                    "tipo": "faq",
                    "fonte": faq["pergunta"],
                },
            )
        )

    return documentos


def dividir_em_chunks(documentos):
    """Divide os documentos em trechos menores (seção 13.1)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return text_splitter.split_documents(documentos)


def criar_retriever(protocolos, faqs):
    """
    Cria a base vetorial FAISS e o retriever (seções 13.2 a 13.4).

    Retorna o retriever configurado com k documentos.
    """
    documentos = criar_documentos(protocolos, faqs)
    chunks = dividir_em_chunks(documentos)

    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K},
    )


def formatar_documentos(documentos):
    """Formata os documentos recuperados para uso como contexto (seção 14)."""
    textos = []

    for documento in documentos:
        fonte = documento.metadata.get("fonte", "Fonte não identificada")

        textos.append(
            f"Fonte: {fonte}\n"
            f"Conteúdo: {documento.page_content}"
        )

    return "\n\n".join(textos)
