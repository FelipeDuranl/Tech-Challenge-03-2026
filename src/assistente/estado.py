"""
Estado compartilhado do fluxo LangGraph.

Espelha a seção 15 do notebook.
"""

from typing import List, Optional, TypedDict


class EstadoAssistente(TypedDict):
    id_paciente: str
    pergunta: str

    paciente: Optional[dict]

    contexto_paciente: str
    contexto_documentos: str

    fontes: List[str]

    resposta: str
    resposta_validada: str

    log: dict
