"""
Construção do fluxo do assistente com LangGraph.

Espelha a seção 15.7 do notebook:

START -> buscar_paciente -> recuperar_contexto -> gerar_resposta
      -> validar_resposta -> registrar_log -> END
"""

from langgraph.graph import END, START, StateGraph

from src.assistente.estado import EstadoAssistente
from src.assistente.nos import criar_nos


def criar_assistente(pacientes, retriever, gerar_resposta_llm):
    """Monta e compila o grafo do assistente."""
    nos = criar_nos(pacientes, retriever, gerar_resposta_llm)

    grafo = StateGraph(EstadoAssistente)

    grafo.add_node("buscar_paciente", nos["buscar_paciente"])
    grafo.add_node("recuperar_contexto", nos["recuperar_contexto"])
    grafo.add_node("gerar_resposta", nos["gerar_resposta"])
    grafo.add_node("validar_resposta", nos["validar_resposta"])
    grafo.add_node("registrar_log", nos["registrar_log"])

    grafo.add_edge(START, "buscar_paciente")
    grafo.add_edge("buscar_paciente", "recuperar_contexto")
    grafo.add_edge("recuperar_contexto", "gerar_resposta")
    grafo.add_edge("gerar_resposta", "validar_resposta")
    grafo.add_edge("validar_resposta", "registrar_log")
    grafo.add_edge("registrar_log", END)

    return grafo.compile()
