"""
Nós do fluxo LangGraph.

Espelha as seções 15.2 a 15.6 do notebook. Os nós são criados por uma
fábrica (`criar_nos`) que recebe as dependências — pacientes, retriever
e gerador de respostas — e devolve as funções de cada etapa.
"""

from src.assistente.auditoria import registrar_log
from src.assistente.estado import EstadoAssistente
from src.assistente.seguranca import validar_resposta
from src.config import (
    INSTRUCAO_ASSISTENTE,
    TEMPLATE_ENTRADA_ASSISTENTE,
    TEMPLATE_PROMPT,
)
from src.dados.hospital import buscar_paciente, formatar_paciente
from src.rag.base_conhecimento import formatar_documentos


def criar_nos(pacientes, retriever, gerar_resposta_llm):
    """Retorna o dicionário {nome_do_no: funcao} para montagem do grafo."""

    def no_buscar_paciente(estado: EstadoAssistente):
        paciente = buscar_paciente(pacientes, estado["id_paciente"])
        contexto_paciente = formatar_paciente(paciente)

        return {
            "paciente": paciente,
            "contexto_paciente": contexto_paciente,
        }

    def no_recuperar_contexto(estado: EstadoAssistente):
        documentos = retriever.invoke(estado["pergunta"])

        contexto_documentos = formatar_documentos(documentos)

        fontes = [
            documento.metadata.get("fonte", "Fonte não identificada")
            for documento in documentos
        ]

        return {
            "contexto_documentos": contexto_documentos,
            "fontes": fontes,
        }

    def no_gerar_resposta(estado: EstadoAssistente):
        # A instrução e o formato da entrada são os mesmos utilizados na
        # geração do corpus hospitalar de fine-tuning (src/dados/
        # corpus_hospitalar.py), evitando divergência entre treino e inferência.
        entrada = TEMPLATE_ENTRADA_ASSISTENTE.format(
            paciente=estado["contexto_paciente"],
            contexto=estado["contexto_documentos"],
            pergunta=estado["pergunta"],
        )

        prompt = TEMPLATE_PROMPT.format(INSTRUCAO_ASSISTENTE, entrada, "")

        resposta = gerar_resposta_llm(prompt)

        return {
            "resposta": resposta,
        }

    def no_validar_resposta(estado: EstadoAssistente):
        return {
            "resposta_validada": validar_resposta(estado["resposta"]),
        }

    def no_registrar_log(estado: EstadoAssistente):
        log = registrar_log(
            id_paciente=estado["id_paciente"],
            pergunta=estado["pergunta"],
            fontes=estado["fontes"],
            resposta_original=estado["resposta"],
            resposta_final=estado["resposta_validada"],
        )

        return {
            "log": log,
        }

    return {
        "buscar_paciente": no_buscar_paciente,
        "recuperar_contexto": no_recuperar_contexto,
        "gerar_resposta": no_gerar_resposta,
        "validar_resposta": no_validar_resposta,
        "registrar_log": no_registrar_log,
    }
