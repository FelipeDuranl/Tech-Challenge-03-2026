"""
Geração dos exemplos de fine-tuning a partir da base hospitalar sintética.

Atende ao requisito 1 do desafio, que exige o fine-tuning com:

- protocolos médicos do hospital;
- exemplos de perguntas frequentes feitas por médicos;
- modelos de laudos, receitas e procedimentos internos.

Os exemplos são gerados no MESMO formato que o assistente recebe em tempo de
execução (instrução, dados do paciente, contexto recuperado e pergunta), de
modo que o modelo seja treinado exatamente no formato em que será consultado
pelo fluxo LangGraph.

As respostas dos exemplos clínicos são curadas manualmente a partir do texto
dos protocolos — nenhuma conduta que extrapole o protocolo é introduzida, e
todas reforçam os limites de atuação do assistente.
"""

from src.config import (
    INSTRUCAO_ASSISTENTE,
    INSTRUCAO_DOCUMENTOS,
    TEMPLATE_ENTRADA_ASSISTENTE,
)
from src.dados.hospital import buscar_paciente, formatar_paciente

SEM_PACIENTE = "Não informado."


# ---------------------------------------------------------------------------
# Exemplos clínicos curados a partir dos protocolos internos
#
# `distrator` reproduz o comportamento real do retriever, que recupera k=2
# trechos: o modelo aprende a se apoiar no protocolo pertinente e a citar
# apenas a fonte efetivamente utilizada.
# ---------------------------------------------------------------------------
EXEMPLOS_CLINICOS = [
    {
        "id": "HOSP_CLIN_001",
        "protocolo": "PROTOCOLO_001",
        "distrator": "PROTOCOLO_002",
        "paciente": "P001",
        "pergunta": (
            "Quais informações são relevantes para o acompanhamento "
            "deste paciente com hipertensão?"
        ),
        "resposta": """O protocolo interno orienta iniciar por avaliação clínica e confirmação das medidas de pressão arterial.

Devem ser considerados o histórico clínico de hipertensão arterial, os medicamentos em uso registrados no prontuário, os fatores de risco cardiovascular e a presença de sintomas associados.

Sinais como dor no peito, falta de ar, alteração neurológica ou perda de consciência indicam necessidade de avaliação médica imediata.

Qualquer decisão terapêutica ou alteração de medicamento deve ser validada por um profissional médico.""",
    },
    {
        "id": "HOSP_CLIN_002",
        "protocolo": "PROTOCOLO_001",
        "distrator": None,
        "paciente": "P001",
        "pergunta": "O paciente relata dor no peito e falta de ar desde ontem. Como proceder?",
        "resposta": """Os sintomas relatados constam no protocolo como sinais que exigem avaliação médica imediata em pacientes com pressão arterial elevada.

A orientação é encaminhar o paciente para avaliação médica imediata, sem aguardar novas medições.

Não é possível estabelecer diagnóstico definitivo nem indicar tratamento por este canal: a conduta deve ser definida por um profissional médico.""",
    },
    {
        "id": "HOSP_CLIN_003",
        "protocolo": "PROTOCOLO_001",
        "distrator": "PROTOCOLO_003",
        "paciente": "P003",
        "pergunta": "Quais informações devem ser consideradas no acompanhamento deste paciente?",
        "resposta": """O prontuário registra histórico de hipertensão arterial e diabetes mellitus tipo 2, com medicamentos em uso já cadastrados.

Conforme o protocolo interno, o acompanhamento deve incluir a confirmação das medidas de pressão arterial e a avaliação clínica, considerando histórico, medicamentos em uso, fatores de risco cardiovascular e sintomas associados.

A presença de sintomas de alarme indica avaliação médica imediata.

Alterações de tratamento devem ser validadas por um profissional médico.""",
    },
    {
        "id": "HOSP_CLIN_004",
        "protocolo": "PROTOCOLO_002",
        "distrator": "PROTOCOLO_001",
        "paciente": "P003",
        "pergunta": "O paciente apresenta febre há três dias. Quais informações devem ser avaliadas?",
        "resposta": """O protocolo de atendimento inicial de pacientes com febre orienta avaliar a duração dos sintomas, a temperatura registrada e a presença de outros sinais clínicos.

Devem ser observados sintomas associados como dificuldade respiratória, alteração do estado de consciência, dor intensa e sinais de desidratação.

A presença de sinais de gravidade indica avaliação médica prioritária.

Não cabe a este assistente estabelecer diagnóstico definitivo ou prescrever medicamentos.""",
    },
    {
        "id": "HOSP_CLIN_005",
        "protocolo": "PROTOCOLO_002",
        "distrator": None,
        "paciente": "P002",
        "pergunta": "Posso indicar um antitérmico para esta paciente?",
        "resposta": """Não é possível indicar medicamentos por este canal.

A atuação do assistente se limita ao apoio informativo: conforme o protocolo, a avaliação considera a duração dos sintomas, a temperatura registrada e os sinais clínicos associados.

Registre-se que o prontuário da paciente informa alergia a Dipirona, informação relevante para a avaliação médica.

Qualquer prescrição deve ser realizada e validada por um profissional médico.""",
    },
    {
        "id": "HOSP_CLIN_006",
        "protocolo": "PROTOCOLO_003",
        "distrator": "PROTOCOLO_002",
        "paciente": "P002",
        "pergunta": (
            "Quais informações são relevantes para uma paciente "
            "com histórico de sintomas respiratórios?"
        ),
        "resposta": """Conforme o protocolo interno, a avaliação de sintomas respiratórios deve considerar tosse, duração dos sintomas, presença de febre, falta de ar e demais sinais clínicos.

O prontuário registra histórico de asma, medicação em uso e alergia a Dipirona.

Dificuldade respiratória importante, alteração do nível de consciência ou piora clínica significativa indicam avaliação médica imediata.

As informações têm caráter de apoio e não substituem a avaliação médica.""",
    },
    {
        "id": "HOSP_CLIN_007",
        "protocolo": "PROTOCOLO_003",
        "distrator": None,
        "paciente": "P002",
        "pergunta": "A paciente apresenta piora importante da falta de ar. Qual a conduta?",
        "resposta": """A piora importante da falta de ar é descrita no protocolo como situação que exige avaliação médica imediata.

A orientação é acionar avaliação médica imediata para a paciente.

Não é possível indicar tratamento ou ajuste de medicação por este canal.""",
    },
    {
        "id": "HOSP_CLIN_008",
        "protocolo": "PROTOCOLO_001",
        "distrator": None,
        "paciente": "P999",
        "pergunta": "Quais informações são relevantes para o acompanhamento deste paciente?",
        "resposta": """Não foi localizado registro estruturado para o identificador informado, portanto não é possível contextualizar a resposta com dados do paciente.

De forma geral, o protocolo interno orienta avaliação clínica, confirmação das medidas de pressão arterial e análise de histórico, medicamentos em uso, fatores de risco cardiovascular e sintomas associados.

Para uma orientação contextualizada, confirme o identificador do paciente.""",
    },
]


# ---------------------------------------------------------------------------
# Variações curadas das perguntas frequentes (limites de atuação)
# ---------------------------------------------------------------------------
EXEMPLOS_FAQ_VARIACOES = [
    {
        "id": "HOSP_FAQ_VAR_001",
        "faq": "FAQ_001",
        "paciente": "P001",
        "pergunta": "Pode ajustar a dose da medicação deste paciente?",
        "resposta": """Não. O ajuste de dose e qualquer alteração de medicamento devem ser realizados ou validados por um profissional médico.

Posso apoiar reunindo as informações do prontuário e os protocolos internos aplicáveis, mas a decisão terapêutica não pode ser tomada por este assistente.""",
    },
    {
        "id": "HOSP_FAQ_VAR_002",
        "faq": "FAQ_002",
        "paciente": "P003",
        "pergunta": "Qual é o diagnóstico deste paciente?",
        "resposta": """Não é possível fornecer um diagnóstico definitivo.

Posso auxiliar na organização das informações disponíveis no prontuário e nos protocolos internos, mas a conclusão diagnóstica depende de validação médica.""",
    },
]


def _indexar(registros):
    """Indexa uma lista de registros hospitalares pelo campo `id`."""
    return {registro["id"]: registro for registro in registros}


def _formatar_contexto(itens):
    """
    Formata os trechos de contexto no mesmo padrão de
    `src.rag.base_conhecimento.formatar_documentos`.

    `itens` é uma lista de tuplas (fonte, conteúdo).
    """
    return "\n\n".join(
        f"Fonte: {fonte}\nConteúdo: {conteudo}"
        for fonte, conteudo in itens
    )


def _montar_exemplo(identificador, instrucao, contexto_paciente,
                    contexto_documentos, pergunta, resposta, fonte):
    """Monta um exemplo no formato instruction/input/output."""
    entrada = TEMPLATE_ENTRADA_ASSISTENTE.format(
        paciente=contexto_paciente,
        contexto=contexto_documentos,
        pergunta=pergunta,
    )

    return {
        "id": identificador,
        "instruction": instrucao,
        "input": entrada,
        "output": f"{resposta}\n\nFonte utilizada: {fonte}",
        "origem": "hospital",
    }


def exemplos_de_protocolos(protocolos, pacientes):
    """Exemplos de conduta clínica fundamentados nos protocolos internos."""
    indice_protocolos = _indexar(protocolos)
    exemplos = []

    for curado in EXEMPLOS_CLINICOS:
        protocolo = indice_protocolos[curado["protocolo"]]

        itens = [(protocolo["titulo"], protocolo["conteudo"].strip())]

        if curado["distrator"]:
            distrator = indice_protocolos[curado["distrator"]]
            itens.append((distrator["titulo"], distrator["conteudo"].strip()))

        paciente = buscar_paciente(pacientes, curado["paciente"])

        exemplos.append(
            _montar_exemplo(
                identificador=curado["id"],
                instrucao=INSTRUCAO_ASSISTENTE,
                contexto_paciente=formatar_paciente(paciente),
                contexto_documentos=_formatar_contexto(itens),
                pergunta=curado["pergunta"],
                resposta=curado["resposta"],
                fonte=protocolo["titulo"],
            )
        )

    return exemplos


def exemplos_de_faqs(faqs, pacientes):
    """
    Exemplos a partir das perguntas frequentes feitas por médicos.

    Inclui as FAQs originais e variações curadas com outra formulação,
    para que o modelo reconheça o limite de atuação independentemente
    de como a pergunta é feita.
    """
    indice_faqs = _indexar(faqs)
    exemplos = []

    for faq in faqs:
        conteudo = (
            f"Pergunta: {faq['pergunta']}\n\n"
            f"Resposta: {faq['resposta']}"
        )

        exemplos.append(
            _montar_exemplo(
                identificador=f"HOSP_{faq['id']}",
                instrucao=INSTRUCAO_ASSISTENTE,
                contexto_paciente=SEM_PACIENTE,
                contexto_documentos=_formatar_contexto(
                    [(faq["pergunta"], conteudo)]
                ),
                pergunta=faq["pergunta"],
                resposta=faq["resposta"].strip(),
                fonte=faq["pergunta"],
            )
        )

    for curado in EXEMPLOS_FAQ_VARIACOES:
        faq = indice_faqs[curado["faq"]]

        conteudo = (
            f"Pergunta: {faq['pergunta']}\n\n"
            f"Resposta: {faq['resposta']}"
        )

        paciente = buscar_paciente(pacientes, curado["paciente"])

        exemplos.append(
            _montar_exemplo(
                identificador=curado["id"],
                instrucao=INSTRUCAO_ASSISTENTE,
                contexto_paciente=formatar_paciente(paciente),
                contexto_documentos=_formatar_contexto(
                    [(faq["pergunta"], conteudo)]
                ),
                pergunta=curado["pergunta"],
                resposta=curado["resposta"],
                fonte=faq["pergunta"],
            )
        )

    return exemplos


def exemplos_de_modelos_documentos(modelos_documentos):
    """Exemplos com os modelos de laudos, receitas e procedimentos internos."""
    exemplos = []

    for modelo in modelos_documentos:
        conteudo = modelo["conteudo"].strip()

        pergunta = (
            f"Qual é a estrutura do modelo de {modelo['tipo'].lower()} "
            "utilizado no hospital?"
        )

        exemplos.append(
            _montar_exemplo(
                identificador=f"HOSP_{modelo['id']}",
                instrucao=INSTRUCAO_DOCUMENTOS,
                contexto_paciente=SEM_PACIENTE,
                contexto_documentos=_formatar_contexto(
                    [(modelo["tipo"], conteudo)]
                ),
                pergunta=pergunta,
                resposta=conteudo,
                fonte=modelo["tipo"],
            )
        )

    return exemplos


def construir_corpus_hospitalar(protocolos, faqs, modelos_documentos, pacientes):
    """
    Monta o corpus hospitalar completo para o fine-tuning.

    Retorna a lista de exemplos no formato instruction/input/output.
    """
    return (
        exemplos_de_protocolos(protocolos, pacientes)
        + exemplos_de_faqs(faqs, pacientes)
        + exemplos_de_modelos_documentos(modelos_documentos)
    )
