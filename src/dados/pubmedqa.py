"""
Carregamento, curadoria e preparação do dataset PubMedQA.

Espelha as seções 4, 5 e 6 do notebook.
"""

import json
from collections import Counter

from sklearn.model_selection import train_test_split

from src.config import (
    CAMINHO_PUBMEDQA,
    RANDOM_STATE_SPLIT,
    TEMPLATE_PROMPT,
    TEST_SIZE,
)

CAMPOS_OBRIGATORIOS = [
    "QUESTION",
    "CONTEXTS",
    "final_decision",
    "LONG_ANSWER",
]


def carregar_pubmedqa(caminho=CAMINHO_PUBMEDQA):
    """Carrega o arquivo original do PubMedQA (PQA-L)."""
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def verificar_registros_incompletos(pubmedqa):
    """Curadoria: identifica registros sem os campos obrigatórios (seção 4.4)."""
    registros_incompletos = []

    for identificador, registro in pubmedqa.items():
        for campo in CAMPOS_OBRIGATORIOS:
            if not registro.get(campo):
                registros_incompletos.append({
                    "id": identificador,
                    "campo": campo,
                })

    return registros_incompletos


def preparar_dataset(pubmedqa):
    """Cria a representação simplificada dos dados (seção 4.5)."""
    dataset_pubmedqa = []

    for identificador, registro in pubmedqa.items():
        contexto = "\n\n".join(registro["CONTEXTS"])

        exemplo = {
            "id": identificador,
            "question": registro["QUESTION"],
            "context": contexto,
            "answer": registro["final_decision"],
            "explanation": registro["LONG_ANSWER"],
        }

        dataset_pubmedqa.append(exemplo)

    return dataset_pubmedqa


def dividir_treino_teste(dataset_pubmedqa):
    """Divisão estratificada 80/20 com semente fixa (seção 5)."""
    return train_test_split(
        dataset_pubmedqa,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE_SPLIT,
        stratify=[exemplo["answer"] for exemplo in dataset_pubmedqa],
    )


def distribuicao_classes(dataset):
    """Distribuição das respostas yes/no/maybe (seções 4.2 e 5.1)."""
    return Counter(exemplo["answer"] for exemplo in dataset)


def preparar_exemplo(exemplo):
    """Estrutura instruction/input/output (seção 6.1)."""
    return {
        "id": exemplo["id"],

        "instruction": (
            "Responda à pergunta utilizando apenas "
            "o contexto científico fornecido."
        ),

        "input": (
            f"Contexto:\n{exemplo['context']}\n\n"
            f"Pergunta:\n{exemplo['question']}"
        ),

        "output": (
            f"Resposta: {exemplo['answer']}\n\n"
            f"Justificativa: {exemplo['explanation']}"
        ),

        "origem": "pubmedqa",
    }


def formatar_para_treinamento(exemplo):
    """Concatena instruction/input/output no template textual (seção 6.4)."""
    texto = TEMPLATE_PROMPT.format(
        exemplo["instruction"],
        exemplo["input"],
        exemplo["output"],
    )

    return {
        **exemplo,
        "text": texto,
    }


def pipeline_preparacao(caminho=CAMINHO_PUBMEDQA):
    """
    Executa o pipeline completo de preparação dos dados:

    carregamento -> curadoria -> simplificação -> divisão -> formatação.

    Retorna (dataset_treino_formatado, dataset_teste_formatado).
    """
    pubmedqa = carregar_pubmedqa(caminho)

    incompletos = verificar_registros_incompletos(pubmedqa)
    if incompletos:
        print("Registros com campos obrigatórios ausentes:", len(incompletos))

    dataset = preparar_dataset(pubmedqa)
    dataset_treino, dataset_teste = dividir_treino_teste(dataset)

    dataset_treino_formatado = [
        formatar_para_treinamento(preparar_exemplo(exemplo))
        for exemplo in dataset_treino
    ]

    dataset_teste_formatado = [
        formatar_para_treinamento(preparar_exemplo(exemplo))
        for exemplo in dataset_teste
    ]

    return dataset_treino_formatado, dataset_teste_formatado
