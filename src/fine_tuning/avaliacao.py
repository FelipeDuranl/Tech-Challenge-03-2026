"""
Avaliação quantitativa do modelo no conjunto de teste do PubMedQA.

Espelha a seção 11 do notebook. As mesmas funções servem para avaliar
o modelo especializado (fine-tuned) e o modelo base (baseline),
permitindo a comparação antes/depois do fine-tuning.
"""

import re

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from src.config import (
    AVALIACAO_MAX_LENGTH_ENTRADA,
    AVALIACAO_MAX_NEW_TOKENS,
    TEMPLATE_PROMPT,
)

CLASSES = ["yes", "no", "maybe"]


def extrair_decisao(texto):
    """Extrai a decisão final (yes/no/maybe) da resposta gerada (seção 11.4)."""
    padrao = r"Resposta:\s*(yes|no|maybe)"

    resultado = re.search(padrao, texto, flags=re.IGNORECASE)

    if resultado:
        return resultado.group(1).lower()

    return None


def avaliar_modelo(modelo, tokenizer, dataset_teste,
                   max_new_tokens=AVALIACAO_MAX_NEW_TOKENS,
                   descricao="Avaliando modelo"):
    """
    Gera respostas para todos os exemplos do conjunto de teste
    e extrai a decisão prevista (seção 11.6).
    """
    resultados_avaliacao = []

    for exemplo in tqdm(dataset_teste, desc=descricao):

        prompt = TEMPLATE_PROMPT.format(
            exemplo["instruction"],
            exemplo["input"],
            "",
        )

        entradas = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=AVALIACAO_MAX_LENGTH_ENTRADA,
        ).to(modelo.device)

        saidas = modelo.generate(
            **entradas,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        quantidade_tokens_entrada = entradas["input_ids"].shape[1]
        tokens_resposta = saidas[0][quantidade_tokens_entrada:]

        resposta_gerada = tokenizer.decode(
            tokens_resposta,
            skip_special_tokens=True,
        )

        resultados_avaliacao.append({
            "id": exemplo["id"],
            "esperado": extrair_decisao(exemplo["output"]),
            "previsto": extrair_decisao(resposta_gerada),
            "resposta_gerada": resposta_gerada,
        })

    return resultados_avaliacao


def calcular_metricas(resultados_avaliacao):
    """Calcula accuracy, precision, recall e F1 macro (seção 11.8)."""
    y_real = [r["esperado"] for r in resultados_avaliacao]

    y_previsto = [
        r["previsto"] if r["previsto"] is not None else "invalid"
        for r in resultados_avaliacao
    ]

    quantidade_invalidas = sum(
        r["previsto"] is None for r in resultados_avaliacao
    )

    metricas = {
        "total": len(resultados_avaliacao),
        "respostas_invalidas": quantidade_invalidas,
        "accuracy": accuracy_score(y_real, y_previsto),
        "precision_macro": precision_score(
            y_real, y_previsto, labels=CLASSES,
            average="macro", zero_division=0,
        ),
        "recall_macro": recall_score(
            y_real, y_previsto, labels=CLASSES,
            average="macro", zero_division=0,
        ),
        "f1_macro": f1_score(
            y_real, y_previsto, labels=CLASSES,
            average="macro", zero_division=0,
        ),
    }

    relatorio = classification_report(
        y_real, y_previsto, labels=CLASSES, zero_division=0,
    )

    return metricas, relatorio


def imprimir_comparacao(metricas_base, metricas_ft):
    """Tabela comparativa entre o modelo base e o modelo fine-tuned."""
    print(f"{'Métrica':<22}{'Modelo base':>14}{'Fine-tuned':>14}")
    print("-" * 50)

    for chave in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(
            f"{chave:<22}"
            f"{metricas_base[chave]:>14.4f}"
            f"{metricas_ft[chave]:>14.4f}"
        )

    print(
        f"{'respostas_invalidas':<22}"
        f"{metricas_base['respostas_invalidas']:>14}"
        f"{metricas_ft['respostas_invalidas']:>14}"
    )
