"""
Montagem do conjunto de treinamento combinando as duas fontes de dados.

O fine-tuning utiliza:

- **PubMedQA (PQA-L)** — raciocínio clínico sobre literatura biomédica,
  no formato `Resposta: yes/no/maybe` + justificativa;
- **Corpus hospitalar sintético** — protocolos internos, perguntas frequentes
  e modelos de documentos, no formato de apoio clínico utilizado pelo
  assistente em produção.

As duas fontes usam instruções distintas, o que permite ao modelo condicionar
o formato da saída à tarefa solicitada sem que um corpus degrade o outro.
"""

import random
from collections import Counter

from src.config import (
    CAMINHO_DADOS_SINTETICOS,
    CAMINHO_PUBMEDQA,
    REPETICOES_CORPUS_HOSPITALAR,
    SEED_EMBARALHAMENTO,
)
from src.dados.corpus_hospitalar import construir_corpus_hospitalar
from src.dados.hospital import carregar_dados_sinteticos
from src.dados.pubmedqa import formatar_para_treinamento, pipeline_preparacao


def montar_dataset_treinamento(caminho_pubmedqa=CAMINHO_PUBMEDQA,
                               caminho_hospital=CAMINHO_DADOS_SINTETICOS,
                               repeticoes_hospital=REPETICOES_CORPUS_HOSPITALAR,
                               seed=SEED_EMBARALHAMENTO):
    """
    Monta o conjunto de treinamento combinado e o conjunto de teste do PubMedQA.

    O corpus hospitalar é curado e pequeno (poucas dezenas de exemplos). Para
    que tenha peso relevante frente aos 800 exemplos do PubMedQA, ele é
    repetido `repeticoes_hospital` vezes antes do embaralhamento.

    O conjunto de teste permanece exclusivamente com exemplos do PubMedQA,
    preservando a comparabilidade das métricas com a avaliação anterior e
    garantindo que nenhum exemplo de teste participe do treinamento.

    Retorna (dataset_treino_formatado, dataset_teste_formatado).
    """
    treino_pubmedqa, teste_pubmedqa = pipeline_preparacao(caminho_pubmedqa)

    protocolos, faqs, modelos_documentos, pacientes = carregar_dados_sinteticos(
        caminho_hospital
    )

    corpus_hospitalar = construir_corpus_hospitalar(
        protocolos, faqs, modelos_documentos, pacientes
    )

    corpus_hospitalar_formatado = [
        formatar_para_treinamento(exemplo)
        for exemplo in corpus_hospitalar
    ]

    dataset_treino = (
        treino_pubmedqa
        + corpus_hospitalar_formatado * repeticoes_hospital
    )

    random.Random(seed).shuffle(dataset_treino)

    return dataset_treino, teste_pubmedqa


def composicao(dataset):
    """Contagem de exemplos por origem — útil para documentar o corpus."""
    return Counter(exemplo["origem"] for exemplo in dataset)
