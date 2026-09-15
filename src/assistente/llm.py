"""
Geração de respostas com o modelo de linguagem especializado.

Espelha a seção 14.3 do notebook.
"""

from src.config import (
    GERACAO_MAX_LENGTH_ENTRADA,
    GERACAO_MAX_NEW_TOKENS,
    GERACAO_REPETITION_PENALTY,
    MAX_SEQ_LENGTH,
)


def carregar_modelo_especializado(caminho="tinyllama-pubmedqa-lora"):
    """Carrega o modelo salvo após o fine-tuning e o prepara para inferência."""
    from unsloth import FastLanguageModel

    modelo, tokenizer = FastLanguageModel.from_pretrained(
        model_name=caminho,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(modelo)

    return modelo, tokenizer


def criar_gerador(modelo, tokenizer):
    """
    Retorna a função `gerar_resposta_llm(prompt)` com a mesma lógica de
    limpeza da resposta utilizada no notebook.
    """

    def gerar_resposta_llm(prompt):

        entradas = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=GERACAO_MAX_LENGTH_ENTRADA,
        ).to(modelo.device)

        saidas = modelo.generate(
            **entradas,
            max_new_tokens=GERACAO_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=GERACAO_REPETITION_PENALTY,
        )

        quantidade_tokens_entrada = entradas["input_ids"].shape[1]
        tokens_resposta = saidas[0][quantidade_tokens_entrada:]

        resposta = tokenizer.decode(
            tokens_resposta,
            skip_special_tokens=True,
        )

        linhas = resposta.strip().splitlines()

        if linhas and linhas[0].lower().startswith("resposta:"):
            linhas = linhas[1:]

        resposta = "\n".join(linhas).strip()

        if resposta.lower().startswith("justificativa:"):
            resposta = resposta[len("Justificativa:"):].strip()

        marcadores_parada = [
            "\nPergunta:",
            "\n### Comment:",
            "\n### Instruction:",
            "\n### Input:",
        ]

        for marcador in marcadores_parada:
            if marcador in resposta:
                resposta = resposta.split(marcador)[0]

        return resposta.strip()

    return gerar_resposta_llm
