"""
Validação de segurança das respostas do assistente.

Espelha a seção 15.5 do notebook. Define os limites de atuação:
o assistente nunca prescreve ou altera medicamentos diretamente —
respostas com esse padrão recebem alerta de validação humana.
"""

TERMOS_RESTRITOS = [
    "você deve tomar",
    "tome ",
    "aumente a dose",
    "reduza a dose",
    "suspenda o medicamento",
    "pare de tomar",
]

AVISO_VALIDACAO = (
    "\n\nATENÇÃO: qualquer decisão relacionada "
    "a medicamentos ou tratamento deve ser "
    "validada por um profissional de saúde."
)


def validar_resposta(resposta):
    """
    Verifica se a resposta contém recomendações restritas.

    Retorna a resposta (com alerta adicionado, quando necessário).
    """
    resposta_minuscula = resposta.lower()

    possui_recomendacao_restrita = any(
        termo in resposta_minuscula
        for termo in TERMOS_RESTRITOS
    )

    if possui_recomendacao_restrita:
        resposta += AVISO_VALIDACAO

    return resposta
