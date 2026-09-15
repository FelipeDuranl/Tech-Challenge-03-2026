"""
Registro de logs para rastreamento e auditoria.

Espelha a seção 15.6 do notebook. Cada execução do assistente gera
uma linha JSON (formato JSONL) com data/hora, paciente, pergunta,
fontes utilizadas e respostas antes e depois da validação.
"""

import json
from datetime import datetime
from pathlib import Path

from src.config import CAMINHO_LOGS


def registrar_log(id_paciente, pergunta, fontes,
                  resposta_original, resposta_final,
                  caminho=CAMINHO_LOGS):
    """Grava o registro de auditoria da execução e o retorna."""
    log = {
        "data_hora": datetime.now().isoformat(),
        "id_paciente": id_paciente,
        "pergunta": pergunta,
        "fontes": fontes,
        "resposta_original": resposta_original,
        "resposta_final": resposta_final,
    }

    Path(caminho).parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(log, ensure_ascii=False) + "\n")

    return log
