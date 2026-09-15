"""
Base hospitalar sintética: protocolos, FAQs, modelos de documentos
e registros fictícios de pacientes.

Espelha as seções 12 e 18 do notebook. Os dados são carregados do
arquivo `data/dados_hospitalares_sinteticos.json`, que contém
exclusivamente registros fictícios (nenhum dado real de paciente).
"""

import json

from src.config import CAMINHO_DADOS_SINTETICOS


def carregar_dados_sinteticos(caminho=CAMINHO_DADOS_SINTETICOS):
    """Carrega protocolos, faqs, modelos de documentos e pacientes."""
    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    return (
        dados["protocolos"],
        dados["faqs"],
        dados["modelos_documentos"],
        dados["pacientes"],
    )


def buscar_paciente(pacientes, id_paciente):
    """Consulta o registro estruturado de um paciente (seção 14)."""
    for paciente in pacientes:
        if paciente["id"] == id_paciente:
            return paciente

    return None


def formatar_paciente(paciente):
    """Formata os dados do paciente para uso como contexto (seção 14)."""
    if paciente is None:
        return "Paciente não encontrado."

    historico = ", ".join(paciente["historico"]) or "Não informado"
    medicamentos = ", ".join(paciente["medicamentos"]) or "Não informado"
    alergias = ", ".join(paciente["alergias"]) or "Nenhuma informada"

    return f"""
ID: {paciente['id']}
Idade: {paciente['idade']}
Sexo: {paciente['sexo']}
Histórico clínico: {historico}
Medicamentos em uso: {medicamentos}
Alergias: {alergias}
Observações: {paciente['observacoes']}
""".strip()
