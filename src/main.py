"""
Demonstração do assistente médico completo.

Requer:
- GPU disponível (o modelo é carregado em 4 bits);
- o modelo especializado salvo em `tinyllama-pubmedqa-lora`
  (gerado pelo pipeline de fine-tuning ou pelo notebook).

Uso:
    python -m src.main --paciente P001 --pergunta "Quais informações são relevantes para o acompanhamento?"
"""

import argparse

from src.assistente.grafo import criar_assistente
from src.assistente.llm import carregar_modelo_especializado, criar_gerador
from src.config import CAMINHO_MODELO_SALVO
from src.dados.hospital import carregar_dados_sinteticos
from src.rag.base_conhecimento import criar_retriever


def main():
    parser = argparse.ArgumentParser(
        description="Assistente médico com LLM especializada, LangChain e LangGraph."
    )
    parser.add_argument("--paciente", default="P001", help="ID do paciente (ex.: P001)")
    parser.add_argument(
        "--pergunta",
        default=(
            "O paciente possui hipertensão. "
            "Quais informações são relevantes para o acompanhamento?"
        ),
        help="Pergunta clínica",
    )
    parser.add_argument(
        "--modelo",
        default=str(CAMINHO_MODELO_SALVO),
        help="Caminho do modelo especializado salvo",
    )
    args = parser.parse_args()

    print("Carregando dados hospitalares sintéticos...")
    protocolos, faqs, _, pacientes = carregar_dados_sinteticos()

    print("Criando base vetorial e retriever...")
    retriever = criar_retriever(protocolos, faqs)

    print("Carregando modelo especializado...")
    modelo, tokenizer = carregar_modelo_especializado(args.modelo)
    gerar_resposta_llm = criar_gerador(modelo, tokenizer)

    print("Montando fluxo LangGraph...")
    assistente = criar_assistente(pacientes, retriever, gerar_resposta_llm)

    resultado = assistente.invoke({
        "id_paciente": args.paciente,
        "pergunta": args.pergunta,
    })

    print("\n" + "=" * 70)
    print("PACIENTE:", resultado["id_paciente"])

    print("\nFONTES UTILIZADAS:")
    for fonte in resultado["fontes"]:
        print("-", fonte)

    print("\nRESPOSTA:")
    print(resultado["resposta_validada"])

    print("\nLOG DE AUDITORIA:")
    for chave, valor in resultado["log"].items():
        print(f"{chave}: {valor}")


if __name__ == "__main__":
    main()
