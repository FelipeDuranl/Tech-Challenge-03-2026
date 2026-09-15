"""
Configurações centrais do projeto.

Todos os parâmetros abaixo espelham exatamente os valores utilizados
no notebook `notebooks/tech_challenge_fase3.ipynb`.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_DADOS_SINTETICOS = RAIZ_PROJETO / "data" / "dados_hospitalares_sinteticos.json"
CAMINHO_PUBMEDQA = RAIZ_PROJETO / "pubmedqa" / "data" / "ori_pqal.json"
CAMINHO_LOGS = RAIZ_PROJETO / "logs" / "logs_auditoria.jsonl"
CAMINHO_MODELO_SALVO = RAIZ_PROJETO / "tinyllama-pubmedqa-lora"

# ---------------------------------------------------------------------------
# Divisão treino/teste (seção 5 do notebook)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE_SPLIT = 42

# ---------------------------------------------------------------------------
# Modelo base e fine-tuning (seções 8 a 10 do notebook)
# ---------------------------------------------------------------------------
MODELO_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_RANDOM_STATE = 3407

TREINO_OUTPUT_DIR = "outputs"

# 1024 (e não 512) porque os exemplos clínicos do corpus hospitalar chegam a
# ~850 tokens: em 512 o trecho `### Response:` seria truncado e o modelo não
# aprenderia a resposta. Também reduz o truncamento dos contextos do PubMedQA.
TREINO_MAX_LENGTH = 1024
TREINO_BATCH_SIZE = 2
TREINO_GRAD_ACCUM = 4
TREINO_WARMUP_STEPS = 5
TREINO_MAX_STEPS = 60
TREINO_LEARNING_RATE = 2e-4
TREINO_LOGGING_STEPS = 1
TREINO_OPTIM = "adamw_8bit"
TREINO_WEIGHT_DECAY = 0.01
TREINO_LR_SCHEDULER = "linear"
TREINO_SEED = 3407

# ---------------------------------------------------------------------------
# Template de prompt (seção 6.4 do notebook)
# ---------------------------------------------------------------------------
TEMPLATE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# ---------------------------------------------------------------------------
# Corpus hospitalar no fine-tuning (seção 6.6 do notebook)
#
# O corpus hospitalar é pequeno e curado manualmente. Para que tenha peso
# relevante frente aos 800 exemplos do PubMedQA, os exemplos são repetidos
# antes do embaralhamento do conjunto de treino.
# ---------------------------------------------------------------------------
REPETICOES_CORPUS_HOSPITALAR = 5
SEED_EMBARALHAMENTO = 42

# Instrução usada tanto na geração dos exemplos de treino quanto em tempo de
# execução pelo nó `gerar_resposta` — treino e inferência compartilham o mesmo
# formato, evitando divergência entre o que o modelo aprendeu e o que recebe.
INSTRUCAO_ASSISTENTE = """Analise as informações do paciente utilizando apenas o contexto fornecido.

Forneça uma resposta objetiva de apoio clínico.

Não realize diagnóstico definitivo.
Não prescreva medicamentos.
Não altere tratamentos.

Qualquer decisão clínica deve ser validada por um profissional de saúde.

Ao final, indique a fonte utilizada."""

INSTRUCAO_DOCUMENTOS = """Apresente a estrutura do modelo de documento interno do hospital.

Mantenha os campos padronizados, sem preencher dados de pacientes.

Ao final, indique a fonte utilizada."""

TEMPLATE_ENTRADA_ASSISTENTE = """DADOS DO PACIENTE:
{paciente}

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}"""

# ---------------------------------------------------------------------------
# RAG (seção 13 do notebook)
# ---------------------------------------------------------------------------
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 2

# ---------------------------------------------------------------------------
# Geração de respostas (seção 14.3 do notebook)
# ---------------------------------------------------------------------------
GERACAO_MAX_NEW_TOKENS = 100
GERACAO_REPETITION_PENALTY = 1.15
GERACAO_MAX_LENGTH_ENTRADA = 1024

# ---------------------------------------------------------------------------
# Avaliação no PubMedQA (seção 11 do notebook)
#
# Mantido em 512 propositalmente: o protocolo de avaliação permanece idêntico
# ao das execuções anteriores, de modo que as métricas continuem comparáveis
# entre as versões do corpus de treinamento.
# ---------------------------------------------------------------------------
AVALIACAO_MAX_LENGTH_ENTRADA = 512
AVALIACAO_MAX_NEW_TOKENS = 80
