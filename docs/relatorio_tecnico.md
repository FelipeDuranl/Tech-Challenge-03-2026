# Tech-Challenge-03-2026

# Relatório Técnico

## 1. Visão Geral da Solução

Nesta fase, o projeto foi dividido em quatro etapas:

1. **Preparação e curadoria dos dados:** organização de uma base científica pública, a PubMedQA, e construção de uma base hospitalar sintética contendo protocolos, perguntas frequentes, modelos de documentos e registros fictícios de pacientes.
2. **Fine-tuning da LLM:** especialização de um modelo com os dados médicos preparados, utilizando LoRA e quantização em 4 bits para viabilizar o treinamento no Google Colab.
3. **Construção do assistente com LangChain:** integração do modelo especializado ao RAG e à consulta de registros estruturados de pacientes.
4. **Orquestração e segurança com LangGraph:** organização do fluxo de decisão em etapas independentes, com validação de segurança, indicação das fontes e registro de auditoria.

O fluxo combina fine-tuning, RAG e orquestração para criar um assistente de **apoio à decisão clínica**, nunca um substituto da avaliação médica.

---

## 2. Preparação e Curadoria dos Dados

O assistente precisa de dois tipos de conhecimento, por isso trabalhamos com duas fontes distintas.

### 2.1 PubMedQA — raciocínio clínico

Utilizamos o subconjunto **PQA-L**, com 1.000 pares de pergunta e resposta baseados em publicações biomédicas. A distribuição das classes é desbalanceada:

| Classe | Quantidade | Proporção |
|---|---|---|
| yes | 552 | 55,2% |
| no | 338 | 33,8% |
| maybe | 110 | 11,0% |

A curadoria verificou os campos obrigatórios em todos os registros — nenhum incompleto foi encontrado. Os trechos de contexto foram unificados em texto contínuo e o identificador original preservado para rastreabilidade. A divisão foi **estratificada** em 800 exemplos de treino e 200 de teste (`random_state=42`), e o conjunto de teste não participou do treinamento.

### 2.2 Base hospitalar sintética

Toda a base hospitalar é **sintética**, criada exclusivamente para o projeto — a anonimização é garantida por construção, pois não há dado real a ser anonimizado.

| Conjunto | Qtd. | Conteúdo |
|---|---|---|
| Protocolos médicos | 3 | Hipertensão arterial, febre e sintomas respiratórios |
| Perguntas frequentes | 3 | Limites de atuação do assistente |
| Modelos de documentos | 4 | Relatório médico, laudo de exame, receituário e registro de procedimento |
| Pacientes fictícios | 3 | Histórico, medicamentos, alergias e observações |

### 2.3 Corpus hospitalar para o fine-tuning

O desafio exige que o fine-tuning use os **dados próprios do hospital**, que o PubMedQA não contém. Geramos a partir da base hospitalar um corpus de **17 exemplos de instrução** curados manualmente:

| Origem | Exemplos | Papel no treinamento |
|---|---|---|
| Protocolos internos | 8 | Conduta clínica fundamentada no protocolo, com dados do paciente |
| Perguntas frequentes | 5 | Limites de atuação (3 FAQs + 2 variações de formulação) |
| Modelos de documentos | 4 | Estrutura de laudos, receitas e procedimentos internos |

Três decisões de projeto:

- **Mesmo formato em treino e inferência** — os exemplos reproduzem exatamente o que o assistente recebe em produção (instrução, dados do paciente, contexto recuperado e pergunta), usando constantes compartilhadas com o nó de geração do LangGraph;
- **Instruções distintas por tarefa** — o corpus hospitalar usa instrução diferente da do PubMedQA, para o modelo condicionar o formato da saída à tarefa sem que um corpus degrade o outro;
- **Distratores no contexto** — metade dos exemplos clínicos inclui um protocolo irrelevante, reproduzindo o `k=2` do retriever, ensinando o modelo a citar apenas a fonte pertinente.

Por ser pequeno, o corpus é repetido 5 vezes antes do embaralhamento, representando **9,6%** do treino final (800 do PubMedQA + 85 hospitalares = **885**). O **teste permanece 100% PubMedQA**, garantindo que nenhum exemplo de teste participe do treinamento.

Todos os exemplos seguem o formato *instruction / input / output* no template Alpaca, idêntico no treinamento e na inferência.

---

## 3. Fine-tuning da LLM

| Item | Configuração |
|---|---|
| Modelo base | TinyLlama 1.1B Chat v1.0 |
| Biblioteca | Unsloth (`FastLanguageModel`) |
| Quantização | 4 bits |
| Técnica | LoRA (r=16, alpha=16, dropout=0) |
| Módulos adaptados | q/k/v/o_proj, gate/up/down_proj |
| Parâmetros treináveis | 12.615.680 de 1.112.664.064 (**1,13%**) |
| Sequência máxima | 1024 |
| Batch | 2 × acumulação 4 (efetivo 8) |
| Learning rate | 2e-4, scheduler linear, 5 passos de warmup |
| Otimizador | `adamw_8bit`, weight decay 0,01 |
| Passos | 60 (semente 3407) |

Escolhemos um modelo de **pequeno porte** para viabilizar a demonstração completa no Colab (GPU T4). O `max_length` é **1024** porque os exemplos clínicos chegam a ~850 tokens: com 512, o trecho `### Response:` seria truncado e o modelo não aprenderia a resposta.

**Resultado:** loss final ≈ **1,478**, tempo total ≈ **183 segundos**. Os 60 passos percorrem 480 dos 885 exemplos, o equivalente a **0,54 época**.

Ambiente da execução: Tesla T4 (14,56 GB) no Google Colab, Unsloth 2026.9.4, Transformers 5.5.0, PyTorch 2.11.0+cu128. O T4 não suporta `bfloat16`, então o treino usou `fp16`.

<!-- Inserir aqui o print da curva de loss do treinamento (seção 10.2 do notebook) -->

---

## 4. Avaliação do Modelo e Interpretação dos Resultados

A avaliação usou os 200 exemplos de teste. O prompt foi construído **sem** a resposta esperada, a decisão (`yes`/`no`/`maybe`) foi extraída por expressão regular e comparada com o gabarito. Respostas sem classe identificável contam como **inválidas** — métrica que mede se o fine-tuning ensinou a estrutura de saída.

### 4.1 Comparação entre o modelo original e o especializado

Aplicamos **exatamente o mesmo protocolo** ao modelo base, sem fine-tuning. A única diferença é a presença dos adaptadores LoRA.

| Métrica | Modelo base | Fine-tuned | Ganho |
|---|---|---|---|
| Accuracy | 0,0000 | **0,6400** | +0,6400 |
| Precision (macro) | 0,0000 | 0,4733 | +0,4733 |
| Recall (macro) | 0,0000 | 0,4646 | +0,4646 |
| F1-Score (macro) | 0,0000 | 0,4674 | +0,4674 |
| Respostas inválidas | 200/200 (100%) | 19/200 (9,5%) | **-90,5 p.p.** |

O modelo base, nunca exposto ao formato esperado, produziu **200 respostas inválidas em 200** — nenhuma pôde ser classificada, zerando todas as métricas.

<!-- Inserir aqui o print da tabela comparativa (seção 19.2 do notebook) -->

### 4.2 Desempenho por classe (modelo especializado)

| Classe | Precision | Recall | F1-Score | Suporte |
|---|---|---|---|---|
| yes | 0,72 | 0,79 | 0,76 | 110 |
| no | 0,69 | 0,60 | 0,65 | 68 |
| maybe | 0,00 | 0,00 | 0,00 | 22 |

<!-- Inserir aqui o print da matriz de confusão (seção 11.9 do notebook) -->

### 4.3 Interpretação

O fine-tuning teve dois efeitos: o **aprendizado do formato de saída** (de 100% para 9,5% de respostas inválidas) e o **aprendizado da tarefa clínica** (accuracy de 0 para 0,64). As limitações, porém, são igualmente relevantes:

- **A classe `maybe` não foi aprendida** (F1 = 0) — nenhum dos 22 exemplos foi classificado corretamente. Com apenas 88 exemplos de treino dessa classe (11%) e 60 passos, o modelo tendeu às majoritárias. É o que explica a distância entre a accuracy (0,640) e o F1 macro (0,467): **a accuracy sozinha mascara o fato de uma classe ter sido ignorada**.

- **Atribuição do ganho** — em relação à configuração anterior (só PubMedQA, `max_length=512`), a accuracy subiu de 0,620 para 0,640 e o F1 macro de 0,450 para 0,467. Como **duas variáveis mudaram ao mesmo tempo**, não é possível atribuir o ganho a uma delas isoladamente.

- **Qualidade da geração é irregular** — nos testes do fluxo, o paciente P001 gerou resposta coerente e fiel ao protocolo, enquanto P002 e P003 produziram apenas fragmentos. É esperado em um modelo de 1,1B treinado por 60 passos, e mostra que o valor demonstrado está na **arquitetura do fluxo** mais do que na fluência do gerador.

**Possíveis melhorias:** treinar por mais épocas; balancear as classes; usar modelo base maior (ex.: Llama 3 8B); ampliar o corpus hospitalar; e aumentar o `max_new_tokens`, que hoje corta respostas no meio da frase.

---

## 5. Assistente Médico com LangChain

O assistente combina três componentes:

- **Base estruturada de pacientes** — registros fictícios consultados por identificador, simulando um prontuário. É o que permite contextualizar a resposta com informações do paciente;
- **Base de conhecimento vetorial (RAG)** — configuração abaixo;
- **LLM especializada** — o TinyLlama com LoRA treinado no PubMedQA e no corpus hospitalar.

| Etapa | Configuração |
|---|---|
| Documentos indexados | 6 (3 protocolos + 3 perguntas frequentes) |
| Divisão em chunks | `RecursiveCharacterTextSplitter` (500 caracteres, overlap 50) → 7 chunks |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Base vetorial | FAISS |
| Recuperação | `k=2` trechos mais relevantes |

Os **modelos de documentos ficam fora da base vetorial**: por serem formulários de campos vazios, competem por similaridade com perguntas clínicas e poluem a recuperação. Esse conhecimento entra pelo fine-tuning, não pelo RAG.

---

## 6. Orquestração do Fluxo com LangGraph

O fluxo é um grafo de estados com cinco nós sequenciais. Cada nó atualiza um estado compartilhado (`EstadoAssistente`) que transporta paciente, pergunta, contextos, fontes, respostas e log.

![Diagrama do fluxo LangChain/LangGraph](diagrama_fluxo.png)

| Nó | Responsabilidade |
|---|---|
| `buscar_paciente` | Localiza o registro estruturado e o formata como contexto |
| `recuperar_contexto` | Consulta a base FAISS e coleta os metadados de fonte |
| `gerar_resposta` | Monta o prompt e chama a LLM |
| `validar_resposta` | Aplica as regras de segurança pós-geração |
| `registrar_log` | Grava o registro de auditoria em JSONL |

---

## 7. Segurança, Explainability e Auditoria

**Limites de atuação** em duas camadas independentes, para que a falha de uma não comprometa a segurança:

- **No prompt** — não realizar diagnóstico definitivo, não prescrever, não alterar tratamentos;
- **Pós-geração** — a resposta é inspecionada contra padrões restritos ("aumente a dose", "suspenda o medicamento" etc.); se detectados, um alerta de validação humana é anexado antes da entrega.

**Explainability:** cada documento carrega metadados (`id`, `tipo`, `fonte`). As fontes recuperadas são exibidas junto à resposta e registradas no log. O corpus de fine-tuning ainda treina o modelo a encerrar suas respostas indicando a fonte utilizada.

**Auditoria:** cada execução gera uma linha JSON em `logs/logs_auditoria.jsonl` com data/hora, paciente, pergunta, fontes, resposta original e resposta final após validação.

### Exemplo de execução real (paciente P001)

```
PERGUNTA:
Quais informações são relevantes para o acompanhamento de um paciente com hipertensão?

FONTES:
- Atendimento inicial de hipertensão arterial
- Atendimento inicial de pacientes com febre

RESPOSTA:
O paciente deve ser avaliado considerando os fatores de risco cardiovascular,
presença de sintomas associados, histórico clínico e medicamentos em uso.
Casos com pressão arterial muito elevada acompanhadas de sintomas como dor no
peito, falta de ar, alteração neurológica ou perda de cons
```

A resposta é aderente ao protocolo recuperado, não prescreve conduta e direciona casos graves para avaliação médica. O retriever trouxe dois trechos e o modelo se apoiou no pertinente, ignorando o de febre. Dois defeitos ficam visíveis: a resposta é **cortada no meio da palavra** (`max_new_tokens=100`) e o pós-processamento, sensível a maiúsculas, deixou vazar o marcador `pergunta:` em minúsculo em outro caso de teste.

<!-- Inserir aqui o print da execução do fluxo (seção 15.9 ou 16 do notebook) -->

---

## 8. Desafios Enfrentados

- **Escolha do modelo base:** modelos maiores produziriam respostas muito melhores, mas inviabilizariam o treino no Colab. Optamos pelo TinyLlama 1.1B com 4 bits e LoRA, assumindo a perda de qualidade textual em troca de demonstrar o pipeline completo.

- **Incluir os dados do hospital no fine-tuning:** inicialmente eles eram usados só no RAG, o que não atendia ao requisito do desafio. Foi preciso construir um corpus de instruções a partir dessa base e integrá-lo ao treino.

- **Truncamento dos exemplos:** os exemplos clínicos chegam a ~850 tokens e o treino estava limitado a 512 — na prática, a resposta seria cortada e o modelo não aprenderia nada com eles. Ampliamos o `max_length` para 1024.

- **Divergência entre treino e inferência:** o assistente produzia respostas incoerentes porque o modelo fora treinado só no formato `yes/no/maybe`, mas era consultado em português para apoio clínico em texto livre. A solução foi construir o corpus hospitalar no mesmo formato usado em produção.

- **Ruído na recuperação do RAG:** os formulários de documentos eram recuperados como contexto mais relevante para perguntas clínicas. Optamos por mantê-los fora da base vetorial.

- **Onde aplicar a segurança:** confiar apenas no prompt era arriscado, já que o modelo pode ignorá-lo. Implementamos também uma validação pós-geração, independente do modelo.

---

## Conclusão

O projeto cobre de ponta a ponta o ciclo exigido: curadoria de dados médicos, fine-tuning com LoRA, integração via LangChain com prontuário e base vetorial, orquestração com LangGraph, segurança em duas camadas, explainability por fontes e auditoria por logs.

A comparação com o modelo base mostrou que o fine-tuning foi decisivo — o modelo original não conseguia sequer produzir uma resposta no formato esperado (200 inválidas em 200), enquanto o especializado atingiu accuracy de 0,64. O F1 macro de 0,47, porém, deixa claro que a classe minoritária não foi aprendida, reforçando que **a accuracy isolada pode mascarar limitações relevantes** em problemas desbalanceados. Aprendemos também que a consistência entre o formato de treino e o de inferência pesa tanto quanto a qualidade dos dados: boa parte dos problemas iniciais vinha desse descompasso, não do modelo.

Assim como nas fases anteriores, o sistema funciona como **ferramenta de apoio à decisão clínica**, não como substituto da avaliação médica. As respostas indicam as fontes, passam por validação e ficam registradas para auditoria — mas a conduta final continua sendo responsabilidade do profissional de saúde.
