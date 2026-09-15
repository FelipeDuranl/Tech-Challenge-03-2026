"""
Pipeline de fine-tuning com Unsloth + LoRA.

Espelha as seções 7 a 10 do notebook. Requer GPU (ex.: Google Colab T4).
"""

from datasets import Dataset

from src.config import (
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_RANDOM_STATE,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MODELO_BASE,
    TREINO_BATCH_SIZE,
    TREINO_GRAD_ACCUM,
    TREINO_LEARNING_RATE,
    TREINO_LOGGING_STEPS,
    TREINO_LR_SCHEDULER,
    TREINO_MAX_LENGTH,
    TREINO_MAX_STEPS,
    TREINO_OPTIM,
    TREINO_OUTPUT_DIR,
    TREINO_SEED,
    TREINO_WARMUP_STEPS,
    TREINO_WEIGHT_DECAY,
)


def carregar_modelo_base(model_name=MODELO_BASE):
    """Carrega o modelo base com quantização em 4 bits (seção 8)."""
    from unsloth import FastLanguageModel

    modelo, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    return modelo, tokenizer


def configurar_lora(modelo):
    """Adiciona os adaptadores LoRA ao modelo base (seção 9)."""
    from unsloth import FastLanguageModel

    modelo = FastLanguageModel.get_peft_model(
        modelo,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=LORA_RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )

    return modelo


def criar_trainer(modelo, tokenizer, dataset_treino_formatado):
    """Configura o SFTTrainer (seção 10)."""
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported

    hf_treino = Dataset.from_list(dataset_treino_formatado)

    configuracao_treinamento = SFTConfig(
        output_dir=TREINO_OUTPUT_DIR,

        dataset_text_field="text",
        max_length=TREINO_MAX_LENGTH,

        per_device_train_batch_size=TREINO_BATCH_SIZE,
        gradient_accumulation_steps=TREINO_GRAD_ACCUM,

        warmup_steps=TREINO_WARMUP_STEPS,
        max_steps=TREINO_MAX_STEPS,

        learning_rate=TREINO_LEARNING_RATE,

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        logging_steps=TREINO_LOGGING_STEPS,

        optim=TREINO_OPTIM,
        weight_decay=TREINO_WEIGHT_DECAY,
        lr_scheduler_type=TREINO_LR_SCHEDULER,

        seed=TREINO_SEED,
        report_to="none",
    )

    return SFTTrainer(
        model=modelo,
        processing_class=tokenizer,
        train_dataset=hf_treino,
        args=configuracao_treinamento,
    )


def executar_fine_tuning(dataset_treino_formatado,
                         caminho_saida="tinyllama-pubmedqa-lora"):
    """
    Executa o pipeline completo de fine-tuning e salva o modelo.

    Retorna (modelo, tokenizer, resultado_treinamento).
    """
    modelo, tokenizer = carregar_modelo_base()
    modelo = configurar_lora(modelo)

    trainer = criar_trainer(modelo, tokenizer, dataset_treino_formatado)
    resultado_treinamento = trainer.train()

    trainer.save_model(caminho_saida)
    tokenizer.save_pretrained(caminho_saida)

    print("Modelo especializado salvo com sucesso.")

    return modelo, tokenizer, resultado_treinamento
