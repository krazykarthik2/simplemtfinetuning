"""LoRA fine-tuning for English->Telugu machine translation (formal register).

The L-SPEAR translation stage uses an Indic-translation model so the Telugu
output follows the formal (shisthita bhasha) register instead of the casual
register. We LoRA-tune a base seq2seq MT model on a *formal-subset* of a real
human-translated parallel corpus (Samanantar En-Te), selected by formal Telugu
morpho-syntactic markers, so the whole translation style becomes formal.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

logger = logging.getLogger("lspear.mt")

# --------------------------------------------------------------------------- #
# Formal-register selection                                                    #
# --------------------------------------------------------------------------- #
# News/administrative reporting verbs typical of formal Telugu.
FORMAL_VERBS = (
    "తెలిపారు", "తెలిపారని", "పేర్కొన్నారు", "పేర్కొన్నారని", "వెల్లడించారు",
    "వెల్లడి", "వివరించారు", "వెలువరించారు", "ప్రకటించారు", "స్పష్టం చేశారు",
    "తెలిపింది", "పేర్కొంది", "చేయబడుతోంది", "చేయబడింది", "చేయబడ్డారు",
    "ఉంటుందని తెలిపారు", "అభిప్రాయాన్ని వ్యక్తం చేశారు",
)
# Official / administrative vocabulary.
OFFICIAL_NOM = (
    "ప్రభుత్వం", "మంత్రి", "శాసనసభ", "న్యాయస్థానం", "అధికారి", "విధానం",
    "సంస్థ", "నిబంధనలు", "శాసనం", "పారిశ్రామిక", "మంత్రిత్వ", "కమిషన్",
    "వాణిజ్య", "పన్ను", "ఉత్పత్తి", "అభివృద్ధి", "న్యాయమూర్తి", "న్యాయపరమైన",
    "ముఖ్యమంత్రి", "శాఖ", "నిర్ణయం", "చట్టం", "అధికారికం", "కార్యక్రమం",
)
# Casual / colloquial markers that disqualify a sentence from the formal subset.
COLLOQUIAL = (
    "అన్నాడు", "చేశాడు", "పోయింది", "ఏంటి", "అనుకుంటున్న", "ఉంది లే",
    "లేదండీ", "అమ్మాయ్", "అబ్బాయ్", "ఫ్రూట్", "బిల్లు కట్టాలి", "అండి",
    "తెలుసా", "చూడండి", "ఇక్కడున్న", "అక్కడున్న", "వచ్చేస్తా", "వెళ్తాం",
)


def _is_formal(tgt: str) -> bool:
    """True if the Telugu side looks like the formal register."""
    if any(c in tgt for c in COLLOQUIAL):
        return False
    if any(v in tgt for v in FORMAL_VERBS):
        return True
    if any(w in tgt for w in OFFICIAL_NOM):
        return True
    return False


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class MTConfig:
    # --- model / language ------------------------------------------------ #
    base_model: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "eng_Latn"
    tgt_lang: str = "tel_Telu"

    # --- LoRA ------------------------------------------------------------- #
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj", "k_proj", "out_proj")

    # --- data ------------------------------------------------------------- #
    dataset: str = "samanantar"          # real parallel corpus (en -> te)
    max_train_samples: int = 120_000
    max_eval_samples: int = 2_000
    max_src_len: int = 64
    max_tgt_len: int = 64

    # --- training --------------------------------------------------------- #
    output_dir: str = "outputs/mt_formal_te"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 200
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "bleu"
    greater_is_better: bool = True
    predict_with_generate: bool = True
    generation_num_beams: int = 4
    generation_max_length: int = 64
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True
    seed: int = 42
    report_to: str = "none"

    @classmethod
    def from_yaml(cls, path: str) -> "MTConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        fields = {f.name for f in dataclasses.fields(cls)}
        known = {k: v for k, v in raw.items() if k in fields}
        cfg = cls(**known)
        cfg.resolve_paths(path)
        return cfg

    def resolve_paths(self, config_path: str) -> None:
        base = os.path.dirname(os.path.abspath(config_path))
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(base, self.output_dir)

    def save(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
def _keep_formal(example: dict) -> bool:
    src, tgt = example["src"], example["tgt"]
    sw, tw = len(src.split()), len(tgt.split())
    if not (6 <= sw <= 40) or not (4 <= tw <= 60):
        return False
    return _is_formal(tgt)


def build_mt_dataset(cfg: MTConfig) -> Tuple:
    """Load the real Samanantar En-Te corpus, keep the formal-register subset,
    and return (train, eval) HF datasets of En->formal-Te pairs."""
    from datasets import load_dataset

    logger.info("loading %s (te) ...", cfg.dataset)
    ds = load_dataset("ai4bharat/samanantar", "te", split="train")
    logger.info("raw rows: %d -> filtering for formal register ...", len(ds))
    ds = ds.filter(_keep_formal, num_proc=cfg.dataloader_num_workers)
    logger.info("formal-subset rows: %d", len(ds))

    train = ds.select(range(min(cfg.max_train_samples, len(ds))))
    val = ds.select(range(len(train), min(len(train) + cfg.max_eval_samples, len(ds))))
    logger.info("train=%d eval=%d (formal register subset)", len(train), len(val))
    return train, val


def tokenize_mt(examples: dict, tokenizer, cfg: MTConfig) -> dict:
    tokenizer.src_lang = cfg.src_lang
    tokenizer.tgt_lang = cfg.tgt_lang
    src = tokenizer(
        examples["src"], padding=False, truncation=True, max_length=cfg.max_src_len
    )
    tgt = tokenizer(
        text_target=examples["tgt"], padding=False, truncation=True,
        max_length=cfg.max_tgt_len,
    )
    labels = [
        [tok if tok != tokenizer.pad_token_id else -100 for tok in lab]
        for lab in tgt["input_ids"]
    ]
    return {
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "labels": labels,
    }


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def make_mt_compute_metrics(tokenizer, cfg: MTConfig):
    def compute_metrics(pred):
        import sacrebleu

        pred_ids = pred.predictions
        if isinstance(pred_ids, (tuple, list)):
            pred_ids = pred_ids[0]
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = [[t for t in lab if t != -100] for lab in pred.label_ids]
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        bleu = sacrebleu.corpus_bleu(pred_str, [label_str]).score
        chrf = sacrebleu.corpus_chrf(pred_str, [label_str]).score
        return {"bleu": bleu, "chrf": chrf}

    return compute_metrics


# --------------------------------------------------------------------------- #
# Model / trainer                                                              #
# --------------------------------------------------------------------------- #
def build_mt_model(cfg: MTConfig):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.base_model, dtype=torch.float16 if cfg.fp16 else torch.float32
    )
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied | trainable: %s | total: %s | ratio: %.4f%%",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )
    return model


def train(cfg: MTConfig) -> str:
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.output_dir, "config.yaml"))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = build_mt_model(cfg)

    train_ds, eval_ds = build_mt_dataset(cfg)
    tok_train = train_ds.map(
        lambda ex: tokenize_mt(ex, tokenizer, cfg),
        batched=True, remove_columns=train_ds.column_names,
        num_proc=cfg.dataloader_num_workers,
    )
    tok_eval = eval_ds.map(
        lambda ex: tokenize_mt(ex, tokenizer, cfg),
        batched=True, remove_columns=eval_ds.column_names,
        num_proc=cfg.dataloader_num_workers,
    )
    # small overlap so eval loss stays meaningful; use a fresh slice
    del train_ds, eval_ds

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        predict_with_generate=cfg.predict_with_generate,
        generation_num_beams=cfg.generation_num_beams,
        generation_max_length=cfg.generation_max_length,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=cfg.report_to or "none",
        seed=cfg.seed,
    )

    from transformers import DataCollatorForSeq2Seq

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
        compute_metrics=make_mt_compute_metrics(tokenizer, cfg),
        processing_class=tokenizer,
    )

    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved LoRA adapter + tokenizer to %s", final_dir)
    return final_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(MTConfig.from_yaml(args.config))
