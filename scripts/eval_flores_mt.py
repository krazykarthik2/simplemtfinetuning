"""Evaluate LoRA MT on FLORES-200 devtest (formal news register), base vs LoRA."""
import argparse
import torch

MODEL = "facebook/nllb-200-distilled-600M"


def build_test_set(max_n=200):
    from datasets import load_dataset

    en = load_dataset("Muennighoff/flores200", "eng_Latn", split="devtest", trust_remote_code=True)
    te = load_dataset("Muennighoff/flores200", "tel_Telu", split="devtest", trust_remote_code=True)
    en_map = {r["id"]: r["sentence"] for r in en}
    pairs = [(en_map[r["id"]], r["sentence"]) for r in te if r["id"] in en_map]
    return pairs[:max_n]


def translate_batch(model, tokenizer, texts, device, bs=16):
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "tel_Telu"
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            inp = tokenizer(
                texts[i : i + bs], return_tensors="pt", truncation=True,
                max_length=64, padding=True,
            ).to(device)
            out = model.generate(
                **inp,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("tel_Telu"),
                num_beams=1,
                max_length=64,
            )
            outs.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-n", type=int, default=200)
    args = ap.parse_args()

    import sacrebleu
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = build_test_set(args.max_n)
    srcs, refs = zip(*pairs)

    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)

    base_out = translate_batch(base, tok, list(srcs), device)
    print("BASE  BLEU: %.2f  chrF: %.2f" % (
        sacrebleu.corpus_bleu(base_out, [list(refs)]).score,
        sacrebleu.corpus_chrf(base_out, [list(refs)]).score,
    ))

    lora = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    lora_out = translate_batch(lora, tok, list(srcs), device)
    print("LORA  BLEU: %.2f  chrF: %.2f" % (
        sacrebleu.corpus_bleu(lora_out, [list(refs)]).score,
        sacrebleu.corpus_chrf(lora_out, [list(refs)]).score,
    ))


if __name__ == "__main__":
    main()
