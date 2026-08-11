"""Verify the LoRA adapter shifts en->te output toward the formal register.

Compares base NLLB vs LoRA-tuned NLLB on formal/admin English inputs.
"""
import argparse
import torch

MODEL = "facebook/nllb-200-distilled-600M"

TESTS = [
    "The Government of India has announced new regulations for small manufacturing units.",
    "The court has issued a ruling regarding the property dispute.",
    "The minister will visit the factory next week.",
    "All workers must comply with the safety guidelines.",
    "The company has reported a significant increase in production.",
    "The department has decided to establish a new industrial zone.",
    "The chief minister will inaugurate the new plant on Monday.",
    "The tribunal has ordered an inquiry into the accident.",
    "Please provide the documents for verification before the deadline.",
    "The scheme aims to improve working conditions in the textile sector.",
]


def translate(model, tokenizer, texts, device):
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "tel_Telu"
    results = []
    with torch.no_grad():
        for t in texts:
            inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=64).to(device)
            out = model.generate(
                **inp,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("tel_Telu"),
                num_beams=1,
                max_length=64,
            )
            results.append(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to LoRA adapter dir")
    args = ap.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("=" * 80)
    print("BASE NLLB (no LoRA)")
    print("=" * 80)
    base_out = translate(base, tok, TESTS, device)
    for en, te in zip(TESTS, base_out):
        print(f"EN: {en}\nTE: {te}\n")

    lora = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    print("=" * 80)
    print(f"LORA-TUNED (formal register)  [{args.adapter}]")
    print("=" * 80)
    lora_out = translate(lora, tok, TESTS, device)
    for en, te in zip(TESTS, lora_out):
        print(f"EN: {en}\nTE: {te}\n")


if __name__ == "__main__":
    main()
