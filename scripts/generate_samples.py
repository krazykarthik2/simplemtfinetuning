"""Generate English->Telugu sample translations with the final LoRA adapter.

Writes ./outputs/samples.txt as CSV (english,telugu).
"""
import csv
import os
import torch

BASE = "facebook/nllb-200-distilled-600M"
ADAPTER = "configs/outputs/mt_formal_te/checkpoint-final"
OUT = "outputs/samples.txt"

SAMPLES = [
    "The chief minister inaugurated the new hospital in the capital city.",
    "The government announced a new policy for farmers yesterday.",
    "The court delivered its verdict on the long-standing case.",
    "The minister stated that the budget will be presented next month.",
    "Officials confirmed that the investigation is still ongoing.",
    "The company reported a significant increase in its annual profits.",
    "The commission has approved the new industrial project.",
    "The state government has implemented new tax regulations.",
    "The minister explained the details of the new scheme to the assembly.",
    "The department issued new guidelines for the upcoming elections.",
    "The committee reviewed the progress of the development program.",
    "The chief minister said the welfare scheme will benefit all citizens.",
    "How are you doing today?",
    "Let's meet for coffee tomorrow.",
    "The weather is really nice today.",
    "I am waiting for the bus right now.",
    "Call me when you reach home.",
    "What did you have for breakfast?",
    "I will come to your house this evening.",
    "The food at that restaurant is very good.",
    "My phone battery is low.",
    "Can you help me with this?",
    "I am going to the market to buy some vegetables.",
    "The movie starts at seven in the evening.",
    "My son scored good marks in his exams.",
    "Please turn off the lights before you leave.",
    "Warning: coolant temperature high, please check the cooling system.",
    "Machine 3 stopped due to low oil pressure.",
    "Conveyor belt speed is above the normal limit.",
    "The compressor pressure is dropping, inspect the valves.",
    "Emergency stop activated on production line 2.",
    "Vibration levels on motor 4 exceed the safe threshold.",
    "The reactor temperature reached 120 degrees, reduce the load.",
    "Maintenance is scheduled for pump 7 at 2 PM.",
    "Power supply interrupted, switching to backup generator.",
    "Lubrication cycle completed for press machine 5.",
    "Hydraulic pressure in the press is below the required level.",
    "Quality check failed for batch number 12.",
    "The furnace temperature is rising too fast, slow the feed rate.",
]

def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base_orig = AutoModelForSeq2SeqLM.from_pretrained(BASE, dtype=dtype).to(device).eval()
    base_lora = AutoModelForSeq2SeqLM.from_pretrained(BASE, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(BASE)
    lora = PeftModel.from_pretrained(base_lora, ADAPTER).to(device).eval()

    tok.src_lang = "eng_Latn"
    tok.tgt_lang = "tel_Telu"

    def translate(model):
        with torch.no_grad():
            inp = tok(
                SAMPLES, return_tensors="pt", truncation=True,
                max_length=64, padding=True,
            ).to(device)
            out = model.generate(
                **inp,
                forced_bos_token_id=tok.convert_tokens_to_ids("tel_Telu"),
                num_beams=1,
                max_length=64,
            )
            return tok.batch_decode(out, skip_special_tokens=True)

    orig_te = translate(base_orig)
    lora_te = translate(lora)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["english", "telugu_original", "telugu_lora"])
        for en, te0, te1 in zip(SAMPLES, orig_te, lora_te):
            writer.writerow([en, te0, te1])
    print(f"wrote {len(SAMPLES)} rows to {OUT}")


if __name__ == "__main__":
    main()
