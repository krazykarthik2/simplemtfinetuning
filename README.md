# L-SPEAR: Low resource Speech Adaptation Regulation

LoRA fine-tuning pipeline for the L-SPEAR framework (see `L-SPEAR Framework.pdf`).
The framework targets offline Telugu ASR + TTS + translation for MSME
manufacturing environments (Dairy, Paper Mills, Textile, Foundries, Food
Processing, Furniture, Printing/Packaging, Agro-Mills, Plastic, Pharma).

SVD weight factorization is intentionally deferred - this repo only covers the
**LoRA fine-tuning** stage of the framework.

> **Data rule (strict):** only real content. No synthetic, generated or dummy
> data is used anywhere in this project. Training/eval corpora are real Telugu
> speech + human transcripts (FLEURS), and the `local` mode reads your own
> real shop-floor recordings.

## Pipeline implemented

```
Audio Capture -> Preprocessing -> Log-Mel Features -> ASR (Whisper + LoRA)
             -> Text Normalization -> [IndicTrans2] -> [Telugu VITS TTS]
```

* **Noise injection layer** (`lspear/augment.py`): shop-floor acoustic profile -
  machine hum (50/100/150 Hz), colored floor noise, concrete echoes, gain and
  speed perturbation - applied to the waveform during fine-tuning.
* **LoRA adaptation** (`lspear/model.py`): base Whisper frozen; only the
  attention projection matrices of encoder+decoder are adapted (r=16, alpha=32),
  so the model trains on ~0.4% of its parameters.
* **WER / CER evaluation** (`lspear/evaluate.py`) on the test split.

## Quick start

```bash
pip install -r requirements.txt

# 1. fine-tune (FLEURS Telugu)
python -m lspear.train --config configs/lspear_fleurs.yaml

# 2. evaluate WER/CER on the test split
python -m lspear.evaluate --config configs/lspear_fleurs.yaml \
    --adapter outputs/lspear_fleurs/checkpoint-final \
    --out outputs/eval_results.json

# 3. run inference on a shop-floor audio file
python -m lspear.pipeline --config configs/lspear_fleurs.yaml \
    --adapter outputs/lspear_fleurs/checkpoint-final --audio path/to/audio.wav
```

## Using your own MSME data (the real L-SPEAR scenario)

Record real shop-floor audio (3-4 h per the framework) with Telugu transcripts,
then create a CSV with columns `path,text` (relative paths resolved against
`data_dir`) and switch the config to:

```yaml
dataset: local
metadata_csv: data/msme_records/metadata.csv
data_dir: data/msme_records
```

## Model

| component | choice |
|---|---|
| ASR | `openai/whisper-small` (Telugu) + LoRA |
| LoRA | r=16, alpha=32, dropout=0.05, target `q/k/v/out_proj` (enc+dec) |
| Translation | `ai4bharat/indictrans2-indic-indic-1B` (lazy, optional) |
| TTS | `facebook/mms-tts-tel` (offline VITS, lazy, optional) |
