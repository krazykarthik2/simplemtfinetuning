# TODO

## English->Telugu MT LoRA (formal register)

User request: LoRA fine-tune the en->te translation model so the Telugu output
is in FORMAL tone (shisthita bhasha). Just the LoRA fine-tuning -- no full
pipeline rebuild.

- [ ] Verify `ai4bharat/indictrans2-en-indic-1B` loads + translates under transformers 5.x
- [ ] Pick real formal-register En->Te dataset (Samanantar formal domains / FLORES)
- [ ] Build MT LoRA trainer (en->formal-Te) in `lspear/mt.py`
- [ ] Train LoRA on en->formal-Te data (GPU 1, setsid)
- [ ] Verify formal-tone output + adapter saved
