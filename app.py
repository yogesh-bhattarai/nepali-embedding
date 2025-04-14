from transformers import AutoTokenizer, AutoModelForMaskedLM,pipeline
import torch

model = AutoModelForMaskedLM.from_pretrained("./nepali_bert")
tokenizer= AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

sentence = "सगरमाथा संसारको [MASK] हिमाल हो।"
result= fill_mask(sentence)

for res in result[:3]:
    print(f"{res['sequence']} (Confidence: {res['score']:.4f})")
