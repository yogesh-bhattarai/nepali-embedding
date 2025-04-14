from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,Trainer, TrainingArguments
from datasets import load_dataset

dataset= load_dataset("text", data_files={"train":"nepali_sample.txt"})
#print(dataset['train'][1])

model_name = "bert-base-multilingual-cased"
tokenizer= AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def tokenizer_function(exammples):
    return tokenizer(exammples["text"])

tokenized = dataset.map(tokenizer_function, batched=True, remove_columns=["text"])

data_collator= DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15,mlm=True)

training_args = TrainingArguments(
    output_dir="./nepali_bert",            
    per_device_train_batch_size=1,          
    num_train_epochs=10,                    
    logging_steps=1,                        
    save_steps=5,                     
    eval_steps=5,                           
    save_total_limit=1,                    
)

trainer= Trainer(
    model= model,
    args= training_args,
    train_dataset= tokenized['train'],
    tokenizer= tokenizer,
    data_collator= data_collator,
)

trainer.train()
trainer.save_model("./nepali_bert")
tokenizer.save_pretrained("./nepali-bert-small")
print(" Fine-tuning completed on small Nepali dataset!")