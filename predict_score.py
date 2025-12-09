import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2PreTrainedModel, DebertaV2Model
import json
import os

class DebertaForWSD(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                labels=None, target_mask=None, **kwargs):
        outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        batch_size, seq_len, hidden_size = sequence_output.size()
        pooled_output_list = []
        for i in range(batch_size):
            mask = target_mask[i] == 1
            if mask.sum() == 0:
                target_emb = sequence_output[i, 0, :].unsqueeze(0)
            else:
                target_emb = sequence_output[i][mask]
                target_emb = torch.mean(target_emb, dim=0, keepdim=True)
            pooled_output_list.append(target_emb)
            
        pooled_output = torch.cat(pooled_output_list, dim=0)
        logits = self.classifier(pooled_output)
        return logits

def predict_item(model, tokenizer, item, device):
    # Construct Context
    parts = [
        item.get('precontext', ''),
        item.get('sentence', ''),
        item.get('ending', '')
    ]
    context = " ".join([p for p in parts if p]).strip()
    
    target_word = item.get('homonym')
    gloss = item.get('judged_meaning') 

    if not target_word or not gloss:
        # Skip invalid items
        return 1

    # Tokenize
    inputs = tokenizer(
        context, 
        gloss, 
        return_tensors="pt", 
        max_length=256, 
        truncation=True, 
        padding="max_length"
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

  
    target_mask = torch.zeros_like(input_ids)
    sep_pos = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()
    
    target_subtokens = tokenizer(" " + target_word, add_special_tokens=False)['input_ids']
    if not target_subtokens:
        target_subtokens = tokenizer(target_word, add_special_tokens=False)['input_ids']

    found = False
    for i in range(1, sep_pos - len(target_subtokens) + 1):
        if input_ids[0, i:i+len(target_subtokens)].tolist() == target_subtokens:
            target_mask[0, i:i+len(target_subtokens)] = 1
            found = True
            break
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            target_mask=target_mask
        )
        probs = F.softmax(logits, dim=1)
        raw_prob = probs[0][1].item()

    # score = round(raw_prob * 5)
    if raw_prob < 0.019188896: score = 2
    elif raw_prob < 0.304312307: score = 3
    elif raw_prob < 0.996154601: score = 4
    else: score = 5

    # return int(score)
    return score

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return [data[key] for key in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)]
        return data
    except json.JSONDecodeError:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid line: {line[:50]}...")
        return data

if __name__ == "__main__":
    MODEL_PATH = "/home/chirooth/experiments/GlossDeBERTa/results/gloss_deberta_full_7layers/results/merged_model"
    INPUT_FILE = "./data/dev.json"
    OUTPUT_FILE = "predictions.jsonl"

    print(f"Loading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = DebertaForWSD.from_pretrained(MODEL_PATH)
    model.to(device)

    print(f"Reading data from {INPUT_FILE}...")
    data = load_data(INPUT_FILE)
    
    # Validation check
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], str):
            print("Error: Data loaded as a list of strings. Parsing strings to JSON...")
            try:
                data = [json.loads(s) for s in data]
            except:
                print("Failed to parse strings. Check input file format.")
    
    print(f"Generating predictions for {len(data)} items...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for idx, item in enumerate(data):
            try:
                final_score = predict_item(model, tokenizer, item, device)
                output_obj = {"id": str(idx), "prediction": final_score}
                out_f.write(json.dumps(output_obj) + "\n")
                if idx % 50 == 0:
                    print(f"Processed {idx}/{len(data)}")
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                out_f.write(json.dumps({"id": str(idx), "prediction": 1}) + "\n")

    print(f"\nDone! Results saved to {OUTPUT_FILE}")
