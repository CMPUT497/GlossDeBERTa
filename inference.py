import torch
import torch.nn as nn
from transformers import AutoTokenizer, DebertaV2PreTrainedModel, DebertaV2Model
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
import numpy as np

# --- 1. Define the Model Class (Must match training exactly) ---
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
        
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # Average the embeddings of the target word using target_mask
        batch_size, seq_len, hidden_size = sequence_output.size()
        pooled_output_list = []
        
        for i in range(batch_size):
            mask = target_mask[i] == 1
            if mask.sum() == 0:
                # Fallback: take CLS token if mask is empty
                target_emb = sequence_output[i, 0, :].unsqueeze(0) 
            else:
                target_emb = sequence_output[i][mask]
                target_emb = torch.mean(target_emb, dim=0, keepdim=True)
            pooled_output_list.append(target_emb)
            
        pooled_output = torch.cat(pooled_output_list, dim=0)
        logits = self.classifier(pooled_output)
        return logits

# --- 2. Inference Logic ---
def predict_sense(model, tokenizer, sentence, target_word, device):
    # 1. Get all candidate senses from WordNet
    senses = wn.synsets(target_word)
    if not senses:
        print(f"No senses found in WordNet for '{target_word}'")
        return

    print(f"\nContext: \"{sentence}\"")
    print(f"Target: \"{target_word}\" ({len(senses)} candidates)\n")

    candidates = []
    
    # 2. Prepare inputs for every candidate gloss
    for synset in senses:
        gloss = synset.definition()
        
        # Tokenize: [CLS] Sentence [SEP] Gloss [SEP]
        inputs = tokenizer(
            sentence, 
            gloss, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True, 
            padding="max_length"
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
        
        # Create Target Mask (Find where the target word is in the tokens)
        target_mask = torch.zeros_like(input_ids)
        
        # Simple heuristic to find target word index in the tokenized sentence
        # (Note: In a production app, you would want more robust alignment)
        sent_tokens = tokenizer.tokenize(sentence)
        target_subtokens = tokenizer.tokenize(" " + target_word) # DeBERTa adds space prefix often
        
        # Fallback if space prefix didn't match
        if not target_subtokens: 
             target_subtokens = tokenizer.tokenize(target_word)

        target_ids = tokenizer.convert_tokens_to_ids(target_subtokens)
        
        # Scan input_ids for the sequence of target_ids
        sep_pos = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()
        
        found = False
        for i in range(1, sep_pos):
            # Try to match the sequence
            if input_ids[0, i:i+len(target_ids)].tolist() == target_ids:
                target_mask[0, i:i+len(target_ids)] = 1
                found = True
                break
        
        if not found:
            # Fallback: Just look for the first subtoken (less accurate but prevents crash)
            first_id = target_ids[0]
            for i in range(1, sep_pos):
                if input_ids[0, i] == first_id:
                     target_mask[0, i] = 1
                     break

        # 3. Run Model
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                target_mask=target_mask
            )
            
            # Get probability of class "1" (True) or just raw logit
            # We assume label 1 = Correct Sense
            probs = F.softmax(logits, dim=1)
            score = probs[0][1].item() 
            
        candidates.append((score, synset.name(), gloss))

    # 4. Rank and Print Results
    candidates.sort(key=lambda x: x[0], reverse=True)

    print(f"{'Score':<10} | {'Synset':<20} | {'Definition'}")
    print("-" * 100)
    for score, name, definition in candidates:
        print(f"{score:.4f}     | {name:<20} | {definition}")
        
    print(f"\n>> Winner: {candidates[0][1]} ({candidates[0][2]})")

# --- 3. Main Execution ---
if __name__ == "__main__":
    # Pointing to your DEBUG model
    MODEL_PATH = "./results/gloss_deberta_output_debug/1" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit()
    
    # Load Model
    model = DebertaForWSD.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    # --- Examples to Try ---
    
    # Example 1: "Bank" (Financial)
    predict_sense(
        model, tokenizer, 
        sentence="I went to the bank to deposit my money.", 
        target_word="bank", 
        device=device
    )

    # Example 2: "Bank" (River)
    predict_sense(
        model, tokenizer, 
        sentence="He sat on the bank of the river and watched the water.", 
        target_word="bank", 
        device=device
    )