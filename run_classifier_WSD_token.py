# coding=utf-8

"""GlossDeBERTa finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict
import csv
import logging
import os
import random
import sys
import pandas as pd
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

# Modern Transformers imports
from torch.optim import AdamW 
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    get_linear_schedule_with_warmup,
    DebertaV2Model,
    DebertaV2PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, start_id, end_id, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.start_id = start_id
        self.end_id = end_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, label_data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, label_data_dir):
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir, label_data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class WSD_token_Processor(DataProcessor):
    """Processor for the WSD data set."""

    def get_train_examples(self, data_dir, label_data_dir):
        train_data = pd.read_csv(data_dir, sep="\t", na_filter=False).values
        with open(os.path.join(label_data_dir,"lemma2index_dict.pkl"), 'rb') as p:
            lemma2index_dict = pickle.load(p)
        return self._create_examples(train_data, "train", lemma2index_dict)

    def get_dev_examples(self, data_dir, label_data_dir):
        dev_data = pd.read_csv(data_dir, sep="\t", na_filter=False).values
        with open(os.path.join(label_data_dir,"lemma2index_dict.pkl"), 'rb') as p:
            lemma2index_dict = pickle.load(p)
        return self._create_examples(dev_data, "dev", lemma2index_dict)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type, lemma2index_dict):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = str(line[2])
            text_b = str(line[3])
            start_id = int(line[4])
            end_id = int(line[5])
            label = str(line[1])

            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("start_id=", start_id)
                print("end_id=", end_id)
                print("label=", label)

            examples.append(
                InputExample(guid=guid, text_a=text_a, start_id=start_id, end_id=end_id, 
                text_b=text_b, label=label))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, target_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.target_mask = target_mask

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputFeatures`."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens = example.text_a.split(' ')
        target_start = example.start_id
        target_end = example.end_id
        bert_tokens = []

        # Start with CLS token
        bert_tokens.append(tokenizer.cls_token)
        
        target_to_tok_map_start = 0
        target_to_tok_map_end = 0

        # We tokenize word by word to map the start_id/end_id correctly
        for length in range(len(orig_tokens)):
            if length == target_start:
                target_to_tok_map_start = len(bert_tokens)
            if length == target_end:
                target_to_tok_map_end = len(bert_tokens)
            
            sub_tokens = tokenizer.tokenize(orig_tokens[length])
            if not sub_tokens:
                sub_tokens = [tokenizer.unk_token]
            bert_tokens.extend(sub_tokens)

        if target_end == len(orig_tokens):
            target_to_tok_map_end = len(bert_tokens)

        # Use SEP token
        bert_tokens.append(tokenizer.sep_token)
        
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Account for [CLS], [SEP], [SEP]
            _truncate_seq_pair(bert_tokens, tokens_b, max_seq_length - 1)
            bert_tokens += tokens_b + [tokenizer.sep_token]
            segment_ids = [0] * len(bert_tokens) # DeBERTa V3 typically uses type 0
        else:
            if len(bert_tokens) > max_seq_length:
                bert_tokens = bert_tokens[:max_seq_length]
            segment_ids = [0] * len(bert_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # The mask has 1 for real target
        target_mask = [0] * max_seq_length
        # Ensure indices are within bounds
        real_start = target_to_tok_map_start
        real_end = min(target_to_tok_map_end, max_seq_length)
        
        for i in range(real_start, real_end):
            target_mask[i] = 1
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in bert_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("target_mask: %s" % " ".join([str(x) for x in target_mask]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              target_mask=target_mask))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        # Heuristic: Always pop from the longer sequence
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# --- Custom Model Class for DeBERTa WSD ---
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
        
        # Custom Logic from GlossBERT: average target token embeddings
        batch_size, seq_len, hidden_size = sequence_output.size()
        
        pooled_output_list = []
        for i in range(batch_size):
            # Extract the embeddings where target_mask is 1
            mask = target_mask[i] == 1
            if mask.sum() == 0:
                # Fallback if mask is empty (shouldn't happen if data is correct)
                # Just take CLS or first token
                target_emb = sequence_output[i, 0, :].unsqueeze(0) 
            else:
                target_emb = sequence_output[i][mask] # [num_target_tokens, hidden]
                target_emb = torch.mean(target_emb, dim=0, keepdim=True) # [1, hidden]
            
            pooled_output_list.append(target_emb)
            
        pooled_output = torch.cat(pooled_output_list, dim=0) # [batch, hidden]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["WSD"],
                        help="The name of the task to train.")
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files.")
    parser.add_argument("--eval_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files.")
    parser.add_argument("--label_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The label data dir. (./wordnet)")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory.")
    parser.add_argument("--bert_model", 
                        default="microsoft/deberta-v3-base", 
                        type=str, 
                        help="Path to pre-trained model or shortcut name.")
    
    ## Other parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")        
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")  
    
    # NEW PARAMETER FOR TRUNCATION
    parser.add_argument("--num_layers",
                        default=None,
                        type=int,
                        help="Number of hidden layers to use (e.g., 7). If None, uses full model.")
    
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")
    if args.do_train:
        assert args.train_data_dir is not None, "train_data_dir can not be None"
    if args.do_eval:
        assert args.eval_data_dir is not None, "eval_data_dir can not be None"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processors = {
        "WSD": WSD_token_Processor
    }

    output_modes = {
        "WSD": "classification"
    }

    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Use AutoTokenizer for DeBERTa compatibility
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)

    # Training set
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_data_dir, args.label_data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    logger.info(f"Loading model: {args.bert_model}")
    model = DebertaForWSD.from_pretrained(
        args.bert_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # --- TRUNCATION LOGIC ---
    if args.num_layers is not None:
        logger.info(f"Truncating model to first {args.num_layers} layers.")
        # Slice the ModuleList to keep only the first N layers
        model.deberta.encoder.layer = nn.ModuleList(model.deberta.encoder.layer[:args.num_layers])
        model.config.num_hidden_layers = args.num_layers
    # ------------------------
    
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer and scheduler
    optimizer = None
    scheduler = None
    
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(num_train_optimization_steps * args.warmup_proportion), 
            num_training_steps=num_train_optimization_steps
        )

    # Load data
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_target_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.eval_data_dir, args.label_data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_target_mask)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)

    # Train
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        model.train()
        epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch += 1
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, target_mask = batch

                outputs = model(
                    input_ids=input_ids, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                    labels=label_ids, 
                    target_mask=target_mask
                )
                loss = outputs.loss

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model
            model_output_dir = os.path.join(args.output_dir, str(epoch))
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            
            model_to_save.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)

            if args.do_eval:
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                with open(os.path.join(args.output_dir, "results_"+str(epoch)+".txt"),"w") as f:
                    for input_ids, input_mask, segment_ids, label_ids, target_mask in tqdm(eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        target_mask = target_mask.to(device)

                        with torch.no_grad():
                            outputs = model(
                                input_ids=input_ids, 
                                token_type_ids=segment_ids, 
                                attention_mask=input_mask, 
                                labels=label_ids,
                                target_mask=target_mask
                            )
                        
                        logits = outputs.logits
                        logits_ = F.softmax(logits, dim=-1)
                        logits_ = logits_.detach().cpu().numpy()
                        label_ids_ = label_ids.to('cpu').numpy()
                        outputs_idx = np.argmax(logits_, axis=1)
                        
                        for output_i in range(len(outputs_idx)):
                            f.write(str(outputs_idx[output_i]))
                            for ou in logits_[output_i]:
                                f.write(" " + str(ou))
                            f.write("\n")
                        
                        tmp_eval_accuracy = np.sum(outputs_idx == label_ids_)
                        
                        if output_mode == "classification":
                            loss_fct = CrossEntropyLoss()
                            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                        elif output_mode == "regression":
                            loss_fct = MSELoss()
                            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
                        
                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                loss = tr_loss/nb_tr_steps if args.do_train else None

                result = OrderedDict()
                result['eval_loss'] = eval_loss
                result['eval_accuracy'] = eval_accuracy
                result['global_step'] = global_step
                result['loss'] = loss

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a+") as writer:
                    writer.write("epoch=%s\n"%str(epoch))
                    logger.info("***** Eval results *****")
                    for key in result.keys():
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                
                # Set back to train mode
                model.train()

    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.eval_data_dir, args.label_data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_target_mask)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        with open(os.path.join(args.output_dir, "results.txt"),"w") as f:
            for input_ids, input_mask, segment_ids, label_ids, target_mask in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                target_mask = target_mask.to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids, 
                        token_type_ids=segment_ids, 
                        attention_mask=input_mask, 
                        labels=label_ids,
                        target_mask=target_mask
                    )

                logits = outputs.logits
                logits_ = F.softmax(logits, dim=-1)
                logits_ = logits_.detach().cpu().numpy()
                label_ids_ = label_ids.to('cpu').numpy()
                outputs_idx = np.argmax(logits_, axis=1)
                for output_i in range(len(outputs_idx)):
                    f.write(str(outputs_idx[output_i]))
                    for ou in logits_[output_i]:
                        f.write(" " + str(ou))
                    f.write("\n")
                tmp_eval_accuracy = np.sum(outputs_idx == label_ids_)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
                
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result = OrderedDict()
        result['eval_loss'] = eval_loss
        result['eval_accuracy'] = eval_accuracy
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results *****")
            for key in result.keys():
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()