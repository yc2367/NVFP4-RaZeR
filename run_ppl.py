import torch
import torch.nn as nn

from datasets import load_dataset

import argparse
from tqdm import tqdm
from loguru import logger
import os
import json
import random
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

from utils import (
    load_model_and_tokenizer, 
    add_common_args, 
    add_quant_args, 
    get_quant_config,
    set_seed,
    model2path
)
from quantize import quant_weight

    
@torch.no_grad()
def eval_ppl(model, tokenizer, args):
    results = {}
    for task_eval in args.datasets:
        if task_eval == "wikitext":
            # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seq_len = args.seq_len
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seq_len
            nlls = []
            loss_fct = nn.CrossEntropyLoss()
            for i in tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seq_len) : ((i + 1) * model.seq_len)
                ][:, 1:]
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seq_len
                nlls.append(neg_log_likelihood.item())

            ppl = torch.exp(torch.tensor(nlls).sum() / (nsamples * model.seq_len))
            results["wikitext"] = ppl.item()
            print(f'Wikitext-2 perplexity: {ppl.item()}')
            print('\n')

        elif task_eval == "c4":
            model_net = model_name_or_path.split('/')[-1]
            model_family = '_'.join(model_net.lower().split('-')[:-1])
            model.seq_len = args.seq_len

            valenc = []
            testenc = load_dataset("allenai/c4", data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split="validation")
            for _ in range(256): # run 256 samples
                while True:
                    i = random.randint(0, len(testenc) - 1)
                    tmp = tokenizer(testenc[i]['text'], return_tensors='pt')
                    if tmp.input_ids.shape[1] > (model.seq_len+1):
                        break
                i = random.randint(0, tmp.input_ids.shape[1] - model.seq_len - 1)
                j = i + model.seq_len
                valenc.append(tmp.input_ids[:, i:j])
            testenc = torch.hstack(valenc)
            
            nsamples = testenc.numel() // model.seq_len
            loss_fct = nn.CrossEntropyLoss()
            nlls = []
            with tqdm(range(nsamples)) as progress:
                for i in progress:
                    batch = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)].to(model.device)
                    with torch.no_grad():
                        lm_logits = model(batch, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)][:, 1:].to(model.device)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * model.seq_len
                    nlls.append(neg_log_likelihood.item())
                    progress.set_description(f"Evaluating")

            ppl = torch.exp(torch.tensor(nlls).sum() / (nsamples * model.seq_len))
            results['c4'] = ppl.item()
            print(f'C4 perplexity: {ppl.item()}')
            print('\n')

    return results
    

if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    # Set random seed
    set_seed(0)

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_quant_args(parser)
    parser.add_argument('--datasets', type=lambda s: [item for item in s.split(',')], default=['wikitext'], help="Task to be evaled")
    parser.add_argument('--seq_len', type=int, help='sequence length for ppl evaluation', default=2048)
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    parser.add_argument("--output_dir", type=str, default="results/ppl", help="output directory")
    args = parser.parse_args()  
    
    quant_config = get_quant_config(args)
    model_name = args.model_name
    model_name_or_path = model2path[model_name]

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    logger.info(f"#################### Model Info ####################")
    logger.info(f"* Model: {model_name_or_path}")
    logger.info(f"* Datasets: {args.datasets}")
    logger.info(f"* Sequence length {args.seq_len}")

    logger.info("#################### Creating output directory ... ####################")
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.use_fp16:
        output_file_name = "Baseline_FP16.txt"
    elif not args.kv_quant:
        output_file_name = f"w{args.w_bits}_g{args.w_groupsize}_{args.w_dtype}__a{args.a_bits}_g{args.a_groupsize}_{args.a_dtype}.txt"
    else:
        output_file_name = f"w{args.w_bits}_g{args.w_groupsize}_{args.w_dtype}__akv{args.a_bits}_g{args.a_groupsize}_{args.a_dtype}.txt"

    output_file_path = os.path.join(output_dir, f"{output_file_name}")
    # check if result file exists
    if os.path.isfile(output_file_path):
        print(f'Found existing output file  {output_file_name}  for this experiment. Exit!\n\n')
        exit()
    print(f'Results will be saved to the output file:  {output_file_name}\n')

    logger.info(f"#################### Quantization Info ####################")
    print(f"==================================================")
    print(f"Weight Quantization Data Type:      {quant_config.w_dtype}")
    print(f"Weight Quantization Bits:           {quant_config.w_bits}")
    print(f"Weight Quantization Group Size:     {quant_config.w_groupsize}")
    print()
    print(f"Activation Quantization Data Type:  {quant_config.a_dtype}")
    print(f"Activation Quantization Bits:       {quant_config.a_bits}")
    print(f"Activation Quantization Group Size: {quant_config.a_groupsize}")
    print()
    print(f"KV-cache Quantization:              {quant_config.kv_quant}")
    print(f"==================================================")

    logger.info("#################### Loading model and tokenizer ... ####################")
    model, tokenizer = load_model_and_tokenizer(model_name, quant_config=quant_config, use_fp16=args.use_fp16)
    quant_weight(model, quant_config)

    logger.info("#################### Start running perplexity evaluation ... ####################")
    res = eval_ppl(model, tokenizer, args)

    # Save results to JSON file
    with open(output_file_path, "w") as f:
        for dataset, ppl in res.items():
            logger.info(f"{dataset} PPL: {ppl}")
            f.write(f"{dataset.ljust(10)} PPL: {ppl}\n")
    
    print(f"Results saved to {output_file_path} \n\n")
