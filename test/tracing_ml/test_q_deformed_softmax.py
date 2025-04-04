import gc
import math
import os.path
import sys
import typing
from typing import Optional, Union

import entmax
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback, \
    Cache, TrainerState, TrainerControl
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

DEVICE = 'mps'

save_steps =30

TOKEN="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"

def reset_q_params(q_params, lower_clamp=1.0):
    for q in q_params: # TODO: don't let it go down after it goes up
        print(q)
        with open('q.txt', 'a') as q_write:
            q_write.write(str(float(q.data[0].item())))
            q_write.write('\n')
        # q_item = q.data[0].item()
        # if torch.any(torch.isnan(q)):
        #     q.data[0] = 1.0
        # if q_item > 1.9999 or q_item < lower_clamp:
        #     q.data.clamp_(lower_clamp, 1.9999)  # TODO: seems like early it might want low, then push up ? also delegating optimizer, copy adam and then set q lr separate

class EntMaxBisectDelegator(torch.nn.Module):
    def __init__(self, q: torch.nn.Parameter):
        super().__init__()
        self.q = q
        self.e = entmax.EntmaxBisect(alpha=q)

    def forward(self, in_value):
        # TODO: try adding gradient w.r.t. q - q-derivative! derivative of exponential w.r.t. q... stretchiness
        #       of the space could provide information about previous and next ? D f = eigen f -> for q-exponential
        #       compare to q = 1, is like derivative w.r.t. q, so update gradient calculation to calc gradient based on q value
        #       - also chain rule ?
        if torch.allclose(self.q, torch.tensor(1.0), atol=1e-15):
            self.q.data = torch.tensor([1.000001]).to(device=DEVICE)

        r = self.e(in_value)

        return r

def perform_attn(query, key, value, input_shape, attn_mask,
                 scale, softmax_fn, o_proj, training, dropout_p, is_causal=True, enable_gqa=True):
    # [batch, num_heads, seq_len, head_dim]
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=DEVICE)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(device=DEVICE)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax_fn(attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=training)
    return attn_weight @ value, attn_weight

class PatchedLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, from_val, q: torch.nn.Parameter):
        super().__init__()
        config = from_val.config
        self.config = from_val.config
        self.layer_idx = from_val.layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = from_val.scaling
        self.attention_dropout = config.attention_dropout
        self.is_causal = from_val.is_causal

        self.q_proj = from_val.q_proj
        self.k_proj = from_val.k_proj
        self.v_proj = from_val.v_proj
        self.o_proj = from_val.o_proj
        self.softmax = EntMaxBisectDelegator(q)
        self.f = from_val
        self.f._modules.clear()

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: typing.Tuple[torch.Tensor, torch.Tensor],
            attention_mask: typing.Optional[torch.Tensor],
            past_key_value: typing.Optional[Cache] = None,
            cache_position: typing.Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor], typing.Optional[typing.Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = perform_attn(
            query_states,
            key_states,
            value_states,
            input_shape,
            attention_mask,
            scale=self.scaling,
            softmax_fn=self.softmax,
            o_proj=self.o_proj,
            training=self.training,
            dropout_p=0.0 if not self.training else self.attention_dropout
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

count = 0
def do_train():

    #############################
    # 1. Define KAN-based Modules
    #############################

    # Load the facebook/natural_reasoning dataset from Hugging Face
    dataset = load_dataset("facebook/natural_reasoning", token=TOKEN).shuffle()


    # Assume the dataset has a 'text' field. Tokenize for causal LM.
    model_name = 'meta-llama/Llama-3.2-1B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Create the collate_fn function to lazily tokenize during training
    def test_collate_fn(batch):
        # For each example in the batch, extract the response text and tokenize
        question_answers = []
        for example in batch:
            question_answer = f"<question>{example['question']}</question>"
            question_answer += '\n'
            question_answer += f"<response>{example['responses'][0]['response']}</response>"
            question_answer += '\n'
            question_answer += f"<answer>{example['reference_answer']}</answer>"
            question_answers.append(question_answer)

        tokenized_batch = tokenizer(question_answers, padding=True, truncation=True, return_tensors="pt", max_length=8192)
        print("Returning next tokenized batch.")
        labels = tokenized_batch["input_ids"].clone()  # Clone input_ids
        labels[:, :-1] = tokenized_batch["input_ids"][:, 1:]  # Shift labels one position to the right
        labels[:, -1] = tokenizer.pad_token_id  # Pad the last token (important!)
        tokenized_batch["labels"] = labels # Add labels to the tokenized batch
        tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"].to(dtype=torch.bfloat16)
        return tokenized_batch

    # Step 3: Create the collate_fn function to lazily tokenize during training
    def collate_fn(batch):
        # For each example in the batch, extract the response text and tokenize

        question_answers = []
        for example in batch:
            question_answer = f"<question>{example['question']}</question>"
            question_answer += '\n'
            question_answer += f"<response>{example['responses'][0]['response']}</response>"
            question_answers.append(question_answer)

        tokenized_batch = tokenizer(question_answers, padding=True, truncation=True, return_tensors="pt", max_length=8192)
        print("Returning next tokenized batch.")
        labels = tokenized_batch["input_ids"].clone()  # Clone input_ids
        labels[:, :-1] = tokenized_batch["input_ids"][:, 1:]  # Shift labels one position to the right
        labels[:, -1] = tokenizer.pad_token_id  # Pad the last token (important!)
        tokenized_batch["labels"] = labels # Add labels to the tokenized batch
        tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"].to(dtype=torch.bfloat16)
        return tokenized_batch

    # Step 4: Create DataLoader with the custom collate_fn
    train_dataloader = DataLoader(dataset["train"], batch_size=1, collate_fn=collate_fn, shuffle=True, in_order=False)
    next(iter(train_dataloader))
    eval_dataloader = DataLoader(dataset["train"], batch_size=1, collate_fn=test_collate_fn, shuffle=True, in_order=False)
    next(iter(eval_dataloader))

    eval_size = 5

    class EvalDataLoaderProvider:
        def __init__(self):
            self.num_val = 0
            self.train_ = dataset['train']
            self.max_num = len(self.train_)

        def retrieve(self):
            next_value = Subset(self.train_, range((self.num_val * eval_size) + (save_steps + 1), ((self.num_val * eval_size) + eval_size) + (save_steps + 1)))
            self.num_val += 1
            return DataLoader(next_value, batch_size=1, collate_fn=test_collate_fn, shuffle=True, in_order=False)

    start = "/Users/hayde/IdeaProjects/drools/tracing_ml/"
    model = LlamaForCausalLM.from_pretrained(device_map="auto", token=TOKEN, use_cache=False,
                                             pretrained_model_name_or_path=f'{start}work/checkpoint-30')

    # load the model with the q values
    loaded = torch.load(f'{start}work/checkpoint-30/pytorch_model.bin')
    def do_save_q():
        for j, p in enumerate(q_params):
            torch.save(p, f'model.layers.{j}.self_attn.softmax.q.pt')

    def do_load_q(load_name):
        return torch.load(load_name)

    out_layers = []
    q_params = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            name = f'model.layers.{i}.self_attn.softmax.q'
            if name in loaded.keys():
                next_q_param = loaded[name]
            else:
                if os.path.exists(f'{name}.pt'):
                    next_q_param: torch.nn.Parameter = do_load_q(f'{name}.pt') # TODO: ??? resets q values every other - ok for now, like resetting some weights to 0 and meta-learning to rebuild
                    next_q_param = next_q_param.requires_grad_(True)
                else:
                    next_q_param = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
            q_params.append(next_q_param)
            layer.self_attn = PatchedLlamaAttention(layer.self_attn, next_q_param)

        out_layers.append(layer)

    do_save_q()
    model.model.layers = torch.nn.ModuleList([o for o in out_layers])
    out_layers.clear()

    loaded = None

    # Setup evaluation
    bleu = evaluate.load("bleu")
    rogues = evaluate.load("rouge")



    class OnEval(TrainerCallback):

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            with open('eval.txt', 'a') as f:
                f.write(f"Eval metrics: {state.global_step}, {kwargs['metrics']}\n")
            super().on_evaluate(args, state, control, **kwargs)
            do_save_q()
            sys.exit(0)

    class MyTrainer(Trainer):

        eval_data_loader_provider = EvalDataLoaderProvider()


        def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
            return self.eval_data_loader_provider.retrieve()

        def compute_loss(self, model, inputs, num_items_in_batch=1, return_outputs=False): # Add return_outputs
            if not hasattr(self, 'count'):
                self.count = 0

            self.count += 1

            if self.count >= 50:
                print("Quitting after count!")
                sys.exit(0)


            with open('out_file.txt', 'a') as f:
                f.write("Performing forward pass")
                f.write("\n")

            outputs = model(**inputs)

            self.write_metrics(inputs, outputs)

            ce_loss = outputs.loss  # Standard cross-entropy loss

            reset_q_params(q_params)

            with open('loss_file.txt', 'a') as loss_file:
                total_loss_str = "Total loss: " + str(ce_loss)
                loss_file.writelines(total_loss_str)
                loss_file.write("\n")

            gc.collect()
            if return_outputs:
                return ce_loss, outputs
            return ce_loss

        def write_metrics(self, inputs, outputs):
            token_ids = torch.argmax(outputs.logits, dim=-1)
            decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            target = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
            with open('bleu-metrics.txt', 'a') as f:
                b = bleu.compute(predictions=decoded, references=target)
                f.write(f"{b}\n")
            with open('rouge-metrics.txt', 'a') as f:
                r = rogues.compute(predictions=decoded, references=target)
                f.write(f"{r}\n")
            with open('out_file.txt', 'a') as f:
                f.write(f"Finished forward pass with outputs {str(outputs)}\n")
                f.write(f'{str(decoded)}\n')
                f.write("\n")

    save_strategy = "steps"
    logging_strategy ="no"

    model.gradient_checkpointing_enable()

    out_dir = f'{start}work'

    training_args = TrainingArguments(
        resume_from_checkpoint=f'{start}work/checkpoint-30',
        save_safetensors=False,
        learning_rate=5e-5,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        save_steps=save_steps,
        save_total_limit=1,
        output_dir=out_dir,
        num_train_epochs=3,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        prediction_loss_only=True,
        # logging_steps=50,
        eval_steps=31,
        evaluation_strategy="steps",
        remove_unused_columns=False)

    trainer = MyTrainer(
        model=model,
        eval_dataset=dataset['train'],
        train_dataset=dataset["train"],
        data_collator=collate_fn,
        args=training_args,
        callbacks=[OnEval()])

    trainer.train()


do_train()



