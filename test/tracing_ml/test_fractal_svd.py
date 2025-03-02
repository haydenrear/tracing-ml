# !pip3 install torch torchaudio torchvision torchtext torchdata
# !pip3 install datasets
import os
import typing

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
from datasets import load_dataset

def do_train():

    #############################
    # 1. Define KAN-based Modules
    #############################

    # GELU approximation for initialization
    def gelu_approx(x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))

    # # KAN Activation: intended to start out like GELU
    # class KANActivation(nn.Module):
    #     def __init__(self, hidden_dim):
    #         super().__init__()
    #         # Initialize parameters so that initially:
    #         #   x * sin(1 * x) + cos(0 * x)  =  x * sin(x) + 1
    #         self.alpha = nn.Parameter(torch.ones(hidden_dim))   # ~1
    #         self.beta = nn.Parameter(torch.zeros(hidden_dim))     # ~0
    #
    #     def forward(self, x):
    #         result = x * torch.sin(self.alpha * x) + torch.cos(self.beta * x)
    #         for i in range(4):
    #             result += torch.sin(i * self.alpha * x) + torch.cos(i * self.beta * x)
    #
    #         return result
    #
    #     @torch.no_grad()
    #     def initialize_like_gelu(self, sample_input):
    #         # Compute GELU and KAN outputs on sample input, then adjust parameters slightly.
    #         gelu_output = gelu_approx(sample_input)
    #         kan_output = self.forward(sample_input)
    #         diff = gelu_output - kan_output
    #         adjustment = 0.01 * diff.mean()
    #         self.alpha.add_(adjustment)
    #         self.beta.add_(adjustment)

    # KAN Function used in attention transformation
    class KANFunction(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.alpha = nn.Parameter(torch.randn(dim))
            self.beta = nn.Parameter(torch.randn(dim))

        def forward(self, x):
            result = x * torch.sin(self.alpha * x) + torch.cos(self.beta * x)
            for i in range(2):
                result += torch.sin(i * self.alpha * x) + torch.cos(i * self.beta * x)
            return result

    from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # KAN-enhanced Attention module to replace standard MultiheadAttention
    class KANAttention(LlamaAttention):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)  # Important: Call the superclass constructor
            self.kan = KANFunction(config.hidden_size)  # Initialize KAN function with correct hidden dim

        def forward(self,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_value = None,
                    cache_position = None,
                    **kwargs):

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # Apply KAN before projections
            kan_hidden_states = self.kan(hidden_states)

            query_states = self.q_proj(kan_hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(kan_hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(kan_hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface: typing.Callable = eager_attention_forward  # Or other attention function

            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    print(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    import torch

    def local_unitary_loss(W, iterations=3, lambda_local=0.1):
        """
        Applies recursive SVD within a layer to enforce near-unitarity.
        Penalizes deviations of singular values from 1.
        """
        loss = 0.0
        W_current = W.clone().detach()
        U, S, V = torch.svd(W_current)

        for _ in range(iterations):
            # Penalize deviation of each singular value from 1
            loss += lambda_local * torch.sum((S - 1) ** 2)
            # Optionally, update W_current to refine the transformation
            U, S, V = torch.svd(W_current)
            W_current = U @ torch.diag(S) @ V.t()

        return loss / iterations

    def global_volume_loss(weight_matrices, base_exponent=0.5, lambda_global=1e-5):
        """
        Computes a global loss over selected layers based on the log-volume
        (i.e., sum of logarithms of singular values) of each layer.
        """
        total_loss = 0.0
        for l, W in enumerate(weight_matrices):
            U, S, V = torch.svd(W)
            log_volume = torch.sum(torch.log(S + 1e-8))
            # Define a target log-volume evolution; e.g., linear with layer depth:
            target_log_volume = base_exponent * (l + 1)
            total_loss += lambda_global * (log_volume - target_log_volume) ** 2

        return total_loss / len(weight_matrices)

    def combined_loss(weight_matrices, iterations=3, lambda_local=1, base_exponent=1, lambda_global=1):
        """
        Combines the primary task loss with both local and global losses.
        """
        local_loss_total = 0.0
        for W in weight_matrices:
            local_loss_total += local_unitary_loss(W, iterations, lambda_local)

        local_loss_total / len(weight_matrices)

        global_loss_total = global_volume_loss(weight_matrices, base_exponent, lambda_global)

        return (local_loss_total + global_loss_total) / 2

    # Example usage in a training loop:
    # weight_matrices might be a list of weight matrices from sampled layers.
    # task_loss is the standard loss from your model (e.g., cross-entropy).
    # total_loss = combined_loss(task_loss, weight_matrices)


    def recursive_svd_regularization_successive(weights, depth=3, alpha=0.1):
        loss = 0
        W = weights.clone().detach()
        print("Performing SVD Regularization")

        for i in range(depth):
            print(f"Performing SVD Regularization {i}")

            U, S, V = torch.svd(W)

            S_norm = S / torch.norm(S)

            if i > 0:
                loss += alpha * torch.sum((S_norm[1:] - S_norm[:-1]) / (S_norm[:-1] + 1e-8))  # Prevent division by zero

            W = U @ torch.diag(S) @ V.t()

        print(f"Performed Recursive SVD Regularization - Loss: {loss}")
        return loss

    def exponential_logarithmic_loss(singular_values, target_exponent=1.0, weight=0.1):
        # Compute the 'volume' as the product of singular values (or sum of logs)
        log_volume = torch.sum(torch.log(singular_values + 1e-8))
        # Expected log-volume growth (could be set per layer)
        expected_log_volume = target_exponent * singular_values.shape[0]
        loss = weight * (log_volume - expected_log_volume) ** 2
        return loss

    def local_unitary_loss(W, iterations=3, lambda_unitary=0.1):
        """
        Applies recursive SVD to enforce local unitary behavior,
        and returns a loss that penalizes deviation of singular values from 1.
        """
        loss = 0.0
        W_current = W.clone().detach()

        for _ in range(iterations):
            U, S, V = torch.svd(W_current)
            # Loss: deviation from 1 for each singular value
            loss += lambda_unitary * torch.sum((S - 1) ** 2)
            # Optionally, recompute W: force near-unitarity
            W_current = U @ torch.diag(S) @ V.t()

        return loss

    def layer_volume_loss(singular_values, layer_index, base_exponent=0.5, weight=0.1):
        # Compute the log-volume (sum of logs of singular values)
        log_volume = torch.sum(torch.log(singular_values + 1e-8))
        # Define the expected log-volume as a function of layer depth (layer_index)
        # For example, a linear function: T(i) = base_exponent * (i + 1)
        target_log_volume = base_exponent * (layer_index + 1)
        # Loss: penalize deviation from target
        loss = weight * (log_volume - target_log_volume) ** 2
        return loss

    def recursive_svd_regularization(weights, depth=3, alpha=0.1):
        loss = 0
        W = weights.clone().detach()
        print("Performing SVD Regularization")

        # Compute initial SVD
        U, PREV_S, V = torch.svd(W)
        PREV_S_norm = PREV_S / torch.norm(PREV_S)
        #  todo: exponential_logarithmic_loss(...) across layers.

        for i in range(depth):
            print(f"Performing SVD Regularization {i}")

            # Perform SVD on updated W
            U, S, V = torch.svd(W)

            S_norm = S / torch.norm(S)


            if i > 0:  # Skip the first iteration since there's no previous S to compare
                loss += alpha * torch.sum((S_norm[1:] - PREV_S_norm[:-1]) / (S_norm[:-1] + 1e-8))  # Prevent division by zero

            # Update previous singular values for next iteration
            PREV_S_norm = S_norm

            # Recompute W using the current SVD result
            print("Performing diagonalization")
            W = U @ torch.diag(S) @ V.t()

        return loss

    #############################
    # 2. Load Dataset and Preprocess
    #############################

    # Load the facebook/natural_reasoning dataset from Hugging Face
    dataset = load_dataset("facebook/natural_reasoning", token=TOKEN)


    # Assume the dataset has a 'text' field. Tokenize for causal LM.
    model_name = 'meta-llama/Llama-3.2-1B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    def causal_mask(seq_len):
        """
        Creates a lower-triangular causal mask (1s where attention is allowed, 0s elsewhere).
        """
        return torch.tril(torch.ones(seq_len, seq_len))


    # Step 3: Create the collate_fn function to lazily tokenize during training
    def collate_fn(batch):
        # For each example in the batch, extract the response text and tokenize

        texts = [example["responses"][0]['response'] for example in batch]

        tokenized_batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        print("Returning next tokenized batch.")
        labels = tokenized_batch["input_ids"].clone()  # Clone input_ids
        labels[:, :-1] = tokenized_batch["input_ids"][:, 1:]  # Shift labels one position to the right
        labels[:, -1] = tokenizer.pad_token_id  # Pad the last token (important!)
        tokenized_batch["labels"] = labels # Add labels to the tokenized batch
        tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"].to(dtype=torch.bfloat16)
        return tokenized_batch

    # Step 4: Create DataLoader with the custom collate_fn
    train_dataloader = DataLoader(dataset["train"], batch_size=1, collate_fn=collate_fn)
    next(iter(train_dataloader))

    #############################
    # 3. Load and Modify LLaMA Model
    #############################

    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", token=TOKEN)

    # Replace GELU with KANActivation and MultiheadAttention with KANAttention.
    # Note: This loop depends on model implementation details.
    # For this example we assume that the LLaMA model's transformer block
    # is accessible via model.model.layers.
    for i, layer in enumerate(model.model.layers):
        # Replace the activation in the MLP block:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
            # The gate_proj is usually followed by an activation (often GELU)
            # Replace it with our KANActivation:
            print("Replacing KAN Activation for MLP layer")
            # hidden_dim = layer.mlp.gate_proj.out_features
            # layer.mlp.activation = KANActivation(hidden_dim)
        # Replace self-attention with KANAttention in the attention block:
        if hasattr(layer, "self_attn"):
            print("Replacing KAN Activation for self_attn layer")
            # hidden_dim = layer.self_attn.q_proj.out_features
            # num_heads = layer.self_attn.config.num_key_value_heads
            # layer.self_attn = KANAttention(layer.self_attn.config, i)

    # Initialize KANActivation layers to mimic GELU.
    def initialize_kan_layers(model):
        sample_input = torch.randn(8192)  # A sample 1D tensor
        # for layer in model.model.layers:
        #     if hasattr(layer, "mlp") and hasattr(layer.mlp, "activation"):
        #         act = layer.mlp.activation
        #         print("Found MLP Layer")
                # if isinstance(act, KANActivation):
                #     print("Found KAN Activation")
                #     act.initialize_like_gelu(sample_input)

    initialize_kan_layers(model)

    #############################
    # 4. Define Loss Function with Recursive SVD Regularization
    #############################


    with open('out_file.txt', 'w') as o:
        pass
    with open('loss_file.txt', 'w') as oo:
        pass

    def do_compute_loss(model, inputs, labels, svd_depth=3, apply_to_all_layers=True):

        with open('out_file.txt', 'a') as f:
            f.write("Performing forward pass")
            f.write("\n")
        outputs = model(**inputs)
        token_ids = torch.argmax(outputs.logits, dim=-1)  # Get highest-probability token for each position
        decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        with open('out_file.txt', 'a') as f:
            f.write(f"Finished forward pass with outputs {str(outputs)}\n")
            f.write(f'{str(decoded)}\n')
            f.write("\n")
        ce_loss = outputs.loss  # Standard cross-entropy loss

        layers = [layer.self_attn.q_proj.weight for layer in model.model.layers]
        svd_loss = combined_loss(layers, svd_depth)

        # svd_loss = 0
        # if apply_to_all_layers:
            # for layer in model.model.layers:
            #     svd_loss += recursive_svd_regularization_successive(layer.self_attn.q_proj.weight, depth=svd_depth)
        # else:
        #     svd_loss += recursive_svd_regularization_successive(model.model.layers[-1].self_attn.q_proj.weight, depth=svd_depth)

        svd_loss = svd_loss / (ce_loss + svd_loss)
        total_loss = ce_loss + svd_loss
        with open('loss_file.txt', 'a') as loss_file:
            total_loss_str = "Total loss: " + str(total_loss)
            loss_file.writelines(total_loss_str)
            loss_file.write("\n")

        return total_loss

    #############################
    # 5. Fine-Tuning Setup with Trainer
    #############################


    out_dir = "work/kan"

    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch): # Add return_outputs
            print("In compute loss.")
            labels = inputs.get("labels")
            # forward pass
            loss = do_compute_loss(model, inputs, labels, svd_depth=1, apply_to_all_layers=False)
            return loss
    save_steps = 1
    save_strategy = "no"
    logging_strategy ="no"

    training_args = TrainingArguments(
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        save_steps=save_steps,
        save_total_limit=1,
        # output_dir=out_dir,
        num_train_epochs=3,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        # logging_steps=50,
        evaluation_strategy="no",
        remove_unused_columns=False)

    trainer = MyTrainer(
        model=model,
        train_dataset=dataset["train"],
        data_collator=collate_fn,
        args=training_args)


    # while True:
    #     try:
    trainer.train()
        # except Exception as e:
        #     raise e
            # print(f"Starting training again f{str(e)}")
            #
            # model = LlamaForCausalLM.from_pretrained(os.path.join(out_dir, 'checkpoint-1'), device_map="auto", token=TOKEN)
            #
            # training_args = TrainingArguments(
            #     save_strategy=save_strategy,
            #     save_steps=save_steps,
            #     logging_strategy=logging_strategy,
            #     save_total_limit=1,
            #     # output_dir=out_dir,
            #     num_train_epochs=3,
            #     overwrite_output_dir=True,
            #     per_device_train_batch_size=1,
            #     # logging_steps=50,
            #     evaluation_strategy="no",
            #     remove_unused_columns=False)
            #
            # trainer = MyTrainer(
            #     model=model,
            #     train_dataset=dataset["train"],
            #     data_collator=collate_fn,
            #     args=training_args)

do_train()



