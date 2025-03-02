import os
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


from torchcfm import ConditionalFlowMatcher  # Import CFM module

def do_train():
    TOKEN = "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"

    #############################
    # 1. Load Dataset and Tokenizer
    #############################

    model_name = 'meta-llama/Llama-3.2-1B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", token=TOKEN)

    dataset = load_dataset("facebook/natural_reasoning", token=TOKEN)

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
    # 2. Define Gradient History Buffer
    #############################

    class GradientHistory:
        def __init__(self, max_steps=10):
            self.history = []
            self.max_steps = max_steps

        def __len__(self):
            return len(self.history)

        def update(self, gradients):
            if len(self.history) >= self.max_steps:
                self.history.pop(0)  # Remove oldest entry
            self.history.append(gradients)

        def get_history(self):
            if len(self.history) == 0:
                return torch.zeros_like(self.history[0]).to(device)  # Return zeros if no history
            return torch.stack(self.history, dim=0).to(device)  # Shape: (num_steps, num_params)

    gradient_history = GradientHistory(max_steps=5)

    #############################
    # 3. Define CFM Model
    #############################

    with open('out_file.txt', 'w') as o:
        pass
    with open('loss_file.txt', 'w') as oo:
        pass

    class GradientCFM(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GradientCFM, self).__init__()
            self.cfm = ConditionalFlowMatcher()

        def forward(self, gradient_history):
            return self.cfm(gradient_history)

    # Initialize CFM model
    gradient_cfm = GradientCFM(input_dim=4096, hidden_dim=512, output_dim=4096).to(device)

    #############################
    # 4. Define GradientDataset for Trainer
    #############################

    class GradientDataset(Dataset):
        def __init__(self, data_loader, tokenizer, gradient_history):
            self.model = model
            self.size = len(dataset)
            self.dataset = iter(data_loader)
            self.tokenizer = tokenizer
            self.gradient_history = gradient_history

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            tokenized = next(self.dataset)

            # Forward pass (LLaMA is frozen)
            # self.model.eval()

            collapsed_gradient = self.add_to_history(tokenized)

            while len(self.gradient_history) < 10:
                collapsed_gradient = self.add_to_history(tokenized)

            # Get historical gradients
            historical_gradients = self.gradient_history.get_history()

            return {
                "historical_gradients": historical_gradients,
                "true_gradients": collapsed_gradient
            }

        def add_to_history(self, tokenized):
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            outputs = self.model(**tokenized)
            ce_loss = outputs.loss
            # Compute gradients
            parameters = self.model.parameters()
            grads = torch.autograd.grad(ce_loss, parameters, retain_graph=True)
            grad_tensor = torch.stack([i for i in grads[-3:]], dim=-1)
            # Collapse gradients (sum or mean)
            collapsed_gradient = torch.mean(grad_tensor, dim=-1).to(device)
            # Store in history
            self.gradient_history.update(collapsed_gradient)
            return collapsed_gradient

    #############################
    # 5. Load LLaMA Model and Freeze It
    #############################


    # for param in model.parameters():
    #     param.requires_grad = False  # Freeze LLaMA model

    train_dataset = GradientDataset(train_dataloader, tokenizer, gradient_history)

    #############################
    # 6. Train CFM Using Hugging Face Trainer
    #############################

    class CFMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            with open('out_file.txt', 'a') as f:
                f.write("Computing loss.")
                f.write("\n")

            historical_gradients = inputs["historical_gradients"]
            true_gradients = inputs["true_gradients"]

            # Predict gradient evolution with CFM
            predicted_gradients = model(historical_gradients)

            # Compute MSE loss
            loss = torch.nn.functional.mse_loss(true_gradients, predicted_gradients)
            with open('out_file.txt', 'a') as f:
                f.write(f"Loss found {str(loss)}.")
                f.write("\n")

            return loss

    training_args = TrainingArguments(
        output_dir="./cfm_checkpoints",
        save_strategy="no",
        logging_strategy="no",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        evaluation_strategy="no",
        remove_unused_columns=False
    )

    trainer = CFMTrainer(
        model=gradient_cfm,
        train_dataset=train_dataset,
        args=training_args
    )

    print("Starting CFM Pretraining...")
    trainer.train()
    print("CFM Pretraining Complete.")

do_train()
