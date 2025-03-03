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
# 3. Define CFM Model
#############################

with open('out_file.txt', 'w') as o:
    pass
with open('loss_file.txt', 'w') as oo:
    pass

# Initialize CFM model
# for param in model.parameters():
#     param.requires_grad = False  # Freeze LLaMA model

# Load the facebook/natural_reasoning dataset from Hugging Face
dataset = load_dataset("facebook/natural_reasoning", token=TOKEN)


# Assume the dataset has a 'text' field. Tokenize for causal LM.
model_name = 'meta-llama/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token


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
# 6. Train CFM Using Hugging Face Trainer
#############################
import torch.nn.functional as F

import torch
import torch.nn as nn

class ThetaSeriesLayer(nn.Module):
    def __init__(self, num_terms):
        super(ThetaSeriesLayer, self).__init__()
        self.num_terms = num_terms
        self.coefficients = nn.Parameter(torch.randn(num_terms)) #learnable coefficients
        self.tau = nn.Parameter(torch.randn(1))

    def forward(self, x):
        #Simplified theta series example, needs to be adjusted.
        result = torch.zeros_like(x)
        for n in range(self.num_terms):
            result += self.coefficients[n] * torch.exp(-n*x * self.tau)
        return result

class MyNetwork(nn.Module):
    def __init__(self, num_terms):
        super(MyNetwork, self).__init__()
        self.theta_layer = ThetaSeriesLayer(num_terms)
        self.fc = nn.Linear(1, 1) # Example output layer

    def forward(self, x):
        theta_output = self.theta_layer(x)
        output = self.fc(theta_output)
        return output

def modular_transform(tau, z):
    """Applies a simple modular transformation (example)."""
    # Replace with your actual transformation
    transformed_tau = (tau + 1)
    transformed_z = z
    return transformed_tau, transformed_z

def expected_transformation(theta_output, tau, z):
    """Calculates the expected transformation of the output."""
    # Replace with your actual transformation rule
    expected_output = theta_output #example, most theta functions transform more complex.
    return expected_output

def invariance_loss(theta_series_layer, tau, z):
    """Calculates the invariance loss."""
    original_output = theta_series_layer(tau, z)
    transformed_tau, transformed_z = modular_transform(tau, z)
    transformed_output = theta_series_layer(transformed_tau, transformed_z)
    expected_output = expected_transformation(original_output, tau, z)
    loss = nn.MSELoss()(transformed_output, expected_output)
    return loss

# Example Usage within training loop.
def training_loop(model, optimizer, tau, z):
    optimizer.zero_grad()
    prediction_loss = your_prediction_loss_function(model(tau,z), your_target)
    inv_loss = invariance_loss(model.theta_layer, tau, z)
    total_loss = prediction_loss + inv_loss
    total_loss.backward()
    optimizer.step()

class LaplacesDemon(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Assume your inputs include "input_ids" and "labels"
        # Temperature values
        t = 0.5  # base temperature (this is a hyperparameter you can adjust)



        # Forward pass through the model to obtain logits
        outputs = model(**inputs)

        # Retrieve labels from the inputs
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Inputs must contain 'labels'")

        model.config.temperature = 20
        model.generation_config.temperature = 20
        model.model.temperature = 20
        higher_temp_outputs = model(**inputs)

        loss_lower_temp = outputs.loss
        loss_higher_temp = higher_temp_outputs.loss

        model.config.temperature = 1
        model.generation_config.temperature = 1
        model.model.temperature = 1

        return torch.nn.functional.relu(loss_higher_temp - loss_lower_temp)


import torch

def is_converged(theta_output, running_average, threshold):
    """Checks if the theta series has converged."""
    difference = torch.abs(theta_output - running_average)
    return difference < threshold

def update_running_average(theta_output, running_average, alpha):
    """Updates the running average."""
    return alpha * theta_output + (1 - alpha) * running_average

# In your training loop:
running_average = torch.zeros_like(model.theta_layer(input_data)) #initialize average.
threshold = 0.01
alpha = 0.1 #smoothing factor.

theta_output = model.theta_layer(input_data)
running_average = update_running_average(theta_output, running_average, alpha)

if not is_converged(theta_output, running_average, threshold):
    add_theta_term(model.theta_layer) #add a term if not converged.
    optimizer.param_groups[0]['lr'] *= 1.5  # Increase learning rate

def should_branch(entropy_map, threshold):
    """Checks if branching is needed."""
    branching_score = calculate_branching_score(entropy_map)
    return branching_score > threshold

def add_theta_term(theta_series_layer):
    """Adds a new term to the theta series."""
    # Implement your term addition logic here
    new_coefficient = nn.Parameter(torch.randn(1))
    theta_series_layer.coefficients = nn.Parameter(torch.cat((theta_series_layer.coefficients, new_coefficient)))

# In your training loop:
if should_branch(entropy_map, threshold):
    add_theta_term(model.theta_layer)
    optimizer.param_groups[0]['lr'] *= 1.5  # Increase learning rate
    # Log the branching event

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./cfm_checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    logging_strategy="no",
    evaluation_strategy="no",
    remove_unused_columns=False
)

# Assuming you have defined your dataset (e.g., GradientDataset) and your model is instantiated as `your_model`
trainer = LaplacesDemon(
    model=model,
    train_dataset=dataset["train"],
    data_collator=collate_fn,
    args=training_args)

trainer.train()


import torch
import torch.nn as nn

class ThetaSeriesLayer(nn.Module):
    def __init__(self, num_terms):
        super(ThetaSeriesLayer, self).__init__()
        self.num_terms = num_terms
        self.coefficients = nn.Parameter(torch.randn(num_terms))

    def forward(self, x):
        result = torch.zeros_like(x)
        for n in range(self.num_terms):
            result += self.coefficients[n] * torch.exp(-n * x)  # Simplified example
        return result

def calculate_growth_rate(tensor):
    """Approximates growth rate by counting independent features."""
    # A simple proxy: count the number of distinct values in the tensor
    distinct_values = torch.unique(tensor)
    return distinct_values.numel()

def expected_growth_rate_lower_bound(num_terms, weight):
    """Calculates a lower bound for expected growth (example)."""
    # Replace with your actual dimension formula or lower bound
    return weight * num_terms  # Example linear growth

def invariance_loss(theta_layer, x):
    """Example invariance loss (simplified)."""
    original_output = theta_layer(x)
    transformed_x = x + 1.0  # Simplified transformation
    transformed_output = theta_layer(transformed_x)
    return nn.MSELoss()(original_output, transformed_output)

class MyNetwork(nn.Module):
    def __init__(self, num_terms, weight):
        super(MyNetwork, self).__init__()
        self.theta_layer = ThetaSeriesLayer(num_terms)
        self.fc = nn.Linear(1, 1)
        self.weight = weight

    def forward(self, x):
        theta_output = self.theta_layer(x)
        output = self.fc(theta_output)
        return output

    def growth_loss(self, x):
        actual_growth = calculate_growth_rate(self.theta_layer(x))
        expected_growth = expected_growth_rate_lower_bound(self.theta_layer.num_terms, self.weight)
        return nn.MSELoss()(torch.tensor([actual_growth], dtype=torch.float32), torch.tensor([expected_growth], dtype=torch.float32))

def training_loop(model, optimizer, input_data, target_data):
    optimizer.zero_grad()
    prediction_loss = nn.MSELoss()(model(input_data), target_data)
    inv_loss = invariance_loss(model.theta_layer, input_data)
    growth_loss = model.growth_loss(input_data)

    total_loss = prediction_loss + inv_loss + growth_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), model.growth_loss(input_data).item()

# Example Usage
num_terms = 10
weight = 2  # Example weight parameter
model = MyNetwork(num_terms, weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

input_data = torch.randn(10, 1)
target_data = torch.randn(10, 1)

for epoch in range(100):
    total_loss, growth_loss = training_loop(model, optimizer, input_data, target_data)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Total Loss: {total_loss}, Growth Loss: {growth_loss}")

    # Monitoring growth rate directly
    actual_growth = calculate_growth_rate(model.theta_layer(input_data))
    expected_growth = expected_growth_rate_lower_bound(model.theta_layer.num_terms, model.weight)
    if epoch % 10 == 0:
        print(f"actual growth: {actual_growth}, expected growth: {expected_growth}")


import torch

def volume_change(tensor):
    """Calculates volume change using the determinant of the covariance matrix."""
    if tensor.ndim < 2:
        return torch.tensor(0.0) #Cannot calculate covariance of single value.
    if tensor.shape[0] < 2:
        return torch.tensor(0.0) #Cannot calculate covariance with only one point.
    # Calculate covariance matrix
    covariance_matrix = torch.cov(tensor.T)
    # Calculate determinant
    volume = torch.det(covariance_matrix)
    return volume

def volume_loss(theta_layer, x, expected_volume):
    """Calculates volume loss."""
    output = theta_layer(x)
    actual_volume = volume_change(output)
    return nn.MSELoss()(actual_volume, torch.tensor([expected_volume], dtype=torch.float32))

# Example Usage
# Add this to your training loop:
def training_loop(model, optimizer, input_data, target_data, expected_volume):
    optimizer.zero_grad()
    prediction_loss = nn.MSELoss()(model(input_data), target_data)
    inv_loss = invariance_loss(model.theta_layer, input_data)
    growth_loss = model.growth_loss(input_data)
    vol_loss = volume_loss(model.theta_layer, input_data, expected_volume)

    total_loss = prediction_loss + inv_loss + growth_loss + vol_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), model.growth_loss(input_data).item(), vol_loss.item()

#then in your training loop, pass in the expected volume.

import torch
import torch.nn as nn

# Assuming you have your theta series layer and other functions defined

class ReconstructionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ReconstructionNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim) # Example reconstruction network
    def forward(self, x):
        return self.fc(x)

def training_loop_adversarial(model, reconstruction_network, optimizer_model, optimizer_recon, input_data, target_data):
    optimizer_model.zero_grad()
    optimizer_recon.zero_grad()

    # Forward pass
    theta_output = model.theta_layer(input_data)
    model_output = model(input_data)
    unmodeled_component = input_data - theta_output #example unmodeled component
    reconstructed_data = reconstruction_network(unmodeled_component)

    # Losses
    prediction_loss = nn.MSELoss()(model_output, target_data)
    reconstruction_loss = nn.MSELoss()(reconstructed_data, input_data)

    # Adversarial loss (maximize reconstruction loss)
    total_loss = prediction_loss + reconstruction_loss

    # Backpropagation
    total_loss.backward()
    optimizer_model.step()
    optimizer_recon.step()

    import torch
import torch.nn as nn

# Assuming you have your theta series layer and reconstruction network
def laplaces_demon(model, reconstruction_network, optimizer_model, optimizer_recon, input_data):
    optimizer_model.zero_grad()
    optimizer_recon.zero_grad()

    theta_output = model.theta_layer(input_data)
    unmodeled_component = input_data - theta_output
    reconstructed_data = reconstruction_network(unmodeled_component)

    reconstruction_loss = nn.MSELoss()(reconstructed_data, input_data)
    # TODO: should reconstruction loss be minimized or maximized from un-modeled?
    #       if it's maximized then it's the amount that's incoherent
    #       if it's maximized and ability to predict OOD is maximized then
    #       it only captures the general portion.
    #       maximization of reconstruction loss, or some version of it could be important here to enforce
    #       general
    invariance_loss = invariance_loss(model.theta_layer, input_data) #your invariance loss function.

    total_loss = -reconstruction_loss + invariance_loss #negative reconstruction loss.

    total_loss.backward()
    optimizer_model.step()
    optimizer_recon.step()

import torch

# Use the properties of the underlying torus to infer how modular the structure is
def fundamental_domain_loss(tau):
    """Calculates the fundamental domain loss."""
    real_tau = tau.real
    imag_tau = tau.imag
    magnitude_tau = torch.sqrt(real_tau**2 + imag_tau**2)

    magnitude_check = magnitude_tau >= 1.0
    real_part_check = (real_tau >= -0.5) & (real_tau <= 0.5)

    if magnitude_check and real_part_check:
        return torch.tensor(0.0)  # tau is within the fundamental domain
    else:
        # Calculate a loss value (e.g., distance from the boundary)
        magnitude_violation = torch.relu(1.0 - magnitude_tau)
        real_violation_left = torch.relu(-0.5 - real_tau)
        real_violation_right = torch.relu(real_tau - 0.5)
        return magnitude_violation + real_violation_left + real_violation_right

# Example Usage within training loop.
def training_loop(model, optimizer, tau, z, target):
    optimizer.zero_grad()
    prediction_loss = your_prediction_loss_function(model(tau,z), target)
    fd_loss = fundamental_domain_loss(tau)
    total_loss = prediction_loss + fd_loss
    total_loss.backward()
    optimizer.step()