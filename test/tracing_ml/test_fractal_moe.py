import copy

import torch
import torch.nn as nn
from transformers import AutoTokenizer,  AutoModelForCausalLM
import torch
from transformers import T5Tokenizer
import datasets
from torch.utils.data import DataLoader

# CUDA = "cuda:0"
#
# device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.cuda.set_device(0)

import copy

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

MODEL = "google-t5/t5-small"
DIM=512

class FractalMixtureOfExpertsLayer(nn.Module):
    # maybe n_models should be 12
    def __init__(self, num_experts, num_layers, num_router_models, kan_wave_dim,
                 kan_num_scales=3, kan_num_orientations=3, n_models=3):
        super(FractalMixtureOfExpertsLayer, self).__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.num_router_models = num_router_models
        self.kan_wave_dim = kan_wave_dim

        # Initialize the pre-trained mixture of experts model
        # Initialize the KAN-Wave layers
        self.kan_wave_layers = [KanWave(kan_num_scales, kan_num_orientations, kan_wave_dim) for _ in range(n_models)]
        self.models = []
        self.aggregator_model = torch.nn.Linear(DIM * 2, 2048)
        self.aggregator_intermediate = torch.nn.ModuleList([torch.nn.Linear(2048, 2048) for _ in range(4)])
        self.ln = torch.nn.ReLU()
        self.aggregator_out = torch.nn.Linear(2048, DIM)
        self.pw_model = torch.nn.Linear(DIM, DIM)
        self.n_models = n_models

    def lm_head(self, kan, moe_output):
        self.init_models()
        return self.models[0].lm_head(kan + moe_output)

    def initial(self):
        self.init_models()
        return self.models[0](input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    def forward(self, input_tokens, attention_mask, moe_output):
        self.init_models()

        output = moe_output.decoder_hidden_states[-1]
        waves = [torch.nn.functional.softmax(torch.mean(wave_layer(output), dim=1), dim=-1)
                 for wave_layer in self.kan_wave_layers]
        concat_models = self.concat_models(waves, attention_mask, input_tokens, moe_output)
        first = output
        for i in concat_models:
            first = self.aggregator_model(torch.concat([first, i], dim=-1))
            for m in self.aggregator_intermediate:
                first = self.ln(m(first))
            first = self.aggregator_out(first)

        return first

    def init_models(self):
        if len(self.models) != self.n_models:
            self.models = [
                # extreme case "self-similarity" - does this work digitally???
                AutoModelForSeq2SeqLM.from_pretrained(MODEL, token=TOKEN)
                for _ in range(self.n_models)]
            for m in self.models:
                m.to(device)

    def concat_models(self, argmax, attention_mask, input_tokens, first):
        return [self.second(argmax[i],
                            lambda: self.models[i](decoder_input_ids=input_tokens, input_ids=input_tokens, attention_mask=attention_mask, output_hidden_states=True)
                            if i != 0 else first)
                for i in range(self.n_models)]

    def second(self, argmax_2, model):
        f = (model().decoder_hidden_states[-1] * torch.nn.functional.softmax(argmax_2, dim=-1))
        return f



class MotherWaveletRouter(nn.Module):
    def __init__(self, num_scales, num_orientations, kernel_size, num_wavelets=12):
        super(MotherWaveletRouter, self).__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        self.num_wavelets = num_wavelets

        # Learnable parameters for Gaussian centers and variances
        self.means = nn.Parameter(torch.randn(self.num_wavelets, 1, 1))
        self.vars = nn.Parameter(torch.ones(self.num_wavelets, 1, 1))

        self.base_wavelet = self._create_base_wavelet()
        self.ln = torch.nn.LayerNorm(512)
        self.p_ln = torch.nn.LayerNorm(512)

    def _create_base_wavelet(self):
        x = torch.linspace(-1, 1, steps=self.kernel_size)
        y = torch.linspace(-1, 1, steps=self.kernel_size)
        grid_x, grid_y = torch.meshgrid(x, y)
        base = torch.exp(-(grid_x**2 + grid_y**2))
        return base  # Shape: (kernel_size, kernel_size)

    def forward(self, x):
        channels, height, width = x.size()
        wavelet_transform = torch.zeros(self.num_wavelets, height, width, device=x.device)

        x = x.unsqueeze(0).expand(self.num_wavelets, x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.num_wavelets):
            # Create Gaussian-modulated wavelet
            gaussian = torch.exp(-((x[i, :, :] - self.means[i])**2) / (2 * self.vars[i]**2))
            wavelet = torch.matmul(gaussian, self.base_wavelet)
            wavelet = wavelet.unsqueeze(0)  # Shape: (1, 64, 512)
            wavelet = self.ln(wavelet)

            # Apply convolution
            wavelet_out = torch.nn.functional.conv2d(x[i, :, :, :], wavelet, padding=(32, 256))
            wavelet_transform[i, :, :] = self.p_ln(wavelet_out[:, :64, :512])

        return torch.mean(wavelet_transform, dim=0)

class KanWave(nn.Module):
    def __init__(self, num_scales, num_orientations, kan_dim):
        super(KanWave, self).__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.kan_dim = kan_dim
        self.mother_wavelet = MotherWaveletRouter(num_scales, num_orientations, kan_dim)
        self.kan_layer = nn.Linear(DIM, DIM)

    def forward(self, x):
        # Compute the wavelet transform using the mother wavelet
        print("KAN Forward ", x.shape)
        wavelet_transform = self.mother_wavelet(x)
        # Apply the KAN layer
        kan_output = self.kan_layer(wavelet_transform)

        return kan_output


import torch
from transformers import T5Tokenizer
import datasets
from torch.utils.data import DataLoader

# Set the hyperparameters
batch_size = 32
num_epochs = 20000
num_experts = 8
num_layers = 4
num_router_models = 2
kan_wave_dim = 512
kan_num_orientations=8
max_length=64
num_models=12
kan_num_scales=num_models


# Initialize the model
model = FractalMixtureOfExpertsLayer(num_experts, num_layers, num_router_models,
                                     kan_wave_dim, kan_num_orientations, kan_num_scales,
                                     n_models=num_models)
# model.state_dict(torch.load("/content/drive/MyDrive/out.pt"))

model.init_models()
model.to(device)


# Load the Wikipedia dataset
dataset = datasets.load_dataset("wikipedia", "20220301.en")
tokenizer = AutoTokenizer.from_pretrained(MODEL,
                                          token=TOKEN, device=device)

# Create the data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=9e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)

from transformers import Trainer, TrainingArguments
num = 0
import torch


# Create a custom dataset class to load the text column
class WikipediaTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx = 0

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        encoding = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoding

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.idx < self.__len__() - 1:
            self.idx += 1
            yield self[self.idx]

        else:
            print("Restarting")
            self.idx = 0
            yield self[self.idx]

# class MotherWaveletRouterOther(nn.Module):
#     def __init__(self, num_scales, num_orientations, kernel_size):
#         super(MotherWaveletRouterOther, self).__init__()
#         self.num_scales = num_scales
#         self.num_orientations = num_orientations
#         self.kernel_size = kernel_size
#         self.num_wavelets = num_scales * num_orientations

#         # Initialize learnable Gaussian parameters: mean and variance
#         self.means = nn.Parameter(torch.randn(self.num_wavelets, 1, 1))  # Example dimensions
#         self.vars = nn.Parameter(torch.ones(self.num_wavelets, 1, 1))   # Ensure positivity

#         # Define base wavelet shape, e.g., Morlet wavelet
#         self.base_wavelet = self._create_base_wavelet()

#     def _create_base_wavelet(self):
#         # Define a base wavelet; for example, a simple Gaussian wavelet
#         x = torch.linspace(-1, 1, steps=self.kernel_size)
#         y = torch.linspace(-1, 1, steps=self.kernel_size)
#         grid_x, grid_y = torch.meshgrid(x, y)
#         base = torch.exp(-(grid_x**2 + grid_y**2))
#         return base  # Shape: (kernel_size, kernel_size)

#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         wavelet_transform = torch.zeros(batch_size, self.num_wavelets, height, width, device=x.device)

#         for i in range(self.num_wavelets):
#             # Create Gaussian-modulated wavelet
#             gaussian = torch.exp(-((x[:, i, :, :] - self.means[i])**2) / (2 * self.vars[i]**2))
#             wavelet = self.base_wavelet * gaussian
#             wavelet = wavelet.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)

#             # Apply convolution
#             wavelet_out = torch.nn.functional.conv2d(x, wavelet, stride=1, padding=self.kernel_size//2)
#             wavelet_transform[:, i, :, :] = wavelet_out.squeeze(1)

#         return wavelet_transform


# Create the custom dataset instance
custom_dataset = WikipediaTextDataset(dataset["train"])

# Train the model
while True:
    model.train()
    total_loss = 0
    for batch in custom_dataset:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        moe_output = model.initial()
        kan = model(input_ids, attention_mask, moe_output)
        # Forward pass todo:contrastive loss 4 localglobal proxy b
        head = model.lm_head(kan, kan + moe_output.decoder_hidden_states[-1])
        outputs = torch.softmax(head, dim=-1)

        labels = input_ids[:, 1:]  # Shift the input IDs to the right by one position
        loss = torch.nn.functional.cross_entropy(outputs[:, :-1, :].transpose(2, 1), labels)
        maxed = torch.argmax(outputs, dim=-1)
        print(tokenizer.decode(token_ids=[input_ids[0, 0]]), tokenizer.decode(token_ids=maxed[0]))

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Update the total loss
        total_loss += loss.item()

    # Update the learning rate
    scheduler.step()

    # Print the loss
    print(f'Epoch {num}, Loss: {total_loss}')
    total_loss = 0
    num += 1
    # if num % 1000 == 0:
    #     torch.save(model.state_dict(), "/content/drive/MyDrive/out.pt")



    # Save the model
    # torch.save(model.state_dict(), f'model_{epoch + 1}.pth')

