import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import matplotlib.pyplot as plt

## Classes for different model architectures that induce non-convex loss landscapes
## If possible, please visualize the loss landscapes of these models 
class SimpleNet(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # [B, 16, 10, 10]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # [B, 32, 10, 10]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 32, 1, 1]
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(x.size(0), 1, 10, 10)  # reshape [B, 100] to [B, 1, 10, 10]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)  # flatten [B, 32]
        return self.fc(x)

class TransformerNet(nn.Module):
    def __init__(self, seq_len=10, embed_dim=10):
        super().__init__()
        self.embedding = nn.Linear(embed_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), 10, 10)  # [B, 100] → [B, 10, 10] (sequence of 10 tokens)
        x = self.embedding(x)         # [B, 10, 64]
        x = self.transformer(x)       # [B, 10, 64]
        x = x.transpose(1, 2)         # [B, 64, 10]
        x = self.pool(x).squeeze(-1)  # [B, 64]
        return self.fc(x)             # [B, 1]

def run_input_noise_sequential(model, x, y, optim='sgd', steps=100, full_batch=100, sub_batch=10, noise_std=0.1):
    ''' Run input noise perturbation with sequential updates 
        (For some models that are not compatible with inputs having batch size > 1)'''
    if optim not in ['sgd', 'adam']:
        raise ValueError("Optimizer must be 'sgd' or 'adam'")
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_hisory = []
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0
        for i in range(0, full_batch, sub_batch):
            x_part = x[i:i+sub_batch]
            y_part = y[i:i+sub_batch]
            x_noisy = x_part + torch.randn_like(x_part) * noise_std
            loss = F.mse_loss(model(x_noisy), y_part) / (full_batch / sub_batch)
            loss.backward()  # Gradients accumulate
            total_loss += loss.item()
        optimizer.step()
        loss_hisory.append(total_loss)
    return time.time() - start, loss_hisory

def run_gradient_noise(model, x, y, optim='sgd', steps=100, noise_std=0.1, K=10):
    ''' Run gradient noise perturbation,
        and batch is considered as a single update for this configuration'''
    if optim not in ['sgd', 'adam']:
        raise ValueError("Optimizer must be 'sgd' or 'adam'")
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    min_loss = float('inf')
    start = time.time()
    loss_hisory = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()

        # Save clean gradients
        base_grad = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

        # Vectorized noise accumulation over K
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                if p.grad is not None:
                    # Add noise to the gradient
                    g = base_grad[i]
                    gradient = g + torch.randn_like(g) * noise_std * (1/math.sqrt(K))
                    p.grad.copy_(gradient)

        optimizer.step()
        loss_hisory.append(loss.item())

    return time.time() - start, loss_hisory


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_iter = 1000
batch_sizes = [100] # Whole batch size for input noise perturbation
sub_batch_sizes = [100] # Sub-batch sizes
optimizer_types = ['sgd', 'adam'] 
model_types = ['SimpleNet', 'CNNNet', 'TransformerNet'] # Model architectures to test
results = []
result_path = 'results/250625_noisy_opt' # Results directory

## TODO: Here, I only did with MSE loss, but you can add other losses as well. 
## Landscape would be different by model * loss type, so you can add more combinations.
## TODO: Learning rate 

if not os.path.exists(result_path):
    os.makedirs(result_path)
    
for optimizer_type in optimizer_types:
    print(f"Testing optimizer: {optimizer_type}")

    for model_class in model_types:
        print(f"Testing model: {model_class.__name__}")

        for batch in batch_sizes:
            for sub_batch in sub_batch_sizes:
                model_input = model_class().to(device)
                model_grad = model_class().to(device)

                x = torch.randn(batch, 100).to(device)
                y = torch.randn(batch, 1).to(device)

                ## Perturb noise in input space and backpropagate
                t_in, loss_history_in = run_input_noise_sequential(model_input, x, y, steps=num_iter, optim=optimizer_type, sub_batch=sub_batch)
                
                ## Perturb noise in gradient space and backpropagate
                t_gr, loss_history_gr = run_gradient_noise(model_grad, x, y, steps=num_iter, optim=optimizer_type, K=sub_batch)

                results.append({
                    'Optimizer': optimizer_type,
                    'Model': model_class.__name__,
                    'Batch Size': batch,
                    'Sub Batch Size': sub_batch,
                    'Time (Input Noise)': round(t_in, 4),
                    'Time (Grad Noise)': round(t_gr, 4),
                    'Min Loss (Input Noise)': round(min(loss_history_in), 6),
                    'Min Loss (Grad Noise)': round(min(loss_history_gr), 6)
                })

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(loss_history_in, label='Input Noise Loss')
                plt.title(f'{model_class.__name__} - {optimizer_type} - sub_batch {sub_batch} - Input Noise')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(loss_history_gr, label='Gradient Noise Loss')
                plt.title(f'{model_class.__name__} - {optimizer_type} - sub_batch {sub_batch} - Grad Noise')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                # plt.show()
                # plt.pause(1)
                plt.savefig(f'{result_path}/loss_curve_{model_class.__name__}_{optimizer_type}_sub{sub_batch}.png')
                plt.close()
                print(f"Completed {model_class.__name__} with {optimizer_type} for batch {batch} and sub-batch {sub_batch}")

df = pd.DataFrame(results)
print(df)
