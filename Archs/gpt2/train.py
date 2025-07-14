import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tiktoken
import os
from model import GPTModel, create_dataloader_v1, GPTConfig, visualize_model_weights, print_model_summary
import time
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10, save_path='gpt2_model.pth'):
    model.train()
    model.to(device)
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    epoch_losses = []
    batch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            batch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
              f"Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{save_path}_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_loss,
    }, save_path)
    print(f"Final model saved as {save_path}")
    
    return model, epoch_losses, batch_losses

def plot_loss_curves(epoch_losses, batch_losses, save_path='training_loss.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(batch_losses, 'r-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss per Batch')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Loss curves saved as {save_path}")

def generate_sample_text(model, tokenizer, device, prompt="The", max_new_tokens=50, context_size=256):
    model.eval()
    
    encoded_prompt = tokenizer.encode(prompt, allowed_special={"<|EOS|>"})
    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    
    print(f"Generating text with prompt: '{prompt}'")
    print("Generated text:")
    print(prompt, end="")
    
    with torch.no_grad():
        generated_idx = model.generate(idx, max_new_tokens, context_size)
    
    generated_tokens = generated_idx[0].cpu().numpy()
    generated_text = tokenizer.decode(generated_tokens)
    
    print(generated_text[len(prompt):])
    return generated_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    text_file = "verdict.txt"
    if not os.path.exists(text_file):
        print(f"Error: {text_file} not found!")
        return
    
    text_data = load_text_data(text_file)
    print(f"Loaded text data: {len(text_data)} characters")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"Vocabulary size: {vocab_size}")
    
    config = GPTConfig(
        vocab_size=vocab_size,
        context_length=256,
        emb_dim=384,
        n_layers=6,
        n_heads=6,
        drop_rate=0.1,
        qkv_bias=False
    )
    model = GPTModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print_model_summary(model)
    # visualize_model_weights(model, save_path="gpt2_model_weights_hist_before_training.png")
    
    batch_size = 8
    max_length = 256
    stride = 128
    
    train_loader = create_dataloader_v1(
        text_data, 
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    print(f"Created data loader with {len(train_loader)} batches")
    
    learning_rate = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 100
    model, epoch_losses, batch_losses = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        save_path='gpt2_verdict_model.pth'
    )
    
    print_model_summary(model)
    # visualize_model_weights(model, save_path="gpt2_model_weights_hist_after_training.png")
    print("\n" + "="*50)
    print("PLOTTING LOSS CURVES")
    print("="*50)
    plot_loss_curves(epoch_losses, batch_losses, 'gpt2_training_loss.png')
    
    print("\n" + "="*50)
    print("GENERATING SAMPLE TEXT")
    print("="*50)
    
    prompts = ["The", "I", "Jack", "Mrs.", "Well"]
    for prompt in prompts:
        try:
            generated_text = generate_sample_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=30,
                context_size=256
            )
            print("\n" + "-"*30 + "\n")
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")

if __name__ == "__main__":
    main() 
