import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
from utils import get_data, DepthDataset, FlyDepth

ROOT_DIR = "C:/Users/deshu/.cache/kagglehub/datasets/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/"
IMG_HEIGHT = 480
IMG_WIDTH = 640
BATCH_SIZE = 4

nyu_train_df, nyu_test_df = get_data(ROOT_DIR)

train_dataset = DepthDataset(nyu_train_df, ROOT_DIR, IMG_HEIGHT, IMG_WIDTH)
test_dataset = DepthDataset(nyu_test_df, ROOT_DIR, IMG_HEIGHT, IMG_WIDTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlyDepth(cnn_out_channels=64, gnn_hidden_dim=64, num_gnn_layers=2, fly_embed_dim=64).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() 

epochs = 10

model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for rgb_tensor, depth_tensor in train_loader:
        rgb_tensor = rgb_tensor.to(device)           
        depth_tensor = depth_tensor.to(device).float() 
        
        optimizer.zero_grad()
        pred_depth = model(rgb_tensor)                 
        loss = loss_fn(pred_depth, depth_tensor)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * rgb_tensor.size(0)
    
    avg_loss = epoch_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")



save_path = "cnn2gnn_flyprior.pth"
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, save_path)
print("Model saved at", save_path)




# import os
# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from utils import get_data, DepthDataset, FlyDepth  
# from config import get_config, get_weights_file_path

# def train_model(config):
#     os.makedirs(config['model_folder'], exist_ok=True)

#     nyu_train_df, nyu_test_df = get_data(config['root_dir'])

#     train_dataset = DepthDataset(nyu_train_df, 
#                                  config['root_dir'], 
#                                  img_height=config['img_height'], 
#                                  img_width=config['img_width'])
    
#     test_dataset = DepthDataset(nyu_test_df, 
#                                 config['root_dir'], 
#                                 img_height=config['img_height'], 
#                                 img_width=config['img_width'])
    
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model = FlyDepth(cnn_out_channels=64, gnn_hidden_dim=64, num_gnn_layers=2, fly_embed_dim=64).to(device)
    
#     optimizer = optim.Adam(model.parameters(), lr=config['lr'])
#     loss_fn = nn.MSELoss()  
    
#     start_epoch = 0
#     if config['preload'] is not None:
#         checkpoint_path = get_weights_file_path(config, config['preload'])
#         print(f"Preloading model from {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
    
#     print(f"Train samples: {len(train_dataset)}")
#     print(f"Test samples: {len(test_dataset)}")
    
#     # Training Loop
#     model.train()
#     for epoch in range(start_epoch, config['num_epochs']):
#         epoch_loss = 0.0
#         for rgb_tensor, depth_tensor in train_loader:
#             rgb_tensor = rgb_tensor.to(device)           # [B, 3, H, W]
#             depth_tensor = depth_tensor.to(device).float() # [B, 1, H, W]
            
#             optimizer.zero_grad()
#             pred_depth = model(rgb_tensor)  # Forward pass; shape: [B, 1, H_out, W_out]
#             loss = loss_fn(pred_depth, depth_tensor)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item() * rgb_tensor.size(0)
        
#         avg_loss = epoch_loss / len(train_dataset)
#         print(f"Epoch {epoch+1}/{config['num_epochs']}, Average Loss: {avg_loss:.4f}")
        
#         # Save model checkpoint at the end of each epoch.
#         save_path = get_weights_file_path(config, f"{epoch+1:02d}")
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': avg_loss,
#         }, save_path)
#         print(f"Model saved at {save_path}")
    
#     print("Training complete!")
    
#     # Optionally, you could add a testing function here to evaluate on the test set.
#     return model, train_loader, test_loader

# if __name__ == '__main__':
#     config = get_config()
#     train_model(config)
