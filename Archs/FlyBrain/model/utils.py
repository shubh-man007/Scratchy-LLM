import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.ndimage
import cv2   
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
from torchvision.models import resnet18
from torchvision import transforms
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx


def get_data(ROOT_DIR):
    if os.path.exists(ROOT_DIR):
        CSV_PATH_TRAIN = os.path.join(ROOT_DIR, "data/nyu2_train.csv")
        CSV_PATH_TEST = os.path.join(ROOT_DIR, "data/nyu2_test.csv")
        nyu_train_df = pd.read_csv(CSV_PATH_TRAIN,   
                           sep=',',
                           header=None, 
                           names=["rgb", "depth"])

        nyu_test_df = pd.read_csv(CSV_PATH_TEST, 
                                sep=',', 
                                header=None,
                                names=["rgb", "depth"])
        
        return nyu_train_df, nyu_test_df
    
    else:
        print(f"Root Directory: {ROOT_DIR}\n not found.")




class DepthDataset(Dataset):
    def __init__(self, dataframe, ROOT_DIR, img_height=240, img_width=320, transform=None):
        """
        :param dataframe: pd.DataFrame [rgb, depth]
        :param root_dir: folder dataframe
        :param img_height, img_width: resize to
        :param transform: augmentation
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = ROOT_DIR
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb_path = os.path.join(self.root_dir, row["rgb"])   
        #print(rgb_path, os.path.exists(rgb_path))
        depth_path = os.path.join(self.root_dir, row["depth"])
        #print(depth_path, os.path.exists(depth_path))
        

        # (BGR -> RGB)
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print("Warning: Failed to load image:", rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (self.img_width, self.img_height))

        # Depth 
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_img = cv2.resize(depth_img, (self.img_width, self.img_height))

        # to float32
        rgb_img = rgb_img.astype(np.float32) / 255.0
        depth_img = depth_img.astype(np.float32)

        # to tensor PyTorch: (C,H,W)
        rgb_tensor = torch.from_numpy(np.transpose(rgb_img, (2,0,1)))   # (3, H, W)
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0)         # (1, H, W)

        return rgb_tensor, depth_tensor
    



def build_grid_edge_index(H, W, connectivity=4):
    nodes = H * W
    indices = np.arange(nodes).reshape(H, W)
    edge_list = []
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")
    for i in range(H):
        for j in range(W):
            current = indices[i, j]
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor = indices[ni, nj]
                    edge_list.append([current, neighbor])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index




class FlyDepth(nn.Module):
    def __init__(self, cnn_out_channels=64, gnn_hidden_dim=64, num_gnn_layers=2, fly_embed_dim=64):
        super(FlyDepth, self).__init__()
        # 1. CNN Backbone: Use a pretrained ResNet-18.
        backbone = resnet18(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Reduce channels from 512 to cnn_out_channels.
        self.cnn_reducer = nn.Conv2d(512, cnn_out_channels, kernel_size=1)
        
        # 2. GNN Layers: Stack of GCNConv layers.
        self.gnn_layers = nn.ModuleList()
        in_channels = cnn_out_channels
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(GCNConv(in_channels, gnn_hidden_dim))
            in_channels = gnn_hidden_dim
        
        # 3. FlyBrain Prior: A learnable embedding per node.
        # Assume fixed grid size after CNN (e.g., if feature map is 20x30, then num_nodes=600).
        self.num_nodes = 600  # Adjust this value to match your CNN feature map size.
        self.fly_prior = nn.Embedding(self.num_nodes, fly_embed_dim)
        
        # 4. Fusion: Fuse GNN output with the fly prior.
        self.fusion = nn.Linear(gnn_hidden_dim + fly_embed_dim, gnn_hidden_dim)
        
        # 5. Decoder: Upsample fused grid features back to a dense depth map.
        self.decoder = nn.Sequential(
            nn.Conv2d(gnn_hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)  # Output single-channel depth map.
        )
    
    def forward(self, x):
        """
        x: Input RGB image tensor of shape [B, 3, H_img, W_img]
        Returns:
            Depth map tensor of shape [B, 1, H_img, W_img]
        """
        B, C, H_img, W_img = x.shape
        # 1. Extract CNN features.
        feat_map = self.cnn_backbone(x)   # [B, 512, H_feat, W_feat]
        feat_map = self.cnn_reducer(feat_map)  # [B, cnn_out_channels, H_feat, W_feat]
        B, C_feat, H_feat, W_feat = feat_map.shape  # Use C_feat for channel dimension.
        num_nodes = H_feat * W_feat  # Expected to match self.num_nodes (or adjust accordingly).
        
        # Build grid graph edge_index (same for all images).
        edge_index = build_grid_edge_index(H_feat, W_feat, connectivity=4).to(x.device)
        
        outputs = []
        for b in range(B):
            # Reshape CNN feature map into node features: [num_nodes, C_feat]
            node_feats = feat_map[b].view(C_feat, -1).t()  # [num_nodes, C_feat]
            
            # 2. Pass node features through GNN layers.
            for conv in self.gnn_layers:
                node_feats = conv(node_feats, edge_index)
                node_feats = F.relu(node_feats)
            
            # 3. Get fly brain prior embedding.
            if num_nodes != self.num_nodes:
                fly_embed = self.fly_prior.weight[:num_nodes]
            else:
                fly_embed = self.fly_prior.weight  # [num_nodes, fly_embed_dim]
            
            # 4. Fusion: Concatenate and apply linear fusion.
            fused = torch.cat([node_feats, fly_embed], dim=1)  # [num_nodes, gnn_hidden_dim + fly_embed_dim]
            fused = self.fusion(fused)  # [num_nodes, gnn_hidden_dim]
            
            # 5. Reshape fused node features back to grid: [1, gnn_hidden_dim, H_feat, W_feat]
            fused_grid = fused.t().view(1, -1, H_feat, W_feat)
            
            # 6. Decode fused features into a depth map.
            depth_map = self.decoder(fused_grid)  # [1, 1, H_out, W_out]
            # Upsample the predicted depth map to match the original input resolution.
            depth_map = F.interpolate(depth_map, size=(H_img, W_img), mode='bilinear', align_corners=True)
            outputs.append(depth_map)
        
        outputs = torch.cat(outputs, dim=0)  
        return outputs




def guided_filter(I, p, radius=64, eps=1e-3):
    """
    Perform guided filtering to refine the depth map.
    :param I: Guidance image (grayscale)
    :param p: Input image to be filtered (depth map)
    :param radius: Window radius
    :param eps: Regularization term
    :return: Filtered image
    """
    mean_I = scipy.ndimage.uniform_filter(I, radius)
    mean_p = scipy.ndimage.uniform_filter(p, radius)
    corr_I = scipy.ndimage.uniform_filter(I * I, radius)
    corr_Ip = scipy.ndimage.uniform_filter(I * p, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = scipy.ndimage.uniform_filter(a, radius)
    mean_b = scipy.ndimage.uniform_filter(b, radius)

    q = mean_a * I + mean_b
    return q

def apply_guided_filter(rgb_img, pred_heatmap):
    # Convert RGB to grayscale
    guide_gray = np.mean(rgb_img, axis=2)  
    return guided_filter(guide_gray, pred_heatmap)

def enhance_sharpness(image, alpha=1.5, beta=-0.5):
    """
    Apply unsharp masking to enhance the sharpness of the image.
    :param image: Input grayscale image.
    :param alpha: Weight for the original image.
    :param beta: Weight for the blurred image.
    :return: Sharpened image.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 3)  # Gaussian Blur
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def test_and_visualize(model, data_loader, device, save_dir="output"):
    model.eval()
    os.makedirs(f"{save_dir}/rgb_images", exist_ok=True)
    os.makedirs(f"{save_dir}/predicted_depth", exist_ok=True)
    os.makedirs(f"{save_dir}/filtered_depth", exist_ok=True)

    with torch.no_grad():
        for batch_idx, (rgb_tensor, depth_tensor) in enumerate(data_loader):
            rgb_tensor = rgb_tensor.to(device)
            depth_tensor = depth_tensor.to(device).float()
            
            pred_depth = model(rgb_tensor)  

            pred_depth = pred_depth.cpu().numpy()
            depth_tensor = depth_tensor.cpu().numpy()
            rgb_tensor = rgb_tensor.cpu().numpy()

            idx = 0
            rgb_img = np.transpose(rgb_tensor[idx], (1, 2, 0))  
            gt_depth = depth_tensor[idx, 0, :, :]
            pred_heatmap = pred_depth[idx, 0, :, :]
            
            pred_heatmap_norm = cv2.normalize(pred_heatmap, None, 0, 255, cv2.NORM_MINMAX)
            pred_heatmap_norm = np.uint8(pred_heatmap_norm)

            refined_depth = apply_guided_filter(rgb_img, pred_heatmap_norm)

            refined_depth_sharp = enhance_sharpness(refined_depth)

            rgb_save_path = os.path.join(save_dir, "rgb_images", f"rgb_{batch_idx}.png")
            depth_save_path = os.path.join(save_dir, "predicted_depth", f"depth_{batch_idx}.png")
            filtered_save_path = os.path.join(save_dir, "filtered_depth", f"filtered_depth_{batch_idx}.png")

            cv2.imwrite(rgb_save_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_save_path, pred_heatmap_norm)
            cv2.imwrite(filtered_save_path, refined_depth_sharp)

            print(f"Saved: {rgb_save_path}, {depth_save_path}, and {filtered_save_path}")

            plt.figure(figsize=(24, 6))

            plt.subplot(1, 4, 1)
            plt.imshow(rgb_img)
            plt.title("RGB Image")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(gt_depth, cmap='viridis')
            plt.title("Ground Truth Depth")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(pred_heatmap, cmap='viridis')
            plt.title("Predicted Depth Heatmap")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(refined_depth_sharp, cmap='viridis')
            plt.title("Refined Depth (Guided + Sharpened)")
            plt.axis("off")

            plt.show()
            break




def load_and_preprocess_image(image_path, img_height=480, img_width=640):
    """
    Load an image from disk, resize, normalize, and convert to a torch tensor.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))  
    img_tensor = img_tensor.unsqueeze(0)  
    
    return img_tensor, img  


def test_image_path(model, device, image_path, save_dir="output", img_height=480, img_width=640):
    """
    Load an image from a given path, run the model to predict depth, and apply guided filtering
    and sharpness enhancement to refine the depth map.
    Saves and visualizes the results.
    """
    os.makedirs(os.path.join(save_dir, "rgb_images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "predicted_depth"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "filtered_depth"), exist_ok=True)
    
    img_tensor, rgb_img = load_and_preprocess_image(image_path, img_height, img_width)
    img_tensor = img_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        pred_depth = model(img_tensor)
    
    pred_depth = torch.nn.functional.interpolate(pred_depth, size=(img_height, img_width), mode='bilinear', align_corners=True)
    
    pred_depth_np = pred_depth.squeeze().cpu().numpy()  
    
    pred_depth_norm = cv2.normalize(pred_depth_np, None, 0, 255, cv2.NORM_MINMAX)
    pred_depth_norm = np.uint8(pred_depth_norm)
    
    refined_depth = apply_guided_filter(rgb_img, pred_depth_norm)
    
    refined_depth_sharp = enhance_sharpness(refined_depth)
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    rgb_save_path = os.path.join(save_dir, "rgb_images", f"{base_filename}_rgb.png")
    depth_save_path = os.path.join(save_dir, "predicted_depth", f"{base_filename}_depth.png")
    filtered_save_path = os.path.join(save_dir, "filtered_depth", f"{base_filename}_filtered.png")
    
    cv2.imwrite(rgb_save_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_save_path, pred_depth_norm)
    cv2.imwrite(filtered_save_path, refined_depth_sharp)
    
    print(f"Saved: {rgb_save_path}, {depth_save_path}, and {filtered_save_path}")
    
    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(rgb_img)
    plt.title("RGB Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(pred_depth_np, cmap='viridis')
    plt.title("Predicted Depth Heatmap")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(refined_depth, cmap='viridis')
    plt.title("Refined Depth (Guided Filter)")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(refined_depth_sharp, cmap='viridis')
    plt.title("Refined Depth (Sharpness Enhanced)")
    plt.axis("off")
    
    plt.show()
