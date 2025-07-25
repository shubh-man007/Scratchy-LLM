from utils import get_data, DepthDataset, FlyDepth, test_and_visualize, test_image_path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim


model = FlyDepth(cnn_out_channels=64, gnn_hidden_dim=64, num_gnn_layers=2, fly_embed_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("cnn2gnn_flyprior.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
last_loss = checkpoint['loss']

model.to(device)

ROOT_DIR = "C:/Users/deshu/.cache/kagglehub/datasets/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/"
IMG_HEIGHT = 480
IMG_WIDTH = 640
BATCH_SIZE = 4

nyu_train_df, nyu_test_df = get_data(ROOT_DIR)
test_dataset = DepthDataset(nyu_test_df, ROOT_DIR, IMG_HEIGHT, IMG_WIDTH)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_and_visualize(model, test_loader, device)

test_image_file = "C:/Users/deshu/Desktop/FlyBrain/Testing_Images/1.jpg" 
test_image_path(model, device, test_image_file, save_dir="output", img_height=480, img_width=640)
