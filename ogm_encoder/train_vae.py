import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from vae import VAE
from dataset import OccupancyGridDataset

# Command line arguments
parser = argparse.ArgumentParser(description='Train a VAE with a specific dataset')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train (default: 50)')
parser.add_argument('--batch-size', type=int, default=1024, help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('--data-dir', type=str, default='/home/aleksi/ogm_images/', help='directory for training data')
parser.add_argument('--save-dir', type=str, default='outputs', help='directory to save output images and model checkpoints')
parser.add_argument('--validation-split', type=float, default=0.1, help='proportion of data to use for validation')
parser.add_argument('--latent-shape', type=int, default=64, help='number of latent variables')
parser.add_argument('--initial-weights', type=str, default=None, help='path to initial model weights to load (optional)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
os.makedirs(args.save_dir, exist_ok=True)

# Set up datasets
dataset = OccupancyGridDataset(folder_path=args.data_dir)  # Use your custom dataset
train_size = int((1 - args.validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Model, optimizer, and loss function setup
model = VAE(input_shape=dataset.image_size, latent_shape=args.latent_shape).to(device)
if args.initial_weights:
    model.load_state_dict(torch.load(args.initial_weights))
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_loader) * args.epochs
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
reconstruction_loss = nn.CrossEntropyLoss(reduction='sum')


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        CEL = reconstruction_loss(recon_batch, target)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = CEL + KLD
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()  # Update learning rate
        
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch} Average Training Loss: {avg_train_loss:.6f}')
    writer.add_scalar('Loss/train', avg_train_loss, epoch)


def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            recon_batch, mu, logvar = model(data)
            CEL = reconstruction_loss(recon_batch, target)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            val_loss += CEL.item() + KLD.item()
        
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f'Epoch: {epoch} Average Validation Loss: {avg_val_loss:.6f}')
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)

    # Save reconstructed images for visual inspection
    if epoch % 5 == 0:
        # Number of images to save
        n = min(data.size(0), 8)
        # Convert model probabilities to class indices [N, C, H, W] -> [N, H, W]
        _, predicted = torch.max(recon_batch.data, 1)  # Get the max indices along the channel dimension
        # Map class indices back to grayscale values for visualization
        # Mapping: 0 -> 0 (black), 1 -> 128 (gray), 2 -> 255 (white)
        predicted_images = predicted.float() / 2.0 * 255.0  # Normalize and scale to [0, 255]
        comparison_images = []
        for i in range(n):
            original_img = data[i].float() * 255.0  # Scale back to [0, 255], assume single channel for simplicity
            reconstructed_img = predicted_images[i].unsqueeze(0)  # Already has correct scaling
            comparison_pair = torch.cat((original_img, reconstructed_img), 2)  # Concatenate along width
            comparison_images.append(comparison_pair)

        comparison_grid = torch.cat(comparison_images, 1)
        save_image(comparison_grid.cpu(), os.path.join(args.save_dir, f'reconstruction_epoch_{epoch}.png'), nrow=1)


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validate(epoch)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'vae_{epoch}.ckpt'))

writer.close()
