
%pylab inline
import torch
%load_ext tensorboard
import torch.utils.tensorboard as tb
import tempfile
log_dir=tempfile.mkdtemp()
%tensorboard --logdir {log_dir} --reload_interval 1
import os
from torchvision.utils import make_grid

# Import necessary packages
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

# Define the Generator and Discriminator networks
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Create a directory for TensorBoard logs
# log_dir = "runs/gan_mnist"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a SummaryWriter
writer = SummaryWriter(log_dir)


#Set hyperparameters and instantiate the models
batch_size = 100
epochs = 150
lr = 0.0003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device----" )
print( device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
loss_function = nn.BCELoss()

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Define the training loop
def train_discriminator(real_images, fake_images):
    d_optimizer.zero_grad()

    real_outputs = discriminator(real_images)
    real_loss = loss_function(real_outputs, torch.ones(real_images.size(0), 1).to(device))
    real_loss.backward()

    fake_outputs = discriminator(fake_images)
    fake_loss = loss_function(fake_outputs, torch.zeros(fake_images.size(0), 1).to(device))
    fake_loss.backward()

    d_optimizer.step()

    return real_loss + fake_loss

def train_generator(fake_images):
    g_optimizer.zero_grad()

    outputs = discriminator(fake_images)
    loss = loss_function(outputs, torch.ones(fake_images.size(0), 1).to(device))
    loss.backward()

    g_optimizer.step()

    return loss

# Train the GAN
for epoch in range(epochs):
    for batch_idx, (images, _) in enumerate(train_loader):
              # Train the Discriminator
        real_images = images.view(images.size(0), -1).to(device)
        noise = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(noise)
        d_loss = train_discriminator(real_images, fake_images.detach())

        # Train the Generator
        noise = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(noise)
        g_loss = train_generator(fake_images)

         # Write losses to TensorBoard
        writer.add_scalar("D-Loss", d_loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar("G-Loss", g_loss.item(), epoch * len(train_loader) + batch_idx)


        # Add generated images to TensorBoard every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch: {batch_idx}, D-Loss: {d_loss.item()}, G-Loss: {g_loss.item()}")          
            with torch.no_grad():
                noise = torch.randn(64, 100).to(device)
                generated_images = generator(noise).view(-1, 1, 28, 28)
                img_grid = make_grid(generated_images, normalize=True)
                writer.add_image("Generated Images", img_grid, epoch * len(train_loader) + batch_idx)

    # Save the model weights
    if (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

# Close the SummaryWriter
writer.close()
print("GAN Training Completed!")

       
