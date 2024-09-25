import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from models.gan_model import Generator, Discriminator

# Hyperparameters
epochs = 10
lr = 0.0002
batch_size = 64
nz = 100  # Size of latent vector (input to the generator)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (assuming your abstract art dataset is in 'data/abstract_art/')
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root='data/abstract_art', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate models
netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr)
optimizerD = optim.Adam(netD.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        real_images = data[0].to(device)
        batch_size = real_images.size(0)

        # Train Discriminator
        netD.zero_grad()
        label = torch.full((batch_size,), 1, device=device)
        output = netD(real_images).view(-1)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        label.fill_(0)
        output = netD(fake_images.detach()).view(-1)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()

        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake_images).view(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}] Loss D: {lossD_real + lossD_fake}, Loss G: {lossG}')
    
    # Save some generated images at the end of each epoch
    vutils.save_image(fake_images, f'outputs/epoch_{epoch}.png', normalize=True)

# Save the model after training
torch.save(netG.state_dict(), 'models/generator.pth')
torch.save(netD.state_dict(), 'models/discriminator.pth')
