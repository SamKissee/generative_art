import torch
from models.gan_model import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Load the trained Generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load('models/generator.pth'))

# Generate and save images
def generate_images(num_images=5):
    noise = torch.randn(num_images, 100, 1, 1, device=device)
    fake_images = netG(noise).detach().cpu()

    for i in range(num_images):
        vutils.save_image(fake_images[i], f'outputs/generated_{i}.png', normalize=True)

    # Display the generated images
    for i in range(num_images):
        img = fake_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1]
        plt.imshow(img)
        plt.show()

generate_images(5)
