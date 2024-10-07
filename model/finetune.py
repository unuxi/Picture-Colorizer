
from colorizers import *

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import torch.optim as optim
from skimage import io

from torchvision.transforms import Resize

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            Resize((256, 256)),  # resize all images to 256x256
            transforms.ToTensor()
        ])
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')  # convert image to RGB
        image = self.transform(image)
    
        # Permute the image tensor to (Height, Width, Channels) before converting to LAB color space
        image = image.permute(1, 2, 0)
        image = color.rgb2lab(image)  # convert image to LAB color space
    
    # Permute the image tensor back to (Channels, Height, Width) before returning
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image



# Create the dataset and dataloader
dataset = ImageDataset('/Users/robin/Documents/Studium/WInfo_Master/Semester_2/ML2/Projekt/student_dataset/train/images/', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load the pre-trained model
model = siggraph17(pretrained=True)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10  # specify the number of epochs for training
for epoch in range(num_epochs):
    for i, images in enumerate(dataloader):
        # Move tensors to the configured device
        images = images.to(device)
        l, ab = images[:, 0, :, :].unsqueeze(1), images[:, 1:, :, :]

        # Forward pass
        outputs = model(l)
        loss = criterion(outputs, ab)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
            
    torch.save(model, f'model_{epoch+1}.pth')


