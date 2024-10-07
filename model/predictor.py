import torch
import numpy as np
import os
from PIL import Image
from skimage import color

# Define the device for the computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = torch.load('/Users/robin/Documents/Studium/WInfo_Master/Semester_2/ML2/Projekt/model_2.pth')  # replace with the path to your model
model = model.to(device)
model.eval()  # set the model to evaluation mode

# Define the directory containing your grayscale images
gray_img_dir = '/Users/robin/Documents/Studium/WInfo_Master/Semester_2/ML2/Projekt/student_dataset/test_color/images'  # replace with your directory path

# Create an empty list to store the colorized images
colorized_images = []

# Loop over all images in the directory
for img_name in sorted(os.listdir(gray_img_dir)):
    img_path = os.path.join(gray_img_dir, img_name)
    
    # Load the grayscale image
    img_gray = Image.open(img_path).convert('L')
    img_gray = np.array(img_gray.resize((224, 224))) / 255.0  # normalize to [0,1]
    
    # Convert the grayscale image to LAB color space
    img_lab = color.gray2rgb(img_gray)
    img_lab = color.rgb2lab(img_lab)
    img_lab = torch.from_numpy(img_lab).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Separate the L and AB channels
    l = img_lab[:, 0:1, :, :]
    ab = img_lab[:, 1:3, :, :]
    
    # Colorize the image with the model
    ab_pred = model(l)
    img_lab_pred = torch.cat((l, ab_pred), dim=1)
    img_rgb_pred = color.lab2rgb(img_lab_pred.cpu().data.numpy()[0].transpose((1, 2, 0)))
    
    # Append the colorized image to the list
    colorized_images.append(img_rgb_pred)

# Convert the list of colorized images to a numpy array
colorized_images_np = np.stack(colorized_images)

# Ensure that the array has the correct dimensions
assert colorized_images_np.shape == (50, 224, 224, 3)

# Convert the array to uint8
colorized_images_np = (colorized_images_np * 255).astype(np.uint8)

# Save the array to a .npy file
np.save('vorhersage.npy', colorized_images_np)
