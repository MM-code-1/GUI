import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################
def select_file():
    global selected_file_path
    file_path = filedialog.askopenfilename(title="Select a 3dv file with a size of 36,600 KB", filetypes=[("All Files", "*.*")])
    if file_path:
        validate_file(file_path)
        selected_file_path = Path(file_path)
#########################################################
def validate_file(file_path):
    file_size_kb = os.path.getsize(file_path) / 1024
    if not file_path.endswith('.3dv') or file_size_kb != 36600:
        messagebox.showerror("Invalid file", "The selected file is not valid. The file must be a 3dv file and have a size of 36,600 KB")
    else:
        messagebox.showinfo("Valid file", "The selected file is valid")
#########################################################
def show_oct_images():
    global selected_file_path
    # Check if a 3dv file has been selected
    if not selected_file_path:
        messagebox.showerror("Error", "Select a 3dv file first")
        return

    # Obtain images from the selected 3dv file
    x_res = 800         # AB-Scan
    y_res = 16          # BC-Scan: number of images
    z_res = 1464        # Depth
   
    data = selected_file_path.read_bytes()

    # Reshape the data to match the image dimensions
    img_data = np.frombuffer(data, dtype=np.uint16).reshape(y_res, z_res, x_res)

    # Plot the image 
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(img_data[i * 4 + j], cmap='gray', aspect='auto')
            axs[i, j].axis('off')
    
    plt.show()

###########################################################################################################################

# Function to save images
def save_images(img_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(16):
        img_path = os.path.join(save_dir, f"image_{i}.png")
        plt.imsave(img_path, img_data[i], cmap='gray')

###############################################################################################################################

def preprocess_images(img_dir):
    num_images_per_exam = 16
    exam_images = []

    for i in range(num_images_per_exam):
        img_path = os.path.join(img_dir, f"image_{i}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        crop_bottom = int(0.6 * height)
        cropped_img = img[:-crop_bottom, int(0.25 * width):width - int(0.25 * width)]
        resized_img = cv2.resize(cropped_img, (224, 224))
        exam_images.append(resized_img)

    stacked_exam_images = np.stack(exam_images, axis=0)
    return stacked_exam_images


###############################################################################################################################

class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()

        # Load the EfficientNet-B0 model without pretrained weights
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        
        # Modify the first convolutional layer to accept 16-channel input
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=16, 
            out_channels=self.efficientnet.features[0][0].out_channels,
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding,
            bias=self.efficientnet.features[0][0].bias
        )
        
        # Get the number of features for the classifier layer
        num_features = self.efficientnet.classifier[1].in_features

        # Modify the classifier layer for a single output
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 1)
        )

        # Additional fully connected layer to process the combined features
        self.final_fc = nn.Linear(3, 1)  # 1 feature from EfficientNet + 2 for the eye parameters

    def forward(self, x, eye):
        batch_size = x.size(0)

        # Process the image through the EfficientNet model
        x = self.efficientnet(x)

        # Ensure the eye tensor is the right shape
        eye = eye.view(batch_size, -1)  # Reshape to [batch_size, 2]

        # Concatenate the eye parameter with the output from EfficientNet
        combined = torch.cat((x, eye), dim=1)  # Shape: [batch_size, 3]

        # Process the combined features through the final fully connected layer
        out = self.final_fc(combined)

        return out

###############################################################################################################################

# Function to predict corneal ectasia risk
def predict_esi():
    global selected_file_path

    # Check if a .3dv file has been selected
    if not selected_file_path:
        messagebox.showerror("Error", "Please select a 3dv file first")
        return

    # Ask the user to press 'r' for right eye or 'l' for left eye
    eye_input = simpledialog.askstring("Eye", "Enter r for right eye or l for left eye")
    if eye_input and eye_input.lower() in ['r', 'l']:
        eye_tensor = torch.tensor([1, 0], dtype=torch.float32) if eye_input.lower() == 'r' else torch.tensor([0, 1], dtype=torch.float32)
    else:
        messagebox.showerror("Error", "Invalid input. Enter r for right eye or l for left eye")
        return

    # Obtain images from the selected 3dv file
    x_res = 800         # AB-Scan
    y_res = 16          # BC-Scan: number of images
    z_res = 1464        # Depth
    data = selected_file_path.read_bytes()

    # Reshape the data to match the image dimensions
    img_data = np.frombuffer(data, dtype=np.uint16).reshape(y_res, z_res, x_res)
    
    # Ask user where to save the images
    save_dir = filedialog.askdirectory(title="Select a directory to save the images extracted from the selected 3dv file")
    if not save_dir:
        messagebox.showerror("Error", "Select a directory to save the images extracted from the selected 3dv file")
        return

    # Save the images
    save_images(img_data, save_dir)
    
    # Preprocess the images
    processed_images = preprocess_images(save_dir)

    # Load the saved model 
    model_path = r'C:\Maziar Mirsalehi\Python\EfficientNetB0 model\EfficientNetB0_model_14.01.2025.pth'
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

   
    # Add batch dimension
    processed_images = torch.tensor(processed_images, dtype=torch.float32).to(device)                                     
    processed_images = processed_images.unsqueeze(0)     # Add batch dimension                                                                
    eye_tensor = eye_tensor.to(device)

    # Predict
    with torch.no_grad():
        output = model(processed_images, eye_tensor)
        prediction = output.item()

    # Display prediction, classification, threshold and information
    if prediction >= 32.12:
        messagebox.showinfo("Predicted corneal ectasia and diagnosis", f"Score: {prediction:.2f}\n\nDiagnosis: Ectasia\n\nThreshold: 32.00\n\nInformation: Scores at or above the threshold indicate ectasia, while scores below the threshold suggest either suspicion of ectasia or no detectable ectasia")
    else:
        messagebox.showinfo("Predicted corneal ectasia and diagnosis", f"Score: {prediction:.2f}\n\nDiagnosis: Suspicion of ectasia or no ectasia\n\\Threshold: 32.00\n\nInformation: Scores at or above the threshold indicate ectasia, while scores below the threshold suggest either suspicion of ectasia or no detectable ectasia")

###########################################################################################################################
# Setup GUI
root = tk.Tk()
root.title("Corneal Ectasia Risk Predictor")
root.geometry("600x500")
root.configure(bg="#f0f4f7")  

# Create a frame for centering content
center_frame = tk.Frame(root, bg="#f0f4f7")
center_frame.pack(expand=True, pady=50)

# Add text
text_label = tk.Label(center_frame, text="Corneal Ectasia Risk Predictor", font=("Helvetica Neue", 62, "bold"), fg="#333333", bg="#f0f4f7")
text_label.grid(row=0, column=0, columnspan=3, padx=20, pady=100)


# Create and place rectangular buttons
button_font = ("Helvetica Neue", 35, "bold")
button_fg = "#ffffff"  # White text color
button_width = 12
button_height = 3
button_padx = 5
button_pady = 10

# Place the three buttons on the same row
select_btn = tk.Button(center_frame, text="File", command=select_file, bg="blue", fg=button_fg, font=button_font, width=button_width, height=button_height)
select_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

oct_btn = tk.Button(center_frame, text="Images", command=show_oct_images, bg="green", fg=button_fg, font=button_font, width=button_width, height=button_height)
oct_btn.grid(row=1, column=1, padx=button_padx, pady=button_pady)

predict_btn = tk.Button(center_frame, text="Prediction", command=predict_esi, bg="red", fg=button_fg, font=button_font, width=button_width, height=button_height)
predict_btn.grid(row=1, column=2, padx=button_padx, pady=button_pady)


root.mainloop()