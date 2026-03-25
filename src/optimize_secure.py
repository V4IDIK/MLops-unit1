import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
import os

# 1. Define a Simple Neural Network for MNIST
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Fast Gradient Sign Method (FGSM) Adversarial Attack
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def main():
    print("Downloading MNIST Dataset and initializing model...")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- PART A: FAST TRAINING ---
    print("\n--- Training Model (1 Epoch) ---")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == 100: # Cut short for speed
            break 
    print("Training complete.")

    # --- PART B: OPTIMIZATION (Pruning & Quantization) ---
    print("\n--- Optimizing Model ---")
    
    # 1. Pruning: Remove 30% of the weights in the first layer to make it lighter
    prune.random_unstructured(model.fc1, name="weight", amount=0.3)

    prune.remove(model.fc1, 'weight')
    
    print("Pruning: Removed 30% of connections in the first layer.")

    # 2. Quantization: Convert 32-bit floats to 8-bit integers (massive size reduction)
    model.eval()

    torch.backends.quantized.engine = 'qnnpack'
    
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/base_model.pth')
    torch.save(quantized_model.state_dict(), 'models/quantized_model.pth')
    
    base_size = os.path.getsize('models/base_model.pth') / 1024
    quant_size = os.path.getsize('models/quantized_model.pth') / 1024
    print(f"Quantization: Reduced model size from {base_size:.0f} KB to {quant_size:.0f} KB")

    # --- PART C: SECURITY (Adversarial Attack) ---
    print("\n--- Testing Security Robustness (FGSM Attack) ---")
    model.eval()
    correct_clean = 0
    correct_hacked = 0
    epsilon = 0.25 # Strength of the attack (invisible noise)

    for data, target in test_loader:
        data.requires_grad = True
        
        # Test 1: Clean Data
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() == target.item():
            correct_clean += 1
            
            # Test 2: Generate Hack (FGSM)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            
            # Predict on the hacked image
            hacked_output = model(perturbed_data)
            hacked_pred = hacked_output.max(1, keepdim=True)[1]
            if hacked_pred.item() == target.item():
                correct_hacked += 1
                
        # Only test 1000 images for speed
        if correct_clean >= 1000:
            break

    print(f"Accuracy on Clean Images:   {(correct_clean/1000)*100:.1f}%")
    print(f"Accuracy on Hacked Images:  {(correct_hacked/1000)*100:.1f}%")
    print("\nNotice how the invisible noise completely fools the network!")

if __name__ == "__main__":
    main()