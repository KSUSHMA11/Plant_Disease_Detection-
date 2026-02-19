
import torch
import timm
import os

def create_dummy():
    print("Creating dummy Swin Transformer model...")
    # Initialize model with random weights
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=38)
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/best_model_swin.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved dummy model to {save_path}")

if __name__ == "__main__":
    create_dummy()
