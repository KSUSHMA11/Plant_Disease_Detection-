import torch
import timm
import os

def create_dummies():
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Swin Transformer
    print("Creating dummy Swin Transformer model...")
    model_swin = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=38)
    save_path_swin = "checkpoints/best_model_swin.pth"
    torch.save(model_swin.state_dict(), save_path_swin)
    print(f"Saved dummy Swin model to {save_path_swin}")

    # 2. Vision Transformer (ViT)
    print("Creating dummy ViT model...")
    model_vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=38)
    save_path_vit = "checkpoints/best_model_vit.pth"
    torch.save(model_vit.state_dict(), save_path_vit)
    print(f"Saved dummy ViT model to {save_path_vit}")

if __name__ == "__main__":
    create_dummies()
