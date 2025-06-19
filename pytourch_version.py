import torch
import torch_geometric

print("Torch:", torch.__version__)
print("PyTorch Geometric:", torch_geometric.__version__)
print("CUDA disponível:", torch.cuda.is_available())
