import torch
print("torch version:", torch.__version__)
print("cuda in torch:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())