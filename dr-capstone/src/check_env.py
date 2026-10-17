import sys
import torch, torchvision, timm, cv2, albumentations, sklearn, pandas

print("Python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("timm        :", timm.__version__)
print("opencv      :", cv2.__version__)
print("sklearn     :", sklearn.__version__)

if torch.cuda.is_available():
    print("Device      : CUDA ->", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("Device      : Apple MPS")
else:
    print("Device      : CPU only")

x = torch.randn(2, 3, 224, 224)
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
print("Forward pass:", model(x).shape)  # expect torch.Size([2, 5])