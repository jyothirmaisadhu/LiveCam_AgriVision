
import time
import torch
import torch.nn as nn
from torchvision import models

start = time.time()

def status(message):
    elapsed = time.time() - start
    print(f"[{elapsed:6.1f}s] {message}", flush=True)

try:
    status("Step 1/5: Checking CPU device...")
    device = torch.device("cpu")
    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print("Torchvision:", __import__("torchvision").__version__)

    status("Step 2/5: Loading MobileNetV3-Small...")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 10
    )
    model = model.to(device)
    print("Model ready for 10 output classes.")

    status("Step 3/5: Creating a small test batch...")
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1], dtype=torch.long)
    print("Batch shape:", tuple(images.shape))

    status("Step 4/5: Running forward pass and calculating loss...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    outputs = model(images)
    loss = criterion(outputs, labels)
    print("Output shape:", tuple(outputs.shape))
    print("Test loss:", round(loss.item(), 4))

    status("Step 5/5: Running one training step...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    status("SUCCESS: Model forward and backward passes completed.")
    print(f"Total elapsed time: {time.time() - start:.1f}s")

except Exception as error:
    status(f"FAILED: {type(error).__name__}: {error}")
    raise
