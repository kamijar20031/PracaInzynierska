from model import *
from tqdm import tqdm
from torch.amp import autocast, GradScaler

batch_size = 48
train_workers = 10

testDataset = DocLayNetDataset(COCO_TEST)
test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=train_workers, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLOBasedModel()
model.to(device)
model.eval()

model.load_state_dict(torch.load("model.pt"))

val_loss = 0
batch = 0
with torch.no_grad():
    for images, targets in tqdm(test_loader, desc=f"Test evaluation"):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = lossForYOLO(outputs, targets)
        val_loss += loss
        batch += 1
val_avg = val_loss/batch
print(f"LOSS: {val_loss}, AVG: {val_loss/batch}")