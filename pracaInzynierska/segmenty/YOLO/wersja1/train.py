from model import *
from torch.amp import autocast, GradScaler
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

train_epochs=100
learning_rate = 0.0005
patience = 10
batch_size = 48
train_workers = 10

bestLoss = float("inf")
counter = 0
print("Przygotowywanie datasetu...")
# Dataset
trainDataset = DocLayNetDataset(COCO_TRAIN)
valDataset = DocLayNetDataset(COCO_VAL)

print("Przygotowywanie DataLoader...")

train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=True)
val_loader = DataLoader(valDataset, batch_size=batch_size, shuffle=False, num_workers=train_workers, persistent_workers=True)

print("Przygotowywanie modelu...")

model = YOLOBasedModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience//3)
scaler = GradScaler("cuda")
last = 0
print("Rozpoczecie trenowania...")
for epoch in range(train_epochs):
    model.train()
    epoch_loss = 0
    batch = 0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast("cuda"):
            # Funkcja liniowa zwraca: batch, sequence_width, klasy (litery) w formie logitow
            outputs = model(images)
            loss = lossForYOLO(outputs, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        batch += 1
    print(f"EPOCH {epoch+1}: {epoch_loss} AVG: {epoch_loss/batch}")
    model.eval()
    val_loss = 0
    batch = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} validation"):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = lossForYOLO(outputs, targets)
            val_loss += loss
            batch += 1
    val_avg = val_loss/batch
    print(f"LOSS: {val_loss}, AVG: {val_loss/batch}")
    if val_loss < bestLoss: 
        bestLoss = val_loss 
        last = epoch
        counter = 0 
        torch.save(model.state_dict(), "model.pt") 
        print("Saved best model")
    else: 
        counter+=1 
        print(f"No improvement from previous epochs, last: {last+1}")

    print(f"LR: {optimizer.param_groups[0]['lr']}")
    
    if counter >= patience: 
        print("EARLY STOPPING!")
        break
    scheduler.step(val_loss)