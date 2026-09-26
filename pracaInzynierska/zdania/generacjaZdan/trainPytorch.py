from modelPytorch import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

DATASET_NAME = "temp"
height = 32
width = 512
train_epochs=100
learning_rate = 0.0005
patience = 20
split = 0.9
batch_size = 16
train_workers = 10

best_cer = float("inf")
counter = 0
print("Przygotowywanie datasetu...")
# Dataset
dataset = OCRDataset(DATASET_NAME, width, height, mytransforms = [randomRotate, randomSharpen, randomBrightness])
train_dataset, val_dataset = dataset.split(split)
print("Przygotowywanie DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_workers, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=train_workers, collate_fn=collate_fn, persistent_workers=True)
num_classes = len(dataset.char_to_idx) + 1 
print("Zapisywanie w pickle...")
pickle.dump(dataset.char_to_idx, open('vocab.pkl', 'wb'))

print("Przygotowywanie modelu...")
# Model
model = OCRModel(num_classes=num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
lossFn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=patience/2)
scaler = GradScaler("cuda")

print("Rozpoczecie trenowania...")
for epoch in range(train_epochs):
    model.train()
    epoch_loss = 0
    for images, labels, label_lengths, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda"):
            # Funkcja liniowa zwraca: batch, sequence_width, klasy (litery) w formie logitow
            outputs = model(images)
            input_lengths = torch.full(
                size=(outputs.size(0),),
                fill_value=outputs.size(1),
                dtype=torch.long,
                device=device
            )
            # CTCLoss musi przyjac Sequence_width, batch, klasy w formie logarytmow prawdopodobienstw
            loss = lossFn(outputs.log_softmax(2).permute(1,0,2), labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    print(f"EPOCH {epoch+1}: {epoch_loss}")
    model.eval()
    val_cer = 0
    batches = 0
    with torch.no_grad():
        for images, labels, label_lengths, o_labels in tqdm(val_loader, desc=f"Epoch {epoch+1} validation"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            input_lengths = torch.full(
                size=(outputs.size(0),),
                fill_value=outputs.size(1),
                dtype=torch.long,
                device=device,
            )
            loss = lossFn(outputs.log_softmax(2).permute(1,0,2), labels, input_lengths, label_lengths)
            batch_cer = calculate_cer(outputs, o_labels, dataset.idx_to_char)
            val_cer += batch_cer
            batches +=1
    val_cer /= batches
    print(f"CER: {val_cer}")
    if val_cer < best_cer: 
        best_cer = val_cer 
        counter = 0 
        torch.save(model.state_dict(), "model.pt") 
        print("Saved best model")
    else: 
        counter+=1 
        print("No improvement from previous epochs")
    if counter >= patience: 
        print("EARLY STOPPING!") 
        break
    scheduler.step(val_cer)