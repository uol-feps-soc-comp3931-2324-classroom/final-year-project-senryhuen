import time
import torch
from torch.utils.data import random_split, DataLoader

from spectrograminversion import DNN
from dataset import DNNDataset


# load model
model = DNN()
model.train()

device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("no device found, falling back to CPU")

if device:
    model.to(device)

# load/split dataset
data = DNNDataset()
trainset, testset = random_split(data, [0.9, 0.1])
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
# TODO: split validation set?

# training params
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 10**-0.5)

# TODO: load saved model weights

# training loop
best_loss = None
num_batches = len(train_loader)
num_epochs = 50
for epoch in range(num_epochs):
    for it, (inputdata, target) in enumerate(train_loader):
        start = time.time()
        print(f"Iteration: {it+1} out of {num_batches}")

        # check target doesn't contain any NaNs
        if torch.isnan(target).any().item():
            print("target is NaN\n")
            continue

        # move inputdata and target to device
        if device:
            inputdata = inputdata.to(device)
            target = target.to(device)

        optimizer.zero_grad()

        # get prediction for inputdata
        y_pred = model(inputdata)

        if torch.isnan(y_pred).any().item():
            print("y_pred is NaN\n")
            continue

        # compute/print loss between prediction and target
        loss = criterion(y_pred, target)
        loss_val = loss.item()
        print(f"loss: {loss_val}")

        print("updating weights...")

        # backward pass, update weights
        loss.backward()
        optimizer.step()

        # save model if lower loss
        if not best_loss or loss_val < best_loss:
            best_loss = loss_val
            print("saving model weights...")
            torch.save(model.state_dict(), "spectrograminversion/degli_dnn_state.pt")

        end = time.time()
        print(f"done in {end-start}s\n")

    scheduler.step()
