import time
import torch
from torch.utils.data import random_split, DataLoader

from spectrograminversion import DNN
from dataset import DNNDataset


# load model
model = DNN()
mps_device = torch.device("mps")
model.to(mps_device)

# load/split dataset
data = DNNDataset()
trainset, testset = random_split(data, [0.9, 0.1])
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
# TODO: split validation set?

# training params
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# training loop
# TODO: multiple epochs?
# TODO: load saved model weights
best_loss = None
num_batches = len(train_loader)
for it in range(num_batches):
    start = time.time()
    print(f"Iteration: {it+1} out of {num_batches}")

    # load inputdata and target
    inputdata, target = next(iter(train_loader))
    inputdata = inputdata.to(mps_device)
    target = target.to(mps_device)

    # get prediction for inputdata
    y_pred = model(inputdata)

    # print loss between prediction and target
    loss = criterion(y_pred, target)
    loss_val = loss.item()
    print(f"loss: {loss_val}")

    print("updating weights...")

    # zero gradients, backward pass, update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save model if lower loss
    if not best_loss or loss_val < best_loss:
        best_loss = loss_val
        print("saving model weights...")
        torch.save(model.state_dict(), "spectrograminversion/degli_dnn_state.pt")

    end = time.time()
    print(f"done in {end-start}s\n")
