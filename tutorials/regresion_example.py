import torch
from torch import nn
import matplotlib.pyplot as plt
import os
try:
    import torch_npu
except:
    pass
os.environ['KMP_DUPLICATE_LIB_OK']='True'


try:
    device = "npu:5" if torch.npu.is_available() else "cpu"
except:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Create weight and bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.0000002

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will happen later on (shapes within linear layers)
y = weight * X + bias 

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


torch.manual_seed(42)
model_0 =nn.Sequential(
    nn.Linear(in_features=1, out_features=50),
    nn.Linear(in_features=50, out_features=1)
).to(device)

loss_fn = nn.L1Loss()


optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) 
epochs = 100000

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.no_grad():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
        print(f'loss:',loss.detach().cpu(),  'test_loss: ',test_loss.detach().cpu()) 
