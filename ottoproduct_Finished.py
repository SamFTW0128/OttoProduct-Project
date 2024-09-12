import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from torch import nn , optim
import torch.nn.functional as F
import time
import pandas as pd


device = torch.device("mps")
Batch_Size = 64
#Train Data Pre-process

train_dataset = np.loadtxt("/Users/samftw0128/Documents/Visual Studio Code/SOFTMAXEXERCISE/train.csv",
                            skiprows=1,
                            delimiter=",", 
                            dtype=str)

out=[]

for i in train_dataset[0:]:
    #Get the i row of data
    tmp=i[:]
    #because class1 has 6postions WTFFFFFFFFFFF this took me bloody 5 hours to figure out why XD,
    #i am so god dam dumb
    tmp[-1] = tmp[-1][6]
    out.append(list(np.float32(np.array(tmp[0:]))))
np.savetxt("/Users/samftw0128/Documents/Visual Studio Code/SOFTMAXEXERCISE/train_test.csv", out, delimiter=",")


#Test data pre-process
test_dataset = np.loadtxt("/Users/samftw0128/Documents/Visual Studio Code/SOFTMAXEXERCISE/test.csv",
                          skiprows=1,
                          delimiter=",",
                          dtype=str)

out=[]

for i in test_dataset[0:]:
    tmp = i[:]
    out.append(list(np.float32(np.array(tmp[0:]))))
xy=np.savetxt("/Users/samftw0128/Documents/Visual Studio Code/SOFTMAXEXERCISE/test_test.csv", out, delimiter=",")

#####################################################################################################

#Loading the processed train_Dataset and test_dataset
xy = np.loadtxt("/Users/samftw0128/Documents/Visual Studio Code/SOFTMAXEXERCISE/train_test.csv", 
                           delimiter=",",
                           dtype=np.float32)
#xy_test = np.loadtxt("/Users/samftw0128/Documents/Visual Studio Code/SOFTMAXEXERCISE/test_test.csv",
                           #delimiter=",",
                           #dtype=np.float32)

class Otto(Dataset):

    def __init__(self):
        self.len = xy.shape[0]
        #Features
        self.x = torch.tensor(xy[:, 1:-1])
        #Targets
        self.y = torch.LongTensor(xy[:, -1])
        

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return len(xy)
    
class Otto_test(Dataset):

    def __init__(self):
        self.len = xy.shape[0]
        #Features
        self.x = torch.tensor(xy[:, 1:-1])
        #Targets
        self.y = torch.LongTensor(xy[:, -1])
        

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return len(xy)
  

train_dataset = Otto()

test_dataset = Otto_test()

train_loader = DataLoader(dataset = train_dataset,
                          batch_size=Batch_Size,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=Batch_Size,
                         shuffle=False,
                         num_workers=4)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(93, 256)
        self.l2 = nn.Linear(256,128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 32)
        self.l5 = nn.Linear(32, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#TRAINING USING ALL DATA
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
#TESTING USING HALF OF TRAINING DATA
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        #sum up batch loss
        test_loss = criterion(output, target)
        test_loss.item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')
                
if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 20):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')