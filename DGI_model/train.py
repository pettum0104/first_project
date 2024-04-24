from encoder import *
import gc
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
import scipy
import os

DATASET_PATH = 'D:/matrix150'
HIDDEN_CHANNELS = 128

def train(model, dataset, epochs, optimizer, device):
    sampler = SubsetRandomSampler(range(len(dataset)))
    for epoch in range(epochs):
        ave_loss = 0
        for i in sampler:
            data = dataset[i]
            model.train()
            x_gpu = data.X.to(device)
            a_gpu = data.A.to(device)
            pos_z, neg_z, summary = model(x_gpu, a_gpu)
            loss = model.loss(pos_z, neg_z, summary)
            loss_value = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            del pos_z, neg_z, summary, x_gpu, a_gpu
            gc.collect()

            ave_loss += loss_value

        print(f'\nEpoch {epoch} loss: {ave_loss/len(dataset)}\n')


class LargeDatasetB(Dataset):
    def __init__(self, file_path, N):
        self.file_path = file_path
        self.N = N

    def __getitem__(self, index):
        index = 150 * (index//self.N) + index%self.N
        l = os.listdir(self.file_path)
        A_name = l[2*index]
        X_name = l[2*index + 1]

        sparse_matrix = scipy.sparse.load_npz(self.file_path+'/'+A_name)
        row, col = sparse_matrix.nonzero()
        A = torch.tensor(np.column_stack((row, col)).T)

        X = np.load(self.file_path+'/'+X_name)
        X = torch.tensor(X, dtype=torch.float32)
        X_enc = F.one_hot(X[:,0].type(torch.int64))
        X = torch.cat((X_enc, X[:,1][:,None]), dim=-1)
            
        return Data(A=A, X=X, A_name=A_name)

    def __len__(self):
        return self.N*os.listdir(self.file_path)/150
    
    
def main():
    dataset = LargeDatasetB(DATASET_PATH, 150)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        
    def corruption(x, edge_index):
        z = dataset[random.randint(0, len(dataset)-1)]
        return z.X.to(device), z.A.to(device)
    
    n = HIDDEN_CHANNELS
    encoder = Encoder(input_size = 4, hidden_channels=n, conv=GCNConv, n_output=n*2)

    model = DeepGraphInfomax(
        hidden_channels=n*2, encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    )
    model.to(device)
    epochs = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    train(model, dataset, epochs, optimizer, device)
