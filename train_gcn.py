import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt

# Carregar o snapshot de uma semana específica
ano, semana = 2022, 15  # Exemplo
caminho = f'data/output/snapshots/dataset_{ano}_S{semana}.pt'

dados = torch.load(caminho)
print(f'Dados carregados para {ano}-S{semana}')


# Transformar em objeto PyG
X = torch.tensor(dados['X'], dtype=torch.float)
A = torch.tensor(dados['A'], dtype=torch.float)
y = torch.tensor(dados['y'], dtype=torch.float)

# Converter matriz de adjacência para edge_index (PyG)
edge_index = (A > 0).nonzero(as_tuple=False).t().contiguous()

# Criar objeto Data do PyG
data = Data(x=X, edge_index=edge_index, y=y)

print(data)


# Definir o modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)  # Saída → regressão (previsão de casos)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.lin(x)

        return x.squeeze()  # Saída 1D


# Inicializar modelo
model = GCN(num_features=X.shape[1], hidden_channels=32)

device = torch.device('cpu')  # ⚙️ CPU
model = model.to(device)
data = data.to(device)
y = y.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.MSELoss()

# Loop de treino
print('Treinando...')

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = loss_fn(out, y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        print(f'Epoch {epoch} | Loss: {loss.item():.4f}')

print('Treinamento concluído.')


# Avaliação
model.eval()
with torch.no_grad():
    pred = model(data)

mae = F.l1_loss(pred, y).item()
rmse = torch.sqrt(F.mse_loss(pred, y)).item()

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')


# Visualizar previsões

plt.figure(figsize=(8, 6))
plt.scatter(y.cpu().numpy(), pred.cpu().numpy())
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title(f'Previsão de Malária - {ano}-S{semana}')
plt.grid()
plt.show()
