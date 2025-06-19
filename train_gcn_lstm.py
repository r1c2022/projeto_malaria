from src.models.gcn_lstm import GCN_LSTM
from src.loaders.loader_snapshots import carregar_sequencia

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


# Definir sequência temporal
sequencia = [(2022, 10), (2022, 11), (2022, 12), (2022, 13), (2022, 14), (2022, 15)]

# Carregar dados
data_seq, y_real = carregar_sequencia(sequencia)

# Inicializar modelo
num_features = data_seq[0].num_node_features

model = GCN_LSTM(
    num_features=num_features,
    gcn_hidden=32,
    lstm_hidden=64,
    lstm_layers=1
)

device = torch.device('cpu')
model = model.to(device)
y_real = y_real.to(device)
data_seq = [d.to(device) for d in data_seq]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.MSELoss()

# Loop de treino
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()

    out = model(data_seq)
    loss = loss_fn(out, y_real)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        print(f'Epoch {epoch} | Loss: {loss.item():.4f}')

print('Treinamento concluído.')

# Avaliação
model.eval()
with torch.no_grad():
    pred = model(data_seq)

mae = F.l1_loss(pred, y_real).item()
rmse = torch.sqrt(F.mse_loss(pred, y_real)).item()

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# Visualizar previsões
plt.figure(figsize=(8, 6))
plt.scatter(y_real.cpu().numpy(), pred.cpu().numpy())
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title(f'Previsão de Malária - Sequência {sequencia}')
plt.grid()
plt.show()

# Salvar modelo
os.makedirs('data/output/models', exist_ok=True)
torch.save(model.state_dict(), 'data/output/models/gcn_lstm_model.pt')
print('Modelo salvo em data/output/models/gcn_lstm_model.pt')
