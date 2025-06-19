from src.models.gcn_lstm import GCN_LSTM
from src.loaders.loader_forecasting import carregar_sequencia_para_previsao

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


# Definir sequência temporal e target
#equencia_input = [(2022, 10), (2022, 11), (2022, 12), (2022, 13), (2022, 14)]
#semana_target = (2022, 15) 


# Sequência de entrada — todas as semanas de 2020 e 2021
sequencia_input = [(2020, s) for s in range(1, 54)] + [(2021, s) for s in range(1, 54)]
semana_target = (2022, 1)



# Carregar dados
data_seq, y_real = carregar_sequencia_para_previsao(sequencia_input, semana_target)

print(f'Dados carregados: Sequência {sequencia_input} → Prevendo {semana_target}')



#Inicializar modelo
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


#Loop de treino
print('Iniciando treinamento para previsão da semana seguinte...')

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

print('====================')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print('====================')


# Visualizar previsões
plt.figure(figsize=(8, 6))
plt.scatter(y_real.cpu().numpy(), pred.cpu().numpy())
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title(f'Previsão de Malária - Input {sequencia_input} → Prev {semana_target}')
plt.grid()
plt.show()


# Salvar modelo
os.makedirs('data/output/models', exist_ok=True)
nome_modelo = f'gcn_lstm_forecast_{semana_target[0]}_S{semana_target[1]}.pt'
torch.save(model.state_dict(), f'data/output/models/{nome_modelo}')
print(f'Modelo salvo em data/output/models/{nome_modelo}')



from src.visualizador.visualizador import plotar_erro_temporal
from src.visualizador.visualizador import plotar_grafo_heatmap
from src.visualizador.visualizador import gerar_animacao_grafo


# Plotar grafo com casos de malária na semana

# Grafo com intensidade dos casos
plotar_grafo_heatmap(2022, 15)  

# Plotar grafo com erro da previsão
plotar_grafo_heatmap(2022, 15, y_real=y_real, y_pred=pred)

# Gerar animação da evolução do grafo
semanas = [(2022, s) for s in range(10, 16)]
gerar_animacao_grafo(semanas, nome_arquivo='grafo_evolucao.gif')

# Plotar erro ao longo do tempo (exemplo manual)
lista_semanas = ['2022-S10', '2022-S11', '2022-S12', '2022-S13', '2022-S14', '2022-S15']
lista_mae = [10.2, 8.5, 9.3, 7.8, 6.5, 6.0]
lista_rmse = [12.4, 10.7, 11.5, 9.6, 8.2, 7.9]

plotar_erro_temporal(lista_semanas, lista_mae, lista_rmse)
