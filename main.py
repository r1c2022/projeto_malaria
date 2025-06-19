from src.pipelines.pipeline_malaria import tratar_dados_malaria
from src.pipelines.pipeline_clima import carregar_e_tratar_clima
from src.pipelines.pipeline_grafo import carregar_grafo
from src.pipelines.tabela_ibge import carregar_tabela_ibge, mapear_nome_para_codigo_com_ibge
from src.pipelines.dataset_gnn import construir_dataset_grafo
from src.pipelines.logger_config import configurar_logger
from src.exporter.exportador import exportar_dataset_por_semana

# Inicializar Logger
logger = configurar_logger()
logger.info('Pipeline iniciado')

# Paths dos arquivos
path_malaria = 'data/malaria/2020-2023.csv'
path_clima = 'data/clima/clima_semanal.csv'
path_grafo = 'data/grafo/grafo.GraphML'
path_tabela_ibge = 'data/ibge/ibge_municipios.csv'


# Processamento dos Dados de Malária
logger.info('Processando dados de malária...')
malaria, dados_incompletos = tratar_dados_malaria(path_malaria)
logger.info(f'Dados de malária processados: {malaria.shape[0]} registros completos.')

if not dados_incompletos.empty:
    logger.warning(f'Existem {dados_incompletos.shape[0]} registros de malária incompletos.')


# Processamento dos Dados Climáticos
logger.info('Carregando e processando dados climáticos...')
clima = carregar_e_tratar_clima(path_clima)


# Carregar Tabela IBGE
logger.info('Carregando tabela IBGE...')
tabela_ibge = carregar_tabela_ibge(path_tabela_ibge)


# Mapear Clima → Cod_Municipio
logger.info('Realizando mapeamento dos dados climáticos para Cod_Municipio...')
clima_mapeado = mapear_nome_para_codigo_com_ibge(clima, tabela_ibge)

if clima_mapeado['Cod_Municipio'].isna().sum() > 0:
    logger.warning(f'Existem {clima_mapeado["Cod_Municipio"].isna().sum()} registros de clima sem Cod_Municipio.')


# Carregar Grafo de Mobilidade
logger.info('Carregando grafo de mobilidade...')
grafo = carregar_grafo(path_grafo)


# Construir Dataset para GNN
logger.info('Construindo dataset integrado com grafo...')
dataset = construir_dataset_grafo(grafo, malaria, clima_mapeado)
logger.info(f'Dataset para GNN construído com {len(dataset)} semanas processadas.')

# Finalização
logger.info('Pipeline finalizado com sucesso.')

# Exportar o dataset para .pt
#exportar_dataset_gnn_para_pt(dataset, 'data/output/dataset_gnn.pt')
exportar_dataset_por_semana(dataset, pasta_saida='data/output/snapshots')

