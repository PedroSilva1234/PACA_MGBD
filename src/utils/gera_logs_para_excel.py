import os
import json
import logging
import subprocess
from numpy import power
import pandas as pd

def rodar_prof5fake(caminho_prof5fake, caminho_log):
    # Verifica se o arquivo .log realmente existe antes de rodar
    if not os.path.exists(caminho_log):
        logging.error(f"[SGD Metric] Arquivo de log não encontrado: {caminho_log}")
        return
    
    # Monta o comando com o arquivo .log específico, conforme exigido pelo seu prof5fake
    comando = ['python3', caminho_prof5fake, caminho_log, "--modelo", "APPROX.json"]

    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True)
        logging.info(f"[SGD Metric] Sucesso! Saída:\n{resultado.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"[SGD Metric] Erro ao executar prof5fake.py:\n{e.stderr}")
    except FileNotFoundError:
        logging.error("❌ Comando 'python3' não encontrado. Tente trocar para 'python'.")

def compilar_logs_para_excel(diretorio_logs, caminho_saida):
    """Lê todos os arquivos .json do diretório e compila as métricas em uma planilha Excel."""
    dados = []
    
    logging.info("\n[Excel] Iniciando a compilação dos arquivos JSON gerados...")
    
    for nome_arquivo in os.listdir(diretorio_logs):
        if nome_arquivo.endswith('.json'):
            caminho_arquivo = os.path.join(diretorio_logs, nome_arquivo)
            
            try:
                with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                    log = json.load(f)
                    summary = log.get('summary', {})
                    
                    # Extrai os valores usando as chaves do log
                    linha = {
                        'Arquivo': nome_arquivo,
                        'IPC': summary.get('IPC'),
                        'Total Inst': summary.get('Total_Inst_Log'), 
                        'cycles': summary.get('cycles'),
                        'avg_power_cycle': summary.get('avg_power_cycle'),
                        'total_energy': summary.get('energy_total'),
                        'power': summary.get('power')    
                    }
                    dados.append(linha)
            except json.JSONDecodeError:
                logging.error(f"[Excel] Erro ao ler ou decodificar o arquivo: {nome_arquivo}")
            except Exception as e:
                logging.error(f"[Excel] Erro inesperado no arquivo {nome_arquivo}: {e}")

    # Se houver dados processados, gera o arquivo Excel
    if dados:
        df = pd.DataFrame(dados)
        df.to_excel(caminho_saida, index=False)
        logging.info(f"✅ Sucesso! Planilha '{caminho_saida}' gerada com {len(dados)} logs computados.")
    else:
        logging.warning("⚠️ Nenhum arquivo .json válido foi encontrado para compilar.")

if __name__ == "__main__":
    # Configuração básica de logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Diretórios base
    diretorio_outputs = r"C:\Programas\LSCAD\IC_Dani\Docker_Gregorio\ILAC\storage\executions\sgd_forcabruta_20260716_144301\outputs"
    diretorio_logs = r"C:\Programas\LSCAD\IC_Dani\Docker_Gregorio\ILAC\storage\executions\sgd_forcabruta_20260716_144301\logs"
    CAMINHO_PROF5FAKE = r'C:\Programas\LSCAD\IC_Dani\Docker_Gregorio\ILAC\src\utils\prof5fake.py'

    # Percorre as pastas
    for raiz, diretorios, arquivos in os.walk(diretorio_outputs):
        # 1º Correção: Fazer um loop dentro da lista de arquivos
        for nome_arquivo in arquivos:
            
            if nome_arquivo.endswith('.error'):
                caminho_completo = os.path.join(raiz, nome_arquivo)
                
                try:
                    with open(caminho_completo, 'r', encoding='utf-8') as arquivo:
                        # Pega o texto e remove espaços em branco/quebras de linha
                        conteudo_texto = arquivo.read().strip()
                        
                        # 2º Correção: Tenta converter o texto para número decimal
                        valor_erro = float(conteudo_texto)
                        
                        # 3º Correção: Agora sim podemos comparar números!
                        if valor_erro <= 5:
                            logging.info(f"\n[SGD Metric] Processando erro de {valor_erro}: {nome_arquivo}")
                            
                            # 4º Correção: Descobrir o nome do log a partir do nome do .error
                            # Tira '.data.error' e coloca '.log'
                            nome_base_log = nome_arquivo.replace('.data.error', '.log')
                            caminho_do_log = os.path.join(diretorio_logs, nome_base_log)
                            
                            # Chama a função passando o log, e não a pasta!
                            rodar_prof5fake(CAMINHO_PROF5FAKE, caminho_do_log)

                except UnicodeDecodeError:
                    logging.warning(f"Aviso: Problema de codificação ao ler {caminho_completo}.")
                except ValueError:
                    # Esse erro acontece se tentar fazer float("EXPLODIU") ou float("NaN")
                    logging.warning(f"Ignorado (Não numérico): {nome_arquivo}")
    
    # -------------------------------------------------------------------------
    # NOVO PASSO: Executa a compilação do Excel no final do processamento
    # -------------------------------------------------------------------------
    # Define onde a planilha será salva. Neste caso, dentro da própria pasta de logs.
    caminho_planilha = os.path.join(diretorio_logs, "resultados_logs.xlsx")
    
    # Chama a função de compilação
    compilar_logs_para_excel(diretorio_logs, caminho_planilha)