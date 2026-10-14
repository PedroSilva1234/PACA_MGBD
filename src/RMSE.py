import math
import os
import re
import logging
import pandas as pd

# 1. Correção: Removido o 'f' (Sintaxe Python)
dados_x = [1.0, 2.0, 3.0, 4.0, 5.0]

def extrair_erros(filepath):
    """Extrai os erros do ficheiro .data usando regex."""
    try:
        with open(filepath, 'r') as f:
            conteudo = f.read()
        
        match = re.search(r"Erros\s*:\s*([-+]?(?:\d*\.\d+|\d+|nan|inf))", conteudo, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
    except Exception:
        return None    

def extrair_parametros(filepath):
    """Extrai os parâmetros m e b do ficheiro .data usando regex."""
    try:
        with open(filepath, 'r') as f:
            conteudo = f.read()
        
        match = re.search(r"Reta com (?:FLOAT|DOUBLE)\s*:\s*y\s*=\s*([-+]?(?:\d*\.\d+|\d+|nan|inf))\s*x\s*\+\s*([-+]?(?:\d*\.\d+|\d+|nan|inf))", conteudo, re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None
    except Exception:
        return None, None

def calcular_rmse(m_ref, b_ref, m_var, b_var):
    """
    Calcula o RMSE (Root Mean Squared Error).
    Retorna o erro absoluto na mesma unidade dos dados preditos.
    """
    erro_quadratico_total = 0.0
    
    for x in dados_x:
        y_ref = (m_ref * x) + b_ref        
        y_var = (m_var * x) + b_var
        
        diferenca = y_ref - y_var
        erro_quadratico_total += (diferenca * diferenca)
        
    mse = erro_quadratico_total / len(dados_x)  
    rmse = math.sqrt(mse)
    
    return rmse

def NRMSE(m_ref, b_ref, m_var, b_var):
    """
    Calcula o NRMSE (Normalized Root Mean Squared Error).
    Normalizado pela média das previsões de referência.
    """
    rmse = calcular_rmse(m_ref, b_ref, m_var, b_var)
    
    # 2. Correção: Normalizando pela média de Y (Gabarito) para ter uma proporção real
    soma_y_ref = sum([(m_ref * x) + b_ref for x in dados_x])
    media_y_ref = soma_y_ref / len(dados_x)
    
    if media_y_ref == 0:
        media_y_ref = 1e-8  # Evita divisão por zero
        
    return abs(rmse / media_y_ref)


if __name__ == "__main__":
    m_ref = 0.785
    b_ref = 2.85

    target_dir = r"C:\Programas\LSCAD\IC_Dani\Docker_Gregorio\ILAC\storage\executions\sgd_forcabruta_20260716_144301\outputs"
    erros_txt = "erros_rmse.txt"
    
    if not os.path.exists(target_dir):
        print(f"Erro: A pasta {target_dir} não existe.")
    else:
        arquivos_data = sorted([f for f in os.listdir(target_dir) if f.endswith('.data')])

        try:
            # 3. Correção: Arquivo aberto FORA do loop para não ser sobrescrito
            with open(erros_txt, 'w') as f:
                for arquivo in arquivos_data:
                    caminho_data = os.path.join(target_dir, arquivo)
                    m_var, b_var = extrair_parametros(caminho_data)
                    hash_id = arquivo.replace('.data', '')

                    # 4. Prevenção de quebra caso o regex retorne None
                    if m_var is None or b_var is None or math.isnan(m_var) or math.isnan(b_var):
                        f.write(f"{hash_id:<70} | RMSE: FALHA NA LEITURA | NRMSE: FALHA\n")
                        continue

                    erro_rmse = calcular_rmse(m_ref, b_ref, m_var, b_var)
                    erro_nrmse = NRMSE(m_ref, b_ref, m_var, b_var)

                    f.write(f"{hash_id:<70} | RMSE: {erro_rmse:.6f} | NRMSE: {erro_nrmse:.6f}\n")
            
            print(f"Sucesso! {len(arquivos_data)} variantes avaliadas e salvas em {erros_txt}.")
            
        except Exception as e:
            logging.error(f"Erro ao escrever no ficheiro {erros_txt}: {e}")