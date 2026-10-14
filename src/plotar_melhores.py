import matplotlib.pyplot as plt
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class ParametrosReta:
    m: float
    b: float
    identificador: str = ""
    estilo: str = "-"       # Padrão: linha sólida
    espessura: float = 1.5  # Padrão: 1.5
    z_order: int = 2        # Padrão: 2 (desenhada acima do grid/pontos base)

if __name__ == "__main__":
    # 1. Dados originais (Pontos do Scatter Plot)
    x = [1, 2, 3, 4, 5]
    y = [2.5, 4.8, 5.9, 7.5, 5.0]
    x_arr = np.array(x)
    
    # 2. Definição das retas
    # Adicionamos estilo='--', espessura maior e z_order baixo para a Reta Exata servir de "fundo"
    retas = [ 
        ParametrosReta(m=0.785, b=2.85, identificador="Reta Exata", estilo="--", espessura=3.0, z_order=2),  
        ParametrosReta(m=0.779372, b=4.28135, identificador="Reta Variante 1"), 
        ParametrosReta(m=0.655908, b=3.245652, identificador="Reta Variante 2"),
        ParametrosReta(m=0.657933, b=3.256318, identificador="Reta Variante 4"),
        ParametrosReta(m=0.785, b=2.85, identificador="Reta Variante 13", z_order=3) # z_order maior desenha por cima
    ]

    # 3. Configuração da Figura
    plt.figure(figsize=(10, 6), dpi=300)

    # A) Plota os pontos reais primeiro
    plt.scatter(x, y, color='tab:blue', label='Training Data: {1,2.5}; {2,4.8}; {3,5.9}; {4,7.5}; {5,5.0}', s=50, zorder=10)

    # B) Plota as Retas (Exata e Variantes) no mesmo loop
    cores = ['tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:green']
    
    for i, reta in enumerate(retas):
        y_variante = reta.m * x_arr + reta.b
        cor_atual = cores[i % len(cores)]
        
        plt.plot(
            x_arr, 
            y_variante, 
            color=cor_atual, 
            label=f"{reta.identificador}: y={reta.m:.2f}x+{reta.b:.2f}", 
            linestyle=reta.estilo,       # Aplica o tracejado na Exata e sólido nas demais
            linewidth=reta.espessura,    # Aplica a espessura configurada
            zorder=reta.z_order          # Garante quem fica por cima
        )

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    
    plt.xlim(0, 6)
    plt.ylim(0, 10.5)
    
    plt.legend(loc='upper left', fontsize=10)

    # 5. Salvar o gráfico
    diretorio_imagens = os.path.join(os.path.dirname(__file__), "graficos_retas")
    
    if not os.path.exists(diretorio_imagens):
        os.makedirs(diretorio_imagens)
        
    nome_imagem = "grafico_melhoresretas.png"
    caminho_imagem = os.path.join(diretorio_imagens, nome_imagem)

    plt.savefig(caminho_imagem, bbox_inches='tight')
    plt.close()

    print(f"Sucesso! Gráfico gerado em: {caminho_imagem}")