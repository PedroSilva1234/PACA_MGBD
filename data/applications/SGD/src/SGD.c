#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Removi o #include "approx.h" para compilação genérica, adicione de volta se precisar

typedef struct {
    float x;
    float y;
} DataPoint;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: ./programa <caminho_saida>\n");
        return 1;
    }
    
    char *caminho_saida = argv[argc - 1];
    FILE *arquivo = fopen(caminho_saida, "w");
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo.\n");
        return 1;
    }  

    // Dados originais totalmente em Float
    DataPoint dados[] = {
        {1.0f, 2.5f}, {2.0f, 4.8f}, {3.0f, 5.9f}, {4.0f, 7.5f}, {5.0f, 5.0f}
    };
    
    int n = 5;
    
    // DEFINIÇÃO DO TAMANHO DO MINI-BATCH
    int batch_size = 2; // Você pode testar com 1 (SGD), 2, 3, etc.

    float m = 0.0f;
    float b = 0.0f;
    float learning_rate = 0.03f;
    int epochs = 100000;

    // Variaveis do RMSProp
    float v_m = 0.0f;
    float v_b = 0.0f;
    float beta = 0.9f;
    float epsilon = 1e-8f;
    
    // Alocação dinâmica para evitar estouro de pilha (stack overflow) com arrays muito grandes
    float *historico_m = malloc(epochs * sizeof(float));
    float *historico_b = malloc(epochs * sizeof(float));

    // Loop de Épocas
    for (int epoch = 0; epoch < epochs; epoch++) {
        
        // Loop dos Mini-batches: avança 'batch_size' passos de cada vez
        for (int i = 0; i < n; i += batch_size) {
            
            float grad_m_total = 0.0f;
            float grad_b_total = 0.0f;
            int dados_no_lote_atual = 0;

            // Loop interno: processa apenas os dados do lote atual
            for (int j = i; j < i + batch_size && j < n; j++) {
                float x_val = dados[j].x;
                float y_val = dados[j].y;

                //anotacao:
                float compontente_X = m * x_val;
                //anotacao:
                float previsao = compontente_X + b;
                //anotacao:
                float error = y_val - previsao;
                //anotacao:
                float grad_m = 2.0f * x_val * (error);
                //anotacao:
                float grad_b = 2.0f * (error);
                //anotacao:
                grad_m_total = grad_m_total - grad_m;
                //anotacao:
                grad_b_total = grad_b_total - grad_b;
                dados_no_lote_atual++;
            }

            // Tira a média com base na quantidade real de dados do lote 
            // (o último lote pode ter menos dados se 'n' não for divisível por 'batch_size')
            grad_m_total /= (float)dados_no_lote_atual;
            grad_b_total /= (float)dados_no_lote_atual;

            // ATUALIZAÇÃO DOS PESOS (RMSProp) ACONTECE AQUI, PARA CADA MINI-BATCH
            v_m = beta * v_m + (1.0f - beta) * grad_m_total * grad_m_total;
            v_b = beta * v_b + (1.0f - beta) * grad_b_total * grad_b_total;

            float adaptive_lr_m = learning_rate / (sqrtf(v_m) + epsilon);
            float adaptive_lr_b = learning_rate / (sqrtf(v_b) + epsilon);

            m -= adaptive_lr_m * grad_m_total;
            b -= adaptive_lr_b * grad_b_total;
        }
        
        // Salva o estado ao final da época
        historico_m[epoch] = m;
        historico_b[epoch] = b;
    }

    fprintf(arquivo, "Treinamento concluido apos %d epocas!\n", epochs);
    fprintf(arquivo, "Reta com FLOAT : y = %f x + %f\n", m, b);

    fclose(arquivo);
    free(historico_m);
    free(historico_b);

    return 0;
}