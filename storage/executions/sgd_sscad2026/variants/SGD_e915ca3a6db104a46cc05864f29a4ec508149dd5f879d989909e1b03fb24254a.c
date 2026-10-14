#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "approx.h"

typedef struct {
    float x;
    float y;
} DataPoint;

int main(int argc, char *argv[]) {

    if (argc < 2) {
        return 1;
    }

    char *caminho_saida = argv[argc - 1];
    FILE *arquivo = fopen(caminho_saida, "w");

    if (arquivo == NULL) {
        return 1;
    }
    
    // Dados originais totalmente em Float (com o sufixo 'f')
    DataPoint dados[] = {
        {1.0f, 2.5f}, {2.0f, 4.8f}, {3.0f, 5.9f}, {4.0f, 7.5f}, {5.0f, 5.0f}
    };
    int n = 5;

    float m = 0.0f; 
    float b = 0.0f; 
    float previsao = 0.0f;
    float learning_rate = 0.03f; 
    float grad_m = 0.0f;
    float grad_b = 0.0f;
    float compontente_X = 0.0f;
    int epochs = 100000; 
    float error = 0.0f;
    
    //variaveis do RMSProp 
    float v_m = 0.0f;
    float v_b = 0.0f;
    float beta = 0.9f;
    float epsilon = 1e-8f;

    float historico_m[100000];
    float historico_b[100000];

    // Loop do Gradiente Descendente
    for (int epoch = 0; epoch < epochs; epoch++) {
        float grad_m_total = 0.0f;
        float grad_b_total = 0.0f;
        
        for (int i = 0; i < n; i++) {
            float x_val = dados[i].x;
            float y_val = dados[i].y;
            
            //anotacao:
            compontente_X = m * x_val;

            //anotacao:
            previsao = compontente_X + b;
            //anotacao:
            error = y_val - previsao;
           //anotacao:
            grad_m = 2.0f * x_val * (error);
            //anotacao:
            grad_b = 2.0f * (error);

            //anotacao:
            grad_m_total = grad_m_total - grad_m;
            //anotacao:
            grad_b_total = grad_b_total - grad_b;
        }
        
        grad_m_total /= (float)n;
        grad_b_total /= (float)n;

        //Atualização dos momentos para RMSProp
        v_m = beta * v_m + (1.0f - beta) * grad_m_total * grad_m_total;
        v_b = beta * v_b + (1.0f - beta) * grad_b_total * grad_b_total;
        
        float adaptive_lr_m = learning_rate / (sqrtf(v_m) + epsilon);
        float adaptive_lr_b = learning_rate / (sqrtf(v_b) + epsilon);

        m -= adaptive_lr_m * grad_m_total;
        b -= adaptive_lr_b * grad_b_total;

        historico_m[epoch] = m;
        historico_b[epoch] = b;
    }


   
    //fprintf(arquivo, "==============================================\n");

    //fprintf(arquivo, "========== Historico do Treinamento ==========\n");
   // for (int epoch = 0; epoch < epochs; epoch++) {
     //   if (epoch % 50 == 0 || epoch == epochs - 1) { 
            // CORREÇÃO 1: Adicionado o ponteiro 'arquivo' aqui!
       //     fprintf(arquivo, "Epoca %04d: m = %10.6f | b = %10.6f\n", epoch, historico_m[epoch], historico_b[epoch]);
        //}
    //}
    //fprintf(arquivo, "==============================================\n");


    // Resultado final (que o Python usa para calcular o erro)
    //fprintf(arquivo, "========== Resultado do SGD ==========\n");
    fprintf(arquivo, "Treinamento concluido apos %d epocas!\n",epochs);
    
    // CORREÇÃO 2: Voltando ao formato exato que o regex do Python consegue ler!
    fprintf(arquivo, "Reta com FLOAT : y = %f x + %f\n", m, b); 
    
    //fprintf(arquivo, "======================================\n");

    fclose(arquivo);

    return 0;
}

