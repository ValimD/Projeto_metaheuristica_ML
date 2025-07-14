# Projeto_metaheuristica_ML

Este repositório contém algoritmos heurísticos para o problema de *wave picking*, contando com algoritmos originais e com adaptações de metaheurísticas já existentes. Os algoritmos originais são:

- Heurísticas Construtivas:
    - Aleatória;
    - Gulosa;
    - Híbrida.

- Heurísticas de Refinamento:
    - Clusterização + VNS;
    - Melhor Vizinhança.

As metaheurísticas adapatadas foram:
- *Adaptive Large Neighborhood Search* (ALNS);
- *FLower Pollination Algorithm* (FPA);
- *Particle Swarm Optimization* (PSO).

## Objetivo

O objetivo do repositório era solucionar o problema do Mercado Livre proposto para o Simbósio Brasileiro de Pesquisa Operacional (SBPO) 2025 (disponível no [repositório](https://github.com/mercadolibre/challenge-sbpo-2025)).

## Organização

Este repositório está organizado da seguinte maneira:

```
Datasets/
└── Datasets do problema em formato .txt.
Metodos/
└── Arquivos .py conténdo os códigos dos algoritmos.
Processa/
└── Arquivo .py responsável por tratar os datasets e salvar os resultados.
Resultados-csv/
└── Arquivos .csv dos resultados de execução. O formato é: dataset,pedidos (separados por -),corredores (separados por -),valor da função objetivo,tempo de execução.
Resultados-txt/
└── Arquivos .txt dos resultados de execução. Os arquivos estão no formato esperado pelo MeLi.
main.py - Código principal do repositório.
```

## Funcionamento

Para execução do programa, siga os passos:

```shell
git clone https://github.com/ValimD/Projeto_metaheuristica_ML.git
cd Projeto_metaheuristica_ML

pip install scikit-learn matplotlib

python main.py <dataset> <nome_arquivo_resultados> <heurística_construtiva_metaheurística> <heurística_refinamento> <semente_aleatoria>
```

Os parâmetros esperados pela `main.py` são:
- Dataset: nome do dataset na pasta `Datasets` sem o .txt;
- Nome arquivo resultados: nome do arquivo em que será salvo os resultados, sem o .txt;
- Heurística construtiva ou metaheurística: algoritmo que será utilizado inicialmente;
    - 0: Híbrida;
    - 1: Aleatória;
    - 2: Gulosa;
    - 3: PSO;
    - 4: FPA;
    - 5: ALNS;
- Heurística de refinamento: algoritmo de refinamento que será utilizado;
    - 1: Melhor Vizinhança;
    - 2: Clusterização + VNS;
    - qualquer: nenhuma.
- Semente: *seed* numérica para a aleatoriedade.

## Autores

[HenriUz](https://github.com/HenriUz)

[gebra04](https://github.com/gebra04)

[ValimD](https://github.com/ValimD)