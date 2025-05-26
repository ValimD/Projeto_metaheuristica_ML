import Processa
from time import perf_counter
from collections import defaultdict

def peso_aresta(problema, corredor_id, pedido_id):
    """
    Peso = quantidade de itens faltantes para suprir o pedido.
    Retorna 9999 se mais de 15 unidades estiverem faltando.
    """
    oferta = problema.aisles[corredor_id]
    demanda = problema.orders[pedido_id]
    faltantes = 0

    for item, qnt in demanda.items():
        faltantes += max(0, qnt - oferta.get(item, 0))
        # if faltantes > 15:
        #     return 9999

    return faltantes


def inicia_grafo(problema):
    """
    Inicializa o grafo bipartido entre corredores e pedidos.
    Cada aresta tem peso igual à quantidade de itens faltantes para suprir o pedido.
    """
    grafo = defaultdict(list)
    for c_id in range(len(problema.aisles)):
        for p_id in range(len(problema.orders)):
            peso = peso_aresta(problema, c_id, p_id)
            grafo[c_id].append((p_id, peso))
        
        grafo[c_id].sort(key=lambda x: x[1])

    return grafo

def main():
    arquivo = 'teste'
    dataset = 'instance_0014'
    problema = Processa.Problema(dataset, arquivo)
    inicio = perf_counter()
    grafo = inicia_grafo(problema)
    tempo = perf_counter() - inicio
    print(f"Tempo total: {tempo} segundos")

if __name__ == "__main__":
    main()