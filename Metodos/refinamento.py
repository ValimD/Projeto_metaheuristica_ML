import Metodos
from collections import defaultdict
from time import perf_counter

"""
Descrição: heurística de refinamento baseada em melhor vizinhança, ela procura qual das vizinhanças possíveis (adicionando corredor, trocando corredores, removendo corredor) tem o melhor valor de função objetivo. Tanto os corredores, como os pedidos adicionados quando possível, são escolhidos de forma gulosa (corredores por peso, e pedidos por quantidade de itens).
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados; instancia do dataclass, contendo os elementos principais e auxiliares da solução inicial.
Saída: instancia do dataclass, contendo os elementos principais e auxiliares da nova solução.
"""
def melhor_vizinhanca(problema, solucao):
    inicio = perf_counter()

    # Removendo os corredores redundantes da solução inicial.
    solucao = Metodos.remove_redundantes(problema, solucao)

    # Calculando a demanda de cada item.
    demanda_por_item = defaultdict(int)             # Dicionário da soma total da demanda de c3ada item em todos os pedidos.
    for pedido in problema.orders:
        for item, qtd in pedido.items():
            demanda_por_item[item] += qtd
            if demanda_por_item[item] > problema.ub:
                demanda_por_item[item] = problema.ub

    # Calculando o peso de cada corredor com base na demanda e na quantidade ofertada.
    # Peso de cada corredor é calculado com base na demanda dos itens e na quantidade que ele oferece.
    peso_corredores = {indice: sum(demanda_por_item[item] * qnt for item, qnt in problema.aisles[indice].items()) for indice in range(problema.a)}

    # Explorando a vizinhança até não encontrar nenhuma melhor.
    vizinhanca_explorada = False                    # Condição de parada do loop, quanda a vizinhanca tiver sido explorada, o loop encerra.
    while not vizinhanca_explorada:
        # Pegando o indice do corredor de maior peso que ainda não está na solução.
        corredor_max = -1
        for indice in range(problema.a):
            if not solucao.corredoresDisp[indice]:
                if corredor_max == -1 or peso_corredores[indice] > peso_corredores[corredor_max]:
                    corredor_max = indice

        # Pegando o índice de um corredor de menor peso.
        corredor_min = -1
        for indice in solucao.corredores:
            if corredor_min == -1 or peso_corredores[indice] < peso_corredores[corredor_min]:
                corredor_min = indice

        # Explorando a vizinhança por meio da adição do novo corredor.
        primeira_vizinhanca = Metodos.adiciona_corredor(problema, solucao, corredor_max)

        # Explorando a vizinhança por meio da troca de corredores, trocando o de menor peso selecionado com o de maior peso não selecionado.
        segunda_vizinhanca = Metodos.troca_corredor(problema, solucao, corredor_max, corredor_min)

        # Explorando a vizinhança por meio da remoção do corredor de menor peso.
        terceira_vizinhanca = Metodos.remove_corredor(problema, solucao, corredor_min)

        # Comparando as soluções, e salvando a atual caso seja melhor.
        primeira_vizinhanca.objetivo = Metodos.funcao_objetivo(problema, primeira_vizinhanca.itensP, primeira_vizinhanca.itensC) / primeira_vizinhanca.qntCorredores
        segunda_vizinhanca.objetivo = Metodos.funcao_objetivo(problema, segunda_vizinhanca.itensP, segunda_vizinhanca.itensC) / segunda_vizinhanca.qntCorredores
        terceira_vizinhanca.objetivo = Metodos.funcao_objetivo(problema, terceira_vizinhanca.itensP, terceira_vizinhanca.itensC) / terceira_vizinhanca.qntCorredores

        melhor_vizinhanca = solucao
        if primeira_vizinhanca.objetivo > segunda_vizinhanca.objetivo and primeira_vizinhanca.objetivo > terceira_vizinhanca.objetivo:
            melhor_vizinhanca = primeira_vizinhanca
        elif segunda_vizinhanca.objetivo > terceira_vizinhanca.objetivo:
            melhor_vizinhanca = segunda_vizinhanca
        else:
            melhor_vizinhanca = terceira_vizinhanca

        if melhor_vizinhanca.objetivo > solucao.objetivo or melhor_vizinhanca.qntItens < problema.lb:
            solucao = melhor_vizinhanca
        else:
            vizinhanca_explorada = True

    fim = perf_counter()
    solucao.tempo += fim - inicio
    return solucao