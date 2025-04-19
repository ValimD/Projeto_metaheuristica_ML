import Metodos
from random import randint

"""
Descrição: heurística construtiva que usa aleatoriedade para escolher os corredores, e usa o método guloso para encontrar os pedidos. Para o método guloso funcionar, os pedidos serão filtrados para ter somente aqueles que contém os itens dos corredores, e depois serão rankeados pela quantidade de itens em cada um.
Entrada: objeto do problema instanciado.
Saída: valor da função objetivo, lista contendo uma lista com os índices dos pedidos e uma contendo os índices dos corredores.
"""
def misto(problema):
    itensC = dict((i, 0) for i in range(problema.i)) # Dicionário da quantidade de itens nos corredores selecionados.
    corredores = [] # Corredores na solução.
    qntCorredores = randint(1, problema.a) # Quantidade inicial de corredores na solução.
    selecionados = [0 for _ in range(problema.a)] # Lista para indicar se o corredor já foi selecionado, evitando o uso de not in.
    # Selecionando os corredores.
    for _ in range(qntCorredores):
        corredor = randint(0, problema.a - 1)
        # Se o corredor já foi selecionado, busca outro.
        while(selecionados[corredor]):
            corredor = randint(0, problema.a - 1)
        # Atualizando variáveis.
        selecionados[corredor] = 1
        corredores.append(corredor)
        # Adicionando itens no dicionário.
        for i in problema.aisles[corredor]:
            itensC[i] += problema.aisles[corredor][i]
    # Filtrando os pedidos.
    ranking = [[i, 0] for i in range(problema.o)] # Lista contendo sublistas de dois elementos. O primeiro é o id do pedido e o segundo é a quantidade de itens nele.
    pedidosRem = [] # Lista para guardar as sublistas de dois elementos que são incompatíveis com os corredores.
    selecionados = [0 for _ in range(problema.o)] # Usando a lista para marcar se o pedido foi removido do ranking ou não.
    for p in ranking:
        for i in problema.orders[p[0]]:
            if itensC[i] < problema.orders[p[0]][i]:
                pedidosRem.append(p)
                selecionados[p[0]] = 1
                break
        if selecionados[p[0]] == 0:
            p[1] = sum(problema.orders[p[0]].values())
    # Removendo do ranking os pedidos incompatíveis com os corredores.
    for p in pedidosRem:
        ranking.remove(p)
    # Rankeando de acordo com a quantidade de itens.
    ranking.sort(key = lambda value: value[1])
    # Adicionando os pedidos mais valiosos até não sobrar mais nenhum.
    itensP = dict((i, 0) for i in range(problema.i)) # Dicionário da quantidade de itens nos pedidos selecionados.
    pedidos = [] # Pedidos na solução.
    qntItens = 0
    objetivo = 0
    while ranking != []:
        # Adicionando o pedido com a maior pontuação.
        pedido = ranking.pop()
        pedidos.append(pedido[0])
        qntItens += pedido[1]
        for i in problema.orders[pedido[0]]:
            itensP[i] += problema.orders[pedido[0]][i]
        # Se o limite inferior não foi alcançado, somente adiciona, caso contrário compara com a solução anterior. 
        if qntItens < problema.lb:
            objetivo = Metodos.funcao_objetivo(problema, itensP, itensC) / qntCorredores
        else:
            nObjetivo = Metodos.funcao_objetivo(problema, itensP, itensC) / qntCorredores
            # Se a nova solução for melhor, atualiza a variável objetivo, caso contrário remove o pedido da solução.
            if nObjetivo > objetivo:
                objetivo = nObjetivo
            else:
                for i in problema.orders[pedido[0]]:
                    itensP[i] -= problema.orders[pedido[0]][i]
                pedidos.pop()
                qntItens -= pedido[1]
    print(objetivo)
    print(pedidos, qntItens, problema.lb, problema.ub)
    print(corredores)
    return objetivo, pedidos, corredores