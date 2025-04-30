import Metodos
from random import randint
from time import perf_counter

"""
Descrição: heurística construtiva que usa aleatoriedade para escolher os corredores, e usa o método guloso para encontrar os pedidos. Para o método guloso funcionar, os pedidos serão filtrados para ter somente aqueles que contém os itens dos corredores, e depois serão rankeados pela quantidade de itens em cada um.
Entrada: objeto do problema instanciado.
Saída: lista contendo os pedidos, os corredores, o valor da função objetivo, e o tempo da execução do método.
"""
def misto_v1(problema):
    inicio = perf_counter()

    itensC = dict((i, 0) for i in range(problema.i))    # Dicionário da quantidade de itens nos corredores selecionados.
    corredores = []                                     # Lista dos índices dos corredores compondo a solução.
    qntCorredores = randint(1, problema.a)              # Quantidade de corredores na solução.
    selecionados = [0 for _ in range(problema.a)]       # Lista para indicar se o corredor já foi selecionado, evitando o uso de not in.

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
    ranking = [[i, 0] for i in range(problema.o)]       # Lista contendo sublistas de dois elementos. O primeiro é o id do pedido e o segundo é a quantidade de itens nele.
    pedidosRem = []                                     # Lista para guardar as sublistas de dois elementos que são incompatíveis com os corredores.
    selecionados = [0 for _ in range(problema.o)]       # Usando a lista para marcar se o pedido foi removido do ranking ou não.

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

    # Rankeando de acordo com a quantidade de itens em ordem crescente.
    ranking.sort(key = lambda value: value[1])

    # Adicionando os pedidos mais valiosos até não sobrar mais nenhum.
    itensP = dict((i, 0) for i in range(problema.i))    # Dicionário da quantidade de itens nos pedidos selecionados.
    pedidos = []                                        # Lista dos índices dos pedidos compondo a solução.
    objetivo = 0                                        # Valor da função objetivo.

    while ranking != []:
        pedido = ranking.pop()
        pedidos.append(pedido[0])
        for i in problema.orders[pedido[0]]:
            itensP[i] += problema.orders[pedido[0]][i]
        # Verificando a qualidade.
        nObjetivo = Metodos.funcao_objetivo(problema, itensP, itensC) / qntCorredores
        if nObjetivo >= objetivo:
            objetivo = nObjetivo
        else:
            for i in problema.orders[pedido[0]]:
                itensP[i] -= problema.orders[pedido[0]][i]
            pedidos.pop()

    fim = perf_counter()
    return [pedidos, corredores, objetivo, fim - inicio]

"""
Descrição: heurística construtiva que usa aleatoriedade para escolher os pedidos, e usa o método guloso para encontrar os corredores. Para o método guloso funcionar, os corredores serão filtrados para ter somente aqueles que contém pelo menos um item dos pedidos, e depois serão rankeados pela quantidade de itens (dos pedidos) em cada um.
Entrada: objeto do problema instanciado.
Saída: lista contendo os pedidos, os corredores, o valor da função objetivo, e o tempo da execução do método.
"""
def misto_v2(problema):
    inicio = perf_counter()

    itensP = dict((i, 0) for i in range(problema.i))    # Universo dos itens nos pedidos selecionados.
    qntItens = 0                                        # Quantidade total de itens nos pedidos, serve para ajudar a corrigir soluções inválidas.
    pedidos = []                                        # Lista dos índices dos pedidos compondo a solução.
    qntPedidos = randint(1, problema.o)                 # Quantidade inicial de pedidos na solução (a quantidade real pode variar na hora de tentar corrigir possíveis soluções inválidas).
    selecionados = [0 for _ in range(problema.o)]       # Lista para indicar se o pedido já foi selecionado, evitando o uso de not in.
    p = 0                                               # Variável de iteração.

    # Selecionando pedidos até a restrição de lb ser atendida ou até atingir a quantidade estimada.
    while qntItens < problema.lb or p < qntPedidos:
        pedido = randint(0, problema.o - 1)
        # Se o pedido já foi selecionado, busca outro.
        while(selecionados[pedido]):
            pedido = randint(0, problema.o - 1)
        # Adicionando o pedido caso ele não viole a restrição de ub. Se violar, o loop é encerrado.
        qntItens += sum(problema.orders[pedido].values())
        if qntItens <= problema.ub:
            selecionados[pedido] = 1
            pedidos.append(pedido)
            p += 1
            for i in problema.orders[pedido]:
                itensP[i] += problema.orders[pedido][i]
        else:
            break

    # Construindo o universo dos corredores com somente os corredores que contém pelo menos um item do universo de itens.
    universoC = []                                      # Universo dos corredores possíveis para a solução.
    for c in range(problema.a):
        contem = False
        for i in problema.aisles[c]:
            if itensP[i] > 0:
                contem = True
        if contem:
            universoC.append(c)


    # Adicionando os corredores mais valiosos até não cobrir todos os itens selecionados.
    universoP = itensP.copy()                           # Cópia do universo dos itens. Necessário para não modificar diretamente o dicionário dos itens dos pedidos.
    itensF = sum(universoP.values())                    # Quantidade de itens no universo que ainda não foram cobertos.
    itensC = dict((i, 0) for i in range(problema.i))    # Universo dos itens nos corredores selecionados.
    corredores = []                                     # Lista dos índices dos corredores compondo a solução.
    qntCorredores = 0                                   # Quantidade de corredores selecionados.

    while itensF > 0:
        # Procurando o melhor corredor para o universo atual.
        melhor = [-1, -1]                               # ID do corredor, e a quantidade de itens que ele satisfaz.
        for c in universoC:
            soma = 0
            for i in problema.aisles[c]:
                if universoP[i] > 0:
                    soma +=  min(problema.aisles[c][i], universoP[i])
            if soma > melhor[1]:
                melhor[0] = c
                melhor[1] = soma
        # Se a variável melhor permanece inalterada, encerra execução.
        if melhor[0] == -1:
            break
        # Atualizando universo e variáveis.
        qntCorredores += 1
        corredores.append(melhor[0])
        universoC.remove(melhor[0])
        itensF -= melhor[1]
        for i in problema.aisles[melhor[0]]:
            universoP[i] -= problema.aisles[melhor[0]][i]
            itensC[i] += problema.aisles[melhor[0]][i]

    objetivo = Metodos.funcao_objetivo(problema, itensP, itensC) / qntCorredores
    fim = perf_counter()
    return [pedidos, corredores, objetivo, fim - inicio]