import Metodos
from collections import defaultdict
from random import choice, choices
from time import perf_counter

"""
Descrição: heurística construtiva híbrida (gulosa + aleatória) para o problema de seleção de corredores e pedidos. A heurística seleciona corredores com probabilidade proporcional à sua contribuição potencial (peso), calculada com base na demanda dos itens. Após selecionar um corredor, os pedidos são escolhidos gulosamente respeitando as restrições, maximizando a função objetivo.
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados.
Saída: lista contendo os pedidos selecionados, corredores selecionados, valor da função objetivo, e tempo de execução da heurística.
"""
def hibrida(problema):
    inicio = perf_counter()

    sol = Metodos.solucao(
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        [],
        [0 for _ in range(problema.a)],
        [],
        [0 for _ in range(problema.o)],
        0,
        0,
        0.0
    )

    # Calculando a demanda de cada item.
    demanda_por_item = defaultdict(int)             # Dicionário da soma total da demanda de cada item em todos os pedidos.
    for pedido in problema.orders:
        for item, qtd in pedido.items():
            demanda_por_item[item] += qtd

    # Calculando o peso de cada corredor com base na demanda e na quantidade ofertada.
    # Peso de cada corredor é calculado com base na demanda dos itens e na quantidade que ele oferece.
    peso_corredores = {indice: sum(demanda_por_item[item] * qnt for item, qnt in problema.aisles[indice].items()) for indice in range(problema.a)}

    # Buscando a melhor solução até ficar 3 iterações seguidas sem encontrar uma melhor.
    tentativas_sem_melhora = 0
    while tentativas_sem_melhora < 3 and peso_corredores:
        copiaSol = sol.clone()

        # Selecionando um corredor ainda não utilizado.
        # Se todos os pesos forem zero, escolhe aleatoriamente. Caso contrário, utiliza seleção ponderada proporcional ao peso.
        total = sum(peso_corredores.values())
        if total == 0:
            corredor = choice(list(peso_corredores.keys()))
        else:
            escolhas, prob = zip(*[(indice, peso / total) for indice, peso in peso_corredores.items()])
            corredor = choices(escolhas, weights=prob, k=1)[0]

        # Atualizando universo dos corredores.
        copiaSol.corredores.append(corredor)
        copiaSol.corredoresDisp[corredor] = 1
        copiaSol.qntCorredores += 1

        for item, qnt in problema.aisles[corredor].items():
            copiaSol.itensC[item] += qnt
            copiaSol.universoC[item] += qnt

        # Filtrando os pedidos possíveis para os corredores selecionados.
        pedidos_viaveis = []                        # Lista de pedidos viáveis com os corredores atualmente selecionados.
        for indice in range(problema.o):
            if not copiaSol.pedidosDisp[indice]:
                valida = True
                for item, qnt in problema.orders[indice].items():
                    if qnt > copiaSol.universoC[item]:
                        valida = False
                        break
                if valida:
                    pedidos_viaveis.append(indice)

        # Selecionando os melhores pedidos (os que tem mais itens e que não quebram a restrição de ub).
        while True:
            # Buscando o melhor pedido para o instante atual.
            melhor_pedido = [-1, -1]                # [índice do melhor pedido, total de itens].
            for indice in pedidos_viaveis:
                if not copiaSol.pedidosDisp[indice]:
                    soma = 0
                    valida = True
                    for item, qnt in problema.orders[indice].items():
                        if qnt > copiaSol.universoC[item]:
                            valida = False
                            break
                        else:
                            soma += qnt
                    if valida and copiaSol.qntItens + soma <= problema.ub and soma > melhor_pedido[1]:
                        melhor_pedido[0] = indice
                        melhor_pedido[1] = soma
            # Encerrando loop caso não tenha encontrado nenhum.
            if melhor_pedido[0] == -1:
                break
            # Atualizando o universo dos pedidos.
            copiaSol.qntItens += melhor_pedido[1]
            copiaSol.pedidosDisp[melhor_pedido[0]] = 1
            copiaSol.pedidos.append(melhor_pedido[0])
            for item, qnt in problema.orders[melhor_pedido[0]].items():
                copiaSol.universoC[item] -= qnt
                copiaSol.itensP[item] += qnt

        # Comparando as soluções, e salvando a atual caso seja melhor.
        copiaSol.objetivo = Metodos.funcao_objetivo(problema, copiaSol.itensP, copiaSol.itensC) / copiaSol.qntCorredores
        if copiaSol.objetivo > sol.objetivo or copiaSol.qntItens < problema.lb:
            sol = copiaSol
            peso_corredores.pop(corredor)
            tentativas_sem_melhora = 0
        else:
            tentativas_sem_melhora += 1

    fim = perf_counter()

    return [sol.pedidos, sol.corredores, sol.objetivo, fim - inicio]


"""
Descrição: heurística construtiva aleatória para o problema de seleção de corredores e pedidos. A heurística seleciona corredores e pedidos de forma aleatória, respeitando as restrições do problema.
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados.
Saída: lista contendo os pedidos selecionados, corredores selecionados, valor da função objetivo, e tempo de execução da heurística.
"""
def aleatorio(problema):
    inicio = perf_counter()
    
    sol = Metodos.solucao(
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        [],
        [0 for _ in range(problema.a)],
        [],
        [0 for _ in range(problema.o)],
        0,
        0,
        0.0
    )

    corredores_disponiveis = set(range(problema.a))
    pedidos_disponiveis = set(range(problema.o))

    while corredores_disponiveis:
        # Selecionando um corredor aleatoriamente.
        corredor = choice(list(corredores_disponiveis))
        corredores_disponiveis.remove(corredor)

        # Atualizando universo dos corredores.
        sol.corredores.append(corredor)
        sol.corredoresDisp[corredor] = 1
        sol.qntCorredores += 1

        for item, qnt in problema.aisles[corredor].items():
            sol.itensC[item] += qnt
            sol.universoC[item] += qnt

        # Selecionando pedidos que melhor se adequam ao corredor aleatoriamente.
        while sol.qntItens < problema.ub:
            # Filtrando os pedidos possíveis para os corredores selecionados.
            pedidos_viaveis = []
            for pedido in pedidos_disponiveis:
                valida = True
                soma_itens = 0
                for item, qnt in problema.orders[pedido].items():
                    if qnt > sol.universoC[item]:
                        valida = False
                        break
                    soma_itens += qnt

                if valida and sol.qntItens + soma_itens <= problema.ub:
                    pedidos_viaveis.append(pedido)

            if not pedidos_viaveis:
                break

            # Selecionando um pedido aleatoriamente entre os viáveis.
            pedido = choice(pedidos_viaveis)
            pedidos_disponiveis.remove(pedido)

            # Atualizando o universo dos pedidos.
            soma_itens = sum(problema.orders[pedido].values())
            sol.qntItens += soma_itens
            sol.pedidosDisp[pedido] = 1
            sol.pedidos.append(pedido)
            for item, qnt in problema.orders[pedido].items():
                sol.universoC[item] -= qnt
                sol.itensP[item] += qnt

        # Calculando o valor da função objetivo.
        sol.objetivo = Metodos.funcao_objetivo(problema, sol.itensP, sol.itensC) / max(1, sol.qntCorredores)

        # Se a função objetivo estiver abaixo de 5, tentar alocar mais pedidos no corredor atual.
        if sol.objetivo < 5:
        if True:
            while True:
                pedidos_viaveis = []
                for pedido in pedidos_disponiveis:
                    valida = True
                    soma_itens = 0
                    for item, qnt in problema.orders[pedido].items():
                        if qnt > sol.universoC[item]:
                            valida = False
                            break
                        soma_itens += qnt

                    if valida and sol.qntItens + soma_itens <= problema.ub:
                        pedidos_viaveis.append(pedido)

                if not pedidos_viaveis:
                    break

                # Selecionando um pedido aleatoriamente entre os viáveis.
                pedido = choice(pedidos_viaveis)
                pedidos_disponiveis.remove(pedido)

                # Atualizando o universo dos pedidos.
                soma_itens = sum(problema.orders[pedido].values())
                sol.qntItens += soma_itens
                sol.pedidosDisp[pedido] = 1
                sol.pedidos.append(pedido)
                for item, qnt in problema.orders[pedido].items():
                    sol.universoC[item] -= qnt
                    sol.itensP[item] += qnt

            # Recalcular a função objetivo após tentar alocar mais pedidos.
            sol.objetivo = Metodos.funcao_objetivo(problema, sol.itensP, sol.itensC) / max(1, sol.qntCorredores)

    fim = perf_counter()

    return [sol.pedidos, sol.corredores, sol.objetivo, fim - inicio]