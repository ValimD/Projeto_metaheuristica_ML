import Metodos
import Processa
from collections import defaultdict, deque
from random import choice, choices, randint, shuffle
from time import perf_counter

def hibrida(problema: Processa.Problema) -> Metodos.Solucao:
    """
    Heurística construtiva híbrida (gulosa + aleatória) para o problema de seleção de corredores e pedidos.

    A heurística seleciona corredores com probabilidade proporcional à sua contribuição potencial (peso), calculada com base na demanda dos itens. Após selecionar um corredor, os pedidos são escolhidos gulosamente respeitando as restrições, maximizando a função objetivo.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).

    Returns:
        solucao (Solucao): Dataclass representando a solução construída, incluindo estruturas auxiliares.
    """

    sol = Metodos.Solucao(
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        [],
        [0 for _ in range(problema.a)],
        [],
        [0 for _ in range(problema.o)],
        0,
        0,
        0.0,
        perf_counter()
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

        # Adicionando pedidos se possível.
        Metodos.adiciona_pedidos(problema, copiaSol)

        # Comparando as soluções, e salvando a atual caso seja melhor.
        copiaSol.objetivo = Metodos.funcao_objetivo(problema, copiaSol.itensP, copiaSol.itensC) / copiaSol.qntCorredores
        if copiaSol.objetivo > sol.objetivo or copiaSol.qntItens < problema.lb or copiaSol.qntItens == 0:
            sol = copiaSol
            peso_corredores.pop(corredor)
            tentativas_sem_melhora = 0
        else:
            tentativas_sem_melhora += 1

    sol.tempo = perf_counter() - sol.tempo

    return sol

def aleatorio(problema: Processa.Problema) -> Metodos.Solucao:
    """
    Heurística construtiva aleatória para o problema de seleção de corredores e pedidos.

    Esta heurística embaralha os corredores e os adiciona um a um à solução. Após cada adição, os pedidos que podem ser atendidos com o universo atual de itens são identificados e embaralhados.

    Enquanto o limite inferior de itens (problema.lb) não for atingido, a heurística tenta adicionar todos os pedidos viáveis. Após o limite, uma quantidade aleatória de pedidos é selecionada.

    A construção é interrompida quando uma nova solução não supera a anterior, assumindo que já foi encontrada uma solução viável. Esse critério evita execuções desnecessárias após atingir uma boa solução.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).

    Returns:
        solucao (Solucao): Dataclass representando a solução construída, incluindo estruturas auxiliares.
    """

    solucao = Metodos.Solucao(
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        dict.fromkeys(range(problema.i), 0),
        [],
        [0 for _ in range(problema.a)],
        [],
        [0 for _ in range(problema.o)],
        0,
        0,
        0.0,
        perf_counter()
    )

    corredores_selecionados = list(range(problema.a))       # Lista dos corredores embaralhados.
    shuffle(corredores_selecionados)

    # Percorrendo os corredores.
    for corredor in corredores_selecionados:
        nova_solucao = solucao.clone()

        # Atualizando universo dos corredores.
        nova_solucao.corredores.append(corredor)
        nova_solucao.corredoresDisp[corredor] = 1
        nova_solucao.qntCorredores += 1

        for item, qnt in problema.aisles[corredor].items():
            nova_solucao.itensC[item] += qnt
            nova_solucao.universoC[item] += qnt

        # Verificando os pedidos disponíveis.
        pedidos_sorteados = []                              # Lista dos pedidos possíveis aleatórios.
        quantidade = 0                                      # Quantidade de pedidos possíveis.
        for indice in range(problema.o):
            if not nova_solucao.pedidosDisp[indice]:
                valida = True
                for item, qnt in problema.orders[indice].items():
                    if qnt > nova_solucao.universoC[item]:
                        valida = False
                        break
                if valida:
                    pedidos_sorteados.append(indice)
                    quantidade += 1

        # Adicionando os pedidos.
        if quantidade:
            if quantidade != 1:
                shuffle(pedidos_sorteados)

            # Define quantos pedidos tentar adicionar.
            limite_pedidos = randint(1, quantidade) if nova_solucao.qntItens > problema.lb else quantidade

            for indice in pedidos_sorteados[:limite_pedidos]:
                if not nova_solucao.pedidosDisp[indice]:
                    valida = True
                    soma = 0
                    for item, qnt in problema.orders[indice].items():
                        soma += qnt
                        if qnt > nova_solucao.universoC[item]:
                            valida = False
                            break
                    if valida and nova_solucao.qntItens + soma <= problema.ub:
                        nova_solucao.qntItens += soma
                        nova_solucao.pedidosDisp[indice] = 1
                        nova_solucao.pedidos.append(indice)
                        for item, qnt in problema.orders[indice].items():
                            nova_solucao.universoC[item] -= qnt
                            nova_solucao.itensP[item] += qnt

        # Verificando a nova solução.
        nova_solucao.objetivo = Metodos.funcao_objetivo(problema, nova_solucao.itensP, nova_solucao.itensC) / nova_solucao.qntCorredores
        if nova_solucao.objetivo > solucao.objetivo or nova_solucao.qntItens < problema.lb or nova_solucao.qntItens == 0:
            solucao = nova_solucao
        else:
            break

    solucao.tempo = perf_counter() - solucao.tempo
    return solucao

def gulosa(problema: Processa.Problema) -> Metodos.Solucao:
    """
    Heurística construtiva gulosa e iterativa com estratégia first fit para o problema de seleção de corredores e pedidos.

    Ranqueia os pedidos com base na concentração de itens dos corredores e ranqueia os corredores com base na concentração de itens dos pedidos,

    escolhe-se o primeiro corredor e preenche-o com os pedidos viáveis em ordem de ranqueamento, após isso o processo se repete, escolhe-se o próximo corredor e o meso é preenchido de forma gulosa,

    se o corredor novo melhorar o valor de objetivo da função, ele é adicionado à solução, se não, o processo se repete novamente, até um total de 3 vezes sem melhoria na função objetivo ou até UB ser atingido.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).

    Returns:
        solucao (Solucao): Dataclass representando a solução construída, incluindo estruturas auxiliares.
    """
    # Inicializa solução
    solucao = Metodos.Solucao(
        {i: 0 for i in range(problema.i)},  # universoC
        {i: 0 for i in range(problema.i)},  # itensC
        {i: 0 for i in range(problema.i)},  # itensP
        [],  # corredores selecionados
        [0] * problema.a,  # corredoresDisp
        [],  # pedidos selecionados
        [0] * problema.o,  # pedidosDisp
        0,  # qntItens
        0,  # qntCorredores
        0.0,  # objetivo
        perf_counter()  # tempo inicial
    )

    # Cálculo de concentração e pesos ponderados
    concentracao_pedidos = defaultdict(lambda: {"total": 0, "contagem": 0})
    for pedido in problema.orders:
        for item, qtd in pedido.items():
            concentracao_pedidos[item]["total"] += qtd
            concentracao_pedidos[item]["contagem"] += 1

    concentracao_corredores = defaultdict(lambda: {"total": 0, "contagem": 0})
    for corredor in problema.aisles:
        for item, qtd in corredor.items():
            concentracao_corredores[item]["total"] += qtd
            concentracao_corredores[item]["contagem"] += 1

    peso_ponderado_pedidos = {}
    peso_ponderado_corredores = {}
    peso_ponderado_pedidos = {item: (concentracao_corredores[item]["total"] / concentracao_corredores[item]["contagem"] if concentracao_corredores[item]["contagem"] != 0 else 0) for item in range (problema.i)}
    peso_ponderado_corredores = {item: (concentracao_pedidos[item]["total"] / concentracao_pedidos[item]["contagem"] if concentracao_pedidos[item]["contagem"] != 0 else 0) for item in range (problema.i)}

    # Funções para ranqueamento
    def peso_pedido(idx):
        return sum(peso_ponderado_pedidos[item] * qtd for item, qtd in problema.orders[idx].items())

    def peso_corredor(idx):
        return sum(peso_ponderado_corredores[item] * qtd for item, qtd in problema.aisles[idx].items())

    pedidos_rankeados = sorted(range(problema.o), key=peso_pedido, reverse=False)
    corredores_rankeados = sorted(range(problema.a), key=peso_corredor, reverse=False)

    # Buscando a melhor solução até ficar 3 iterações seguidas sem encontrar uma melhor.
    tentativas_sem_melhora = 0
    while tentativas_sem_melhora < 3 and corredores_rankeados:

        # Selecionando o corredor de maior nota
        copiaSolucao = solucao.clone()
        corredor = corredores_rankeados.pop()

        # Atualizando universo dos corredores
        copiaSolucao.corredores.append(corredor)
        copiaSolucao.corredoresDisp[corredor] = 1
        copiaSolucao.qntCorredores += 1

        for item, qnt in problema.aisles[corredor].items():
            copiaSolucao.itensC[item] += qnt
            copiaSolucao.universoC[item] += qnt

        # Adicionando pedidos
        pedidos_viaveis = []                        # Lista de pedidos viáveis com os corredores atualmente selecionados.
        for indice in pedidos_rankeados:
            if not copiaSolucao.pedidosDisp[indice]:
                valida = True
                itens_totais = 0
                for item, qnt in problema.orders[indice].items():
                    itens_totais += qnt
                    if qnt > copiaSolucao.universoC[item]:
                        valida = False
                        break
                if valida:
                    pedidos_viaveis.append([indice, itens_totais])

        for pedido in pedidos_viaveis:
            valida = True
            for item, qnt in problema.orders[pedido[0]].items():
                if qnt > copiaSolucao.universoC[item]:
                    valida = False
                    break
            if valida and copiaSolucao.qntItens + pedido[1] <= problema.ub:
                copiaSolucao.qntItens += pedido[1]
                copiaSolucao.pedidosDisp[pedido[0]] = 1
                copiaSolucao.pedidos.append(pedido[0])
                for item, qnt in problema.orders[pedido[0]].items():
                    copiaSolucao.universoC[item] -= qnt
                    copiaSolucao.itensP[item] += qnt

        # Comparando as soluções, e salvando a atual caso seja melhor
        copiaSolucao.objetivo = Metodos.funcao_objetivo(problema, copiaSolucao.itensP, copiaSolucao.itensC) / copiaSolucao.qntCorredores
        if copiaSolucao.objetivo > solucao.objetivo or copiaSolucao.qntItens < problema.lb:
            solucao = copiaSolucao
            tentativas_sem_melhora = 0
        else:
            tentativas_sem_melhora += 1

    solucao.tempo = perf_counter() - solucao.tempo

    return solucao