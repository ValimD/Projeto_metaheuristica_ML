import Metodos
import Processa
from collections import defaultdict
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
        if copiaSol.objetivo > sol.objetivo or copiaSol.qntItens < problema.lb:
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
    Heurística construtiva gulosa para o problema de seleção de corredores e pedidos.

    Os corredores e pedidos são ranqueados de acordo com a concentração de itens, e com isso os melhores são selecionados.

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

    pedidos_restantes    = {i: problema.orders[i]   for i in range(problema.o)}
    corredores_restantes = {i: problema.aisles[i]   for i in range(problema.a)}

    # Calculando a concentração dos itens nos pedidos.
    concentracao_pedidos = defaultdict(lambda: {"total": 0, "pedidos": 0})
    for pedido in problema.orders:
        for item, qtd in pedido.items():
            concentracao_pedidos[item]["total"] += qtd
            concentracao_pedidos[item]["pedidos"] += 1

    # Calculando a concentração dos itens nos corredores.
    concentracao_corredores = defaultdict(lambda: {"total": 0, "corredores": 0})
    for corredor in problema.aisles:
        for item, qtd in corredor.items():
            concentracao_corredores[item]["total"] += qtd
            concentracao_corredores[item]["corredores"] += 1

    # Calculando o peso ponderado de cada item, com base na concentração dos mesmos nos pedidos e nos corredores.
    peso_ponderado_itens = {item: (concentracao_pedidos[item]["total"] / concentracao_pedidos[item]["pedidos"]) * (concentracao_corredores[item]["total"] / concentracao_corredores[item]["corredores"]) for item in range (problema.i)}

    # Função para ranquear os pedidos com base na soma dos pesos dos itens.
    def peso_pedido(indice_pedido):
        pedido = pedidos_restantes[indice_pedido]
        return sum(peso_ponderado_itens[id] * qtd for id, qtd in pedido.items())

    # Função para ranquear os corredores com base na soma dos pesos dos itens.
    def peso_corredor(indice_corredor):
        corredor = corredores_restantes[indice_corredor]
        return sum(peso_ponderado_itens[id] * qtd for id, qtd in corredor.items())

    lista_idx_pedidos_ranqueados = []
    lista_idx_corredores_ranqueados = []

    # Ranqueamento dos pedidos, e o resultado é a lista dos indices em ordem de peso (maior -> menor).
    while pedidos_restantes:
        id_pedido_max = max(pedidos_restantes, key = peso_pedido)
        lista_idx_pedidos_ranqueados.append(id_pedido_max)
        pedidos_restantes.pop(id_pedido_max)

    # Ranqueamento dos corredores, e o resultado é a lista dos indices em ordem de peso (maior -> menor).
    while corredores_restantes:
        id_corredor_max = max(corredores_restantes, key = peso_corredor)
        lista_idx_corredores_ranqueados.append(id_corredor_max)
        corredores_restantes.pop(id_corredor_max)

    universo_temporario = dict.fromkeys(range(problema.i), 0)

    # Seleção dos pedidos para satisfazer os corredores.
    for idx_corredor in lista_idx_corredores_ranqueados:
        corredor = problema.aisles[idx_corredor]
        solucao.corredores.append(idx_corredor)
        solucao.corredoresDisp[idx_corredor] = 1
        solucao.qntCorredores += 1

        # Acumulação dos itens dos corredores.
        for item, qtd in corredor.items():
            solucao.itensC[item] += qtd
            solucao.universoC[item] += qtd
            universo_temporario[item] += qtd

        # Teste dos pedidos em ordem de ranqueamento.
        for idx_pedido in lista_idx_pedidos_ranqueados:
            if solucao.pedidosDisp[idx_pedido]:
                continue
            pedido = problema.orders[idx_pedido]
            # Verificação de UB, se verdadeiro -> pula este pedido e vai para o próximo.
            if solucao.qntItens + sum(pedido.values()) > problema.ub:
                continue
            # Verificação de se o pedido cabe completamente no inventário temporário.
            if all(universo_temporario[item] >= qtd for item, qtd in pedido.items()):
                # Alocação do pedido e consumo do inventário.
                for item, qtd in pedido.items():
                    universo_temporario[item] -= qtd
                    solucao.itensP[item] += qtd
                solucao.pedidos.append(idx_pedido)
                solucao.pedidosDisp[idx_pedido] = 1
                solucao.qntItens += sum(pedido.values())

    solucao.objetivo = solucao.qntItens / solucao.qntCorredores if solucao.qntCorredores else 0.0
    solucao.tempo = perf_counter() - solucao.tempo

    return solucao