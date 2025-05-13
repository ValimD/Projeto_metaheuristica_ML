import Metodos
import numpy as np
import Processa
from collections import defaultdict
from random import choice
from sklearn.cluster import MiniBatchKMeans
from time import perf_counter

def construir_clusters_de_dicts(labels_pedidos, labels_corredores):
    """
    Função responsável por construir dicts de cluster->{lista de índices} para pedidos e corredores.

    Args:
        labels_pedidos ():
        labels_corredores ():
    """

    clusters_ped = defaultdict(list)
    for idx, lab in enumerate(labels_pedidos):
        clusters_ped[lab].append(idx)
    clusters_corr = defaultdict(list)
    for idx, lab in enumerate(labels_corredores):
        clusters_corr[lab].append(idx)
    return clusters_ped, clusters_corr


def atualizaCorredores(solucao: Metodos.Solucao, problema: Processa.Problema, sol_vizinha, novo_c, pos):
    """
    Args:
        solucao (Solucao): Dataclass representando a solução inicial, incluindo estruturas auxiliares.
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        sol_vizinha ():
        novo_c: ID do corredor que entra
        pos: posição na lista solucao.corredores onde trocamos o corredor
    """

    # 1) identifica IDs
    corredor_antigo = solucao.corredores[pos]

    # 2) subtrai as quantidades do corredor antigo
    for item, qtd in problema.aisles[corredor_antigo].items():
        sol_vizinha.universoC[item] -= qtd
        sol_vizinha.itensC[item]   -= qtd

    # 3) adiciona as quantidades do corredor novo
    for item, qtd in problema.aisles[novo_c].items():
        sol_vizinha.universoC[item] = sol_vizinha.universoC.get(item, 0) + qtd
        sol_vizinha.itensC[item]   = sol_vizinha.itensC.get(item, 0)   + qtd

    # 4) atualiza as listas de corredores
    sol_vizinha.corredores[pos] = novo_c
    sol_vizinha.corredoresDisp[corredor_antigo] = False
    sol_vizinha.corredoresDisp[novo_c] = True

    return sol_vizinha


def atualizaPedidos(solucao: Metodos.Solucao, problema: Processa.Problema, sol_vizinha, novo_p, pos):
    """
    Args:
        solucao (Solucao): Dataclass representando a solução inicial, incluindo estruturas auxiliares.
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        sol_vizinha ():
        novo_p: ID do pedido que entra
        pos: posição na lista solucao.pedidos onde ocorre a troca
    """

    # 1) ID do pedido antigo
    pedido_antigo = solucao.pedidos[pos]

    # 2) subtrai as quantidades do pedido antigo
    for item, qtd in problema.orders[pedido_antigo].items():
        sol_vizinha.itensP[item] -= qtd
        # se restar zero, podemos opcionalmente remover a chave:
        if sol_vizinha.itensP[item] == 0:
            del sol_vizinha.itensP[item]

    # 3) adiciona as quantidades do novo pedido
    for item, qtd in problema.orders[novo_p].items():
        sol_vizinha.itensP[item] = sol_vizinha.itensP.get(item, 0) + qtd

    # 4) atualiza a lista de pedidos e o vetor de disponibilidade
    sol_vizinha.pedidos[pos]        = novo_p
    sol_vizinha.pedidosDisp[pedido_antigo] = 0
    sol_vizinha.pedidosDisp[novo_p]       = 1

    # 5) recalcula qntItens como soma de todos os itens em itensP
    sol_vizinha.qntItens = sum(sol_vizinha.itensP.values())

    return sol_vizinha


def gerar_sol_vizinha(solucao: Metodos.Solucao, problema: Processa.Problema, tipo: str, clusters_ped, clusters_corr, label_pedidos, label_corredores) -> Metodos.Solucao | None:
    """
    Função responsável por gerar um vizinho da solução trocando ou um pedido ou um corredor dentro do mesmo cluster.

    Args:
        solucao (Solucao): Dataclass representando a solução inicial, incluindo estruturas auxiliares.
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        tipo (str): String simbolizando qual vai ser a troca.
        clusters_ped ():
        clusters_corr ():
        label_pedidos ():
        label_corredores ():

    Returns:
        sol_vizinho (Solucao | None):
    """

    sol_vizinha = solucao.clone()

    if tipo == 'pedido':
        # Criando lista com os índices dos label_pedidos utilizados na solução
        ativos = list(range(len(sol_vizinha.pedidos)))
        if not ativos:
            return None
        # Escolhendo um pedido ativo aleatoriamente para alterar
        i = choice(ativos)
        # id do pedido atual nessa posição
        pedido_atual = sol_vizinha.pedidos[i]
        # rótulo dele
        lab = label_pedidos[pedido_atual]
        # Criando lista de candidatos dentro do mesmo cluster
        candidatos = [p for p in clusters_ped[lab] if p != sol_vizinha.pedidos[i]]
        if not candidatos:
            return None
        # Escolhendo um pedido candidato aleatoriamente
        novo_p = choice(candidatos)

        sol_vizinha = atualizaPedidos(solucao, problema, sol_vizinha, novo_p, i)

        # Verificando se o novo pedido não ultrapassa a capacidade do corredor
        if sol_vizinha.qntItens > problema.ub or novo_p in solucao.pedidos or sol_vizinha.qntItens < problema.lb:
            return None

        sol_vizinha.pedidos[i] = novo_p
        sol_vizinha.qntItens = sum(sol_vizinha.itensP.values())

        # antes de trocar efetivamente o pedido, verifique viabilidade:
        for item, qtd in problema.orders[novo_p].items():
            demanda_antes = sol_vizinha.itensP.get(item, 0)
            capacidade   = sol_vizinha.itensC.get(item, 0)
            if demanda_antes + qtd > capacidade:
                # não cabe este pedido na oferta atual
                return None

    else:
        ativos = list(range(len(sol_vizinha.corredores)))
        if not ativos:
            return None
        i = choice(ativos)
        pedido_atual = sol_vizinha.corredores[i]
        lab = label_corredores[pedido_atual]
        candidatos = [c for c in clusters_corr[lab] if c != sol_vizinha.corredores[i]]
        if not candidatos:
            return None
        novo_c = choice(candidatos)
        if novo_c in solucao.corredores:
            return None
        else:
            sol_vizinha = atualizaCorredores(solucao, problema, sol_vizinha, novo_c, i)

    return sol_vizinha


def refinamento_cluster_vns(problema: Processa.Problema, solucao: Metodos.Solucao) -> Metodos.Solucao:
    """
    Função responsável por executar um VNS simples alternando entre vizinhança de pedidos e de corredores.

    Redireciona para a melhor_vizinhanca se o tamanho do problema for menor do que 10.

    Args:
        solucao (Solucao): Dataclass representando a solução inicial, incluindo estruturas auxiliares.
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).

    Returns:
        best (Solucao): Dataclass representando a solução refinada, incluindo estruturas auxiliares.
    """

    inicio = perf_counter()
    best = solucao
    iter_sem_melhora = 0
    k = 1

    if problema.a < 10 or problema.o < 10:
        # print("Problema muito pequeno para clusterização.")
        return melhor_vizinhanca(problema, solucao)

    pedidos, corredores = clusterizacao_MBKM(problema)
    clusters_ped, clusters_corr = construir_clusters_de_dicts(pedidos, corredores)


    while iter_sem_melhora < 1000:
        if k%4 < 2:
            tipo = 'pedido'
        else:
            tipo = 'corredor'
        # Gera uma solução vizinha
        viz = gerar_sol_vizinha(best, problema, tipo, clusters_ped, clusters_corr, pedidos, corredores)
        if viz is None:
            k += 1
            iter_sem_melhora += 1
            continue

        viz.objetivo = Metodos.funcao_objetivo(problema, viz.itensP, viz.itensC)/viz.qntCorredores
        k += 1

        # Se melhorou, aceite e reinicie vizinhança
        if viz.objetivo > best.objetivo:
            #print(f"Melhorou: {viz.objetivo:.2f} > {best.objetivo:.2f}")
            best = viz
            clusters_ped, clusters_corr = construir_clusters_de_dicts(pedidos, corredores)
            iter_sem_melhora = 0
        else:
            # muda de vizinhançax'
            iter_sem_melhora += 1

    fim = perf_counter()
    best.tempo += fim - inicio
    # print(f"Refinamento concluído em {fim - inicio:.2f}s, objetivo final = {best.objetivo:.2f}")
    return best


def clusterizacao_MBKM(problema):
    tam = problema.i + 1
    pedidos = []
    corredores = []

    # Vetorização dos pedidos
    for pedido in problema.orders:
        v = np.zeros(tam)
        for item, qtd in pedido.items():
            v[item] = qtd
        pedidos.append(v)

    # Vetorização dos corredores
    for corredor in problema.aisles:
        v = np.zeros(tam)
        for item, qtd in corredor.items():
            v[item] = qtd
        corredores.append(v)

    X_pedidos    = np.vstack(pedidos)    # shape = (n_pedidos, tam)
    X_corredores = np.vstack(corredores) # shape = (n_corredores, tam)

    pedidos    = MiniBatchKMeans(n_clusters=10, batch_size=1024, random_state=0)
    corredores = MiniBatchKMeans(n_clusters=10, batch_size=1024, random_state=0)

    labels_pedidos    = pedidos.fit_predict(X_pedidos)
    labels_corredores = corredores.fit_predict(X_corredores)

    return labels_pedidos, labels_corredores


def melhor_vizinhanca(problema: Processa.Problema, solucao: Metodos.Solucao) -> Metodos.Solucao:
    """
    Heurística de refinamento baseada em melhor vizinhança.

    Procura qual das vizinhanças possíveis (adicionando corredor, trocando corredores, removendo corredor) tem o melhor valor de função objetivo. Tanto os corredores, como os pedidos adicionados quando possível, são escolhidos de forma gulosa (corredores por peso, e pedidos por quantidade de itens).

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução inicial, incluindo estruturas auxiliares.

    Returns:
        solucao (Solucao): Dataclass representando a solução refinada, incluindo estruturas auxiliares.
    """

    inicio = perf_counter()

    # Removendo os corredores redundantes da solução inicial.
    solucao = Metodos.remove_redundantes(problema, solucao)

    # Calculando a demanda de cada item.
    demanda_por_item = defaultdict(int)             # Dicionário da soma total da demanda de cada item em todos os pedidos.
    for pedido in problema.orders:
        for item, qtd in pedido.items():
            demanda_por_item[item] += qtd

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