import Metodos
import numpy as np
from time import perf_counter
from collections import defaultdict
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
from random import choice

def construir_clusters_de_dicts(labels_pedidos, labels_corredores):
    """
    Constrói dicts de cluster->{lista de índices} para pedidos e corredores.
    """
    clusters_ped = defaultdict(list)
    for idx, lab in enumerate(labels_pedidos):
        clusters_ped[lab].append(idx)
    clusters_corr = defaultdict(list)
    for idx, lab in enumerate(labels_corredores):
        clusters_corr[lab].append(idx)
    return clusters_ped, clusters_corr


def atualizaCorredores(solucao, problema, sol_vizinha, novo_c, pos):
    """
    pos: posição na lista solucao.corredores onde trocamos o corredor
    novo_c: ID do corredor que entra
    problema: para acessar problema.aisles
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
    sol_vizinha.corredores[pos]    = novo_c
    sol_vizinha.corredoresDisp[corredor_antigo] = False
    sol_vizinha.corredoresDisp[novo_c]         = True

    return sol_vizinha


def atualizaPedidos(solucao, problema, sol_vizinha, novo_p, pos):
    """
    pos: posição na lista solucao.pedidos onde ocorre a troca
    novo_p: ID do pedido que entra
    problema: para acessar problema.orders
    """

    # 1) identifica ID do pedido antigo
    pedido_antigo = solucao.pedidos[pos]

    # 2) subtrai as quantidades do pedido antigo
    for item, qtd in problema.orders[pedido_antigo].items():
        sol_vizinha.universoP[item] -= qtd
        sol_vizinha.itensP[item]    -= qtd

    # 3) adiciona as quantidades do novo pedido
    for item, qtd in problema.orders[novo_p].items():
        sol_vizinha.universoP[item] = sol_vizinha.universoP.get(item, 0) + qtd
        sol_vizinha.itensP[item]    = sol_vizinha.itensP.get(item,    0) + qtd

    # 4) atualiza a lista de pedidos e o vetor de disponibilidade
    sol_vizinha.pedidos[pos]        = novo_p
    sol_vizinha.pedidosDisp[pedido_antigo] = False
    sol_vizinha.pedidosDisp[novo_p]       = True

    # 5) recalcula qntItens como soma de todos os itens em itensP
    sol_vizinha.qntItens = sum(sol_vizinha.itensP.values())

    return sol_vizinha


def gerar_sol_vizinha(solucao, problema, tipo, clusters_ped, clusters_corr, label_pedidos, label_corredores):
    """
    Gera um vizinho da solucao trocando ou um pedido ou um corredor dentro do mesmo cluster.
    tipo: 'pedido' ou 'corredor'
    Retorna nova_solucao.
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
        
        for item, qnt in novo_p.items():
            # Verificando se o novo pedido não ultrapassa a capacidade do corredor
            if sol_vizinha.itensC[item] < qnt + sol_vizinha.itensP[item]:
                return None

        sol_vizinha.pedidos[i] = novo_p
        sol_vizinha.qntItens = sum(label_pedidos[novo_p].values())
        for item in label_pedidos[novo_p]:
            # Verificando se o novo pedido não ultrapassa a capacidade do corredor
            if label_pedidos[novo_p][item] + sol_vizinha.pedidos[i][item] > solucao.limites[item]:
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


def refinamento_cluster_vns(problema, solucao):
    """
    Executa um VNS simples alternando entre vizinhança de pedidos e de corredores.
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
        tipo = 'pedido' if k < 1 else 'corredor'
        # Gera uma solução vizinha
        viz = gerar_sol_vizinha(best, problema, tipo, clusters_ped, clusters_corr, pedidos, corredores)
        if viz is None:
            if  k == 4:
                k = 1
                iter_sem_melhora += 1
                
            else:
                k += 1
                iter_sem_melhora += 1
            continue
                
            
        viz.objetivo = Metodos.funcao_objetivo(problema, viz.itensP, viz.itensC)
        
        # Se melhorou, aceite e reinicie vizinhança
        if viz.objetivo > best.objetivo:
            best = viz
            clusters_ped, clusters_corr = construir_clusters_de_dicts(pedidos, corredores)
            k = 1
            iter_sem_melhora = 0
        else:
            # muda de vizinhançax'
            k = 1 if k == 4 else k + 1
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
    demanda_por_item = defaultdict(int)             # Dicionário da soma total da demanda de cada item em todos os pedidos.
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