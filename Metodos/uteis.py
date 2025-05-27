import Processa
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Solucao:
    universoC: Dict[int, int]     # Universo dos itens disponíveis nos corredores selecionados.
    itensC: Dict[int, int]        # Universo dos itens totais nos corredores selecionados.
    itensP: Dict[int, int]        # Universo dos itens totais nos pedidos selecionados.
    corredores: List[int]         # Índices dos corredores na solução.
    corredoresDisp: List[int]     # Lista binária representando se o corredor da posição x foi selecionado (1) ou não (0).
    pedidos: List[int]            # Índices dos pedidos na solução.
    pedidosDisp: List[int]        # Lista binária representando se o pedido da posição x for selecionado (1) ou não (0).
    qntItens: int                 # Quantidade total de itens nos pedidos selecionados.
    qntCorredores: int            # Quantidade de corredores selecionados.
    objetivo: float               # Valor da função objetivo para a solução encontrada.
    tempo: float                  # Tempo de execução da heurística.

    def clone(self):
        return Solucao(
            self.universoC.copy(),
            self.itensC.copy(),
            self.itensP.copy(),
            self.corredores[:],
            self.corredoresDisp[:],
            self.pedidos[:],
            self.pedidosDisp[:],
            self.qntItens,
            self.qntCorredores,
            self.objetivo,
            self.tempo
        )

def adiciona_pedidos(problema: Processa.Problema, solucao: Solucao):
    """
    Função responsável por adicionar os pedidos viáveis de forma gulosa, usando a estratégia da cobertura, na solução informada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
    """

    # Filtrando os pedidos possíveis para os corredores selecionados.
    pedidos_viaveis = []                        # Lista de pedidos viáveis com os corredores atualmente selecionados.
    for indice in range(problema.o):
        if not solucao.pedidosDisp[indice]:
            valida = True
            itens_totais = 0
            for item, qnt in problema.orders[indice].items():
                itens_totais += qnt
                if qnt > solucao.universoC[item]:
                    valida = False
                    break
            if valida:
                pedidos_viaveis.append([indice, itens_totais])

    # Selecionando os melhores pedidos (os que tem mais itens e que não quebram a restrição de ub).
    pedidos_viaveis.sort(key = lambda i: i[1], reverse = True)
    for pedido in pedidos_viaveis:
        valida = True
        for item, qnt in problema.orders[pedido[0]].items():
            if qnt > solucao.universoC[item]:
                valida = False
                break
        if valida and solucao.qntItens + pedido[1] <= problema.ub:
            solucao.qntItens += pedido[1]
            solucao.pedidosDisp[pedido[0]] = 1
            solucao.pedidos.append(pedido[0])
            for item, qnt in problema.orders[pedido[0]].items():
                solucao.universoC[item] -= qnt
                solucao.itensP[item] += qnt

def adiciona_corredor(problema: Processa.Problema, solucao: Solucao, corredor_max: int):
    """
    Função responsável por adicionar um novo corredor na solução informada. Não adiciona novos pedidos.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
        corredor_max (int): Índice do corredor que será inserido.
    """

    # Adicionando o novo corredor se ele existe.
    if corredor_max >= 0 and corredor_max < problema.a:
        solucao.corredores.append(corredor_max)
        solucao.corredoresDisp[corredor_max] = 1
        solucao.qntCorredores += 1

        for item, qnt in problema.aisles[corredor_max].items():
            solucao.itensC[item] += qnt
            solucao.universoC[item] += qnt

def troca_corredor(problema: Processa.Problema, solucao: Solucao, corredor_max: int, corredor_min: int):
    """
    Função responsável por trocar dois corredores na solução informada. Também remove todos os pedidos da solução para eles serem adicionados novamente pela estratégia da cobertura.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
        corredor_max (int): Índice do corredor que será inserido.
        corredor_min (int): Índice do corredor que será removido.
    """

    # Trocando os corredores caso o corredor de peso máximo exista.
    if corredor_max >= 0 and corredor_max < problema.a:
        solucao.corredores.remove(corredor_min)
        solucao.corredoresDisp[corredor_min] = 0
        for item, qnt in problema.aisles[corredor_min].items():
            solucao.itensC[item] -= qnt

        solucao.corredores.append(corredor_max)
        solucao.corredoresDisp[corredor_max] = 1
        for item, qnt in problema.aisles[corredor_max].items():
            solucao.itensC[item] += qnt

        # Redefinindo solução para começar a inserir pedidos do 0.
        solucao.universoC = solucao.itensC.copy()
        solucao.pedidos = []
        solucao.pedidosDisp = [0 for _ in range(problema.o)]
        solucao.itensP = dict.fromkeys(range(problema.i), 0)
        solucao.qntItens = 0

def remove_corredor(problema: Processa.Problema, solucao: Solucao, corredor_min: int):
    """
    Função responsável por remover um corredor da solução informada. Também remove todos os pedidos da solução para eles serem adicionados novamente pela estratégia da cobertura.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
        corredor_min (int): Índice do corredor que será removido.
    """

    # Removendo o corredor.
    if solucao.qntCorredores > 1:
        solucao.corredores.remove(corredor_min)
        solucao.corredoresDisp[corredor_min] = 0
        solucao.qntCorredores -= 1
        for item, qnt in problema.aisles[corredor_min].items():
            solucao.itensC[item] -= qnt

        # Redefinindo solução para começar a inserir pedidos do 0.
        solucao.universoC = solucao.itensC.copy()
        solucao.pedidos = []
        solucao.pedidosDisp = [0 for _ in range(problema.o)]
        solucao.itensP = dict.fromkeys(range(problema.i), 0)
        solucao.qntItens = 0

def remove_redundantes(problema: Processa.Problema, solucao: Solucao):
    """
    Função responsável por remover os corredores redundantes da solução informada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
    """

    # Procurando corredores redundantes, sem considerar que a remoção de um anterior possa tornar o atual não redundante.
    redundante_por_item = defaultdict(list)                     # Dicionário mapeando os itens para listas contendo o índice do corredor redundante, e a quantidade do item nele.
    corredores_importantes = [0 for _ in range(problema.a)]     # Lista de 0 e 1, sendo que 1 representa que o corredor é importante.
    for indice in solucao.corredores:
        redundante = True
        for item, qnt in problema.aisles[indice].items():
            if solucao.itensP[item] > solucao.universoC[item] - qnt:
                redundante = False
                corredores_importantes[indice] = 1
                break
        if redundante:
            for item in problema.aisles[indice].keys():
                redundante_por_item[item].append([indice, problema.aisles[indice][item]])

    # Verificando qual dos redundantes selecionados anteriormente deve permanecer na solução. Ordena os redundantes do item atual pela quantidade do item, e vai removendo os com menor quantidade até encontrar um que se torna importante.
    for item, qnt in solucao.itensP.items():
        if qnt != 0:
            qnt_removida = 0
            redundante_por_item[item].sort(key= lambda x : x[1])
            for indice, qnt in redundante_por_item[item]:
                if solucao.itensP[item] > solucao.universoC[item] - (qnt_removida + qnt):
                    corredores_importantes[indice] = 1
                else:
                    qnt_removida += qnt

    # Removendo os corredores redundantes.
    for indice in range(problema.a):
        if not corredores_importantes[indice] and solucao.corredoresDisp[indice]:
            solucao.corredores.remove(indice)
            solucao.corredoresDisp[indice] = 0
            solucao.qntCorredores -= 1
            for item, qnt in problema.aisles[indice].items():
                solucao.universoC[item] -= qnt
                solucao.itensC[item] -= qnt

def funcao_objetivo(problema: Processa.Problema, itensP: dict, itensC: dict) -> int:
    """
    Função que calcula a quantidade total de itens nos pedidos da solução, e verifica se alguma restrição foi violada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        itensP (dict): Dicionário mapeando os itens dos pedidos selecionados para a quantidade total.
        itensC (dict): Dicionário mapeando os itens dos corredores selecionados para a quantidade total.

    Returns:
        soma (int): Quantidade total de itens em itensP `ou` 0 se alguma restrição foi violada.
    """

    # Calculando quantidade total de itens.
    soma = sum(itensP.values())
    # Verificando as restrições de limites.
    if soma < problema.lb or soma > problema.ub:
        return 0
    # Verificando a restrição de capacidade nos corredores.
    for p in itensP:
        if itensP[p] > itensC[p]:
            return 0
    # Retornando a soma.
    return soma


def peso_aresta(problema: Processa.Problema, corredor_id: int, pedido_id: int) -> int:
    """
    Calcula o peso da aresta entre um corredor e um pedido em um grafo bipartido.

    O peso corresponde à quantidade de itens faltantes para suprir o pedido com base na oferta disponível no corredor.

    Args:
        problema (Problema): Instância contendo os dados do problema, incluindo as ofertas dos corredores e as demandas dos pedidos.
        corredor_id (int): Índice do corredor cuja oferta será considerada.
        pedido_id (int): Índice do pedido cuja demanda será avaliada.

    Returns:
        faltantes (int): Peso da aresta entre o corredor e o pedido, representando a quantidade de itens faltantes.
    """

    oferta = problema.aisles[corredor_id]
    demanda = problema.orders[pedido_id]
    faltantes = 0

    for item, qnt in demanda.items():
        faltantes += max(0, qnt - oferta.get(item, 0))

    return faltantes


def inicia_grafo(problema: Processa.Problema) -> Dict[int, List[tuple]]:
    """
    Inicializa o grafo bipartido que relaciona os corredores aos pedidos.

    Para cada corredor, a função cria uma lista de arestas para os pedidos, onde cada aresta é representada por uma tupla (pedido, peso). O peso de cada aresta é determinado pela quantidade de itens faltantes para suprir o pedido com a oferta do corredor. A lista de arestas para cada corredor é ordenada em ordem crescente de peso.

    Args:
        problema (Problema): Instância contendo os dados do problema, incluindo as estruturas 'aisles' (corredores) e 'orders' (pedidos).

    Returns:
        grafo (dict): Um dicionário representando o grafo bipartido, onde a chave é o índice do corredor e o valor é uma lista de tuplas (índice do pedido, peso da aresta).
    """

    grafo = defaultdict(list)
    for c_id in range(len(problema.aisles)):
        for p_id in range(len(problema.orders)):
            peso = peso_aresta(problema, c_id, p_id)
            grafo[c_id].append((p_id, peso))

        grafo[c_id].sort(key=lambda x: x[1])

    return grafo


def jaccard_distance(sol1: Solucao, sol2: Solucao) -> float:
    """
    Calcula a Distância Jaccard entre as máscaras de corredores de duas instâncias de Solucao.

    Args:
        sol1 (Solucao): A primeira solução.
        sol2 (Solucao): A segunda solução.

    Returns:
        float: A Distância Jaccard entre as duas soluções (0.0 se idênticas, 1.0 se disjuntas).
    """

    mask1 = sol1.corredores
    mask2 = sol2.corredores

    if not mask1:  # Trata o caso de máscaras vazias (numero_corredores = 0)
        return 0.0 if not mask2 else 1.0

    intersect_count = 0
    union_count = 0

    for i in range(len(mask1)):
        if not (mask1[i] ^ mask2[i]):
            intersect_count += 1

        if mask1[i] or mask2[i]:
            union_count += 1

    if union_count == 0:
        # Ambas as máscaras são compostas apenas por zeros (nenhum corredor selecionado)
        return 0.0  # Consideradas idênticas

    jaccard_index = intersect_count / union_count
    return 1.0 - jaccard_index