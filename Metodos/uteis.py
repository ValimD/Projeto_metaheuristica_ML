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

def adiciona_pedidos(problema: Processa.Problema, solucao: Solucao) -> Solucao:
    """
    Função responsável por adicionar os pedidos viáveis de forma gulosa na solução informada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.

    Returns:
        solucao (Solucao): Dataclass representando a solução com os novos pedidos, incluindo estruturas auxiliares.
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

    return solucao

def adiciona_corredor(problema: Processa.Problema, solucao: Solucao, corredor_max: int) -> Solucao:
    """
    Função responsável por adicionar um novo corredor na solução informada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
        corredor_max (int): Índice do corredor que será inserido.

    Returns:
        solucao (Solucao): Dataclass representando a solução com o novo corredor, incluindo estruturas auxiliares.
    """

    primeira_vizinhanca = solucao.clone()       # Variáveis da vizinhança baseada na adição de um corredor.

    # Adicionando o novo corredor se ele existe.
    if corredor_max >= 0 and corredor_max < problema.a:
        primeira_vizinhanca.corredores.append(corredor_max)
        primeira_vizinhanca.corredoresDisp[corredor_max] = 1
        primeira_vizinhanca.qntCorredores += 1

        for item, qnt in problema.aisles[corredor_max].items():
            primeira_vizinhanca.itensC[item] += qnt
            primeira_vizinhanca.universoC[item] += qnt

        # Adicionando pedidos se possível.
        primeira_vizinhanca = adiciona_pedidos(problema, primeira_vizinhanca)

    return primeira_vizinhanca

def troca_corredor(problema: Processa.Problema, solucao: Solucao, corredor_max: int, corredor_min: int) -> Solucao:
    """
    Função responsável por trocar dois corredores na solução informada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
        corredor_max (int): Índice do corredor que será inserido.
        corredor_min (int): Índice do corredor que será removido.

    Returns:
        solucao (Solucao): Dataclass representando a solução com os corredores trocados, incluindo estruturas auxiliares.
    """

    segunda_vizinhanca = solucao.clone()        # Variáveis da vizinhança baseada na troca de corredores.

    # Trocando os corredores caso o corredor de peso máximo exista.
    if corredor_max >= 0 and corredor_max < problema.a:
        segunda_vizinhanca.corredores.remove(corredor_min)
        segunda_vizinhanca.corredoresDisp[corredor_min] = 0
        for item, qnt in problema.aisles[corredor_min].items():
            segunda_vizinhanca.itensC[item] -= qnt
            segunda_vizinhanca.universoC[item] -= qnt

        segunda_vizinhanca.corredores.append(corredor_max)
        segunda_vizinhanca.corredoresDisp[corredor_max] = 1
        for item, qnt in problema.aisles[corredor_max].items():
            segunda_vizinhanca.itensC[item] += qnt
            segunda_vizinhanca.universoC[item] += qnt

        # Removendo todos os pedidos inviáveis, sem considerar que a remoção de um anterior possa deixar os próximos viáveis.
        pedidos_inviaves = []                   # Lista de pedidos inviáveis com os corredores atualmente selecionados.

        for indice in segunda_vizinhanca.pedidos:
            for item in problema.orders[indice]:
                if segunda_vizinhanca.itensP[item] > segunda_vizinhanca.universoC[item]:
                    pedidos_inviaves.append(indice)
                    break

        for indice in pedidos_inviaves:
            segunda_vizinhanca.pedidos.remove(indice)
            segunda_vizinhanca.pedidosDisp[indice] = 0
            for item, qnt in problema.orders[indice].items():
                segunda_vizinhanca.universoC[item] += qnt
                segunda_vizinhanca.itensP[item] -= qnt
                segunda_vizinhanca.qntItens -= qnt

        # Adicionando pedidos se possível.
        segunda_vizinhanca = adiciona_pedidos(problema, segunda_vizinhanca)

    return segunda_vizinhanca

def remove_corredor(problema: Processa.Problema, solucao: Solucao, corredor_min: int) -> Solucao:
    """
    Função responsável por remover um corredor da solução informada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.
        corredor_min (int): Índice do corredor que será removido.

    Returns:
        solucao (Solucao): Dataclass representando a solução com o corredor removido, incluindo estruturas auxiliares.
    """

    terceira_vizinhanca = solucao.clone()       # Variáveis da vizinhança baseada na remoção de um corredor.

    # Removendo o corredor.
    if terceira_vizinhanca.qntCorredores > 1:
        terceira_vizinhanca.corredores.remove(corredor_min)
        terceira_vizinhanca.corredoresDisp[corredor_min] = 0
        terceira_vizinhanca.qntCorredores -= 1
        for item, qnt in problema.aisles[corredor_min].items():
            terceira_vizinhanca.itensC[item] -= qnt
            terceira_vizinhanca.universoC[item] -= qnt

        # Removendo todos os pedidos inviáveis, sem considerar que a remoção de um anterior possa deixar os próximos viáveis.
        pedidos_inviaves = []                   # Lista de pedidos inviáveis com os corredores atualmente selecionados.

        for indice in terceira_vizinhanca.pedidos:
            for item in problema.orders[indice]:
                if terceira_vizinhanca.itensP[item] > terceira_vizinhanca.universoC[item]:
                    pedidos_inviaves.append(indice)
                    break

        for indice in pedidos_inviaves:
            terceira_vizinhanca.pedidos.remove(indice)
            terceira_vizinhanca.pedidosDisp[indice] = 0
            for item, qnt in problema.orders[indice].items():
                terceira_vizinhanca.universoC[item] += qnt
                terceira_vizinhanca.itensP[item] -= qnt
                terceira_vizinhanca.qntItens -= qnt

        # Adicionando pedidos se possível.
        terceira_vizinhanca = adiciona_pedidos(problema, terceira_vizinhanca)

    return terceira_vizinhanca

def remove_redundantes(problema: Processa.Problema, solucao: Solucao) -> Solucao:
    """
    Função responsável por remover os corredores redundantes da solução informada.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        solucao (Solucao): Dataclass representando a solução, incluindo estruturas auxiliares.

    Returns:
        solucao (Solucao): Dataclass representando a solução com os corredores removidos, incluindo estruturas auxiliares.
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

    return solucao

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