from dataclasses import dataclass
from typing import Dict, List

@dataclass
class solucao:
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
        return solucao(
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

"""
Descrição: função responsável por adicionar os pedidos gulosamente na solução informada.
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados; instancia do dataclass, contendo os elementos principais e auxiliares da solução.
Saída: instancia do dataclass, contendo os elementos principais e auxiliares da solução com os novos pedidos.
"""
def adiciona_pedidos(problema, solucao):
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

"""
Descrição: função responsável por adicionar um novo corredor na solução informada.
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados; instancia do dataclass, contendo os elementos principais e auxiliares da solução; índice do corredor que será adicionado.
Saída: instancia do dataclass, contendo os elementos principais e auxiliares da solução com o novo corredor.
"""
def adiciona_corredor(problema, solucao, corredor_max):
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

"""
Descrição: função responsável por trocar os dois corredores informados na solução informada.
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados; instancia do dataclass, contendo os elementos principais e auxiliares da solução; índice do corredor que será adicionado; índice do corredor que será removido.
Saída: instancia do dataclass, contendo os elementos principais e auxiliares da solução com o corredor trocado.
"""
def troca_corredor(problema, solucao, corredor_max, corredor_min):
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

"""
Descrição: função responsável por remover um corredor na solução informada.
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados; instancia do dataclass, contendo os elementos principais e auxiliares da solução; índice do corredor que será removido.
Saída: instancia do dataclass, contendo os elementos principais e auxiliares da solução com o novo corredor.
"""
def remove_corredor(problema, solucao, corredor_min):
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

"""
Descrição: função auxiliar responsável por remover os corredores redundantes de uma solução dada.
Entrada: instância do problema contendo corredores, pedidos, limites e demais dados.
Saída: instancia do dataclass, contendo os elementos principais e auxiliares da solução modificada.
"""
def remove_redundantes(problema, solucao):
    # Descobrindo quais corredores são redundantes (não considera casos nos quais se um corredor é marcado como redundante, o próximo pode não ser mais).
    corredores_redundantes = []
    for indice in solucao.corredores:
        redundante = True
        for item, qnt in problema.aisles[indice].items():
            if solucao.itensP[item] > solucao.universoC[item] - qnt:
                redundante = False
                break
        if redundante:
            corredores_redundantes.append(indice)

    # Descobrindo qual dos redundantes deve permanecer para não quebrar com os pedidos já selecionados (faz isso vendo qual corredor redundante tem a maior quantidade dos itens solicitados).
    corredores_importantes = [0 for _ in range(problema.a)]     # Lista de 0 e 1, sendo que 1 representa que o corredor é importante.
    for item, qnt in solucao.itensP.items():
        if qnt != 0:
            melhor_corredor = None
            maior_qnt = 0
            for indice in corredores_redundantes:
                valor = problema.aisles[indice].get(item, 0)
                if valor > maior_qnt:
                    melhor_corredor = indice
                    maior_qnt = valor
            if melhor_corredor is not None:
                corredores_importantes[melhor_corredor] = 1

    # Removendo os corredores que realmente são redundantes.
    corredores_final = [c for c in corredores_redundantes if not corredores_importantes[c]]
    for indice in corredores_final:
        solucao.corredores.remove(indice)
        solucao.corredoresDisp[indice] = 0
        solucao.qntCorredores -= 1
        for item, qnt in problema.aisles[indice].items():
            solucao.universoC[item] -= qnt
            solucao.itensC[item] -= qnt

    return solucao

"""
Descrição: função que calcula a quantidade total de itens nos pedidos da solução, e verifica se alguma restrição foi violada.
Entrada: objeto do problema, itens dos pedidos da solução, itens dos corredores da solução.
Saída: quantidade de itens ou 0 caso uma restrição tenha sido violada.

Observação: para otimizar a função, é responsabilidade dos métodos gerenciar os dicionários dos itens, e dividir o valor retornado pela quantidade de corredores.
"""
def funcao_objetivo(problema, itensP, itensC):
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