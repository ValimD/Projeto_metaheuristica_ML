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