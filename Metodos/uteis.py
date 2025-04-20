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