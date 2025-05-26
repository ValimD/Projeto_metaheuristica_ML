import Metodos
import Processa
import statistics
from itertools import islice
from random import randint
from time import perf_counter

def calcula_componente(problema: Processa.Problema, componente_atual: set, particula: dict) -> set:
    """
    Função responsável por montar o conjunto contendo os índices dos corredores que serão adicionados e/ou removidos, usando como base o conjunto informado.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).
        componente_atual (set): Conjunto contendo os índices dos corredores que serão adicionados (conjunto da melhor posição individual, ou global, ou velocidade anterior).
        particula (dict): Dicionário contendo as informações da partícula, como posição atual.

    Returns:
        componente (set): Conjunto contendo os índices dos corredores que podem ser adicionados e removidos.
    """

    # Calculando os corredores que podem ser adicionados na solução (elementos que estão no componente atual, mas que não estão na posição atual).
    componente = componente_atual - particula["Xt"]

    # Verificando o 0 que aparece é para ser removido ou adicionado (-0 não existe na operação seguinte).
    remover_zero = True
    if 0 in componente:
        remover_zero = False

    # Calculando os corredores que podem ser removidos (elementos que estão na posição atual, mas que não estão no componente).
    componente = componente | {-corredor for corredor in particula["Xt"] if corredor not in componente_atual}

    # Como 0 não pode ser representado como negativo (para marcar a remoção), adiciona a quantidade de corredores da instância no lugar (uma vez que os índices vão até essa quantidade - 1).
    if 0 in componente and remover_zero:
        componente.discard(0)
        componente.add(problema.a)

    return componente

def remove_corredor_temp(problema: Processa.Problema, solucao: Metodos.Solucao, corredor: int):
    # Removendo o corredor da solução.
    solucao.corredores.remove(corredor)
    solucao.qntCorredores -= 1
    solucao.corredoresDisp[corredor] = 0.
    for item, qnt in problema.aisles[corredor].items():
        solucao.itensC[item] -= qnt

def adiciona_pedidos_temp(problema: Processa.Problema, solucao: Metodos.Solucao):
    # Redefinindo solução para começar a inserir pedidos do 0.
    solucao.universoC = solucao.itensC.copy()
    solucao.pedidos = []
    solucao.pedidosDisp = [0 for _ in range(problema.o)]
    solucao.itensP = dict.fromkeys(range(problema.i), 0)
    solucao.qntItens = 0

    # Adicionando os pedidos por cobertura máxima.
    Metodos.adiciona_pedidos(problema, solucao)

def pso_discreto(problema: Processa.Problema) -> Metodos.Solucao:
    """
    Metaheurística PSO adaptada para os casos discretos desse problema.

    Args:
        problema (Problema): Instância contendo os dados do problema (corredores, pedidos, limites).

    Returns:
        solucao (Solucao): Dataclass representando a melhor solução da população, incluindo estruturas auxiliares.
    """

    inicio = perf_counter()

    # Preparando ambiente.
    tamanho_enxame = 20                             # Quantidade de partículas geradas para o enxame.
    parcela_enxame = tamanho_enxame * 50 / 100      # Quantidade de partículas geradas aleatórias e quantidade gerada de forma híbrida (híbrida é a menor parte).
    solucao = None                                  # Melhor solução encontrada.
    posicao_global = None                           # Conjunto de corredores da melhor solução.

    constante_cognitivo = 2                         # Componente cognitivo do PSO (c1).
    constante_social = 2                            # Componente social do PSO (c2).
    inercia = 1                                     # Peso da inércia (w).

    # Gerando soluções iniciais.
    objetivos = []                                  # Lista com todas as funções objetivos do enxame atual, para o cálculo desvio padrão.
    enxame = []                                     # Lista armazenando todas as partículas do enxame (dicionários).
    for i in range(tamanho_enxame):
        if i > parcela_enxame:
            particula = Metodos.hibrida(problema)
        else:
            particula = Metodos.aleatorio(problema)

        objetivos.append(particula.objetivo)
        enxame.append({"solucao": particula, "Xt": set(particula.corredores), "P": set(particula.corredores), "Op": particula.objetivo, "Vt_1": set(particula.corredores), "Vt": set(particula.corredores)})
        if solucao == None or enxame[i]["solucao"].objetivo > solucao.objetivo:
            solucao = enxame[i]["solucao"].clone()
            posicao_global = set(particula.corredores)

    print(f"Melhor solução da geração inicial: {solucao.objetivo}")

    # Iniciando iterações.
    desvio = statistics.stdev(objetivos)            # Desvio padrão inicial.
    geracao_maxima = 2000                           # Geração máxima do enxame.
    geracao_atual = 0                               # Geração atual do enxame.
    ponto_convergencia = 3 * geracao_maxima / 4     # Ponto de alteração na inércia.
    while geracao_atual < geracao_maxima and desvio > 0.001:
        print(desvio, solucao.objetivo)
        # Verificando inércia.
        if geracao_atual > ponto_convergencia:
            inercia = 0

        # Calculando as velocidades.
        for i in range(tamanho_enxame):
            # Calculando a inércia. No PSO original é calculada por (w * Vt), enquanto aqui é calculada pegando (inércia) corredores da velocidade atual.
            componente_inercia = calcula_componente(problema, enxame[i]["Vt"], enxame[i])
            componente_inercia = set(islice(componente_inercia, inercia))

            # Calculando o valor cognitivo. No PSO original é calculado por (c1 * [0, 1] * (Pi - Xi_t)), enquanto aqui é calculado pegando [0, c1] corredores da melhor posição da partícula.
            componente_cognitivo = calcula_componente(problema, enxame[i]["P"], enxame[i])
            componente_cognitivo = componente_cognitivo - componente_inercia
            enxame[i]["Vt"] = set(componente_cognitivo)

            componente_cognitivo = set(islice(componente_cognitivo, randint(0, constante_cognitivo)))

            # Calculando o valor social. No PSO original é calculado por (c2 * [0, 1] * (G + Xi_t)), enquanto aqui é calculado pegando [0, c2] corredores da melhor posição do enxame.
            componente_social = calcula_componente(problema, posicao_global, enxame[i])
            componente_social = componente_social - (componente_inercia | componente_cognitivo)
            enxame[i]["Vt"] = enxame[i]["Vt"] | componente_social

            componente_social = set(islice(componente_social, randint(0, constante_social)))

            # Calculando a nova velocidade (Vt_1). Se a quantidade de corredores que serão removidos for maior do que a quantidade atual de corredores, seleciona (quantidade de corredores atual) - 1 para remover.
            velocidade_negativa = set()
            velocidade_positiva = set()
            for velocidades in [componente_inercia, componente_cognitivo, componente_social]:
                if len(velocidades):
                    if next(iter(velocidades)) < 0 or problema.a in velocidades:
                        velocidade_negativa = velocidade_negativa | velocidades
                    else:
                        velocidade_positiva = velocidade_positiva | velocidades

            if len(velocidade_negativa) - len(velocidade_positiva) >= enxame[i]["solucao"].qntCorredores:
                velocidade_negativa = set(islice(velocidade_negativa, enxame[i]["solucao"].qntCorredores - 1))

            velocidade = velocidade_positiva | velocidade_negativa
            enxame[i]["Vt_1"] = velocidade
            enxame[i]["Vt"] = enxame[i]["Vt"] - enxame[i]["Vt_1"]

        # Andando de acordo com as velocidades, corrigindo pedidos e atualizando universo.
        for i in range(tamanho_enxame):
            # Modificando as posições.
            for corredor in enxame[i]["Vt_1"]:
                if corredor < 0:
                    # Removendo corredor.
                    remove_corredor_temp(problema, enxame[i]["solucao"], corredor * -1)
                elif corredor == problema.a:
                    # Removendo corredor 0.
                    remove_corredor_temp(problema, enxame[i]["solucao"], 0)
                else:
                    # Adicionando corredor.
                    enxame[i]["solucao"].corredores.append(corredor)
                    enxame[i]["solucao"].qntCorredores += 1
                    enxame[i]["solucao"].corredoresDisp[corredor] = 1
                    for item, qnt in problema.aisles[corredor].items():
                        enxame[i]["solucao"].itensC[item] += qnt

            # Adicionando novos pedidos usando cobertura.
            adiciona_pedidos_temp(problema, enxame[i]["solucao"])

            # Atualizando universo.
            enxame[i]["Xt"] = set(enxame[i]["solucao"].corredores)
            enxame[i]["solucao"].objetivo = Metodos.funcao_objetivo(problema, enxame[i]["solucao"].itensP, enxame[i]["solucao"].itensC) / enxame[i]["solucao"].qntCorredores

            if enxame[i]["solucao"].objetivo > enxame[i]["Op"]:
                enxame[i]["Op"] = enxame[i]["solucao"].objetivo
                enxame[i]["P"] = set(enxame[i]["Xt"])

            if enxame[i]["solucao"].objetivo > solucao.objetivo:
                solucao = enxame[i]["solucao"].clone()
                posicao_global = set(enxame[i]["Xt"])

            objetivos[i] = enxame[i]["solucao"].objetivo

        desvio = statistics.stdev(objetivos)
        geracao_atual += 1

    fim = perf_counter()
    solucao.tempo = fim - inicio

    print(f"Melhor solução da geração {geracao_atual}: {solucao.objetivo}")
    print(f"Desvio padrão: {desvio}, tempo: {solucao.tempo}")
    return solucao