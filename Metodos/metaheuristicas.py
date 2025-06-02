import matplotlib.pyplot as plt
import Metodos
import numpy as np
import Processa
import statistics
import math
from collections import defaultdict
from itertools import islice
from random import choice, randint, random, sample, uniform, shuffle, choices
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

def PSO(problema: Processa.Problema) -> Metodos.Solucao:
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

    # Iniciando iterações.
    desvio = statistics.stdev(objetivos)            # Desvio padrão inicial.
    geracao_maxima = 2000                           # Geração máxima do enxame.
    geracao_atual = 0                               # Geração atual do enxame.
    ponto_convergencia = 3 * geracao_maxima / 4     # Ponto de alteração na inércia.
    while geracao_atual < geracao_maxima and desvio > 0.001:
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
                    Metodos.remove_corredor(problema, enxame[i]["solucao"], corredor * -1)
                elif corredor == problema.a:
                    # Removendo corredor 0.
                    Metodos.remove_corredor(problema, enxame[i]["solucao"], 0)
                else:
                    # Adicionando corredor.
                    Metodos.adiciona_corredor(problema, enxame[i]["solucao"], corredor)

            # Adicionando novos pedidos usando cobertura.
            enxame[i]["solucao"].universoC = enxame[i]["solucao"].itensC.copy()
            enxame[i]["solucao"].pedidos = []
            enxame[i]["solucao"].pedidosDisp = [0 for _ in range(problema.o)]
            enxame[i]["solucao"].itensP = dict.fromkeys(range(problema.i), 0)
            enxame[i]["solucao"].qntItens = 0
            Metodos.adiciona_pedidos(problema, enxame[i]["solucao"])

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

    return solucao

class FPO:
    def __init__(self, problema: Processa.Problema, iterations_num, pop_size, p=0.4, plot=True):
        """
        Construtor do FPO modificado.

        Agora utiliza a função construtiva híbrida para inicializar a população e operadores de refinamento para gerar vizinhos, manipulando corretamente o conjunto de dados.
        """

        self.problema = problema
        self.iterations_num = iterations_num
        self.pop_size = pop_size
        self.population = None  # lista de objetos Solucao
        self.best = None        # melhor solução encontrada
        self.p = p              # probabilidade de aplicar refinamento global
        self.objetivo = np.zeros(pop_size)
        self.plot = plot
        # self.distancias = {}

        # Para datasets menores, p = 0 garante mais velocidade e qualidade.
        # Para datasets maiores, valores de p menores garantem melhor qualidade mas perdem em tempo de execução
        if self.problema.ub < 500:
            self.p = 0.0

    def run(self) -> Metodos.Solucao:
        self.initialize_population()
        # self.calcular_matriz_distancias_populacao(self)
        self.calculate_obj()
        avg = []
        bests = []
        inicio = perf_counter()
        iterations_without_improve = 0
        self.best = self.population[0]
        for i in range(self.iterations_num):
            current_best_val = self.best.objetivo
            self.check_best()
            # Se há melhoria, reseta contador
            if self.best.objetivo > current_best_val:
                iterations_without_improve = 0
            else:
                iterations_without_improve += 1

            if iterations_without_improve > 0 and iterations_without_improve % 50 == 0:
                if self.p > 1:
                    self.p -= 0.1
                    print(f'p setado para {self.p} na iteração {i}/{self.iterations_num}')
                if iterations_without_improve > 250:
                    print(f'{iterations_without_improve} iterations without improve')
                    break



            self.pollination()
            if self.plot:
                avg.append(np.mean(self.objetivo))
                bests.append(max(self.objetivo))

        fim = perf_counter() - inicio
        # Adiciona o tempo à melhor solução
        self.best.tempo = fim

        # Se habilitado plot o gráfico
        if self.plot:
            plt.plot(avg, label="avg")
            plt.plot(bests, label="best")
            plt.yscale('log')
            plt.legend()
            plt.show()

        return self.best

    def initialize_population(self):
        # Utiliza a função construtiva híbrida para gerar a população inicial de soluções
        # top10 = (int)(self.pop_size/10)
        # top90 = self.pop_size - top10
        # self.population = [Metodos.hibrida(self.problema) for _ in range(top10)] + [Metodos.aleatorio(self.problema) for _ in range(top90)]
        self.population = [Metodos.hibrida(self.problema) for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            self.objetivo[i] = self.population[i].objetivo

    def calculate_obj(self):
        for i in range(self.pop_size):
            self.objetivo[i] = self.population[i].objetivo
        # Define o melhor a partir da população inicial
        best_idx = np.argmax(self.objetivo)
        self.best = self.population[best_idx]

    def check_best(self):
        current_best = max(self.objetivo)
        if self.best:
            if current_best > self.best.objetivo:
                best_idx = np.argmax(self.objetivo)
                self.best = self.population[best_idx]
        else:
            best_idx = np.argmax(self.objetivo)
            self.best = self.population[best_idx]


    def pollination(self) -> None:
        for i in range(self.pop_size):
            # Com probabilidade p, aplica refinamento global; caso contrário, utiliza refinamento local
            if random() < self.p:
                nova_sol = self.global_pollination(i)
            else:
                nova_sol = self.local_pollination(i)
            # Se o novo vizinho possui função objetivo melhor, substitui a solução atual
            if nova_sol:
                if nova_sol.qntCorredores:
                    nova_sol.objetivo = Metodos.funcao_objetivo(self.problema, nova_sol.itensP, nova_sol.itensC)/nova_sol.qntCorredores
                    if nova_sol.objetivo > self.objetivo[i]:
                        self.population[i] = nova_sol
                        self.objetivo[i] = nova_sol.objetivo
                else:
                    nova_sol.objetivo = 0


    def local_pollination_multisol(self, i) -> Metodos.Solucao:
        """
        Args:
            i: identificador da população

        Returns:
            solucao (Solucao): Dataclass representando a solução montada, incluindo estruturas auxiliares.
        """

        # Escolher 1 indivíduo aleatório
        sol1 = self.population[i]
        # Encontrar sol2 de acordo com a distância
        sol2 = self
        nova_sol = sol1.clone()
        # Identificar corredores de sol1 que não estão em sol2
        # TESTAR XOR NO LUGAR
        diferencas = {}
        for i in range(sol1.numCorredores):
            diferencas[i] = sol1.corredores[i] ^ sol2.corredores[i]
        if not diferencas:
            return Metodos.aleatorio(self.problema)
        # Sortear um índice dos corredores diferentes do indivíduo 1 (c1)
        c1 = choice(diferencas)
        # Sortear um índice dos corredores do indivíduo 2 (c2)
        c2 = choice(sol2.corredores)
        # Remover c1 e Adicionar c2
        nova_sol.corredores.remove(c1).append(c2)
        Metodos.remove_corredor(self.problema, nova_sol, c2)
        Metodos.adiciona_corredor(self.problema, nova_sol, c1)
        Metodos.adiciona_pedidos(self.problema, nova_sol)
        # Avaliar nova solução
        nova_sol.objetivo = Metodos.funcao_objetivo(self.problema, nova_sol.itensP, nova_sol.itensC)
        self.objetivo[i] = nova_sol.objetivo
        return nova_sol


    def local_pollination(self, i) -> Metodos.Solucao:
        """
        Args:
            i: identificador da população

        Returns:
            solucao (Solucao): Dataclass representando a solução montada, incluindo estruturas auxiliares.
        """

        populacao = self.population[i]
        nova_sol = populacao.clone()
        tam = self.problema.a - 1
        escolhido = choice(range(tam))

        if nova_sol.corredoresDisp[escolhido] == 0:
            Metodos.adiciona_corredor(self.problema, nova_sol, escolhido)
        else:
            Metodos.remove_corredor(self.problema, nova_sol, escolhido)

        Metodos.adiciona_pedidos(self.problema, nova_sol)

        return nova_sol


    def global_pollination(self, i) -> Metodos.Solucao:
        # Define o número de mudanças/ Força do polinizador
        num_levy = Metodos.get_levy_flight_array()
        diferencas = {}
        copia_sol = self.population[i].clone()
        tam = self.problema.a-1

        for j in range(tam):
            # Descobre as diferenças com a melhor solução
            diferencas[j] = copia_sol.corredoresDisp[j] ^ self.best.corredoresDisp[j]
        # Escolhe índices aleatórios dessas mudanças
        escolhidos = sample(range(tam), min(copia_sol.qntCorredores, num_levy))
        # Aplica mudanças nos corredores selecionados
        for j in range(min(num_levy - 1, len(escolhidos))):
            # 70% de chance de aplicar a mudança
            if random() < 0.7:
                if copia_sol.corredoresDisp[escolhidos[j]] == 0:
                    Metodos.adiciona_corredor(self.problema, copia_sol, escolhidos[j])
                else:
                    Metodos.remove_corredor(self.problema, copia_sol, escolhidos[j])

                Metodos.adiciona_pedidos(self.problema, copia_sol)

        return copia_sol


    def calcular_matriz_distancias_populacao(self) -> None:
        """
        Calcula e armazena as distâncias de Jaccard de cada indivíduo para todos os outros, ordenadas por proximidade.

        A estrutura resultante é armazenada em `self.distancias_ordenadas` (novo nome sugerido para o atributo para refletir a estrutura) e é um dicionário onde:
            - A chave é o índice de um indivíduo.
            - O valor é uma lista de tuplas `(distância, índice_do_vizinho)`, ordenada pela distância em ordem crescente.
        """

        # Etapa 1: Calcular todas as distâncias Jaccard únicas e armazenar temporariamente.
        # Usaremos uma matriz temporária para eficiência no cálculo, evitando redundância.
        # distancia_jaccard(pop[i], pop[j]) é simétrica.
        # Matriz temporária para armazenar as distâncias brutas (i, j)
        dist_matrix_temp = [[0.0] * self.pop_size for _ in range(self.pop_size)]

        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):  # Calcula apenas para j > i
                # Note que 'self.population[i]' deve ser do tipo Solucao
                # que é list[list[int]] conforme definido em uteis.py
                sol_i = self.population[i]
                sol_j = self.population[j]

                dist = Metodos.distancia_jaccard(sol_i, sol_j)
                dist_matrix_temp[i][j] = dist
                dist_matrix_temp[j][i] = dist  # A distância é simétrica

        # Etapa 2: Construir o dicionário de listas de vizinhos ordenadas.
        vizinhancas_ordenadas = {}
        for i in range(self.pop_size):
            distancias_para_i = []
            for j in range(self.pop_size):
                if i == j:
                    continue  # Não incluir um indivíduo em sua própria lista de vizinhos

                distancias_para_i.append((dist_matrix_temp[i][j], j))

            # Ordenar a lista de vizinhos para o indivíduo 'i' pela distância (o primeiro elemento da tupla)
            distancias_para_i.sort(key=lambda x: x[0])
            vizinhancas_ordenadas[i] = distancias_para_i
            self.distancias = vizinhancas_ordenadas

class ALNS:
    def __init__(self, problema, solucao, temperatura_inicial, taxa_resfriamento):
        self.problema           = problema
        self.sol_atual          = solucao
        self.sol_melhor         = solucao.clone()
        self.destruidores       = [self.destruidor_aleatorio, self.destruidor_bx_prod]
        self.reconstrutores     = [self.construtor_guloso, self.construtor_hibrido, self.construtor_aleatorio]
        self.peso_dest          = [1] * len(self.destruidores)
        self.peso_reco          = [1] * len(self.reconstrutores)
        self.temp               = temperatura_inicial
        self.taxa_resf          = taxa_resfriamento

    def destruidor_aleatorio(self, solucao, frac = 0.25):
        n = len(solucao.corredores)
        # se não houver corredores, nada a fazer
        if n == 0:
            return solucao

        # calcula k entre 0 e n, proporcional a frac
        k = int(frac * n)
        # garante ao menos 1 remoção, mas não mais que n
        k = max(1, k) if n >= 1 else 0
        k = min(k, n)

        to_remove = sample(solucao.corredores, k)
        for corredor in to_remove:
            Metodos.remove_corredor(self.problema, solucao, corredor)

        return solucao

    # Se existirem itens no UniversoC, ranqueia os corredores selecionados que mais possuem eles, e removem os 10% piores corredores.
    def destruidor_bx_prod(self, solucao, porcent = 0.25):
        k = max(1, int(porcent * len(solucao.corredores)))

        if not solucao.universoC or sum(solucao.universoC.values()) == 0:
            return solucao

        notas = {}
        # Cálculo das notas
        for corredor in solucao.corredores:
            notas[corredor] = sum(self.problema.aisles[corredor].get(item, 0) * solucao.universoC[item] for item in solucao.universoC)

        # Ranqueamento e seleção dos piores
        piores = sorted(notas, key=notas.get)[:k]

        for corredor in piores:
            Metodos.remove_corredor(self.problema, solucao, corredor)

        return solucao

    def construtor_guloso(self, solucao):
        problema = self.problema
        pedidos_ranqueados, corredores_ranqueados = Metodos.ranqueamento_guloso(self.problema, solucao)
        tentativas_sem_melhora = 0

        while tentativas_sem_melhora < 3 and corredores_ranqueados:
           
            # Selecionando o corredor de maior nota
            copiaSolucao = solucao.clone()
            corredor = corredores_ranqueados.pop()

            # Atualizando universo dos corredores
            if copiaSolucao.corredoresDisp[corredor] == 0:
                copiaSolucao.corredores.append(corredor)
                copiaSolucao.corredoresDisp[corredor] = 1
                copiaSolucao.qntCorredores += 1

                # Atualizando itens dos corredores selecionados
                for item, qnt in problema.aisles[corredor].items():
                    copiaSolucao.itensC[item] += qnt
                    copiaSolucao.universoC[item] += qnt
            
            else:
                continue

            # Adicionando pedidos
            pedidos_viaveis = []                        # Lista de pedidos viáveis com os corredores atualmente selecionados.
            for indice in pedidos_ranqueados:
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

            pedidos_ranqueados, corredores_ranqueados = Metodos.ranqueamento_guloso(self.problema, solucao)

        return solucao

    def construtor_hibrido(self, solucao, alpha = 0.3):
        # Calculando a demanda de cada item.
        demanda_por_item = defaultdict(int)             # Dicionário da soma total da demanda de cada item em todos os pedidos.
        for pedido in self.problema.orders:
            for item, qtd in pedido.items():
                demanda_por_item[item] += qtd

        # Calculando o peso de cada corredor com base na demanda e na quantidade ofertada.
        # Peso de cada corredor é calculado com base na demanda dos itens e na quantidade que ele oferece.
        peso_corredores = {indice: sum(demanda_por_item[item] * qnt for item, qnt in self.problema.aisles[indice].items()) for indice in range(self.problema.a)}

        # Buscando a melhor solução até ficar 3 iterações seguidas sem encontrar uma melhor.
        tentativas_sem_melhora = 0
        while tentativas_sem_melhora < 3 and peso_corredores:
            copiaSol = solucao.clone()

            # Selecionando um corredor ainda não utilizado.
            # Se todos os pesos forem zero, escolhe aleatoriamente. Caso contrário, utiliza seleção ponderada proporcional ao peso.
            total = sum(peso_corredores.values())
            if total == 0:
                corredor = choice(list(peso_corredores.keys()))
            else:
                escolhas, prob = zip(*[(indice, peso / total) for indice, peso in peso_corredores.items()])
                corredor = choices(escolhas, weights=prob, k=1)[0]

            # Atualizando universo dos corredores.
            if copiaSol.corredoresDisp[corredor] == 0:
                copiaSol.corredores.append(corredor)
                copiaSol.corredoresDisp[corredor] = 1
                copiaSol.qntCorredores += 1

                for item, qnt in self.problema.aisles[corredor].items():
                    copiaSol.itensC[item] += qnt
                    copiaSol.universoC[item] += qnt

            else:
                peso_corredores.pop(corredor)
                continue

            # Adicionando pedidos se possível.
            Metodos.adiciona_pedidos(self.problema, copiaSol)

            # Comparando as soluções, e salvando a atual caso seja melhor.
            copiaSol.objetivo = Metodos.funcao_objetivo(self.problema, copiaSol.itensP, copiaSol.itensC) / copiaSol.qntCorredores
            if copiaSol.objetivo > solucao.objetivo or copiaSol.qntItens < self.problema.lb or copiaSol.qntItens == 0:
                solucao = copiaSol
                peso_corredores.pop(corredor)
                tentativas_sem_melhora = 0
            else:
                tentativas_sem_melhora += 1

        return solucao

    def construtor_aleatorio(self, solucao):
        corredores_selecionados = list(range(self.problema.a))       # Lista dos corredores embaralhados.
        shuffle(corredores_selecionados)

            # Percorrendo os corredores.
        for corredor in corredores_selecionados:
            nova_solucao = solucao.clone()

            if nova_solucao.corredoresDisp[corredor] == 0:
                # Atualizando universo dos corredores.
                nova_solucao.corredores.append(corredor)
                nova_solucao.corredoresDisp[corredor] = 1
                nova_solucao.qntCorredores += 1

                for item, qnt in self.problema.aisles[corredor].items():
                    nova_solucao.itensC[item] += qnt
                    nova_solucao.universoC[item] += qnt
            else:
                continue

            # Verificando os pedidos disponíveis.
            pedidos_sorteados = []                              # Lista dos pedidos possíveis aleatórios.
            quantidade = 0                                      # Quantidade de pedidos possíveis.
            for indice in range(self.problema.o):
                if not nova_solucao.pedidosDisp[indice]:
                    valida = True
                    for item, qnt in self.problema.orders[indice].items():
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
                limite_pedidos = randint(1, quantidade) if nova_solucao.qntItens > self.problema.lb else quantidade

                for indice in pedidos_sorteados[:limite_pedidos]:
                    if not nova_solucao.pedidosDisp[indice]:
                        valida = True
                        soma = 0
                        for item, qnt in self.problema.orders[indice].items():
                            soma += qnt
                            if qnt > nova_solucao.universoC[item]:
                                valida = False
                                break
                        if valida and nova_solucao.qntItens + soma <= self.problema.ub:
                            nova_solucao.qntItens += soma
                            nova_solucao.pedidosDisp[indice] = 1
                            nova_solucao.pedidos.append(indice)
                            for item, qnt in self.problema.orders[indice].items():
                                nova_solucao.universoC[item] -= qnt
                                nova_solucao.itensP[item] += qnt

            # Verificando a nova solução.
            nova_solucao.objetivo = Metodos.funcao_objetivo(self.problema, nova_solucao.itensP, nova_solucao.itensC) / nova_solucao.qntCorredores
            if nova_solucao.objetivo > solucao.objetivo or nova_solucao.qntItens < self.problema.lb or nova_solucao.qntItens == 0:
                solucao = nova_solucao
            else:
                break

        return solucao

    def seleciona_operador(self, operadores, pesos):
        total = sum(pesos)
        escolha = uniform(0, total)
        acumulado = 0
        for i, w in enumerate(pesos):
            acumulado += w
            if escolha <= acumulado:
                return i, operadores[i]
        return len(operadores)-1, operadores[-1]

    def atualiza_pesos(self, indice, pesos, recompensa, rho=0.1):
        pesos[indice] = (1 - rho) * pesos[indice] + rho * recompensa

    def aceita_solucao(self, obj_atual, obj_novo):
        if obj_novo >= obj_atual:
            return True
        return random() < math.exp((obj_novo - obj_atual) / self.temp)
    
    def run(self, iteracoes):
        tempo_inicio = perf_counter()
        for i in range(iteracoes):  
            i_des, des = self.seleciona_operador(self.destruidores, self.peso_dest)
            i_rec, rec = self.seleciona_operador(self.reconstrutores, self.peso_reco)

            candidata = self.sol_atual.clone()
            candidata = des(candidata)
            candidata = rec(candidata)

            obj_atual = self.sol_atual.objetivo
            obj_novo = candidata.objetivo

            if self.aceita_solucao(obj_atual, obj_novo):

                self.sol_atual = candidata
                recompensa = 1 + max(0, obj_novo - obj_atual)
                self.atualiza_pesos(i_des, self.peso_dest, recompensa)
                self.atualiza_pesos(i_rec, self.peso_reco, recompensa)

                if obj_novo > self.sol_melhor.objetivo:
                    self.sol_melhor = candidata.clone()

            self.temp *= self.taxa_resf

        self.sol_melhor.tempo += perf_counter() - tempo_inicio

        return self.sol_melhor