import matplotlib.pyplot as plt
import Metodos
import numpy as np
import Processa
import statistics
import math
from collections import defaultdict
from itertools import islice
from random import choice, choices, randint, random, sample
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

    print(f"Melhor solução da geração {geracao_atual}: {solucao.objetivo}")
    print(f"Desvio padrão: {desvio}, tempo: {solucao.tempo}")
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

            if iterations_without_improve == 250:
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
        top10 = (int)(self.pop_size/10)
        top90 = self.pop_size - top10
        self.population = [Metodos.hibrida(self.problema) for _ in range(top10)] + [Metodos.aleatorio(self.problema) for _ in range(top90)]
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
                        if self.objetivo[i] > self.best.objetivo:
                            self.best = self.population[i]
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
    def __init__(self, problema, solucao, temperatura_inicial=100, taxa_resfriamento=0.995):
        self.problema           = problema
        self.sol_atual          = solucao
        self.sol_melhor         = solucao.clone()
        self.destruidores       = [self.destruidor_aleatorio, self.destruidor_bx_prod]
        self.reconstrutores     = [self.construtor_guloso, self.construtor_hibrido, self.construtor_aleatorio]
        self.peso_dest          = [1] * len(self.destruidores)
        self.peso_reco          = [1] * len(self.reconstrutores)
        self.temp               = temperatura_inicial
        self.taxa_resf          = taxa_resfriamento

    def destruidor_aleatorio(self, solucao, frac=0.1):
        n = len(solucao.corredores)
        # se não houver corredores, nada a fazer
        if n == 0:
            return solucao

        # calcula k entre 0 e n, proporcional a frac
        k = int(frac * n)
        # garante ao menos 1 remoção, mas não mais que n
        k = max(1, k) if n >= 1 else 0
        k = min(k, n)

        to_remove = random.sample(solucao.corredores, k)
        for c in to_remove:
            solucao = self.remove_corredor(solucao, c)
        return solucao

    # Se existirem itens no UniversoC, ranqueia os corredores selecionados que mais possuem eles, e removem os 10% piores corredores.
    def destruidor_bx_prod(self, solucao, porcent = 0.1):
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
            solucao = self.remove_corredor(solucao, corredor)

        return solucao

    def construtor_guloso(self, solucao):
        problema = self.problema

        while True:
            # 1) calcula peso ponderado de cada item segundo corredores atuais
            conc = defaultdict(lambda: {"total":0, "contagem":0})
            for c in solucao.corredores:
                for item, q in problema.aisles[c].items():
                    conc[item]["total"]   += q
                    conc[item]["contagem"]+= 1

            peso_item = {
                it: (v["total"]/v["contagem"] if v["contagem"] else 0)
                for it,v in conc.items()
            }

            # 2) filtra candidatos viáveis e calcula score
            candidatos = []
            for p in range(problema.o):
                if solucao.pedidosDisp[p]: 
                    continue
                # verifica se cabe no universoC por item e no ub
                total_p = sum(problema.orders[p].values())
                if solucao.qntItens + total_p > problema.ub:
                    continue
                cabe = all(
                    solucao.universoC.get(it,0) >= qt
                    for it,qt in problema.orders[p].items()
                )
                if not cabe:
                    continue
                # score = soma(qt * peso_item[it])
                score = sum(peso_item.get(it,0) * qt
                            for it,qt in problema.orders[p].items())
                candidatos.append((p, score))

            if not candidatos:
                break

            # 3) ordena decrescente e tenta inserir em ordem
            candidatos.sort(key=lambda x: x[1], reverse=True)
            added = False
            for p,_ in candidatos:
                if self.adiciona_pedido(solucao, p):
                    added = True
                # opcional: break após inserir um
            if not added:
                break

        return solucao
    
    

    def construtor_hibrido(self, solucao, alpha = 0.3):
        problema = self.problema

        while True:
            # 1) Seleciona candidatos viáveis (não inseridos e cabem no universo)
            candidatos = []
            for p in range(problema.o):
                if solucao.pedidosDisp[p] == 0:
                    total_itens = sum(problema.orders[p].values())
                    # cabe no universoC?
                    if total_itens + solucao.qntItens <= problema.ub and all(
                        solucao.universoC.get(item, 0) >= qtd
                        for item, qtd in problema.orders[p].items()
                    ):
                        # quantos corredores novos seriam necessários?
                        novos_corr = [
                            c for c in self._get_corridors_for_order(p)
                            if solucao.corredoresDisp[c] == 0
                        ]
                        delta_corr = len(novos_corr) or 1
                        score = total_itens / delta_corr
                        candidatos.append((p, score, total_itens))

            if not candidatos:
                break

            # 2) Ordena por score decrescente
            candidatos.sort(key=lambda x: x[1], reverse=True)

            # 3) Escolha aleatória entre os top-k
            k = min(5, len(candidatos))
            if random.random() < alpha:
                escolha = random.choice(candidatos[:k])[0]
            else:
                escolha = candidatos[0][0]

            # 4) Tenta inserir; se falhar, encerra
            if not self.adiciona_pedido(solucao, escolha):
                break

        return solucao
    
    def construtor_aleatorio(self, solucao):

        candidatos = [p for p in range(len(solucao.pedidosDisp)) if solucao.pedidosDisp[p] == 0]
        random.shuffle(candidatos)

        for pedido in candidatos:

            if not self.adiciona_pedido(solucao, pedido):
                break

        return solucao

    def seleciona_operador(self, operadores, pesos):
        total = sum(pesos)
        escolha = random.uniform(0, total)
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
        return random.random() < math.exp((obj_novo - obj_atual) / self.temp)

    def e_viavel(self, solucao):
        return self.problema.lb <= solucao.qntItens <= self.problema.ub

    def adiciona_pedido(self, solucao, pedido):

        if solucao.pedidosDisp[pedido] == 1:
            return False
        
        nova_qntItens = solucao.qntItens + sum(self.problema.orders[pedido].values())
        
        if nova_qntItens > self.problema.ub:
            return False
        
        solucao.pedidosDisp[pedido] = 1
        solucao.pedidos.append(pedido)
        solucao.qntItens = nova_qntItens

        for item, qnt in self.problema.orders[pedido].items():
            solucao.universoC[item] -= qnt
            solucao.itensP[item] += qnt

        solucao.objetivo = solucao.qntItens / solucao.qntCorredores if solucao.qntCorredores > 0 else 0
        return True

    def remove_pedido(self, solucao, pedido):
        if solucao.pedidosDisp[pedido] == 0:
            return False
        solucao.pedidosDisp[pedido] = 0
        solucao.pedidos.remove(pedido)

        for item, qnt in self.problema.orders[pedido].items():
            solucao.universoC[item] += qnt
            solucao.itensP[item] -= qnt
            solucao.qntItens -= qnt

        solucao.objetivo = Metodos.funcao_objetivo(self.problema, solucao.itensP, solucao.itensC) / solucao.qntCorredores

        return True

    def remove_corredor(self, solucao, corredor):
        solucao.corredores.remove(corredor)
        solucao.corredoresDisp[corredor] = 0
        solucao.qntCorredores -= 1
        for item, qnt in self.problema.aisles[corredor].items():
            solucao.itensC[item] -= qnt
            solucao.universoC[item] -= qnt

        pedidos_inviaveis = []
        for indice in solucao.pedidos:
            for item, _ in self.problema.orders[indice].items():
                if qnt > solucao.universoC[item]:
                    pedidos_inviaveis.append(indice)
                    break

        for indice in pedidos_inviaveis:
            solucao.pedidos.remove(indice)
            solucao.pedidosDisp[indice] = 0
            for item, qnt in self.problema.orders[indice].items():
                solucao.universoC[item] += qnt
                solucao.itensP[item] -= qnt
                solucao.qntItens -= qnt

        solucao.objetivo = solucao.qntItens/solucao.qntCorredores if solucao.qntCorredores != 0 else 0
        return solucao

    def _get_corridors_for_order(self, pedido_idx):
        """
        Retorna a lista de índices de corredores que contêm
        pelo menos um dos itens requisitados pelo pedido.
        """
        corredores = []
        pedido = self.problema.orders[pedido_idx]
        for c, aisle in enumerate(self.problema.aisles):
            # se este corredor tem algum item do pedido, inclui-o
            for item in pedido:
                if aisle.get(item, 0) > 0:
                    corredores.append(c)
                    break
        return corredores
    

    def run(self, iteracoes):
        tempo_inicio = perf_counter()
        for _ in range(iteracoes):
            i_des, des = self.seleciona_operador(self.destruidores, self.peso_dest)
            i_rec, rec = self.seleciona_operador(self.reconstrutores, self.peso_reco)

            candidata = self.sol_atual.clone()
            candidata = des(candidata)
            candidata = rec(candidata)

            if not self.e_viavel(candidata):
                continue

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

        self.sol_melhor.tempo = perf_counter() - tempo_inicio

        return self.sol_melhor