import Metodos
import Processa
from random import random, choice, choices, sample
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from .levy import get_levy_flight_array


class FPO:
    def __init__(self, problema: Processa.Problema, iterations_num, pop_size, p=0.4, plot=True):
        """
        Construtor do FPO modificado.
        Agora utiliza a função construtiva híbrida para inicializar a população e
        operadores de refinamento para gerar vizinhos, manipulando corretamente o conjunto de dados.
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

    def run(self)->Metodos.Solucao:
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


    def pollination(self)->None:
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
                

    def local_pollination_multisol(self, i)->Metodos.Solucao:
        """
        Args:
            self
            i: identificador da população
        Return:
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
        nova_sol = Metodos.remove_corredor(self.problema, nova_sol, c2)
        nova_sol = Metodos.adiciona_corredor(self.problema, nova_sol, c1)
        # Avaliar nova solução
        nova_sol.objetivo = Metodos.funcao_objetivo(self.problema, nova_sol.itensP, nova_sol.itensC)
        self.objetivo[i] = nova_sol.objetivo
        return nova_sol


    def local_pollination(self, i)->Metodos.Solucao:
        """
        Args:
            self
            i: identificador da população
        Return:
            solucao (Solucao): Dataclass representando a solução montada, incluindo estruturas auxiliares.
        """
        populacao = self.population[i]
        nova_sol = populacao.clone()
        tam = self.problema.a - 1
        escolhido = choice(range(tam))
        
        if nova_sol.corredoresDisp[escolhido] == 0:
            nova_sol = Metodos.adiciona_corredor(self.problema, nova_sol, escolhido)
        else:
            nova_sol = Metodos.remove_corredor(self.problema, nova_sol, escolhido)

        return nova_sol


    def global_pollination(self, i)->Metodos.Solucao:
        # Define o número de mudanças/ Força do polinizador
        num_levy = get_levy_flight_array()
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
                    copia_sol = Metodos.adiciona_corredor(self.problema, copia_sol, escolhidos[j])
                else:
                    copia_sol = Metodos.remove_corredor(self.problema, copia_sol, escolhidos[j])
        return copia_sol
    
            
    def calcular_matriz_distancias_populacao(self) -> None:
        """
        Calcula e armazena as distâncias de Jaccard de cada indivíduo para todos os outros,
        ordenadas por proximidade.

        A estrutura resultante é armazenada em `self.distancias_ordenadas` (novo nome sugerido
        para o atributo para refletir a estrutura) e é um dicionário onde:
            - A chave é o índice de um indivíduo.
            - O valor é uma lista de tuplas `(distância, índice_do_vizinho)`,
            ordenada pela distância em ordem crescente.
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