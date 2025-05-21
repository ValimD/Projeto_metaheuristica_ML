import Metodos
import Processa
from random import random
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

class FPO:
    def __init__(self, problema: Processa.Problema, iterations_num, pop_size, p=0.8, plot=False):
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
        self.obj_vals = np.zeros(pop_size)
        self.plot = plot

    def run(self):
        self.initialize_population()
        self.calculate_obj()
        avg = []
        bests = []
        start = perf_counter()
        iterations_without_improve = 0

        for i in range(self.iterations_num):
            current_best_val = self.best.objetivo
            self.check_best()
            # Se há melhoria, reseta contador
            if self.best.objetivo > current_best_val:
                iterations_without_improve = 0
            else:
                iterations_without_improve += 1

            # Encerra se 50 iterações sem melhoria
            if iterations_without_improve >= 50:
                print("Encerrando execução após 50 iterações sem melhoria.")
                break

            self.pollination()
            if self.plot:
                avg.append(np.mean(self.obj_vals))
                bests.append(max(self.obj_vals))
            if not i % 50:
                print("Iteration {} best result: {}".format(i, self.best.objetivo))

        elapsed = perf_counter() - start
        # Adiciona o tempo à melhor solução
        self.best.tempo = elapsed

        if self.plot:
            plt.plot(avg, label="avg")
            plt.plot(bests, label="best")
            plt.yscale('log')
            plt.legend()
            plt.show()

        print(self.best)
        return self.best

    def initialize_population(self):
        # Utiliza a função construtiva híbrida para gerar a população inicial
        self.population = [Metodos.gulosa(self.problema) for _ in range(self.pop_size)]

    def calculate_obj(self):
        for i, sol in enumerate(self.population):
            self.obj_vals[i] = sol.objetivo
        # Define o melhor a partir da população inicial
        best_idx = np.argmax(self.obj_vals)
        self.best = self.population[best_idx]

    def check_best(self):
        current_best = max(self.obj_vals)
        if current_best > self.best.objetivo:
            best_idx = np.argmax(self.obj_vals)
            self.best = self.population[best_idx]

    def pollination(self):
        for i in range(self.pop_size):
            # Clona a solução atual para gerar um vizinho
            sol_clone = self.population[i].clone()
            # Com probabilidade p, aplica refinamento global; caso contrário, utiliza refinamento local
            # if random() < self.p:
            # new_sol = Metodos.refinamento_cluster_vns(self.problema, sol_clone)
            # else:
            new_sol = Metodos.melhor_vizinhanca(self.problema, sol_clone)
            # Se o novo vizinho possui função objetivo melhor, substitui a solução atual
            if new_sol.objetivo > self.obj_vals[i]:
                self.population[i] = new_sol
                self.obj_vals[i] = new_sol.objetivo