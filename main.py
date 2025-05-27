import Metodos
import Processa
import random
import sys

def main(dataset, arquivo, construtiva, refinamento, semente):
    # Definindo a semente.
    random.seed(semente)

    # Instanciando problema.
    problema = Processa.Problema(dataset, arquivo)

    # Construindo uma solução.
    if construtiva == "0":
        solucao = Metodos.hibrida(problema)
    elif construtiva == "1":
        solucao = Metodos.aleatorio(problema)
    elif construtiva == "2":
        solucao = Metodos.gulosa_v3(problema)
    elif construtiva == "3":
        solucao = Metodos.pso_discreto(problema)
    elif construtiva == "4":
        iterations_number = 1000
        population_size = int(problema.o / 200) + 50
        # p = 0.5
        # Instancia a classe FPO passando o problema e a solução atual para refinar
        fpo_instance = Metodos.FPO(problema, iterations_number, population_size, plot=False)
        solucao = fpo_instance.run()
    elif construtiva == "5":
        solucao = Metodos.gulosa_v3(problema)
        ALNS = Metodos.ALNS(problema, solucao, 100, 0.995)
        solucao = ALNS.run(1000)

    # Refinando a solução.
    if refinamento == "1":
        solucao = Metodos.melhor_vizinhanca(problema, solucao)
    elif refinamento == "2":
        solucao = Metodos.refinamento_cluster_vns(problema, solucao)

    # Salvando resultados.
    problema.result["orders"] = solucao.pedidos
    problema.result["aisles"] = solucao.corredores
    problema.result["objective"] = solucao.objetivo
    problema.result["time"] = solucao.tempo

    # Imprimindo e salvando no arquivo.
    problema.imprimeResultados()
    problema.salvaResultado()

# Verificando argumentos e chamando a main.
if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Uso correto: python3 main.py <dataset> <nome_arquivo_resultados> <heurística_construtiva_metaheurística> <heurística_refinamento> <semente_aleatoria>")
        print("Heurísticas construtivas e metaheurísticas: 0 (híbrida), 1 (aleatória), 2 (gulosa), 3 (PSO discreto), 4 (FPO), 5 (ALNS)")
        print("Heurísticas de refinamento: 0 (nenhuma), 1 (melhor_vizinhanca), 2 (refinamento_cluster_vns)")
    else:
        try:
            main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
        except (IndexError, ValueError):
            print("ERRO: Semente deve ser numérica.")