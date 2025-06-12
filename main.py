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
        solucao = Metodos.gulosa(problema)
    elif construtiva == "3":
        solucao = Metodos.PSO(problema, 30, 2, 2, 1, 2000)
    elif construtiva == "4":
        FPA_instance = Metodos.FPA(problema)
        solucao = FPA_instance.run()
    elif construtiva == "5":
        solucao = Metodos.gulosa(problema)
        ALNS = Metodos.ALNS(problema, solucao, 10, 0.999)
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
        print("Heurísticas construtivas e metaheurísticas: 0 (híbrida), 1 (aleatória), 2 (gulosa), 3 (PSO discreto), 4 (FPA), 5 (ALNS)")
        print("Heurísticas de refinamento: 0 (nenhuma), 1 (melhor_vizinhanca), 2 (refinamento_cluster_vns)")
    else:
        try:
            main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
        except (IndexError, ValueError):
            print("ERRO: Semente deve ser numérica.")