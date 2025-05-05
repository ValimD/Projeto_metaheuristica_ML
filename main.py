import Metodos
import Processa
import sys

def main(dataset, arquivo, construtiva, refinamento):
    # Instanciando problema.
    problema = Processa.Problema(dataset, arquivo)

    # Construindo uma solução.
    if construtiva == "0":
        solucao = Metodos.hibrida(problema)
    elif construtiva == "1":
        solucao = Metodos.aleatorio(problema)
    elif construtiva == "2":
        solucao = Metodos.gulosa(problema)

    # Refinando a solução.
    if refinamento == "1":
        solucao = Metodos.melhor_vizinhanca(problema, solucao)

    # Salvando resultados.
    problema.result["orders"] = solucao.pedidos
    problema.result["aisles"] = solucao.corredores
    problema.result["objective"] = solucao.objetivo
    problema.result["time"] = solucao.tempo

    # Imprimindo e salvando no arquivo.
    #problema.imprimeResultados()
    #problema.salvaResultado()

# Verificando argumentos e chamando a main.
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Uso correto: python3 main.py <dataset> <nome_arquivo_resultados> <heurística_construtiva> <heurística_refinamento>")
        print("Heurísticas construtivas: 0 (híbrida), 1 (aleatória), 2 (gulosa)")
        print("Heurísticas de refinamento: 0 (nenhuma), 1 (melhor_vizinhanca), 2 ()")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])