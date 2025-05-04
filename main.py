import Metodos
import Processa
import sys

def main(dataset, arquivo):
    # Instanciando problema.
    problema = Processa.Problema(dataset, arquivo)
    # Construindo uma solução.
    solucao = Metodos.hibrida(problema)
    # Salvando resultados.
    problema.result["orders"] = solucao.pedidos
    problema.result["aisles"] = solucao.corredores
    problema.result["objective"] = solucao.objetivo
    problema.result["time"] = solucao.tempo
    # Imprimindo e salvando no arquivo.
    problema.imprimeResultados()
    #problema.salvaResultado()

# Verificando argumentos e chamando a main.
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso correto: python3 main.py <dataset> <nome_arquivo_resultados>")
    else:
        main(sys.argv[1], sys.argv[2])