import sys
import Processa
import Metodos

def main(dataset, arquivo):
    # Instanciando problema.
    problema = Processa.Problema(dataset, arquivo)
    # Construindo uma solução.
    resultados = Metodos.hibrida(problema)
    # Salvando resultados.
    problema.result["orders"] = resultados[0]
    problema.result["aisles"] = resultados[1]
    problema.result["objective"] = resultados[2]
    problema.result["time"] = resultados[3]
    # Imprimindo e salvando no arquivo.
    problema.imprimeResultados()
    #problema.salvaResultado()

# Verificando argumentos e chamando a main.
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso correto: python3 main.py <dataset> <nome_arquivo_resultados>")
    else:
        main(sys.argv[1], sys.argv[2])