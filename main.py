import sys
import Processa

def main(dataset, arquivo):
    problem = Processa.Problema(dataset, arquivo)
    problem.imprimeProblema()
    problem.imprimeResultados()
    problem.salvaResultado()

# Verificando argumentos e chamando a main.
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso correto: python3 main.py <dataset> <nome_arquivo_resultados>")
    main(sys.argv[1], sys.argv[2])