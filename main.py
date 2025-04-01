import sys
import Processa

def main(dataset):
    problem = Processa.Problema(dataset)
    problem.imprimeProblema()
    problem.imprimeResultados()
    problem.salvaResultado()

# Verificando argumentos e chamando a main.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: python3 main.py <dataset>")
    main(sys.argv[1])    