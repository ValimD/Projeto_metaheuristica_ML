import sys
import Processa

def main(dataset):
    problem = Processa.Problem(dataset)
    problem.printData()
    problem.printResults()
    problem.saveResults()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: python3 main.py <dataset>")
    main(sys.argv[1])
    