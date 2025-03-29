import os

class Problem():
    def __init__(self, dataset):
        try:
            # Pegando o diretório atual.
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
            dataset_path = os.path.join(base_dir, "Datasets", f"{dataset}.txt")

            # Lendo o dataset.
            with open(dataset_path, "rb") as data:
                lines = data.readlines()
                # Elementos o, i, a.
                first_line = lines[0].strip().split()
                self.o, self.i, self.a = int(first_line[0]), int(first_line[1]), int(first_line[2])

                # Pedidos.
                self.orders = []
                for i in range(self.o):
                    order_line = lines[i + 1].strip().split()
                    order = {int(order_line[1 + k * 2]): int(order_line[2 + k * 2]) for k in range(int(order_line[0]))}
                    self.orders.append(order)

                # Corredores.
                self.aisles = []
                for i in range(self.a):
                    aisle_line = lines[i + 1 + self.o].strip().split()
                    aisle = {int(aisle_line[1 + k * 2]): int(aisle_line[2 + k * 2]) for k in range(int(aisle_line[0]))}
                    self.aisles.append(aisle)

                # Limite inferior e superior.
                last_line = lines[self.o + self.a + 1].strip().split()
                self.lb, self.ub = int(last_line[0]), int(last_line[1])

                # Resultado base.
                self.result = {"dataset": dataset, "orders": [], "aisles": [], "time": 0}
        except FileNotFoundError:
            print("Dataset não existe.")

    # Descrição: função que imprime os dados tratados do dataset.
    def printData(self):
        print(f"o = {self.o}, i = {self.i}, a = {self.a}")
        print(f"\nPedidos: ")
        for i in range(len(self.orders)):
            print(f"{i}: {self.orders[i]}")
        print(f"\nCorredores: ")
        for i in range(len(self.aisles)):
            print(f"{i}: {self.aisles[i]}")
        print(f"\nLimite inferior = {self.lb}, limite superior = {self.ub}")

    # Descrição: função que imprime alguns dos dados que compõem a solução.
    def printResults(self):
        print(f"Dataset sendo utilizado: {self.result["dataset"]}")
        print(f"Pedidos selecionados: {self.result["orders"]}, corredores selecionados: {self.result["aisles"]}")
        print(f"Tempo de execução: {self.result["time"]}")

    # Descrição: função que salva os resultados do método no arquivo Resultados.csv.
    # Adicionar os dados 'feasible' e valor objetivo, que podem ser adquiridos executando o checker.py do MeLi, ou criando o nosso próprio.
    def saveResults(self):
        with open("Resultado.csv", "+a") as file:
            line = f"{self.result["dataset"]},{"-".join(map(str, self.result["orders"]))},{"-".join(map(str, self.result["orders"]))},{self.result["time"]}\n"
            file.write(line)
            file.close()