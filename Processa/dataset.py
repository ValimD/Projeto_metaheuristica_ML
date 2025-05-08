import os

class Problema():
    """
    Lê o dataset do diretório de Datasets, e trata as informações presente nele.

    Args:
        dataset (str): Nome do dataset que será tratado.
        arquivo (str): Nome do arquivo em que os resultados serão salvos.
    """

    def __init__(self, dataset: str, arquivo: str) -> None:
        try:
            # Salvando o nome do arquivo de resultados.
            self.arquivo = arquivo

            # Pegando o diretório atual.
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, "Datasets", f"{dataset}.txt")

            # Lendo o dataset.
            with open(dataset_path, "rb") as data:
                lines = data.readlines()
                # Elementos o|i|a.
                first_line = lines[0].strip().split()
                self.o, self.i, self.a = int(first_line[0]), int(first_line[1]), int(first_line[2])

                # Pedidos - 'o' linhas no formato k|i|q ..., sendo k a quantidade de itens no pedido, e i|q o par item|quantidade.
                self.orders = []
                for i in range(self.o):
                    order_line = lines[i + 1].strip().split()
                    order = {int(order_line[1 + k * 2]): int(order_line[2 + k * 2]) for k in range(int(order_line[0]))}
                    self.orders.append(order)

                # Corredores - 'a' linhas no formato k|i|q ..., sendo k a quantidade de itens no corredor, e i|q o par item|quantidade.
                self.aisles = []
                for i in range(self.a):
                    aisle_line = lines[i + 1 + self.o].strip().split()
                    aisle = {int(aisle_line[1 + k * 2]): int(aisle_line[2 + k * 2]) for k in range(int(aisle_line[0]))}
                    self.aisles.append(aisle)

                # Limite inferior e superior.
                last_line = lines[self.o + self.a + 1].strip().split()
                self.lb, self.ub = int(last_line[0]), int(last_line[1])

                # Resultado base.
                self.result = {"dataset": dataset, "orders": [], "aisles": [], "objective": 0, "time": 0}
        except FileNotFoundError:
            print("Dataset não existe.")
            exit()

    def imprimeProblema(self) -> None:
        """
        Função responsável por imprimir os dados tratados do dataset.
        """

        print(f"o = {self.o}, i = {self.i}, a = {self.a}")
        print(f"\nPedidos: ")
        for i in range(len(self.orders)):
            print(f"{i}: {self.orders[i]}")
        print(f"\nCorredores: ")
        for i in range(len(self.aisles)):
            print(f"{i}: {self.aisles[i]}")
        print(f"\nLimite inferior = {self.lb}, limite superior = {self.ub}")

    def imprimeResultados(self) -> None:
        """
        Função responsável por imprimir os dados que compõem a solução.

        Formato de saída: | Dataset | Pedidos da wave | Corredores da wave | Valor da função objetivo | Tempo de execução |
        """

        print(f"| {self.result['dataset']} | {self.result['orders']} | {self.result['aisles']} | {self.result['objective']} | {self.result['time']} |")
        print()

    def salvaResultado(self) -> None:
        """
        Função responsável por salvar os resultados no arquivo csv.

        Formato: dataset,pedidos (separados por -),corredores (separados por -),valor da função objetivo,tempo de execução.
        """

        with open(f"./Resultados/{self.arquivo}.csv", "+a") as file:
            file.write(f"{self.result['dataset']},{'-'.join(map(str, self.result['orders']))},{'-'.join(map(str, self.result['aisles']))},{self.result['objective']},{self.result['time']}\n")
            file.close()