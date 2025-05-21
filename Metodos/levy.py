import numpy as np


def _phi(alpha, beta):
    """
    Calcula o parâmetro phi para a distribuição de Lévy.

    Este método interno computa phi com base nos parâmetros alpha e beta,
    sendo que phi é definido como beta * tan(π * alpha / 2).

    Args:
        alpha (float): Exponente de estabilidade da distribuição.
        beta (float): Parâmetro de assimetria da distribuição.

    Returns:
        float: Valor calculado de phi.
    """

    return beta * np.tan(np.pi * alpha / 2.0)


def change_par(alpha, beta, mu, sigma, par_input, par_output):
    """
    Ajusta o parâmetro de localização (mu) ao converter entre diferentes parametrizações.

    Esta função realiza ajustes em mu de acordo com os valores de par_input e par_output.
    Se as parametrizações forem iguais, retorna mu inalterado; caso contrário, adiciona ou
    subtrai o termo sigma * phi, onde phi é calculado pela função _phi.

    Args:
        alpha (float): Exponente de estabilidade da distribuição.
        beta (float): Parâmetro de assimetria da distribuição.
        mu (float): Parâmetro de localização da distribuição.
        sigma (float): Parâmetro de escala da distribuição.
        par_input (int): Indica a parametrização de entrada (0 ou 1).
        par_output (int): Indica a parametrização de saída (0 ou 1).

    Returns:
        float: Valor ajustado de mu conforme a conversão de parametrização.
    """
    if par_input == par_output:
        return mu
    elif (par_input == 0) and (par_output == 1):
        return mu - sigma * _phi(alpha, beta)
    elif (par_input == 1) and (par_output == 0):
        return mu + sigma * _phi(alpha, beta)


def random_levy(alpha, beta, mu=0.0, sigma=1.0, shape=(), par=0):
    """
    Gera amostras aleatórias conforme a distribuição de Lévy.

    Caso alpha seja 2, a função gera amostras da distribuição normal padrão (multiplicada por sqrt(2)).
    Para outros valores de alpha, realiza o cálculo utilizando transformadas específicas,
    realizando um ajuste quando alpha está próximo de 1 para evitar problemas numéricos.

    Args:
        alpha (float): Exponente de estabilidade (0 < alpha <= 2).
        beta (float): Parâmetro de assimetria (-1 <= beta <= 1).
        mu (float, opcional): Parâmetro de localização da distribuição. Padrão é 0.0.
        sigma (float, opcional): Parâmetro de escala da distribuição. Padrão é 1.0.
        shape (tuple, opcional): Forma desejada para a saída das amostras. Padrão é ().
        par (int, opcional): Parâmetro para ajuste de parametrização (0 ou 1). Padrão é 0.

    Returns:
        numpy.ndarray or float: Amostra ou conjunto de amostras aleatórias seguindo a distribuição de Lévy.
    """
    loc = change_par(alpha, beta, mu, sigma, par, 0)
    if alpha == 2:
        return np.random.standard_normal(shape) * np.sqrt(2.0)

    radius = 1e-15
    if np.absolute(alpha - 1.0) < radius:
        alpha = 1.0 + radius

    r1 = np.random.random(shape)
    r2 = np.random.random(shape)
    pi = np.pi

    a = 1.0 - alpha
    b = r1 - 0.5
    c = a * b * pi
    e = _phi(alpha, beta)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)
    return loc + sigma * k


def get_levy_flight_array(dim=10):
    """
    Gera um array com passos absolutos de um voo de Lévy.

    Utilizando a função random_levy, gera uma sequência (array) de passos absolutos,
    os quais podem ser utilizados para representar movimentos baseados em vôos de Lévy.

    Args:
        dim (int, opcional): Dimensão (número de passos) do array gerado. Padrão é 10.

    Returns:
        numpy.ndarray: Array contendo os passos absolutos obtidos a partir da distribuição de Lévy.
    """
    return np.array([abs(random_levy(1.5, 0)) for _ in range(dim)])


if __name__ == "__main__":

    for i in range(100):
        print(random_levy(1.5, 0))