import csv
def carrega_do_arquivo():
    X = []
    Y = []
    arquivo = open("acesso.csv", "rb")
    leitor = csv.reader(arquivo)
    leitor.next()
    for home,como_funciona,contato,comprou in leitor:
        dados = [int(home), int(como_funciona), int(contato)]
        X.append(dados)

        Y.append(int(comprou))

    return X,Y


