from sklearn.naive_bayes import MultinomialNB
from arquivo import carrega_do_arquivo

X,Y = carrega_do_arquivo()

treino = X[:90]
treino_respostas = Y[:90]

teste = X[-9:]
teste_respostas  = Y[-9:]

modelo = MultinomialNB()

modelo.fit(treino, treino_respostas)

resultado = modelo.predict(teste)

diferencas = resultado - teste_respostas

print(diferencas)

acertos = [d for d in diferencas if d == 0]

total_itens = len(teste_respostas)

print(total_itens)

qtd_acertos = len(acertos)

porcentagem_de_acertos = 100.0 * qtd_acertos / total_itens

print(porcentagem_de_acertos)


