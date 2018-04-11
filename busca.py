import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import Counter

def fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    diferencas = resultado - teste_marcacoes

    acertos = [d for d in diferencas if d == 0]
    tamanho_acertos = len(acertos)
    tamanho_marcacoes = len(teste_marcacoes)

    porcentagem_de_acerto = 100.00 * tamanho_acertos/tamanho_marcacoes

    print(porcentagem_de_acerto)


df = pd.read_csv('busca.csv')

X_df = df[["home", "busca", "logado"]]
Y_df = df["comprou"]

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

# A eficacia de algoritmos que chutam sempre 1 ou 0
acerto_base = max(Counter(Y).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(Y)


porcentagem_de_treino = 0.9
tamanho_de_treino = int(porcentagem_de_treino * len(X))
tamanho_de_teste = tamanho_de_treino - len(X)

print(len(X))
print(tamanho_de_treino)


treino_dados     = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados     = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]


modelo = MultinomialNB()
fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modelo = AdaBoostClassifier()
fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
