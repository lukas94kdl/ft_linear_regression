from model import load_params, estimate_price


def predict():
    # 1 - Charger les paramètres
    theta0, theta1 = load_params("params.csv")

    # 2 - Demander le kilométrage
    km = float(input("Entrez le kilométrage : "))

    # 3 - Prédire le prix
    price = estimate_price(km, theta0, theta1)

    # 4 - Afficher le résultat
    print(f"Prix estimé : {price:.2f} €")


if __name__ == "__main__":
    predict()
