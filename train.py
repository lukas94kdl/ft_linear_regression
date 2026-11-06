from model import (
    load_data,
    normalize,
    gradient_descent,
    denormalize_theta,
    save_params,
    compute_r2
)


def train():
    # 1 - Charger les données
    x, y = load_data("data.csv")

    # 2 - Normaliser
    x_norm, x_mean, x_std = normalize(x)
    y_norm, y_mean, y_std = normalize(y)

    # 3 - Descente de gradient sur les données normalisées
    theta0, theta1 = gradient_descent(x_norm, y_norm)

    # 4 - Reconvertir theta dans l'espace original
    theta0, theta1 = denormalize_theta(
        theta0, theta1,
        x_mean, x_std,
        y_mean, y_std
    )

    # 5 - Sauvegarder
    save_params(theta0, theta1, "params.csv")

    # 6 - Afficher les résultats
    r2 = compute_r2(x, y, theta0, theta1)

    print(f"Entraînement terminé !")
    print(f"  theta0 = {theta0:.2f}")
    print(f"  theta1 = {theta1:.6f}")
    print(f"  R²     = {r2:.4f}  (1.0 = parfait)")


if __name__ == "__main__":
    train()
