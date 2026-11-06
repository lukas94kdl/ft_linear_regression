# ft_linear_regression

Ce projet entraîne une régression linéaire pour estimer le prix d'une voiture à partir de son kilométrage. L'entraînement se fait par descente de gradient, après normalisation des données, puis les paramètres sont reconvertis et enregistrés pour les prédictions.

## Prérequis

- Python 3.9 ou plus récent
- Aucune dépendance externe

## Utilisation

Depuis le dossier `ft_linear_regression`, lancez l'entraînement :

```bash
python train.py
```

Le script lit `data.csv`, affiche les paramètres appris (`theta0`, `theta1`) et le score R², puis enregistre les paramètres dans `params.csv`.

Pour estimer le prix d'une voiture, lancez ensuite :

```bash
python predict.py
```

Saisissez le kilométrage demandé. Le prix estimé est affiché en euros. Il faut avoir exécuté l'entraînement pour disposer de paramètres à jour dans `params.csv`.

## Données

`data.csv` doit être un fichier CSV contenant les colonnes `km` et `price` :

```csv
km,price
240000,3650
139800,3800
```

Chaque ligne représente une voiture : `km` est son kilométrage et `price` son prix en euros.

## Modèle

La prédiction suit l'équation :

```text
price = theta0 + theta1 * km
```

Le script d'entraînement utilise une descente de gradient sur les données normalisées (taux d'apprentissage `0.01`, 1 000 itérations). Les paramètres sauvegardés dans `params.csv` sont reconvertis pour être directement utilisables avec le kilométrage d'origine.

## Fichiers

- `data.csv` : données d'entraînement
- `model.py` : chargement des données, normalisation, entraînement et prédiction
- `train.py` : point d'entrée de l'entraînement
- `predict.py` : point d'entrée des prédictions interactives
- `params.csv` : paramètres appris, générés ou mis à jour par `train.py`# ft_linear_regression
