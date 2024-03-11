# Analyse de panier 🛒 avec Python et l'algorithme Apriori 🤖

## Description du projet 📖

Le but du projet est de faire une **Market Basket Analysis** ou Analyse de panier avec **Python** pour faire des **suggestions** type "les utilisateurs ont aussi acheté".

Je me suis aidée de cette ressource [Market Basket Analysis](https://medium.com/@khusheekapoor/market-basket-analysis-association-rule-mining-dd632aa31a36/).

### Source de données

Le dataset utilisé est un échantillon d'un dataset de Kaggle : Il contient les transactions, les descriptions produit, et les informations client d'une compagnie **e-commerce** basée en Angleterre.

### Librairies utilisées
* NumPy - pour la manipulation des données. 
* Pandas - pour la manipulation des données.  
* CSV Reader - pour lire le dataset. 
* MLXtend -  pour appliquer l'algorithme Apriori.

## Nettoyage des données/Préparation🧹
 Voici les tâches réalisées pour nettoyer les données dans le cadre de ce projet.
 
* Importation des données. 
* Gestion des valeurs manquantes et doublons.  
* Suppression des colonnes inutiles.

## Analyse de panier 👜

### Transformation des listes en transaction

```python
paniers = donnees.groupby("CustomerID")["Description"].apply(list).reset_index(name="Panier")

# Instancier l'encodeur
encoder = TransactionEncoder()

# Encoder les paniers
paniers_encoded = encoder.fit_transform(paniers['Panier'])

# Créer un DataFrame à partir des données encodées
paniers_df = pd.DataFrame(paniers_encoded, columns=encoder.columns_)
```

