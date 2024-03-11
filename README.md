# Analyse de panier 🛒 avec Python et l'algorithme Apriori 🤖

## Description du projet 📖

Le but du projet est de faire une **Market Basket Analysis** ou Analyse de panier avec **Python** pour faire des **suggestions** type "les utilisateurs ont aussi acheté".

Je me suis aidée de cette ressource pour les notions nécessaires à ce projets (support, associations, lift...) : [Market Basket Analysis](https://medium.com/@khusheekapoor/market-basket-analysis-association-rule-mining-dd632aa31a36/).

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

#### Transformation des listes en transaction

```python
paniers = donnees.groupby("CustomerID")["Description"].apply(list).reset_index(name="Panier")

# Instancier l'encodeur
encoder = TransactionEncoder()

# Encodage des paniers
paniers_encoded = encoder.fit_transform(paniers['Panier'])

# Création du DataFrame à partir des données encodées
paniers_df = pd.DataFrame(paniers_encoded, columns=encoder.columns_)
```

#### Trouver les dimensions du dataframe
```python
df.shape ()
```
(897, 2355) 

On peut voir qu'il y'a 897 transactions et 2355 produits dans le dataset.

### Algorithme Apriori

##### Utilisation de l'algorithme Apriori pour extraire les ensembles d'articles fréquents
```python
frequent_itemsets = apriori(paniers_df, min_support=0.02, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
frequent_itemsets
```
On trouve 537 ensembles fréquents.

#### Trouver le top 5 des produits avec un support minimum de 2%
```python
frequent_itemsets[ (frequent_itemsets['length'] == 1) &
                   (frequent_itemsets['support'] >= 0.02) ][0:5]
```

Les **5** produits qui se vendent le plus sont : 

* Le porte bougie en forme de coeur blanc (support de 16%)
* La chaine de Noel en papier ( support de 13%)
* Le présentoir à gateau " royal family " ( support de 13%)
* La bouilloire scottie dog( support de 12%)
* Le chauffe mains babushka( support de 11%)

#### Trouver les groupes de produits avec une longueur de plus de 1 et avec un support minimal de 5%
```python
frequent_itemsets[(frequent_itemsets['length'] > 1) &
                  (frequent_itemsets['support'] >= 0.05)]
```

On trouve **5** groupes de **2** produits : 

* (La chaine de Noel en papier, La chaine en papier vintage)
* (porte bougie en forme de coeur rouge, porte bougie en forme de coeur blanc)
* (coeur en osier large, coeur en osier petit)	
* (La bouilloire scottie dog, La bouilloire chocolat)
* (La bouilloire retrospot, La bouilloire scottie dog)

### Règles associations

#### Trouver le top 10 des associations avec un support minimal de 2%

```python
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.02)
rules.sort_values(by='confidence', ascending=False)[0:10]
rules
```
Voici le **top 10** des associations dans le dataset : 

1.(POPPY'S PLAYHOUSE BEDROOM, POPPY'S PLAYHOUSE BATHROOM) ...	(POPPY'S PLAYHOUSE KITCHEN)
2.(POPPY'S PLAYHOUSE KITCHEN, POPPY'S PLAYHOUSE ...	(POPPY'S PLAYHOUSE BEDROOM)	
3.(HOT WATER BOTTLE I AM SO POORLY, SCOTTIE DOG HOT WATER BOTTLE) ...	(CHOCOLATE HOT WATER BOTTLE)	
4.(POPPY'S PLAYHOUSE BATHROOM)	(POPPY'S PLAYHOUSE KITCHEN)	
5.(POPPY'S PLAYHOUSE BATHROOM)	(POPPY'S PLAYHOUSE BEDROOM)	
6.(POPPY'S PLAYHOUSE BATHROOM)	(POPPY'S PLAYHOUSE BEDROOM, POPPY'S PLAYHOUSE ...	
7.(POPPY'S PLAYHOUSE BEDROOM, POPPY'S PLAYHOUSE BATHROOM ...	(POPPY'S PLAYHOUSE KITCHEN)	
8.(POPPY'S PLAYHOUSE LIVINGROOM, POPPY'S PLAYHOUSE KITCHEN ...	(POPPY'S PLAYHOUSE BEDROOM)	
9.(POPPY'S PLAYHOUSE LIVINGROOM)	(POPPY'S PLAYHOUSE KITCHEN) 
10.(LARGE POPCORN HOLDER)	(SMALL POPCORN HOLDER)

#### Trouver les règles d'association avec un support d'au moins 2% un lift de plus de 1
```python
rules[(rules['support'] >= 0.02) &
      (rules['lift'] > 1.0)]
```
On trouve **528** associations avec un support minimal de 2% et un lift de plus de 1; cela veut dire que si les ventes d'un produit augmentent celles du produit associé aussi et vice-versa.


## Conclusion

Ce projet d'**analyse de panier** avec Python et l'algorithme Apriori a permis d'extraire des informations précieuses sur les **habitudes d'achat des clients**.
En examinant les **ensembles d'articles fréquents**, il a été possible d'identifier les produits les plus populaires (ex : porte bougie en forme de cœur blanc avec un support de 16%) ainsi que les groupes de produits souvent achetés ensemble ( ex : groupe coeur en osier avec un support de 5% ). 

De plus, grâce aux **règles d'association**, nous avons découvert des liens significatifs entre différents articles comme la relation entre les produits "POPPY'S PLAYHOUSE BEDROOM" et "POPPY'S PLAYHOUSE BATHROOM", qui sont souvent associés au produit "POPPY'S PLAYHOUSE KITCHEN", ce qui pourra pemettre de faire des recommandations pertinentes pour **améliorer les ventes croisées et la satisfaction client**.
