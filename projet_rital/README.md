# Projet TAL — Classification des sentiments

Cette partie du projet porte sur la **classification automatique de critiques de films** en deux classes : **positive** et **négative**, à partir du corpus `movies1000`.

Les notebooks fournis correspondent aux différentes étapes principales de la démarche expérimentale, depuis l’analyse exploratoire jusqu’au modèle final.

---

## Contenu des notebooks

### 1. `eda_sentiment_movies1000.ipynb`
Ce notebook contient l’**analyse exploratoire des données (EDA)** du corpus `movies1000`.

Il comprend notamment :
- le chargement du corpus,
- la vérification de l’équilibre des classes,
- les statistiques descriptives globales,
- l’analyse de la longueur des critiques,
- les visualisations (histogrammes, boxplots),
- les mots les plus fréquents par classe,
- les nuages de mots.

Ce notebook a servi à produire les figures et tableaux de la partie descriptive du rapport.

---

### 2. `movies_baseline_classification_v2.ipynb`
Ce notebook présente les **premières expériences de classification**, avec des modèles classiques de base.

Il contient :
- la préparation initiale des données,
- la vectorisation TF-IDF,
- les modèles de baseline :
  - Naive Bayes,
  - régression logistique,
  - SVM linéaire,
- les premières comparaisons de performances.

Ce notebook permet d’établir une baseline solide pour la suite des expériences.

---

### 3. `movies_03_improvements.ipynb`
Ce notebook contient les **améliorations apportées au meilleur modèle classique**, en particulier au SVM.

Il comprend :
- une recherche d’hyperparamètres,
- l’évaluation de différentes variantes de représentation :
  - word n-grams,
  - char n-grams,
  - LSA,
- l’identification de la meilleure configuration finale du modèle SVM.

Ce notebook correspond à la phase d’optimisation des modèles classiques.

---

### 4. `movies_04_advanced_nlp_fasttext_distilbert_bert.ipynb`
Ce notebook regroupe les expériences avec plusieurs **modèles NLP avancés**.

Il contient les essais réalisés avec :
- FastText,
- DistilBERT,
- BERT.

Ces expériences permettent de comparer les modèles classiques à des approches fondées sur des embeddings ou sur des Transformers généralistes.

---

### 5. `movies_05_xlmr_twitter_roberta_cardiffnlp.ipynb`
Ce notebook présente les expériences avec des **Transformers spécialisés pour l’analyse de sentiment**.

Il contient notamment :
- les tests avec Twitter-XLM-R,
- les essais avec Twitter-RoBERTa de CardiffNLP,
- le réglage fin du modèle,
- l’évaluation des performances sur validation.

Ce notebook correspond à l’étape où les modèles spécialisés commencent à dépasser les approches précédentes.

---

### 6. `movie7_head_tail_roberta_ensemble.ipynb`
Ce notebook contient les expériences liées à l’**adaptation aux textes longs** et aux premières méthodes d’ensemble.

Il comprend :
- l’analyse du problème de troncature des critiques longues,
- la mise en place des variantes **head+tail**,
- la comparaison entre plusieurs découpages, notamment :
  - 256/256,
  - 384/128,
- des essais de combinaison avec le SVM.

Ce notebook est important car il montre comment l’analyse d’erreurs a conduit à une amélioration du modèle.

---

### 7. `movie10_full_ensemble_all_submissions.ipynb`
Ce notebook correspond à la **phase finale du projet**.

Il contient :
- la comparaison des meilleurs modèles retenus,
- les expériences d’ensemble les plus avancées,
- la recherche des meilleures pondérations,
- la sélection du meilleur système final,
- la préparation et le suivi des soumissions sur la plateforme de test.

C’est le notebook qui contient la version la plus complète du système final présenté dans le rapport.

---

## Remarque

Certains notebooks intermédiaires plus anciens ou redondants n’ont pas été retenus dans la version finale rendue, afin de garder uniquement les étapes les plus utiles, les plus propres et les plus représentatives de la démarche expérimentale.

---

## Résumé de la progression

La progression suivie dans cette partie est la suivante :

1. Analyse exploratoire du corpus
2. Baselines classiques
3. Amélioration du SVM
4. Modèles NLP avancés
5. Transformers spécialisés pour le sentiment
6. Adaptation aux textes longs
7. Ensemble final et soumissions test

---

# Projet TAL — Classification des présidents

Cette partie du projet porte sur la **classification automatique de phrases politiques** en deux classes correspondant à leur auteur : **Jacques Chirac** ou **François Mitterrand**.

Les notebooks fournis correspondent aux différentes expériences menées sur cette tâche, depuis la baseline phrase par phrase jusqu’aux variantes contextuelles finales, en passant par l’analyse des frontières entre locuteurs et l’analyse d’erreurs.

---

## Contenu des notebooks

### 1. `pres_phrase_baseline_sans_chunks.ipynb`
Ce notebook contient la **baseline la plus simple**, dans laquelle chaque phrase est traitée indépendamment, sans construction de chunks ni ajout de contexte documentaire.

Il comprend notamment :
- le chargement et la préparation du corpus présidentiel,
- l’encodage des labels,
- une première approche de classification phrase par phrase,
- l’évaluation des performances de base.

Ce notebook sert de point de comparaison pour mesurer l’apport des approches contextuelles plus avancées.

---

### 2. `pres_final_4tests_compare_retrain_submit_corrected (1).ipynb`
Ce notebook correspond à la **version finale principale** de la partie présidents.

Il contient :
- la préparation finale des données,
- le split par document,
- la construction des chunks d’entraînement,
- la comparaison de plusieurs configurations finales,
- le réentraînement des meilleurs modèles,
- l’évaluation locale détaillée,
- la préparation des soumissions sur la plateforme de test.

C’est le notebook principal de cette partie, car il regroupe les expériences comparatives finales et permet d’identifier le meilleur système retenu dans le rapport.

---

### 3. `pres_v14_corrige_weighted_earlystop.ipynb`
Ce notebook présente une variante introduisant un **rééquilibrage de la fonction de coût** ainsi qu’un mécanisme d’**early stopping**.

Il comprend :
- le calcul de poids de classes adaptés au déséquilibre du corpus,
- l’entraînement du modèle avec pondération,
- le suivi des performances sur validation,
- la comparaison avec la version standard.

Ce notebook permet d’étudier l’effet de la pondération sur la détection de la classe minoritaire, Mitterrand.

---

### 4. `pres_v14_corrige_frontieres_chunks.ipynb`
Ce notebook contient les expériences liées à la **prise en compte du contexte**, des **chunks** et des **frontières entre locuteurs**.

Il comprend notamment :
- la construction des chunks d’apprentissage,
- les fenêtres de contexte autour des phrases,
- l’évaluation du comportement du modèle dans les documents mixtes,
- l’étude de l’impact des zones proches des transitions Chirac/Mitterrand.

Ce notebook est important car il montre comment la structure documentaire influence directement la difficulté de la tâche.

---

### 5. `pres_v14_corrige_analyse_erreurs (1).ipynb`
Ce notebook est consacré à l’**analyse des erreurs** du système.

Il contient :
- l’identification des phrases mal classées,
- l’analyse des erreurs selon le type de document,
- l’étude des performances près des frontières entre locuteurs,
- la comparaison entre documents mixtes et documents non mixtes,
- des observations utiles pour interpréter les résultats finaux.

Ce notebook a servi à nourrir la discussion des résultats dans le rapport.

---

## Remarque

Comme pour la partie sentiment, certains fichiers intermédiaires, anciennes versions ou doublons n’ont pas été retenus dans le livrable final, afin de conserver uniquement les notebooks les plus utiles et les plus représentatifs de la démarche expérimentale.

---

## Résumé de la progression

La progression suivie dans cette partie est la suivante :

1. Baseline phrase par phrase
2. Mise en place du split par document
3. Construction de chunks d’apprentissage
4. Intégration du contexte local
5. Expériences avec pondération et early stopping
6. Analyse des frontières entre locuteurs
7. Analyse d’erreurs
8. Comparaison finale des modèles et soumissions test
