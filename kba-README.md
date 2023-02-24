# Precisions on added code during the "kbest approach" graduation project

**Author** : Simon Decomble, Master 2 Machine Learning (2022/2023)

prefix kba = k-best-antecedents approach

model.py : change function in forward

- modif code pour reproductibilité
  - code génération CSV kbest pred
  - code génération CSV kbest gold
  - code génération CSV gold
  - remettre les fonctions d'origine avec leur code natif (mettre un commentaire pour dire de virer le #loss) → faire un diff avec fichiers du repo de base
  - evaluate csv propre nickel

MAX_NB_ANTECEDENTS_EXAMPLE = 210 # the max number of antecedents for the test documents is 262 but in model this value was reduced to 210 because of lack of memory

Code du model de base modifié dans tous les cas pour prendre en compte 200 antécédents, par défaut il en prend 50 max (dans le fichier de conf, max_top_antecedents=50)

## gold logging
`python kba-gold_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000`

## kbest logging (gold boundaries)
- in the *forward* method of model.py, let only `return self.get_predictions_and_loss_gold_bound(*input)` uncommented
- `python kba-kbest_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 gold 200`

## kbest logging (predicted boundaries)
- in the *forward* method of model.py, let only `return self.get_predictions_and_loss(*input)` uncommented
- `python kba-kbest_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 predicted 200`

## evaluation from csv files (gold boundaries)
`python kba-evaluate_from_csv.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 gold 200`

## evaluation from csv files (predicted boundaries)
`python kba-evaluate_from_csv.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 predicted 200`