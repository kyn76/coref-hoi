# Graduation project : improve neural coreference classifier via "k-best" approach

**Author** : Simon Decomble, Master 2 Machine Learning (2022/2023)

## Project introduction

Lorem ipsum

## k-best approach gain evaluation

### Commands to launch evaluations

#### Gold logging

To log into `kba-antecedent-csv/{num_doc}-gold_antecedents.csv` the gold (from oracle) pairs mention-antecedent from the document of number *num_doc* (precise id is in the csv file). If the mention is not anaphoric, it is linked to the dummy antecedent (index -1). 

**Command** :  
`python kba-gold_logging.py CONFIG_NAME SAVED_SUFFIX [GPU_ID]`

(argument between brackets is optional, if not provided, no GPU used)

**Example used for this project** :  
`python kba-gold_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000`

#### k-best logging (gold boundaries)

**Preliminary steps** :  
- in the *forward* method of model.py, let only `return self.get_predictions_and_loss_gold_bound(*input)` uncommented
- at the beginning of model.py : set MAX_TOP_ANTECEDENTS to wanted k value (200 in our results)

**Command** :  
`python kba-kbest_logging.py CONFIG_NAME SAVED_SUFFIX  gold K_VALUE [GPU_ID]`

**Example used for this project** :  
`python kba-kbest_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 gold 200`

#### k-best logging (predicted boundaries)

Constant variable / macro MAX_TOP_ANTECEDENTS used to replace conf["max_top_antecedents"], in clean code, it has to be modified in configuration. Here we use k=50 because there are more predicted antecedents than gold, and memory limits of my machine are reached for lower value of k.

- in the *forward* method of model.py, let only `return self.get_predictions_and_loss(*input)` uncommented
- at the beginning of model.py : set MAX_TOP_ANTECEDENTS to wanted k value : 90 
- `python kba-kbest_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 predicted 90`

#### Evaluation from CSV files (gold boundaries)
`python kba-evaluate_from_csv.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 gold 200`

#### Evaluation from CSV files (predicted boundaries)
`python kba-evaluate_from_csv.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 predicted 50`

### Diff with source repository
List of modified files (compared to https://github.com/lxucs/coref-hoi, see precise diff between source and fork for more details) :
- conll.py : to extract mention detection metrics from the output of Pearl script
- evaluate.py : to add the possibility of CPU evaluation only
- model.py : to add gold boundaries option, reduce pruning and change max_top_antecedents (which is decided by a configuration field, but an existing configuration was used)
- run.py : to add new process methods (logging, evaluation from csv)

The new files and folders are prefixed by `kba-` (which stands for k-best antecedents approach).

### Details and remarks

We tried to modify the source code at minimum and allow reproducibility of experiments through command line, but a few steps are sometimes necessary before, directly in the code. This is because :  
- we used an existing configuration
- we implement the use of gold boundaries in the model forward step, which was not a native feature

The max number of antecedents for the test documents is 262, so the maximum k value for test documents is probably this one. Due to memory limits, the experiment was made with k = 50 for predicted boundaries and k = 200 for gold boundaries.
The forward step of the model is indeed more memory-consuming with predicted boundaries.


