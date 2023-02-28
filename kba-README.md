# Graduation project : improve neural coreference classifier via "k-best" approach

**Author** : Simon Decomble, Master 2 Machine Learning (2022/2023)

## Project introduction

In the field of coreference resolution, Deep Learning models have allowed performance gains recently. However, those approaches tend to be greedy and rely mainly on local coreference decisions. Some work has been made to consider more global information (relative to the coreference clusters), but those higher order methods were not enough conclusive.

In this project, we try to adapt some pre-neural global inference approaches to a recent coreference model (https://aclanthology.org/2020.emnlp-main.686/), and evaluate the impact on performance.

More precisely, the goal is to use the model code from https://github.com/lxucs/coref-hoi. This model reports the k best antecedents predictions for each detected mention, associated to a score to weight their likeliness to be the antecedent or to precise if it is dummy (score of 0, the mention is not anaphoric). But then, this model use only the 1-best predicted antecedents to create the coreference cluster. Here, we want to benefit from the k-best prediction and keep all the information to implement a less "greedy" approach (consider the k-best instead of the 1-best).  

First, the goal of this project is to evaluate the possible benefits from this k-best approach. Then, we will try to implement a prototype of less greedy approach similar to Luo et al's Bell Tree (https://aclanthology.org/P04-1018.pdf) to consider different types of clusterings at the same time during the construction of the tree, and perform a beam search in it (guided by prediction scores criteria).

## k-best approach gain evaluation

In this part, we try to evaluate the benefits of a score ranking approach. First, we log the gold (given by oracle) mention-antecedent pairs in CSV files (one file per test document). This step is named the *gold logging*.  
Then, we log the predicted mention-antecedent pairs, considering for each mention the k best antecedents according to the model, and the associated score. This is what we call the **k-best logging** (one CSV file per test document). This step can be done using the gold mentions boundaries (starts / ends) or the predicted boundaries.  
Finally, using the CSV files, we evaluate the gain of considering the k best antecedents for each mention and not just the first one. The precise process is described in the corresponding section below.  

### Commands to launch evaluations

#### 1. Gold logging

To log into `kba-antecedent-csv/{num_doc}-gold_antecedents.csv` the gold (from oracle) mention-antecedent pairs from the document of number *num_doc* (precise id is in the csv file). If the mention is not anaphoric, it is linked to the dummy antecedent (index -1). 

**Command** :  
`python kba-gold_logging.py CONFIG_NAME SAVED_SUFFIX [GPU_ID]`

(argument between brackets is optional, if not provided, no GPU used)

**Example used in this project** :  
`python kba-gold_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000`

#### 2.a. k-best logging (gold boundaries)

To log into `kba-antecedent-csv/{num_doc}-k_best_ant_gold.csv` the **k** best predicted mention-antecedent pairs(according to antecedent score given by the model) from the document of number *num_doc* (precise id is in the csv file). If the mention is predicted as not anaphoric, it is linked to the dummy antecedent (index : -1, antecedent score : 0). For this logging, the gold spans boundaries (spans start and end given by the oracle) are used.

**Preliminary steps** :  
- in the *forward* method of model.py, let only `option = "gold_boundaries"` uncommented
- at the beginning of model.py : set MAX_TOP_ANTECEDENTS to wanted k value (200 in our evaluation)

**Command** :  
`python kba-kbest_logging.py CONFIG_NAME SAVED_SUFFIX  gold K_VALUE [GPU_ID]`

**Example used in this project** :  
`python kba-kbest_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 gold 200`

#### 2.b. k-best logging (predicted boundaries)

To log into `kba-antecedent-csv/{num_doc}-k_best_ant_gold.csv` the **k** best predicted mention-antecedent pairs(according to antecedent score given by the model) from the document of number *num_doc* (precise id is in the csv file). If the mention is predicted as not anaphoric, it is linked to the dummy antecedent (index : -1, antecedent score : 0). For this logging, the predicted spans boundaries are used. Those spans boundaries are computed at the very beginning of the end-to-end neural approach.

**Preliminary steps** :  
- in the *forward* method of model.py, let only `option = None` uncommented
- at the beginning of model.py : set MAX_TOP_ANTECEDENTS to wanted k value (50 in our evaluation) 

**Command** :  
`python kba-kbest_logging.py CONFIG_NAME SAVED_SUFFIX  predicted K_VALUE [GPU_ID]`

**Example used in this project** :  
`python kba-kbest_logging.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 predicted 50`

#### 3.a. Evaluation from CSV files (gold boundaries)

Once the gold and kbest logging CSV files have been written, this process reads for each document the k-best predicted mention-antecedent pairs and compare them to the real (gold) pairs. If the gold is in the list, it is picked to build the cluster later ; else, the first prediction is used (like in the source repository's approach). We use this process by taking the value of k between 1 and K_MAX_VALUE (e.g. 200) and report the metrics in a csv file. For each mention, we consider k possible prediction, as k increase for each iteration of the process, we expect that the metrics also increase. For the gold boundaries case, as span detection cannot cascade errors on coreference, we expect that the results approach 100% precision/recall/f-score when k approaches 262 (the max number of antecedents observed in the test documents).

**Command** :  
`python kba-evaluate_from_csv.py CONFIG_NAME SAVED_SUFFIX gold K_MAX_VALUE [GPU_ID]`

**Example used in this project** :
`python kba-evaluate_from_csv.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 gold 200`

*Note* : Even if we evaluate from csv files, we need to provide the configuration and saved_suffix to instantiate the model, because the evaluator is necessarily linked to a model in the native implementation.

#### 3.b. Evaluation from CSV files (predicted boundaries)

See the explanation in the previous section.  
The expectations are similar, but we do not really expect that the f-score approaches 100% as the mention detection would not be perfect (span boundaries are not exact for each mention). Mention detection error would be cascaded over mention-antecedent pairs and then coreference results.

**Command** :  
`python kba-evaluate_from_csv.py CONFIG_NAME SAVED_SUFFIX predicted K_MAX_VALUE [GPU_ID]`

**Example used in this project** :
`python kba-evaluate_from_csv.py train_spanbert_large_ml0_d1 May08_12-37-39_54000 predicted 50`

### Diff with source repository
List of modified files (compared to https://github.com/lxucs/coref-hoi, see precise diff between source and fork for more details) :
- conll.py : to extract mention detection metrics from the output of Pearl script
- evaluate.py : to add the possibility of CPU evaluation only
- model.py : to add gold boundaries option, reduce pruning and change max_top_antecedents (which is decided by a configuration field, but an existing configuration was used)
- run.py : to add new process methods (logging, evaluation from csv)

The new files and folders are prefixed by `kba-` (which stands for k-best antecedents approach).

### Extra remarks

A file `kba-checks.py` was used for debugging and some checks about mention-antecedents pairs. Specifically, to make stats about the predicted antecedents in the generated CSV files, compared to the gold, you can :
- generate the gold and kbest logging CSV files (see above)
- launch `python kba-checks.py gold` (gold boundaries) or `python kba-checks.py predicted` (predicted boundaries)

The output will show you how many mention-antecedents pairs are correctly predicted and reported in the prediction CSV file (whatever the rank k), and also if they are associated to positive, null, negative or minus infinity score.  
For the CSV files generated in this project (k = 200 for gold boundaries and k = 50 for predicted boundaries), the two outputs are below :
```
Stats about correct antecedents predictions (gold boundaries), recorded in csv files given the max rank k

  Nb positive scores :     62.78% (12407/19764)
  Nb null scores (dummy) : 22.93% (4532/19764)
  Nb negative scores :     14.29% (2825/19764)
--------------------------
  TOTAL (predicted) :      100.00% (19764/19764)
--------------------------
  Nb unpredicted :         0.00% (0/19764) -> number of missing correct antecedent prediction
  Nb -infinity scores :    0.00% (0/19764)
```
  
```
Stats about correct antecedents predictions (predicted boundaries), recorded in csv files given the max rank k

  Nb positive scores :     62.02% (12258/19764)
  Nb null scores (dummy) : 21.64% (4277/19764)
  Nb negative scores :     10.30% (2035/19764)
--------------------------
  TOTAL (predicted) :      93.96% (18570/19764)
--------------------------
  Nb unpredicted :         6.04% (1194/19764) -> number of missing correct antecedent prediction
  Nb -infinity scores :    0.00% (0/19764)
```

We tried to modify the source code at minimum and allow reproducibility of experiments through command line, but a few steps are sometimes necessary before, directly in the code. This is because :  
- we used an existing configuration
- we implement the use of gold boundaries in the model forward step, which was not a native feature.

The max number of antecedents for the test documents is 262, so the maximum k value for test documents is probably this one. Due to memory limits, the experiment was made with k = 50 for predicted boundaries and k = 200 for gold boundaries.
The forward step of the model is indeed more memory-consuming with predicted boundaries.  

The machine on which the experiments was made has 8 GB of RAM, the processes were made without any memory-consuming application concurrency. The memory consumption for the prediction process (for some documents) was higher than 7 GB if the value of k goes beyond 50 (predicted boundaries) / 200 (gold boundaries).

### Results

The reported metrics are native and computed thanks to a source repo Python's module and a Pearl script. We used those engines to evaluate the impact of our approach and report the metrics in `kba-metrics/k-metrics-gold_boundaries.csv` and `kba-metrics/k-metrics-pred_boundaries.csv`.  

We made the corresponding graphs for the 5 different f-scores :
- Python evaluation averaged
- CoNLL MUC metric
- CoNLL B-CUBED metric
- CoNLL CEAFe metric
- the average of the three previous : CoNLL averaged

#### 1. Gold boundaries

![goldb_eval_avg_f](kba-graphs/goldb_eval_avg_f.png)
![goldb_conll_avg_f](kba-graphs/goldb_conll_avg_f.png)
![goldb_conll_muc_f](kba-graphs/goldb_conll_muc_f.png)
![goldb_conll_bcub_f](kba-graphs/goldb_conll_bcub_f.png)
![goldb_conll_ceaf_f](kba-graphs/goldb_conll_ceafe_f.png)

When we use gold span boundaries, all f-scores approach 100% when k increases. The value k = 50 (which is the default max_top_antecedents value in configuration) seems to show almost perfect results for CoNLL in terms of potential gain : i.e. we can expect to have a clustering really near to the perfect one if we consider 50 antecedents for each mention and pick the best each time.  
For the "eval_avg_f" (Python averaged evaluation), the convergence seems to be slower, and we don't reach the 100% for k = 200, unlike the 4 other f-scores. The reasons behind that are not figured out.

#### 2. Predicted boundaries

![predb_eval_avg_f](kba-graphs/predb_eval_avg_f.png)
![predb_conll_avg_f](kba-graphs/predb_conll_avg_f.png)
![predb_conll_muc_f](kba-graphs/predb_conll_muc_f.png)
![predb_conll_bcub_f](kba-graphs/predb_conll_bcub_f.png)
![predb_conll_ceaf_f](kba-graphs/predb_conll_ceafe_f.png)

When we use predicted span boundaries, the f-scores are also logically increasing with k. However, we only reach scores near 90%. There are two reasons to explain that.  
On the one hand, the evaluation was restricted to k = 50 (because of memory limitation), we can expect to go beyond for higher values of k.  
On the other hand, statistics about correct predicted boundaries in the whole CSV files (see *Extra remarks* above) have shown that there are only about 93% of antecedents that are correctly reported in CSV files. We can assume that the extra 7% are missing because the model did not compute the correct span boundaries (as there are 0% missing when we use gold boundaries).  
So, the main part of explanation of this performance limitation near 90% when k increase is probably caused by a cascading error from some uncorrect spans start/end.  

## Prototype of k-best Bell Tree

`python kba-bell_tree_one_beam.py train_spanbert_large_ml0_d1 May08_12-37-39_54000`

