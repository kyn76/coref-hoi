import csv

NB_DOCUMENTS = 348
NB_TOTAL_MENTIONS = 19764
K = 5

# Old function used for the ranking approach evaluation
def evaluate_antecedents():
    # correct antecedent = same start/end of anaphor match same start/end of antecedent
    with open("metrics/eval_antecedents.md", "w") as out_file:
        out_file.write("# Antecedent Evaluation\n\n")
        for i in range(NB_DOCUMENTS):
            gold_file = open(f"antecedents-csv/{i}-gold_antecedents.csv", "r")
            pred_file = open(f"antecedents-csv/{i}-k_best_antecedents.csv", "r")
            gold_reader = list(csv.DictReader(gold_file))
            pred_reader = list(csv.DictReader(pred_file))
            correct_antecedent_at_rank = {1 : 0,
                                          2 : 0,
                                          3 : 0,
                                          4 : 0,
                                          5 : 0}
            doc_key = None
            nb_gold_antecedents = 0
            for gold_row in gold_reader:
                if doc_key is None:
                    doc_key = gold_row["doc_key"]
                nb_gold_antecedents += 1
                for pred_row in pred_reader:
                    gold_tuple = (gold_row["span_start"], gold_row["span_end"], gold_row["antecedent_start"], gold_row["antecedent_end"])
                    pred_tuple = (pred_row["span_start"], pred_row["span_end"], pred_row["antecedent_start"], pred_row["antecedent_end"])
                    if gold_tuple == pred_tuple:
                        correct_antecedent_at_rank[int(pred_row["antecedent_rank"])] += 1
            
            out_file.write(f"## Document {i} ({doc_key}) :\n")
            for rank in range(1, K+1):
                out_file.write(f"Number of correct antecedent(s) at rank {rank}  : {correct_antecedent_at_rank[rank]}/{nb_gold_antecedents}\n")
            out_file.write(f"Number of not found anaphor(s)/antecedent(s) : {nb_gold_antecedents - sum(correct_antecedent_at_rank.values())}/{nb_gold_antecedents}\n\n")
            gold_file.close()
            pred_file.close()

def count_nb_mentions_in_gold_csv():
    total_mentions = 0
    for i in range(NB_DOCUMENTS):
        gold_file = open(f"antecedents-csv/{i}-gold_antecedents.csv", "r")
        lines = gold_file.readlines()
        total_mentions += len(lines)-1
        gold_file.close()
    assert(total_mentions == NB_TOTAL_MENTIONS)

def check_mentions_in_k_best_ant_gold_csv():
    for i in range(NB_DOCUMENTS):
        print(f"Doc {i+1}/{NB_DOCUMENTS}")
        # get gold spans boundaries from gold_antecedent csv
        gold_spans = [] # list of gold (start, end) 
        gold_file = open(f"antecedents-csv/{i}-gold_antecedents.csv", "r")
        gold_reader = list(csv.DictReader(gold_file))
        for gold_row in gold_reader:
            gold_spans.append((int(gold_row["span_start"]), int(gold_row["span_end"])))
        gold_file.close()

        # get gold boundaries in k_best_ant_gold_bound csv
        kbest_spans = []
        kbest_file = open(f"antecedents-csv/{i}-k_best_ant_gold_bound.csv", "r")
        kbest_reader = list(csv.DictReader(kbest_file))
        for row in kbest_reader:
            span = (int(row["span_start"]), int(row["span_end"]))
            if span not in kbest_spans:
                kbest_spans.append(span)
        kbest_file.close()

        # check if every gold spans are in the 2nd csv file (which is the one used for the following conll evaluation)
        assert(gold_spans == kbest_spans)

def count_recorded_correct_antecedents(gold_boundaries=True):
    nb_neg_scores = 0 #nb correct antecedent with negative scores
    nb_minus_inf_scores = 0 #same with -inf scores
    nb_pos_scores = 0 #same with positive score
    nb_dummy = 0 #same with zero score (dummy antecedent)
    for i in range(NB_DOCUMENTS):
        gold_antecedent_dict = {} # keys : (start, end), values : (antecedent_start, antecedent_end)
        gold_file = open(f"antecedents-csv/{i}-gold_antecedents.csv", "r")
        gold_reader = list(csv.DictReader(gold_file))
        for gold_row in gold_reader:
            gold_antecedent_dict[(int(gold_row["span_start"]), int(gold_row["span_end"]))] = (int(gold_row["antecedent_start"]), int(gold_row["antecedent_end"]))
        gold_file.close()

        if gold_boundaries:
            suffix = "ant_gold_bound"
        else:
            suffix = "antecedents"
        kbest_file = open(f"antecedents-csv/{i}-k_best_{suffix}.csv", "r")
        kbest_reader = list(csv.DictReader(kbest_file))
        for row in kbest_reader:
            span = (int(row["span_start"]), int(row["span_end"]))
            if span not in gold_antecedent_dict:
                continue # wrong boundaries
            if float(row["antecedent_score"]) == 0 and gold_antecedent_dict[span] == (-1,-1): #span has dummy antecedent and gold too
                nb_dummy += 1
                continue
            antecedent = (int(row["antecedent_start"]), int(row["antecedent_end"]))
            if gold_antecedent_dict[span] == antecedent: #correct antecedent
                antecedent_score = float(row["antecedent_score"])
                assert(antecedent_score != 0) #can't have dummy here
                if antecedent_score > 0:
                    nb_pos_scores += 1
                else:
                    nb_neg_scores += 1
                    if antecedent_score == float("-inf"):
                        nb_minus_inf_scores += 1
        kbest_file.close()
        print(f"Running ... (Doc {i+1}/{NB_DOCUMENTS})")

    nb_predicted = nb_pos_scores + nb_neg_scores + nb_dummy
    nb_unpredicted = NB_TOTAL_MENTIONS - nb_predicted

    print("\nStats about correct antecedents predictions, recorded in csv files given the max rank (k=210 as of 21/02/2023)\n")
    print(f"  Nb positive scores :     {nb_pos_scores/NB_TOTAL_MENTIONS:.2%} ({nb_pos_scores}/{NB_TOTAL_MENTIONS})")
    print(f"  Nb null scores (dummy) : {nb_dummy/NB_TOTAL_MENTIONS:.2%} ({nb_dummy}/{NB_TOTAL_MENTIONS})")
    print(f"  Nb negative scores :     {nb_neg_scores/NB_TOTAL_MENTIONS:.2%} ({nb_neg_scores}/{NB_TOTAL_MENTIONS})")
    print("--------------------------")
    print(f"  TOTAL (predicted) :      {nb_predicted/NB_TOTAL_MENTIONS:.2%} ({nb_predicted}/{NB_TOTAL_MENTIONS})")
    print("--------------------------")
    print(f"  Nb unpredicted :         {nb_unpredicted/NB_TOTAL_MENTIONS:.2%} ({nb_unpredicted}/{NB_TOTAL_MENTIONS}) -> number of missing correct antecedent prediction") #complement
    print(f"  Nb -infinity scores :    {nb_minus_inf_scores/NB_TOTAL_MENTIONS:.2%} ({nb_minus_inf_scores}/{NB_TOTAL_MENTIONS})\n")

            


if __name__ == '__main__':
    # count_nb_mentions_in_gold_csv()
    # check_mentions_in_k_best_ant_gold_csv()
    count_recorded_correct_antecedents(gold_boundaries=True)