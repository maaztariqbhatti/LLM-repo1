###Human evaluation of FSD_1777
#Note down the runs
#k=30 llama3
#Pre flood peak
import json
import wandb

#Context Entity recall
GE = 34

#Query
#30 and 40 - 26 
#50 -28
CE_GE = 28
CER = CE_GE/GE

#Answer Entity recall
AE_GE = 12
AER = AE_GE/GE

#F1 Score
true_positives = AE_GE
false_positives = 4
false_negatives = GE - true_positives

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
CE_notin_AE = CE_GE-AE_GE

##update wandb
api = wandb.Api()
run_path = "sns-early-warning/FFSD_1777_Human_Eval/runs/7os1p8ff"
run = api.run(run_path)
run.summary["Human_context_entity_recall"] = CER
run.summary["Human_answer_entity_recall"] = AER
run.summary["Human_f1_score"] = f1_score
run.summary["Human_contextEntity_not_in_answerEntity"] = CE_notin_AE
run.summary.update()

print("CER: ", CER)
print("AER: ", AER)
print("f1_score: ", f1_score)
print("CE_notin_AE: ", CE_notin_AE)


