###Human evaluation of FSD_1777
#Note down the runs
#k=30 llama3
#Pre flood peak
import json
import wandb
import dotenv
dotenv.load_dotenv()

#run path CHECKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
run_path = "sns-early-warning/FSD_1777_Human_Eval_FW_P2/runs/x7aokgu7"
#Context Entity recall
GE = 34

#Query -  CHECK FOR 50
#30,40 - 31
#50 -33
CE_GE = 31
CER = CE_GE/GE

#Answer Entity recall
AE_GE = 13
AER = AE_GE/GE

#F1 Score
true_positives = AE_GE
false_positives = 0
false_negatives = GE - true_positives

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

#Custom 
#Number of entities not recalled in Answer from context
CE_notin_AE = CE_GE-AE_GE
#Entities recalled in answer from context
AE_recall_CE = AE_GE/CE_GE

##update wandb
api = wandb.Api()
run = api.run(run_path)

run.summary["Human_context_entity_recall"] = CER
run.summary["Human_answer_entity_recall"] = AER
run.summary["Human_f1_score"] = f1_score
run.summary["Human_contextEntity_not_in_answerEntity"] = CE_notin_AE
run.summary["Human_answerEntity_recall_contextEntity"] = AE_recall_CE
run.summary.update()

print("CER: ", CER)
print("AER: ", AER)
print("f1_score: ", f1_score)
print("CE_notin_AE: ", CE_notin_AE)
print("AE_recall_CE: ", AE_recall_CE)


