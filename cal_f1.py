import json
import torch
import os
import re, string

from collections import defaultdict, Counter
folder_path   = '{YOUR OUTPUT PATH}'  # Specify the path to your folder
dev_json_path = "./val_v0.2.json"

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def ff1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    
# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if the current item is a file
    if os.path.isfile(file_path):
        # Process the file as needed
        print("Processing file:", filename)

        dev = open("D://application//EvalConvQA-main//EvalConvQA-main//val_v0.2.json")
        dev = dev.readlines()
        dev = [json.loads(x) for x in dev]
        dev = dev[0]["data"]
        res = open(file_path)
        res = res.readlines()
        res = [json.loads(x) for x in res]


    # Load Gold Ans
    dev_qa = {}
    for x in dev:
        for y in x["paragraphs"]:
            for z in y["qas"]:
                zz = z["id"].split("_q#")[0]
                if zz not in dev_qa.keys():
                    dev_qa[zz] = {}
                dev_qa[zz][z["id"]] = z["orig_answer"]["text"]
               
    acc_f1 = 0
    for r in res:
        q    = r["qid"][0][:-4]
        
        ground_truth = list(dev_qa[q].values())
        predictions  = r["best_span_str"]
        
        sub_f1 = 0
        for idx, p in enumerate(predictions):
            tmp = ff1_score(p, ground_truth[idx])
            sub_f1 += tmp
        sub_f1 = sub_f1/len(predictions)
        
       

        # Print the result
        # print("F1 Score: {:.4f}".format(f1))          
        acc_f1 += sub_f1

    print ("{} - Overall F1: {}".format(filename, acc_f1/len(res)))
 
# import nltk
# def calculate_word_level_scores(ground_truth_answers, predicted_answers):
    # total_words = 0
    # true_positives = 0
    # false_positives = 0
    # false_negatives = 0
    
    # for ground_truth, predicted in zip(ground_truth_answers, predicted_answers):
        ## Tokenize ground truth and predicted answers into words
        # gt_words = set(nltk.word_tokenize(ground_truth.lower()))
        # pred_words = set(nltk.word_tokenize(predicted.lower()))
        
        ## Update counts
        # true_positives += len(gt_words.intersection(pred_words))
        # false_positives += len(pred_words - gt_words)
        # false_negatives += len(gt_words - pred_words)
        # total_words += len(gt_words)
    
    ## Calculate precision, recall, and F1 score
    # precision = true_positives / (true_positives + false_positives + 1e-10)
    # recall = true_positives / (true_positives + false_negatives + 1e-10)
    # f1 = (2 * precision * recall) / (precision + recall + 1e-10)
    
    # return precision, recall, f1

# acc_precision = 0
# acc_recall    = 0
# for r in res:
    # print ("*"*100)
    # q    = r["qid"][0][:-4]
    # print (q)
    
    # ground_truth = list(dev_qa[q].values())
    # predictions  = r["best_span_str"]
    
    ## Calculate word-level precision, recall, and F1 score
    # precision, recall, f1 = calculate_word_level_scores(ground_truth, predictions)

    ## Print the results
    # print("Word-level Precision: {:.4f}".format(precision))
    # print("Word-level Recall: {:.4f}".format(recall))
    # print("Word-level F1 Score: {:.4f}".format(f1))
    # acc_precision += precision
    # acc_recall    += recall
    
# of1 = (2 * acc_precision * acc_recall) / (acc_precision + acc_recall + 1e-10)
# print (of1)