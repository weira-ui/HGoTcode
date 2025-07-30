import json
from nltk.metrics import scores

def calculate_em(answer, ground_truth):
    # 处理字符串
    answer_processed = answer.strip().replace(" ", "").lower()
    ground_truth_processed = ground_truth.strip().replace(" ", "").lower()

    # 判断是否相等或一方包含另一方
    if answer_processed == ground_truth_processed or answer_processed in ground_truth_processed or ground_truth_processed in answer_processed :
        return True
    return False

def calculate_f1(answer, ground_truth):
    ref_tokens = set(ground_truth.strip().replace(" ", "").lower().split())
    hyp_tokens = set(answer.strip().replace(" ", "").lower().split())
    p = scores.precision(ref_tokens, hyp_tokens)
    r = scores.recall(ref_tokens, hyp_tokens)
    if p + r > 0:
        f1 = 2 * (p * r) / (p + r)
    else:
        f1 = 0
    return f1

# 读取原始数据
# with open('../2wiki/reslt_document_1.json', 'r') as file:
with open('../hotpotqa/reslt_document.json', 'r') as file:
    data = json.load(file)
# 初始化计数器
true_count = 0
false_count = 0
# 初始化总参考集和预测集
ref_tokens_total = set()
hyp_tokens_total = set()
# 计算EM和F1 Score，并更新answer
for item in data:
    answer=item['answer']
    if "Type" in answer:
        answer=answer.split("Type")[0]
    if "The type" in answer:
        answer=answer.split("The type")[0]
    if ", the type " in answer:
        answer=answer.split(", the type ")[0]
    if " the type " in answer:
        answer=answer.split(" the type ")[0]
    if "was born" in answer:
        answer=answer.split("was born")[0]
    if "Final Answer: " in answer:
        answer=answer.split("Final Answer: ")[1]
    answer=answer.rstrip('\n').rstrip(' ')

    item['answer']=answer
    em = calculate_em(answer, item['ground_truth'])
    item['em'] = em
    if em:
        true_count += 1
        answer = item['ground_truth']
    else:
        false_count += 1
        # answer = "no"
    f1 = calculate_f1(answer, item['ground_truth'])
    item['f1'] = f1

# 统计并输出结果
total_count = len(data)
FN=total_count-true_count-false_count
TP=true_count
FP=false_count
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
F1=2*Precision*Recall/(Precision+Recall)
accuracy_percentage = (true_count / total_count) * 100

print(f"EM: {accuracy_percentage:.2f}")
print(f"Overall Precision: {Precision:.4f}")
print(f"Overall Recall: {Recall:.4f}")
print(f"Overall F1 Score: {F1:.4f}")
# 将结果保存到新的JSON文件
with open('evaluated_data_2wiki_document.json', 'w') as file:
    json.dump(data, file, indent=4)
