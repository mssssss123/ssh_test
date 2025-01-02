import fire
import torch
import json
from tqdm import tqdm
from transformers import pipeline, logging
logging.set_verbosity_error()

def acc(output, data):
    correctness = 'True' if any(answer.lower() in output.lower() for answer in data['answers']) else 'False'
    return correctness

def get_answer(generator, passages, question):

    task_prompt = "Passages:\n{}\nBased on these texts, answer these questions. Avoid unnecessary details:\n{}"

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task_prompt.format(passages, question)},
    ]
    generation = generator(
        prompt,
        do_sample=True,
        temperature=0.1,
        top_p=0.5,
        max_new_tokens=128
    )

    return generation[0]['generated_text'][-1]['content']

def main(
    model_path = "/home/yujingsi/workspace/pretrained-models/Meta-Llama-3.1-8B-Instruct",
    query_path = "/home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results/dpr_wiki_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/med_qa_requery_retrieved_results.jsonl",
    answer_path = "/home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results/dpr_wiki_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/med_qa_requery_top5_answer.jsonl",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = pipeline(
        "text-generation",
        model=model_path,
        device=device,
        torch_dtype=torch.bfloat16
    )
    with open(query_path, 'r') as query_file, open(answer_path, 'w') as answer_file:
        ori_acc = 0
        re_acc = 0
        total_lines = sum(1 for _ in open(query_path, 'r'))
        for line in tqdm(query_file, total=total_lines, desc="Processing Queries"):
            temp = {}
            ori_passages = ''
            re_passages = ''

            item = json.loads(line.strip())
            # temp['id'] = item['id']
            # temp['answers'] = [a['answer'] for a in item['output'] if 'answer' in a]
            temp['answers'] = item['output']
            temp['query'] = item['ori_query']
            temp['requery'] = item['query']
            temp['ori_passages'] = item['ctxs_ori_query'][:5]
            temp['re_passages'] = item['ctxs'][:5]

            for ctx1 in temp['ori_passages']:
                ori_passages += f"{ctx1['retrieval text']}\n"
            for ctx2 in temp['re_passages']:
                re_passages += f"{ctx2['retrieval text']}\n"

            ori_output = get_answer(generator, ori_passages, item['query'])
            re_output = get_answer(generator, re_passages, item['query'])
            temp['ori_output'] = ori_output
            temp['re_output'] = re_output

            print(f"ori_query: {temp['query']}\nori_output: {ori_output}\nrequery: {temp['requery']}\nre_output: {re_output}\n*****************************************************")

            ori_acc += 1 if acc(ori_output, temp) == 'True' else 0
            re_acc += 1 if acc(re_output, temp) == 'True' else 0

            answer_file.write(json.dumps(temp)+'\n')
        print(f'original query acc: {ori_acc/total_lines}')
        print(f'new query acc: {re_acc/total_lines}')


if __name__ == "__main__":
    fire.Fire(main)