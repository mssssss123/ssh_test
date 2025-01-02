import fire
import torch
import json
from tqdm import tqdm
from transformers import pipeline, logging
logging.set_verbosity_error()

def acc(output, data):
    correctness = 'True' if any(answer.lower() in output.lower() for answer in data['answers']) else 'False'
    return correctness

def get_summary(generator, question, answers):

    task_prompt = """You are an AI assistant helping me summary the answer.
I will give you the original query and different answers that are based on the retrieval results from different domain corpora.
Please summarize these answers to the original query, note that just give me the summary without other informations:
Original Query: {query}
Answers: {ans}
Summary:"""

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task_prompt.format(query=question, ans=answers)},
    ]
    generation = generator(
        prompt,
        do_sample=True,
        temperature=0.5,
        top_p=0.5,
        max_new_tokens=1024
    )

    return generation[0]['generated_text'][-1]['content']


def main(
    model_path = "/home/yujingsi/workspace/pretrained-models/Meta-Llama-3.1-8B-Instruct",
    retrieve_path = "/home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results",
    output_path = "/home/yujingsi/workspace/rag/retrieval-scaling/data/summary_answer",
    test_path = "/home/yujingsi/workspace/rag/retrieval-scaling/examples/med_qa.jsonl",
    domains = ['dpr_wiki', 'pubmed'],
    test = 'med_qa'
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = pipeline(
        "text-generation",
        model=model_path,
        device=device,
        torch_dtype=torch.bfloat16
    )

    all_data = []
    with open(test_path, 'r') as init_file:
        for line in init_file:
            temp = {}
            item = json.loads(line.strip())
            temp['query'] = item['query']
            temp['outputs'] = []
            all_data.append(temp)

    for domain in domains:
        with open(f'{retrieve_path}/{domain}_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/{test}_requery_top5_answer.jsonl') as domain_file:
            i = 0
            for line in domain_file:
                item = json.loads(line.strip())
                all_data[i]['outputs'].append(item['re_output'])
                all_data[i]['answers'] = item['answers']

                i += 1

    with open(f'{output_path}/{test}_summary_answer_requery.jsonl', 'w') as output_file:
        accuracy = 0
        i = 0
        for item in all_data:
            temp = {}
            all_answers = '\n'.join(item['outputs'])
            summary_answer = get_summary(generator, item['query'], all_answers)
            temp['outputs'] = item['outputs']
            temp['query'] = item['query']
            temp['answers'] = item['answers']
            temp['summary_answer'] = summary_answer

            accuracy += 1 if acc(summary_answer, temp) == 'True' else 0
            i += 1

            output_file.write(json.dumps(temp)+'\n')
            print(f"query: {temp['query']}\nsummary: {temp['summary_answer']}\n{accuracy}/{i}\n**********************************")
        
        print(f'Acc: {accuracy/i}')

if __name__ == "__main__":
    fire.Fire(main)