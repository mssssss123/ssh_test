import fire
import torch
import json
from tqdm import tqdm
from transformers import pipeline, logging
logging.set_verbosity_error()

def acc(output, data):
    correctness = 'True' if any(answer.lower() in output.lower() for answer in data['answers']) else 'False'
    return correctness

def get_summary(generator, question, passages):

    task_prompt = """You are an AI assistant helping me summary the passages.
I will give you the original query and different passages that are based on the retrieval results from different domain corpora.
Based on the content of the question to be answered, please extract relevant useful information from these pasages and provide a summary of the passages, note that just give me the summary without other informations:
Original Query: {query}
Passages: {p}
Summary:"""

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task_prompt.format(query=question, p=passages)},
    ]
    generation = generator(
        prompt,
        do_sample=True,
        temperature=0.5,
        top_p=0.5,
        max_new_tokens=1024
    )

    return generation[0]['generated_text'][-1]['content']

def get_output(generator, question, passages):
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
            temp['passages'] = []
            all_data.append(temp)

    for domain in domains:
        with open(f'{retrieve_path}/{domain}_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/{test}_requery_top5_answer.jsonl') as domain_file:
            i = 0
            for line in domain_file:
                item = json.loads(line.strip())
                all_data[i]['passages'].extend(item['ori_passages'])
                all_data[i]['answers'] = item['answers']
                i += 1

    with open(f'{output_path}/{test}_summary_document_ori.jsonl', 'w') as output_file:
        accuracy = 0
        i = 0
        for item in all_data:
            temp = {}
            all_passages = '\n'.join([i['retrieval text'] for i in item['passages']])
            summary_passage = get_summary(generator, item['query'], all_passages)
            temp['passages'] = item['passages']
            temp['query'] = item['query']
            temp['answers'] = item['answers']
            temp['summary_passage'] = summary_passage
            output = get_output(generator, item['query'], summary_passage)
            temp['output'] = output

            accuracy += 1 if acc(output, temp) == 'True' else 0
            i += 1

            output_file.write(json.dumps(temp)+'\n')
            print(f"query: {temp['query']}\nsummary passages: {temp['summary_passage']}\noutput:{output}\n{accuracy}/{i}\n**********************************")
        
        print(f'Acc: {accuracy/i}')

if __name__ == "__main__":
    fire.Fire(main)