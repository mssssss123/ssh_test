## gen_new_query

用于生成领域适配的query

用法：
```
python gen_new_query.py \
    --model_path /home/yujingsi/workspace/pretrained-models/Meta-Llama-3.1-8B-Instruct \
    --query_path /home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results/dpr_wiki_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/med_qa_retrieved_results.jsonl \
    --requery_path /home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results/dpr_wiki_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/med_qa_requery.jsonl \
    --corpus_name dprwiki
```

## search
用于检索文档，inference_pipeline需要放在在[retrieval-scaling](https://github.com/RulinShao/retrieval-scaling)的那个论文的目录下面

用法：
```
PYTHONPATH=.  python inference_pipeline/search.py --config-name pubmed_medqa
```

## gen_anwser
用于基于query生成答案

用法：
```
python gen_answer.py \
    --model_path /home/yujingsi/workspace/pretrained-models/Meta-Llama-3.1-8B-Instruct \
    --query_path /home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results/dpr_wiki_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/med_qa_requery_retrieved_results.jsonl \
    --answer_path /home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results/dpr_wiki_0.1_datastore-256_chunk_size-1of8_shards/top_100/0-1-2-3-4-5-6-7/med_qa_requery_top5_answer.jsonl
```

## anwser_summary
用于将基于不同语料库检索生成的答案进行总结

用法:
```
python anwser_summary.py \
    --model_path /home/yujingsi/workspace/pretrained-models/Meta-Llama-3.1-8B-Instruct \
    --retrieve_path /home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results \
    --output_path /home/yujingsi/workspace/rag/retrieval-scaling/data/summary_answer \
    --test_path /home/yujingsi/workspace/rag/retrieval-scaling/examples/med_qa.jsonl \
    --domains ['dpr_wiki', 'pubmed'] \
    --test med_qa
```

## document_summary
用于将基于不同语料库检索道德文档进行总结

用法:
```
python anwser_summary.py \
    --model_path /home/yujingsi/workspace/pretrained-models/Meta-Llama-3.1-8B-Instruct \
    --retrieve_path /home/yujingsi/workspace/rag/retrieval-scaling/data/retrieved_results \
    --output_path /home/yujingsi/workspace/rag/retrieval-scaling/data/summary_answer \
    --test_path /home/yujingsi/workspace/rag/retrieval-scaling/examples/med_qa.jsonl \
    --domains ['dpr_wiki', 'pubmed'] \
    --test med_qa
```