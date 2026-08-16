"""
RAG 检索准确率评估脚本（仅调用 Dify 检索接口，不调用任何大模型 / 不生成答案）

用途：评估知识库向量检索的召回质量，指标包括：
    HitRate@k / Recall@k / Precision@k / MRR / NDCG@k
    按 doc_type（standard / case）分组统计，并输出失败样例供人工分析。

前置条件：
    - Dify 知识库已配置（config/dify.yml 的 api_base / dataset_id，.env 的 DIFY_API_KEY）
    - 测试集 eval/test_set.jsonl（question + expected_docs [+ match_any] [+ doc_type]）

用法：
    python eval_retrieval.py                     # 评估 test_set.jsonl，top_k=5
    python eval_retrieval.py --top-k 3,5,10      # 同时看多个 k
    python eval_retrieval.py --query "道路养护行为的判定标准是什么"  # 单条快速查看
    python eval_retrieval.py --dump              # 只打印每个问题的检索结果，不算指标
    python eval_retrieval.py --limit 10 --save results.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.dify_retriever import DifyRetriever  # noqa: E402


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """NDCG@k（二值相关性）。"""
    k = min(k, len(relevance))
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(sum(relevance), k)))
    return dcg / ideal if ideal > 0 else 0.0


def eval_item(retriever: DifyRetriever, item: dict, k_values: list[int]):
    """对单条测试问题做检索并计算各 k 的指标。

    注意：按文档名（source）去重后计算——同一文档命中的多个片段只算一次，
    避免同文档多片段把 Recall/Precision 撑高。
    """
    docs = retriever.retrieve(
        item["question"],
        doc_type=item.get("doc_type", ""),
    )[: max(k_values)]

    # 按文档名去重，保留第一次出现（检索结果已按分数排序）
    seen = set()
    distinct = []
    for doc in docs:
        name = doc.metadata.get("source", "")
        if name in seen:
            continue
        seen.add(name)
        distinct.append(doc)

    expected = set(item.get("expected_docs", []))
    match_any = bool(item.get("match_any", False))

    names = [d.metadata.get("source", "") for d in distinct]
    relevance = [1 if name in expected else 0 for name in names]

    results = {}
    for k in k_values:
        rel_k = relevance[:k]
        hits = sum(rel_k)
        hit = hits > 0

        if match_any:
            recall = 1.0 if hit else 0.0
        else:
            total = len(expected)
            recall = hits / total if total else 0.0

        # precision 按实际返回的去重文档数计算（可能少于 k）
        precision = hits / len(rel_k) if rel_k else 0.0

        mrr = 0.0
        for rank, rel in enumerate(rel_k, start=1):
            if rel:
                mrr = 1.0 / rank
                break

        results[k] = {
            "hit": hit,
            "hit_rate": 1.0 if hit else 0.0,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "ndcg": ndcg_at_k(relevance, k),
        }

    return names, results


def summarize(records: list[dict], k_values: list[int], title: str) -> None:
    if not records:
        return
    print(f"\n===== {title}（{len(records)} 条）=====")
    print(f"{'k':>4} {'HitRate@k':>10} {'Recall@k':>10} {'Precision@k':>12} {'MRR':>8} {'NDCG@k':>8}")
    for k in k_values:
        avg = lambda key: sum(r["results"][k][key] for r in records) / len(records)
        print(
            f"{k:>4} {avg('hit_rate'):>10.3f} {avg('recall'):>10.3f} "
            f"{avg('precision'):>12.3f} {avg('mrr'):>8.3f} {avg('ndcg'):>8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索准确率评估（不调用大模型）")
    parser.add_argument("--test-set", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_set.jsonl"))
    parser.add_argument("--top-k", default="5", help="逗号分隔的 k 值，如 3,5,10")
    parser.add_argument("--query", default="", help="只评估/查看这一条问题")
    parser.add_argument("--dump", action="store_true", help="只打印检索结果，不算指标")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--save", default="", help="把完整结果保存到 json 文件")
    args = parser.parse_args()

    k_values = [int(x) for x in args.top_k.split(",") if x.strip()]

    retriever = DifyRetriever()
    if not retriever.available:
        print("Dify 未配置或配置不完整：请检查 config/dify.yml 与 .env 的 DIFY_API_KEY")
        sys.exit(1)

    # 单条查询模式
    if args.query:
        docs = retriever.retrieve(args.query)
        print(f"问题：{args.query}\n")
        for i, doc in enumerate(docs, 1):
            score = doc.metadata.get("score")
            print(f"{i}. [{doc.metadata.get('source', '')}] score={score}")
            print("   ", doc.page_content[:100].replace("\n", " "))
        return

    # 测试集模式
    with open(args.test_set, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        items = items[: args.limit]

    if args.dump:
        for item in items:
            docs = retriever.retrieve(item["question"], doc_type=item.get("doc_type", ""))[: max(k_values)]
            print(f"\n[{item.get('id', '')}] ({item.get('doc_type', '')}) {item['question']}")
            print("  期望:", item.get("expected_docs"))
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. [{doc.metadata.get('source', '')}] {doc.metadata.get('score')}")
        return

    records = []
    for item in items:
        names, results = eval_item(retriever, item, k_values)
        records.append({
            "id": item.get("id", ""),
            "doc_type": item.get("doc_type", ""),
            "question": item["question"],
            "expected_docs": item.get("expected_docs", []),
            "match_any": item.get("match_any", False),
            "retrieved_names": names,
            "results": results,
        })

    summarize(records, k_values, "总体")

    for doc_type in ["standard", "case", ""]:
        group = [r for r in records if r["doc_type"] == doc_type]
        label = "doc_type=standard" if doc_type == "standard" else "doc_type=case" if doc_type == "case" else "doc_type=全部"
        if group:
            summarize(group, k_values, label)

    # 失败样例（默认 k 为第一个 k）
    k0 = k_values[0]
    failures = [r for r in records if not r["results"][k0]["hit"]]
    if failures:
        print(f"\n===== 失败样例（Hit@{k0} 未命中，共 {len(failures)} 条）=====")
        for r in failures:
            print(f"\n[{r['id']}] ({r['doc_type']}) {r['question']}")
            print("  期望文档:", r["expected_docs"])
            print("  实际检索:", r["retrieved_names"][:k0])

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已保存到 {args.save}")


if __name__ == "__main__":
    main()