# RAG 检索准确率评估（只测检索，不调用大模型）

## 说明
- 本评估只调用 Dify 知识库的检索接口（向量检索），**不调用任何大模型、不生成答案**。
- 指标含义（文档级别，按 Dify 返回的 `source` 文档名匹配；同一文档的多个命中片段会**按文档名去重**，只算一次）：
  - `HitRate@k`：前 k 条中是否至少命中一条期望文档
  - `Recall@k`：期望文档被找回的比例（`match_any=true` 时只要求命中任一即可）
  - `Precision@k`：前 k 条中期望文档占比
  - `MRR`：第一条命中出现位置的倒数均值
  - `NDCG@k`：考虑排序位置的加权指标

## 前置条件
- `config/dify.yml` 的 `api_base` / `dataset_id` 已配置
- `.env` 的 `DIFY_API_KEY` 已配置

## 用法（在项目根目录用 agent 环境运行）
```powershell
# 完整评估（默认 top_k=5）
python eval\eval_retrieval.py

# 同时看多个 k
python eval\eval_retrieval.py --top-k 3,5,10

# 只看某个问题的检索结果（调试用）
python eval\eval_retrieval.py --query "道路养护行为的判定标准是什么"

# 打印所有测试问题的检索结果（先看看 Dify 实际返回的文档名是否和测试集一致）
python eval\eval_retrieval.py --dump

# 保存完整结果
python eval\eval_retrieval.py --save results.json
```

## 测试集格式（test_set.jsonl）
每行一个 JSON：
```json
{
  "id": "q01",
  "doc_type": "standard",
  "question": "道路养护行为的判定标准是什么？",
  "expected_docs": ["道路养护判定标准.md"],
  "match_any": false
}
```
- `doc_type`：standard / case / 空（不指定则不过滤，和 Dify 里设置的元数据一致）
- `expected_docs`：期望命中的**文档名**，必须与 Dify 里文档名完全一致（用 `--dump` 核对）
- `match_any`：true 表示命中任一期望文档即算相关（适合案例类问题），false 表示期望文档都要命中才算全召回

## 调参建议（改完 Dify 侧记得重新索引再评估）
- `top_k`：看 Recall@k 的边际收益
- `search_method`：semantic_search / full_text_search / hybrid_search（hybrid 需 Dify 配置 rerank 模型）
- `score_threshold`：只调精度，过高会掉召回
- Dify 分段设置：分段长度 / 重叠 / 分隔符
- 嵌入模型：text-embedding-v4 或其他