"""评测集校验：在跑评测**之前**，把写错的数据拦下来。

为什么需要它：`gold_sections` 写错一个字母，那一题就永远召回失败，
而且**不报错** —— 你会以为是检索的问题，去调 BM25、调 rerank，白忙一周。
见 plan/09「数据治理：三件小事」。

规则：
  1. `id` 唯一
  2. 必填字段齐全：id / question / gold_pmc_ids / gold_sections / has_answer
  3. `has_answer=true` → `gold_pmc_ids` 和 `gold_sections` 都必须非空
  4. `has_answer=false` → 两者都必须为空（负例不该有 gold）
  5. `question` 不能是空白串
"""
from __future__ import annotations

REQUIRED_FIELDS = ("id", "question", "gold_pmc_ids", "gold_sections", "has_answer")


def validate_dataset(items: list[dict]) -> list[str]:
  """返回错误信息列表；空列表表示通过。

  · 缺字段 → 报错，错误信息里要有 id（连 id 都没有就用下标）
  · id 重复 → 报错，指出重复的 id
  · has_answer=True 但 gold 为空 → 报错
  · has_answer=False 但 gold 非空 → 报错
  · question 是空白串或不是字符串 → 报错
  · 全部合法 → 返回 []

  用 `seen = set()` 记已出现的 id；错误信息用 f-string 拼清楚。
"""
  errors = []
  seen = set()

  for i, item in enumerate(items):
        # 提前获取 id，用于后续所有报错信息的定位；如果连 id 都没有，就用下标
        item_id = item.get("id", f"[下标{i}]")

        # 1. 必填字段齐全性检查
        missing_fields = [field for field in REQUIRED_FIELDS if field not in item]
        if missing_fields:
            errors.append(f"第 {i} 条数据：'id': {item_id} 缺少必填字段 {', '.join(missing_fields)}")
            # 如果连 has_answer 或 question 都没有，后续的逻辑检查就没有意义了，直接看下一条
            if "has_answer" not in item or "question" not in item:
                continue

        # 2. id 唯一性检查
        if "id" in item:
            if item["id"] in seen:
                errors.append(f"第 {i} 条数据 id 重复：{item['id']}")
            else:
                seen.add(item["id"])
        else:
            errors.append(f"第 {i} 条数据没有 id 字段")

        # 3. has_answer 与 gold 的一致性检查 (使用 .get 防止 KeyError 崩溃)
        has_answer = item["has_answer"]
        gold_pmc = item.get("gold_pmc_ids")
        gold_sec = item.get("gold_sections")
        
        # 统一转成布尔值判断是否非空
        is_pmc_empty = not gold_pmc
        is_sec_empty = not gold_sec

        if has_answer is True:
            if is_pmc_empty or is_sec_empty:
                errors.append(f"第 {i} 条数据：'id': {item_id} has_answer=True 但 gold_pmc_ids 或 gold_sections 为空")
        elif has_answer is False:
            if not is_pmc_empty or not is_sec_empty:
                errors.append(f"第 {i} 条数据：'id': {item_id} has_answer=False 但 gold_pmc_ids 或 gold_sections 非空")
        else:
            errors.append(f"第 {i} 条数据：'id': {item_id} has_answer 必须是布尔值")

        # 4. question 合法性检查 (使用 isinstance 防止非字符串调用 strip 崩溃)
        question = item["question"]
        if not isinstance(question, str):
            errors.append(f"第 {i} 条数据：'id': {item_id} question 不是字符串")
        elif not question.strip():
            errors.append(f"第 {i} 条数据：'id': {item_id} question 是空白串")
  return errors