import os, json, time, re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "deepseek-reasoner"
client = OpenAI(api_key="sk-6cc269bac20345578bf1ed8811acad8e", base_url="https://api.deepseek.com")

def call(system, user, temperature=0.2, max_tokens=None):
    msgs=[]
    if system: msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":user})
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 2: raise
            time.sleep(1.5*(attempt+1))


def extract_json_block(text):
    m = re.search(r"```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if m: return m.group(1)
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return m.group(1) if m else None


def safe_load_json(text):
    block = extract_json_block(text) or text
    block = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u4e00-\u9fff]", "", block)
    return json.loads(block)

MANAGER_SYS = """
You are a rigorous planner. Output MUST be STRICT JSON (UTF-8, no comments, no code fences, no trailing commas).
Do not include explanations. If you are unsure, return a minimal valid JSON.
"""


def manager_generate_roles():
    usr = """
Your task: Define a set of specialized analyst roles to extract investable, generalizable features from a company's earnings conference call (a transcript of management Q&A and prepared remarks, possibly with acoustic cues).

Constraints:
- Roles must be domain-specific and non-overlapping.
- Roles must be applicable across sectors and companies.
- Roles must focus on aspects causally or statistically relevant to future stock performance (e.g., growth, margins, guidance, risks, competitive dynamics, capital allocation, demand, supply chain, product roadmap, regulation, management credibility).
- Avoid company names, tickers, or dataset-specific details.

Return a JSON array ONLY, each item:
{
  "role": "Descriptive analyst title (e.g., 'Profitability Analyst')",
  "focus_area": "Specific analytical focus within earnings calls (e.g., 'gross margin drivers and forward margin guidance consistency')"
}

Produce more than 5 roles.
"""
    out = call(MANAGER_SYS, usr, temperature=0.2)
    return safe_load_json(out)  # -> list[dict]


ANALYST_SYS = """
You output STRICT JSON in the schema:
{"features": ["Sentence 1", "Sentence 2", ...]}

Each feature must be:
- A generalized, reusable statement in the form "X is ...".
- Applicable for earnings call content (prepared remarks + Q&A).
- Free of company names, numbers, or dataset-specific entities.
- Specific enough to be decision-useful (no vague platitudes).
- One short sentence (8-20 words if English).
- Non-overlapping; maximize diversity across subtopics.
- Aligned with the provided role's focus.

No explanations. No code fences.
"""

FILTER_SYS = """
You output STRICT JSON: {"features": ["...", ...]}
Exactly 20 items.

Filtering rules:
- Keep only role-relevant items.
- Remove duplicates and near-duplicates (semantic equivalence).
- Each sentence must be "X is ..." and stand alone.
- No company names, numbers, or dataset-specific entities.
- Prefer forward-looking, causal/driver-oriented, durable statements.
- Ensure topical diversity within the role (cover different subthemes).

No explanations. No code fences.
"""


def analyst_extract(role, focus, temperature=0.2):
    usr = f"""
Role: {role}
Focus: {focus}

Context assumptions:
- Source is an earnings conference call transcript. Topics include revenue growth, margins, guidance, demand/supply, pricing, capex/opex, working capital, regulation, competitive dynamics, product/roadmap, customer cohorts, geographies, FX, macro headwinds/tailwinds, execution risks, management credibility.
- The output must generalize across companies and time. Avoid any reference to specific names, brands, SKUs, tickers, or numeric values.

Task:
- Extract >= 30 features in the exact form "X is ...".
- Focus on forward-looking and durable insights (drivers, directionality, sustainability, risk asymmetry).
- Prefer actionable granularity (e.g., "Gross margin is expanding due to mix," not "Margins are changing").

Return JSON only.
"""
    out = call(ANALYST_SYS, usr, temperature=temperature)
    data = safe_load_json(out)
    feats = [s.strip() for s in data.get("features", []) if isinstance(s, str) and s.strip()]

    # 去重
    seen, clean = set(), []
    for f in feats:
        if f.lower() not in seen:
            seen.add(f.lower()); clean.append(f)
    return clean


def role_filter(role, feats):
    raw_json = json.dumps({"features": feats}, ensure_ascii=False)
    usr = f"Role: {role}\nHere is the raw list:\n{raw_json}"
    out = call(FILTER_SYS, usr, temperature=0.0)  # 过滤建议设为低温
    data = safe_load_json(out)
    feats = [s.strip() for s in data.get("features", []) if isinstance(s, str) and s.strip()]
    return feats[:20]


def multi_agent_pipeline(temperature=0.2, k_per_role=20, parallel=6, outfile="final_pool.json"):
    roles = manager_generate_roles()  # 或者直接用你固定的角色清单
    results = {}

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(analyst_extract, r["role"], r["focus_area"], temperature): r for r in roles}
        for fut in as_completed(futs):
            r = futs[fut]
            feats = fut.result()
            # 二次过滤
            filt = role_filter(r["role"], feats)[:k_per_role]
            results[r["role"]] = filt

    # TODO: 这里可以做一次跨角色的 embedding 去重（语义相似度 > 0.85 删重）
    # final = dedup_across_roles(results)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    multi_agent_pipeline(temperature=0.3, k_per_role=20, parallel=6, outfile="multi_factor_pool_0_2_v1.json")