"""Reproducible P2 evaluation: multilingual vs. English-only CPU embeddings.

Compares `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` against
`BAAI/bge-small-en-v1.5` over the active vectorized companies stored in Postgres.

Metrics reported (all observed, no fabricated benchmarks):
- Embedding throughput for all active descriptions on local CPU.
- Full-catalog Same-Industry Precision@5 over 22 deterministic queries
  (1 English + 1 Malay per industry across all 11 industries).
- Cross-lingual Same-Industry Precision@5 (EN query -> MS candidates only;
  MS query -> EN candidates only) to isolate EN <-> MS alignment.
- Qualitative Top-5 samples for selected EN and MS queries.
"""

import re
import time
from dataclasses import dataclass

import numpy as np
import psycopg
from fastembed import TextEmbedding

from app.config import Settings

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5",
]

# Deterministic 22-query panel: 1 EN + 1 MS per industry (SYN-001550 for F&B EN).
QUERY_IDS = [
    ("Agriculture", "en", "SYN-000010"),
    ("Agriculture", "ms", "SYN-000019"),
    ("Construction", "en", "SYN-000043"),
    ("Construction", "ms", "SYN-000085"),
    ("Education", "en", "SYN-000032"),
    ("Education", "ms", "SYN-000024"),
    ("Food & Beverage", "en", "SYN-001550"),
    ("Food & Beverage", "ms", "SYN-000037"),
    ("Healthcare", "en", "SYN-000017"),
    ("Healthcare", "ms", "SYN-000016"),
    ("Hospitality", "en", "SYN-000025"),
    ("Hospitality", "ms", "SYN-000013"),
    ("Logistics", "en", "SYN-000004"),
    ("Logistics", "ms", "SYN-000121"),
    ("Manufacturing", "en", "SYN-000008"),
    ("Manufacturing", "ms", "SYN-000012"),
    ("Professional Services", "en", "SYN-000006"),
    ("Professional Services", "ms", "SYN-000011"),
    ("Retail", "en", "SYN-000001"),
    ("Retail", "ms", "SYN-000015"),
    ("Technology", "en", "SYN-000061"),
    ("Technology", "ms", "SYN-000007"),
]

MALAY_RE = re.compile(
    r"\b(dan|di|ke|untuk|serta|yang|seluruh|kerja-kerja|perkhidmatan|perniagaan|"
    r"menjual|membekal|pembekalan|pembinaan|pengangkutan|penghantaran|bungkusan|"
    r"makanan|minuman|kilang|pertanian|syarikat|kedai|restoran|butik|penjualan|"
    r"pembuatan|pemborong|peruncitan|runcit|kejuruteraan|awam|saliran|ladang|"
    r"ternakan|penternakan|katering|pengurusan|hartanah|tadika|taska|pusat|tuisyen|"
    r"pembangunan|perisian|perakaunan|pengilangan|pengeluaran|perunding|klinik|"
    r"penjagaan|agensi|farmasi|pengusaha|mengusahakan|sokongan|rangkaian|perubatan|"
    r"barangan|sejuk|beku|tepi|pantai|kelapa|sawit|getah|eksport|pelajar|sekolah)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ActiveCompany:
    registration_number: str
    company_name: str | None
    industry: str | None
    nature_of_business: str
    lang: str


def load_active_companies(database_url: str) -> list[ActiveCompany]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT registration_number, company_name, industry, nature_of_business
                FROM companies
                WHERE status = 'active' AND nature_of_business IS NOT NULL
                ORDER BY registration_number
                """
            )
            rows = cur.fetchall()
    companies: list[ActiveCompany] = []
    for reg, name, industry, desc in rows:
        lang = "ms" if MALAY_RE.search(desc) else "en"
        companies.append(
            ActiveCompany(
                registration_number=reg,
                company_name=name,
                industry=industry,
                nature_of_business=desc,
                lang=lang,
            )
        )
    return companies


def embed_all(model_name: str, texts: list[str]) -> tuple[np.ndarray, float]:
    model = TextEmbedding(model_name=model_name)
    t0 = time.perf_counter()
    vecs = np.asarray(list(model.embed(texts)), dtype=np.float32)
    elapsed = time.perf_counter() - t0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-12)
    return vecs, elapsed


def evaluate_model(model_name: str, companies: list[ActiveCompany], top_k: int = 5) -> None:
    index_by_id = {c.registration_number: idx for idx, c in enumerate(companies)}
    texts = [" ".join(c.nature_of_business.split()) for c in companies]
    vecs, elapsed = embed_all(model_name, texts)

    full_en_hits: list[float] = []
    full_ms_hits: list[float] = []
    xling_en_to_ms_hits: list[float] = []
    xling_ms_to_en_hits: list[float] = []
    failures: list[str] = []

    for expected_ind, lang, qid in QUERY_IDS:
        q_idx = index_by_id[qid]
        q_company = companies[q_idx]
        scores = vecs @ vecs[q_idx]
        scores[q_idx] = -2.0  # exclude self

        # 1. Full catalog top-k (excluding candidates with NULL industry from precision denom?
        #    No: count exact match `c.industry == q.industry` strictly across all top-k slots.)
        top_full = np.argsort(-scores)[:top_k]
        full_p5 = sum(1 for i in top_full if companies[i].industry == expected_ind) / top_k
        if lang == "en":
            full_en_hits.append(full_p5)
        else:
            full_ms_hits.append(full_p5)

        # 2. Cross-lingual top-k (opposite language candidates only)
        opp_mask = np.array([c.lang != lang for c in companies], dtype=bool)
        opp_scores = np.where(opp_mask, scores, -2.0)
        top_xling = np.argsort(-opp_scores)[:top_k]
        xling_p5 = sum(1 for i in top_xling if companies[i].industry == expected_ind) / top_k
        if lang == "en":
            xling_en_to_ms_hits.append(xling_p5)
        else:
            xling_ms_to_en_hits.append(xling_p5)

        if xling_p5 < 0.8:
            sample_top = companies[top_xling[0]]
            failures.append(
                f"  [{lang}->{sample_top.lang}] {qid} ({expected_ind}: "
                f"{q_company.nature_of_business[:45]!r}) -> top1={sample_top.registration_number} "
                f"({sample_top.industry}: {sample_top.nature_of_business[:45]!r}, "
                f"sim={opp_scores[top_xling[0]]:.3f}, xling P@5={xling_p5:.2f})"
            )

    print(f"\n=== {model_name} ===")
    print(f"Catalog size: {len(companies)} active descriptions")
    print(f"CPU embed time: {elapsed:.2f}s ({len(companies) / elapsed:.1f} docs/sec)")
    print(
        f"Full-catalog Same-Industry P@5: "
        f"overall={np.mean(full_en_hits + full_ms_hits):.3f} | "
        f"EN={np.mean(full_en_hits):.3f} | MS={np.mean(full_ms_hits):.3f}"
    )
    print(
        f"Cross-lingual Same-Industry P@5: "
        f"overall={np.mean(xling_en_to_ms_hits + xling_ms_to_en_hits):.3f} | "
        f"EN->MS={np.mean(xling_en_to_ms_hits):.3f} | "
        f"MS->EN={np.mean(xling_ms_to_en_hits):.3f}"
    )

    # Print qualitative cross-lingual top-3 for SYN-001550 (EN) and SYN-000019 (MS)
    for sample_id in ("SYN-001550", "SYN-000019"):
        q_idx = index_by_id[sample_id]
        q_c = companies[q_idx]
        scores = vecs @ vecs[q_idx]
        scores[q_idx] = -2.0
        opp_mask = np.array([c.lang != q_c.lang for c in companies], dtype=bool)
        opp_scores = np.where(opp_mask, scores, -2.0)
        top3 = np.argsort(-opp_scores)[:3]
        print(
            f"  Cross-lingual top-3 for {sample_id} ({q_c.lang}, {q_c.industry}: "
            f"{q_c.nature_of_business!r}):"
        )
        for rank, i in enumerate(top3, 1):
            c = companies[i]
            print(
                f"    {rank}. {c.registration_number} [{c.industry}] "
                f"sim={opp_scores[i]:.4f} — {c.nature_of_business}"
            )

    print(f"Cross-lingual low-precision queries (P@5 < 0.80): {len(failures)} / {len(QUERY_IDS)}")
    for line in failures[:6]:
        print(line)


def main() -> None:
    settings = Settings()
    companies = load_active_companies(settings.effective_database_url)
    en_count = sum(1 for c in companies if c.lang == "en")
    ms_count = sum(1 for c in companies if c.lang == "ms")
    print(f"Loaded {len(companies)} active companies (EN={en_count}, MS={ms_count})")
    for model_name in MODELS:
        evaluate_model(model_name, companies)


if __name__ == "__main__":
    main()
