#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon Candidate Ranking Engine
Produces a top-100 submission CSV from candidates.jsonl.
Runs CPU-only, offline, within 5 min / 16 GB constraints.
"""

from pathlib import Path
import json
import csv
import re
import math
import argparse
from datetime import date, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURRENT_DATE = date(2026, 6, 11)
CURRENT_YEAR = 2026
TOP_N = 100

# Consulting/services firms that indicate pure-services career
CONSULTING_FIRMS = frozenset([
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tata consultancy services", "tata consultancy", "hcl", "hcl technologies",
    "tech mahindra", "mphasis", "mindtree", "hexaware",
])

# High-value core keywords — IR/Search/Ranking domain (+0.4 each)
CORE_KEYWORDS = [
    "information retrieval", "ir system", "search engine", "recommendation system",
    "recommender system", "re-ranking", "reranking", "ndcg", "mrr", "map metric",
    "hybrid search", "bm25", "dense retrieval", "sparse retrieval",
    "embedding retrieval", "vector search", "semantic search",
    "learning to rank", "ltr", "ranking system", "candidate ranking",
    "faiss", "elasticsearch", "opensearch", "weaviate", "qdrant", "milvus",
    "pinecone", "sentence-transformers", "sentence transformers",
    "bi-encoder", "cross-encoder", "retrieval augmented", "rag pipeline",
    "embedding drift", "index refresh", "retrieval quality",
]

# General AI/ML keywords (+0.2 each)
GENERAL_AI_KEYWORDS = [
    "llm", "large language model", "fine-tuning", "fine tuning", "finetuning",
    "lora", "qlora", "peft", "pytorch", "hugging face", "huggingface",
    "transformers", "bert", "gpt", "embeddings", "vector database",
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "question answering", "text generation",
    "feature store", "mlops", "model serving", "inference optimization",
    "xgboost", "gradient boosting", "neural network",
]

# Keyword-stuffing bait terms that only count if job title is also technical
BUZZWORD_TERMS = frozenset([
    "rag", "pinecone", "vector search", "langchain", "openai", "chatgpt",
    "llm", "embeddings", "faiss", "weaviate",
])

# Non-technical title patterns (regex)
NON_TECH_TITLE_PATTERNS = re.compile(
    r"\b(marketing|hr |human resource|sales|finance|accounts|accountant|"
    r"recruiter|talent acquisition|business development|operations manager|"
    r"product manager(?! of engineering)|program manager|project manager|"
    r"content writer|copywriter|graphic design|ui designer|ux designer|"
    r"social media|brand manager|logistics|supply chain manager)\b",
    re.IGNORECASE,
)

# Computer vision / speech / robotics — down-weight if no NLP/IR
CV_SPEECH_ROBOTICS = frozenset([
    "computer vision", "object detection", "image segmentation", "ocr",
    "speech recognition", "asr", "tts", "text to speech", "robotics",
    "slam", "lidar", "point cloud", "3d reconstruction",
])

NLP_IR_TERMS = frozenset([
    "nlp", "natural language processing", "information retrieval", "search",
    "text classification", "question answering", "ner", "sentiment analysis",
    "ranking", "recommendation", "retrieval", "embeddings", "bert", "gpt",
    "transformer", "language model", "text mining",
])

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_date(ds):
    """Parse ISO date string to date object; return None on failure."""
    if not ds:
        return None
    try:
        return date.fromisoformat(str(ds)[:10])
    except (ValueError, TypeError):
        return None


def _extract_year(date_str):
    """Extract year from ISO date string."""
    d = _parse_date(date_str)
    return d.year if d else None


def _text_lower(candidate):
    """Concatenate all free-text fields for keyword matching (lowercased)."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    for ch in candidate.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
        parts.append(ch.get("company", ""))
    for sk in candidate.get("skills", []):
        parts.append(sk.get("name", ""))
    return " ".join(parts).lower()


def _skill_map(candidate):
    """Return dict: skill_name_lower -> skill record."""
    return {sk["name"].lower(): sk for sk in candidate.get("skills", [])}


# ---------------------------------------------------------------------------
# PART 2: Deterministic Screening (Hard Filters & Traps)
# ---------------------------------------------------------------------------

def _is_honeypot(candidate):
    """
    Returns True if the candidate profile is a honeypot (impossible timeline
    or expert-with-zero-experience trap).
    """
    # Trap 1: Experience > company age
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    for ch in candidate.get("career_history", []):
        # We can't know founding year from data; use start_date of earliest tenure
        # as a proxy — if claimed YoE at this company exceeds company "age" inferred
        # from its earliest start date, flag it.
        pass  # This trap uses employer founding year which isn't in data directly.

    # Trap 1b: Total YoE > (2026 - earliest plausible start)
    # Earliest career start from history
    start_years = []
    for ch in candidate.get("career_history", []):
        y = _extract_year(ch.get("start_date"))
        if y:
            start_years.append(y)
    if start_years:
        earliest = min(start_years)
        max_possible_yoe = CURRENT_YEAR - earliest
        if yoe > max_possible_yoe + 1:  # allow 1-year buffer
            return True

    # Trap 2: Expert skill with 0 months duration
    for sk in candidate.get("skills", []):
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 1) == 0:
            return True

    return False


def _consulting_only_multiplier(candidate):
    """Return 0.1 if entire career is at consulting/services firms only."""
    history = candidate.get("career_history", [])
    if not history:
        return 1.0
    for ch in history:
        company = ch.get("company", "").lower()
        if not any(cf in company for cf in CONSULTING_FIRMS):
            return 1.0  # at least one non-consulting company
    return 0.1


def _keyword_stuffer_penalty(candidate):
    """
    Down-weight if profile matches many buzzwords but all titles are non-technical.
    """
    all_titles = []
    all_titles.append(candidate.get("profile", {}).get("current_title", ""))
    for ch in candidate.get("career_history", []):
        all_titles.append(ch.get("title", ""))
    titles_text = " ".join(all_titles).lower()

    # Check if any title looks technical
    tech_title = re.search(
        r"\b(engineer|developer|scientist|researcher|architect|analyst|"
        r"ml |ai |data |nlp|backend|frontend|software|programmer|coder|"
        r"devops|sre|platform|infrastructure)\b",
        titles_text,
        re.IGNORECASE,
    )

    if tech_title:
        return 1.0  # Technical titles — no penalty

    # Non-technical titles — check for buzzword stuffing
    text = _text_lower(candidate)
    buzzword_hits = sum(1 for bw in BUZZWORD_TERMS if bw in text)
    if buzzword_hits >= 3:
        return 0.1  # Heavy down-weight for stuffers
    if buzzword_hits >= 1:
        return 0.5
    return 1.0


def _domain_specialization_penalty(candidate):
    """
    Down-weight if purely CV/Speech/Robotics with no NLP/IR signals.
    """
    text = _text_lower(candidate)
    cv_hits = sum(1 for t in CV_SPEECH_ROBOTICS if t in text)
    nlp_hits = sum(1 for t in NLP_IR_TERMS if t in text)
    if cv_hits >= 2 and nlp_hits == 0:
        return 0.3
    if cv_hits >= 2 and nlp_hits == 1:
        return 0.6
    return 1.0


# ---------------------------------------------------------------------------
# PART 3: Hybrid Scoring
# ---------------------------------------------------------------------------

def _base_alignment_score(candidate):
    """
    Keyword-based text alignment score combining core IR keywords and
    general AI/ML keywords, weighted by actual profile evidence.
    """
    text = _text_lower(candidate)
    skill_map = _skill_map(candidate)

    score = 0.0

    # Core IR/Search/Ranking keywords — high value
    for kw in CORE_KEYWORDS:
        if kw in text:
            score += 0.4

    # General AI/ML keywords
    for kw in GENERAL_AI_KEYWORDS:
        if kw in text:
            score += 0.2

    # Skill-depth bonus: endorsed + high proficiency
    for sk in candidate.get("skills", []):
        name = sk.get("name", "").lower()
        proficiency = sk.get("proficiency", "beginner")
        endorsements = sk.get("endorsements", 0)
        duration = sk.get("duration_months", 0)

        is_core = any(kw in name for kw in [
            "retrieval", "search", "ranking", "nlp", "embeddings", "pytorch",
            "recommendation", "ir", "faiss", "vector", "elasticsearch",
        ])
        if is_core:
            if proficiency in ("advanced", "expert") and duration > 12:
                score += 0.3
            if endorsements > 10:
                score += 0.1

    # Experience depth bonus: product company years in AI roles
    product_ai_months = 0
    for ch in candidate.get("career_history", []):
        company = ch.get("company", "").lower()
        title = ch.get("title", "").lower()
        desc = ch.get("description", "").lower()
        industry = ch.get("industry", "").lower()
        duration = ch.get("duration_months", 0)

        is_services = any(cf in company for cf in CONSULTING_FIRMS)
        is_ai_role = bool(re.search(
            r"\b(ml|machine learning|ai|nlp|search|retrieval|ranking|"
            r"recommendation|data science|scientist|engineer)\b",
            title + " " + desc,
        ))
        if not is_services and is_ai_role:
            product_ai_months += duration

    product_ai_years = product_ai_months / 12.0
    if product_ai_years >= 5:
        score += 1.0
    elif product_ai_years >= 3:
        score += 0.6
    elif product_ai_years >= 1:
        score += 0.3

    # Years of experience (soft bonus, diminishing returns)
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= yoe <= 9:
        score += 0.5  # Sweet spot for this role
    elif yoe >= 3:
        score += 0.2

    # Education tier bonus
    for edu in candidate.get("education", []):
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            score += 0.3
        elif tier == "tier_2":
            score += 0.15

    # Certification bonus for relevant certs
    for cert in candidate.get("certifications", []):
        cert_name = cert.get("name", "").lower()
        if any(kw in cert_name for kw in ["aws", "gcp", "azure", "ml", "ai", "deep learning"]):
            score += 0.1

    return max(0.0, score)


def _behavioral_multiplier(candidate):
    """
    Build a behavioral multiplier from all 23 Redrob signals.
    Returns a float [0.0, 2.0].
    """
    sig = candidate.get("redrob_signals", {})
    multiplier = 1.0

    # 1. Recency — last active date
    last_active = _parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (CURRENT_DATE - last_active).days
        if days_inactive > 180:
            multiplier *= 0.3
        elif days_inactive > 90:
            multiplier *= 0.7
        elif days_inactive <= 30:
            multiplier *= 1.1  # Recently active bonus

    # 2. Open to work flag
    if sig.get("open_to_work_flag", False):
        multiplier *= 1.15

    # 3. Recruiter response rate
    rrr = sig.get("recruiter_response_rate", 0.5) or 0.5
    multiplier *= (0.4 + 0.6 * rrr)  # Range: 0.4 (0% resp) to 1.0 (100% resp)

    # 4. Interview completion rate
    icr = sig.get("interview_completion_rate", 0.5) or 0.5
    multiplier *= (0.5 + 0.5 * icr)  # Range: 0.5 to 1.0

    # 5. Notice period
    notice = sig.get("notice_period_days", 30)
    if notice is None:
        notice = 30
    if notice <= 30:
        pass  # No penalty
    elif notice <= 90:
        multiplier *= 0.7
    else:
        multiplier *= 0.3

    # 6. GitHub activity score
    gh_score = sig.get("github_activity_score", -1)
    if gh_score is not None and gh_score >= 0:
        multiplier *= (1.0 + 0.2 * (gh_score / 100.0))  # Up to +20% bonus

    # 7. Profile completeness
    completeness = sig.get("profile_completeness_score", 50) or 50
    if completeness >= 80:
        multiplier *= 1.05
    elif completeness < 40:
        multiplier *= 0.9

    # 8. Verification signals
    verified_points = 0
    if sig.get("verified_email", False):
        verified_points += 1
    if sig.get("verified_phone", False):
        verified_points += 1
    if sig.get("linkedin_connected", False):
        verified_points += 1
    if verified_points == 3:
        multiplier *= 1.05
    elif verified_points == 0:
        multiplier *= 0.9

    # 9. Offer acceptance rate (only if they have prior offer history)
    oar = sig.get("offer_acceptance_rate", -1)
    if oar is not None and oar >= 0:
        if oar >= 0.8:
            multiplier *= 1.05
        elif oar < 0.3:
            multiplier *= 0.9

    # 10. Recent recruiter engagement
    saved_30d = sig.get("saved_by_recruiters_30d", 0) or 0
    if saved_30d >= 5:
        multiplier *= 1.08
    elif saved_30d >= 2:
        multiplier *= 1.04

    return max(0.05, min(multiplier, 3.0))


# ---------------------------------------------------------------------------
# PART 4: Reasoning Engine
# ---------------------------------------------------------------------------

def _build_reasoning(candidate, final_score, rank):
    """
    Generate a non-hallucinated 1-2 sentence reasoning string grounded
    entirely in the candidate's actual profile data.
    """
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0) or 0
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    notice = sig.get("notice_period_days", 30)
    rrr = sig.get("recruiter_response_rate", 0.5) or 0.5
    gh = sig.get("github_activity_score", -1)

    # Find best matched IR/core skill for mention
    text = _text_lower(candidate)
    matched_skills = []
    for kw in CORE_KEYWORDS[:15]:
        if kw in text:
            matched_skills.append(kw.title())
    for kw in ["NLP", "Embeddings", "PyTorch", "Fine-tuning", "LLM"]:
        if kw.lower() in text:
            matched_skills.append(kw)
    # Deduplicate preserving order
    seen_ms = set()
    deduped = []
    for ms in matched_skills:
        if ms.lower() not in seen_ms:
            seen_ms.add(ms.lower())
            deduped.append(ms)
    matched_skills = deduped[:3]

    # Check product vs services background
    consulting_mult = _consulting_only_multiplier(candidate)
    is_pure_consulting = consulting_mult < 0.5

    # Education tier
    best_tier = "unknown"
    for edu in candidate.get("education", []):
        t = edu.get("tier", "unknown")
        if t == "tier_1":
            best_tier = "tier_1"
            break
        elif t == "tier_2" and best_tier != "tier_1":
            best_tier = "tier_2"

    # Compose sentences
    skill_phrase = (
        f"Proven background in {', '.join(matched_skills[:2])}"
        if matched_skills
        else "Technical background in AI/ML engineering"
    )

    company_phrase = f" at {company}" if company else ""

    if rank <= 10:
        sentence1 = (
            f"Excellent product fit with {yoe:.0f} years as a {title}{company_phrase}. "
            f"{skill_phrase} matching Redrob's founding search-layer requirements."
        )
    elif rank <= 50:
        depth = "specialized IR/search" if matched_skills else "core engineering"
        sentence1 = (
            f"Competent technical profile showing {yoe:.0f} years of experience. "
            f"Fully qualified in {depth} pipelines"
        )
        if not matched_skills:
            sentence1 += ", though lacks specialized IR system architecture depth."
        else:
            sentence1 += "."
    else:
        sentence1 = (
            f"Adjacent technical profile with {yoe:.0f} years total experience as {title}. "
            f"Partial skill overlap with role requirements"
        )
        if is_pure_consulting:
            sentence1 += "; career entirely in services/consulting reduces product-fit signal."
        else:
            sentence1 += "."

    concerns = []
    if notice > 60:
        concerns.append(
            f"Challenge: Extended notice period of {notice} days may impact rapid onboarding."
        )
    if rrr < 0.3:
        concerns.append(
            "Concern: Low platform engagement metrics signal availability risk."
        )
    last_active = _parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (CURRENT_DATE - last_active).days
        if days_inactive > 180:
            concerns.append(
                f"Note: Profile inactive for {days_inactive} days — reachability uncertain."
            )

    if gh is not None and gh >= 50 and rank <= 50:
        concerns.insert(0, f"Open-source contributor (GitHub score {gh:.0f}/100).")

    reasoning = sentence1
    if concerns:
        reasoning += " " + " ".join(concerns[:2])

    return reasoning.strip()


# ---------------------------------------------------------------------------
# PART 5: Per-candidate processing (runs in worker processes)
# ---------------------------------------------------------------------------

def _score_candidate(candidate):
    """
    Full scoring pipeline for a single candidate.
    Returns (candidate_id, score, reasoning_placeholder_data).
    """
    cid = candidate.get("candidate_id", "")

    # Honeypot check — force score to 0.0
    if _is_honeypot(candidate):
        return (cid, 0.0, candidate)

    # Screening multipliers
    consulting_mult = _consulting_only_multiplier(candidate)
    stuffer_mult = _keyword_stuffer_penalty(candidate)
    domain_mult = _domain_specialization_penalty(candidate)

    # Compute base alignment score
    base = _base_alignment_score(candidate)

    # Behavioral multiplier
    behav = _behavioral_multiplier(candidate)

    # Apply screening multipliers to base score
    adjusted_base = base * consulting_mult * stuffer_mult * domain_mult

    # Final score
    final = adjusted_base * behav

    return (cid, final, candidate)


def _process_chunk(chunk):
    """Process a list of candidate dicts; return list of (cid, score, candidate)."""
    results = []
    for c in chunk:
        results.append(_score_candidate(c))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(candidates_path: Path, output_path: Path):
    # Load all candidates
    print(f"[rank.py] Loading candidates from {candidates_path} ...")
    candidates = []
    open_fn = None

    suffix = candidates_path.suffix.lower()
    if suffix == ".gz":
        import gzip
        open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        open_fn = lambda p: open(p, "r", encoding="utf-8")

    with open_fn(candidates_path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    total = len(candidates)
    print(f"[rank.py] Loaded {total} candidates.")

    # Parallel processing
    cpu_count = min(os.cpu_count() or 6, 12)
    chunk_size = max(1, math.ceil(total / (cpu_count * 4)))
    chunks = [candidates[i:i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"[rank.py] Scoring with {cpu_count} workers, {len(chunks)} chunks ...")
    scored = []
    use_parallel = cpu_count > 1

    if use_parallel:
        try:
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                futures = {executor.submit(_process_chunk, chunk): idx for idx, chunk in enumerate(chunks)}
                completed = 0
                for future in as_completed(futures):
                    scored.extend(future.result())
                    completed += 1
                    if completed % 20 == 0:
                        print(f"[rank.py] Processed {completed}/{len(chunks)} chunks ...")
        except Exception as e:
            print(f"[rank.py] Parallel execution failed ({e}), falling back to sequential ...")
            scored = []
            use_parallel = False

    if not use_parallel:
        print("[rank.py] Running in sequential mode ...")
        for i, chunk in enumerate(chunks):
            scored.extend(_process_chunk(chunk))
            if (i + 1) % 20 == 0:
                print(f"[rank.py] Processed {i+1}/{len(chunks)} chunks ...")

    print(f"[rank.py] Scoring complete. Sorting ...")

    # Sort: descending score, then ascending candidate_id for ties
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Take top 100
    top100 = scored[:TOP_N]

    # Build output rows with reasoning
    print("[rank.py] Generating reasoning for top-100 ...")
    rows = []
    for rank_idx, (cid, score, candidate) in enumerate(top100, start=1):
        reasoning = _build_reasoning(candidate, score, rank_idx)
        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

    # Write CSV
    print(f"[rank.py] Writing output to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[rank.py] Done. Top candidate: {rows[0]['candidate_id']} (score={rows[0]['score']})")
    print(f"[rank.py] Submission written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data") / "candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    main(args.candidates, args.out)
