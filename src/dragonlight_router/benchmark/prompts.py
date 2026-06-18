"""Eval prompt bank for IBR automated benchmarking.

Contains ~50 standardized evaluation prompts organized by intent dimension
(task_type x domain). Each prompt is realistic and answerable in <500 tokens.

Spec reference: intent-based-router-v0.1.0-spec.md section 3.2, Method 3.
"""
from __future__ import annotations

from dataclasses import dataclass

from dragonlight_router.core.types import IBR_DOMAINS, IBR_TASK_TYPES


@dataclass(frozen=True)
class EvalPrompt:
    """Single evaluation prompt for benchmarking a model's flavor profile.

    id: unique identifier, e.g. "generation-code-001"
    task_type: which IBR task_type this tests
    domain: which IBR domain this tests
    quality_speed: quality context for the judge
    prompt: the actual prompt to send to the model
    judge_criteria: what the judge should evaluate
    """

    id: str
    task_type: str
    domain: str
    quality_speed: str
    prompt: str
    judge_criteria: str


def _validate_prompt(p: EvalPrompt) -> None:
    """Assert that an EvalPrompt references valid IBR dimensions."""
    assert p.task_type in IBR_TASK_TYPES, (
        f"Invalid task_type '{p.task_type}' in prompt {p.id}"
    )
    assert p.domain in IBR_DOMAINS, (
        f"Invalid domain '{p.domain}' in prompt {p.id}"
    )
    assert p.quality_speed in {"quality", "balanced", "speed"}, (
        f"Invalid quality_speed '{p.quality_speed}' in prompt {p.id}"
    )
    assert len(p.prompt) > 0, f"Empty prompt text in {p.id}"
    assert len(p.judge_criteria) > 0, f"Empty judge_criteria in {p.id}"


# ---------------------------------------------------------------------------
# Generation prompts (code, technical, creative_writing) — 8 total
# ---------------------------------------------------------------------------

_GENERATION_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="generation-code-001",
        task_type="generation",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Write a Python function that implements a thread-safe LRU cache "
            "with configurable max size and TTL expiry. Include type hints and "
            "a docstring."
        ),
        judge_criteria=(
            "Correctness of LRU eviction logic, thread safety mechanism, "
            "TTL expiry handling, type hints, docstring quality."
        ),
    ),
    EvalPrompt(
        id="generation-code-002",
        task_type="generation",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Write a TypeScript function that debounces an async callback, "
            "returning the most recent promise result. Handle cancellation of "
            "in-flight calls."
        ),
        judge_criteria=(
            "Async debounce correctness, cancellation handling, TypeScript "
            "typing, edge case coverage."
        ),
    ),
    EvalPrompt(
        id="generation-code-003",
        task_type="generation",
        domain="code",
        quality_speed="speed",
        prompt=(
            "Write a Python dataclass for a paginated API response with "
            "items, total_count, page, and page_size fields. Add a "
            "has_next_page property."
        ),
        judge_criteria=(
            "Correct dataclass definition, appropriate field types, "
            "has_next_page logic correctness."
        ),
    ),
    EvalPrompt(
        id="generation-technical-001",
        task_type="generation",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "Write a detailed Nginx reverse proxy configuration that load "
            "balances across three upstream servers with health checks, "
            "SSL termination, and rate limiting."
        ),
        judge_criteria=(
            "Configuration correctness, health check syntax, SSL directives, "
            "rate limiting setup, production readiness."
        ),
    ),
    EvalPrompt(
        id="generation-technical-002",
        task_type="generation",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "Write a Dockerfile for a Python 3.12 FastAPI application with "
            "multi-stage build, non-root user, and proper layer caching."
        ),
        judge_criteria=(
            "Multi-stage build correctness, security practices (non-root), "
            "layer ordering for cache efficiency."
        ),
    ),
    EvalPrompt(
        id="generation-creative-001",
        task_type="generation",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Write a short scene (150-200 words) where a lighthouse keeper "
            "discovers that the light has been attracting something other "
            "than ships. Use sensory detail and build tension."
        ),
        judge_criteria=(
            "Sensory detail richness, tension building, prose quality, "
            "originality of the revelation, adherence to word count."
        ),
    ),
    EvalPrompt(
        id="generation-creative-002",
        task_type="generation",
        domain="creative_writing",
        quality_speed="balanced",
        prompt=(
            "Write a product description for a fictional smart water bottle "
            "that tracks hydration and syncs with health apps. Make it "
            "compelling but not hyperbolic."
        ),
        judge_criteria=(
            "Persuasiveness, tone balance (compelling without hype), "
            "feature communication clarity, call to action."
        ),
    ),
    EvalPrompt(
        id="generation-creative-003",
        task_type="generation",
        domain="creative_writing",
        quality_speed="speed",
        prompt=(
            "Write three different email subject lines for a SaaS product "
            "launch announcement. Target: technical decision makers."
        ),
        judge_criteria=(
            "Subject line effectiveness, audience targeting, variety "
            "between the three options, conciseness."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Analysis prompts (code, technical, business) — 6 total
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="analysis-code-001",
        task_type="analysis",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Analyze this Python function for bugs and performance issues:\n\n"
            "def find_duplicates(items):\n"
            "    seen = []\n"
            "    dupes = []\n"
            "    for item in items:\n"
            "        if item in seen:\n"
            "            dupes.append(item)\n"
            "        seen.append(item)\n"
            "    return dupes"
        ),
        judge_criteria=(
            "Identification of O(n^2) performance issue with list membership "
            "check, duplicate duplicates bug, recommendation of set usage, "
            "completeness of analysis."
        ),
    ),
    EvalPrompt(
        id="analysis-code-002",
        task_type="analysis",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "What are the potential race conditions in this pattern?\n\n"
            "class Counter:\n"
            "    def __init__(self): self.count = 0\n"
            "    def increment(self): self.count += 1\n"
            "    def get(self): return self.count"
        ),
        judge_criteria=(
            "Identification of non-atomic increment, explanation of "
            "read-modify-write race, suggestion of threading.Lock or "
            "atomic operations."
        ),
    ),
    EvalPrompt(
        id="analysis-technical-001",
        task_type="analysis",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "A PostgreSQL query that was fast (10ms) is now taking 3 seconds "
            "after the table grew from 100K to 10M rows. The query uses "
            "WHERE status = 'active' AND created_at > now() - interval '7 days' "
            "ORDER BY created_at DESC LIMIT 50. What are the likely causes "
            "and how would you diagnose?"
        ),
        judge_criteria=(
            "Index analysis (composite index suggestion), EXPLAIN ANALYZE "
            "recommendation, statistics staleness, partial index suggestion, "
            "systematic diagnostic approach."
        ),
    ),
    EvalPrompt(
        id="analysis-technical-002",
        task_type="analysis",
        domain="technical",
        quality_speed="speed",
        prompt=(
            "A Docker container keeps getting OOMKilled with a 512MB limit. "
            "The app is a Node.js Express server. What are the three most "
            "likely causes?"
        ),
        judge_criteria=(
            "Identification of memory leak, V8 heap defaults exceeding "
            "container limit, and unbounded caching or connection pooling."
        ),
    ),
    EvalPrompt(
        id="analysis-business-001",
        task_type="analysis",
        domain="business",
        quality_speed="quality",
        prompt=(
            "A B2B SaaS company has 85% gross retention but only 95% net "
            "revenue retention. Monthly churn is 2.5%. Analyze what these "
            "metrics suggest about the business and what levers to pull."
        ),
        judge_criteria=(
            "Correct interpretation of retention metrics, identification "
            "of expansion revenue masking churn, actionable recommendations "
            "for reducing churn, metric relationship analysis."
        ),
    ),
    EvalPrompt(
        id="analysis-business-002",
        task_type="analysis",
        domain="business",
        quality_speed="balanced",
        prompt=(
            "Compare the tradeoffs of usage-based pricing vs. seat-based "
            "pricing for a developer tools API product. Consider revenue "
            "predictability, customer acquisition, and expansion."
        ),
        judge_criteria=(
            "Balanced comparison, consideration of both perspectives, "
            "practical tradeoff analysis, relevant examples."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Refactoring prompts (code) — 5 total
# ---------------------------------------------------------------------------

_REFACTORING_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="refactoring-code-001",
        task_type="refactoring",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Refactor this function to reduce complexity and improve "
            "testability:\n\n"
            "def process_order(order):\n"
            "    if order['type'] == 'digital':\n"
            "        if order['paid']:\n"
            "            send_download_link(order['email'], order['product'])\n"
            "            update_inventory(order['product'], -1)\n"
            "            return 'delivered'\n"
            "        else:\n"
            "            return 'pending_payment'\n"
            "    elif order['type'] == 'physical':\n"
            "        if order['paid']:\n"
            "            if check_stock(order['product']):\n"
            "                create_shipment(order)\n"
            "                update_inventory(order['product'], -1)\n"
            "                return 'shipped'\n"
            "            else:\n"
            "                return 'out_of_stock'\n"
            "        else:\n"
            "            return 'pending_payment'\n"
            "    return 'unknown_type'"
        ),
        judge_criteria=(
            "Complexity reduction (fewer nested conditionals), extraction "
            "of strategy pattern or dispatch table, testability improvement, "
            "preservation of original behavior."
        ),
    ),
    EvalPrompt(
        id="refactoring-code-002",
        task_type="refactoring",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Refactor this to use dataclasses instead of raw dicts:\n\n"
            "def create_user(name, email, role='viewer'):\n"
            "    return {'name': name, 'email': email, 'role': role, "
            "'created': datetime.now(), 'active': True}\n\n"
            "def deactivate_user(user):\n"
            "    user['active'] = False\n"
            "    return user"
        ),
        judge_criteria=(
            "Proper dataclass definition, type hints, handling of "
            "mutability (frozen vs mutable), preservation of functionality."
        ),
    ),
    EvalPrompt(
        id="refactoring-code-003",
        task_type="refactoring",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Refactor this SQL query builder to prevent SQL injection:\n\n"
            "def search_users(name=None, email=None, role=None):\n"
            "    query = 'SELECT * FROM users WHERE 1=1'\n"
            "    if name:\n"
            "        query += f\" AND name LIKE '%{name}%'\"\n"
            "    if email:\n"
            "        query += f\" AND email = '{email}'\"\n"
            "    if role:\n"
            "        query += f\" AND role = '{role}'\"\n"
            "    return db.execute(query)"
        ),
        judge_criteria=(
            "Parameterized query usage, SQL injection prevention, "
            "maintainability of the builder pattern, correctness."
        ),
    ),
    EvalPrompt(
        id="refactoring-code-004",
        task_type="refactoring",
        domain="code",
        quality_speed="speed",
        prompt=(
            "Simplify this boolean logic:\n\n"
            "if not (x != True and y != False):\n"
            "    return True\n"
            "else:\n"
            "    return False"
        ),
        judge_criteria=(
            "Correct simplification preserving original semantics, "
            "readability improvement, explanation of the transformation."
        ),
    ),
    EvalPrompt(
        id="refactoring-code-005",
        task_type="refactoring",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Refactor this to use async/await instead of callbacks:\n\n"
            "function fetchUserData(userId, callback) {\n"
            "    getUser(userId, function(err, user) {\n"
            "        if (err) return callback(err);\n"
            "        getOrders(user.id, function(err, orders) {\n"
            "            if (err) return callback(err);\n"
            "            callback(null, { user, orders });\n"
            "        });\n"
            "    });\n"
            "}"
        ),
        judge_criteria=(
            "Correct async/await conversion, error handling preservation, "
            "readability improvement, proper Promise usage."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Summarization prompts (technical, business, legal) — 6 total
# ---------------------------------------------------------------------------

_SUMMARIZATION_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="summarization-technical-001",
        task_type="summarization",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "Summarize the key differences between REST, GraphQL, and gRPC "
            "for inter-service communication. Focus on when to choose each "
            "and the tradeoffs. Keep it under 200 words."
        ),
        judge_criteria=(
            "Accuracy of protocol differences, appropriate use-case "
            "recommendations, tradeoff clarity, conciseness."
        ),
    ),
    EvalPrompt(
        id="summarization-technical-002",
        task_type="summarization",
        domain="technical",
        quality_speed="speed",
        prompt=(
            "Give a one-paragraph summary of how TLS 1.3 handshake differs "
            "from TLS 1.2. Focus on the latency improvement."
        ),
        judge_criteria=(
            "Accuracy of handshake difference (1-RTT vs 2-RTT), mention "
            "of 0-RTT resumption, conciseness."
        ),
    ),
    EvalPrompt(
        id="summarization-business-001",
        task_type="summarization",
        domain="business",
        quality_speed="quality",
        prompt=(
            "Summarize the key points a startup founder should understand "
            "about SAFE (Simple Agreement for Future Equity) notes vs. "
            "convertible notes. Cover: valuation caps, interest, maturity, "
            "and conversion triggers."
        ),
        judge_criteria=(
            "Accuracy of SAFE vs convertible note differences, coverage "
            "of all four requested topics, practical relevance for founders."
        ),
    ),
    EvalPrompt(
        id="summarization-business-002",
        task_type="summarization",
        domain="business",
        quality_speed="balanced",
        prompt=(
            "Summarize the difference between ARR, MRR, and revenue run "
            "rate. When is each metric most useful? Keep under 150 words."
        ),
        judge_criteria=(
            "Metric definition accuracy, appropriate use-case distinction, "
            "brevity within word limit."
        ),
    ),
    EvalPrompt(
        id="summarization-legal-001",
        task_type="summarization",
        domain="legal",
        quality_speed="quality",
        prompt=(
            "Summarize the key requirements of the EU AI Act's 'high-risk' "
            "classification for AI systems. What obligations does this "
            "impose on developers? Keep under 200 words."
        ),
        judge_criteria=(
            "Accuracy of high-risk classification criteria, coverage of "
            "developer obligations (risk management, data governance, "
            "transparency), conciseness."
        ),
    ),
    EvalPrompt(
        id="summarization-legal-002",
        task_type="summarization",
        domain="legal",
        quality_speed="balanced",
        prompt=(
            "Summarize the difference between copyright, trademark, and "
            "patent protection. Give one example of each. Under 150 words."
        ),
        judge_criteria=(
            "Correct distinction between the three IP types, relevant "
            "examples, brevity."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Creative prompts (creative_writing) — 5 total
# ---------------------------------------------------------------------------

_CREATIVE_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="creative-writing-001",
        task_type="creative",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Write a six-word story that conveys loss without using the "
            "words 'death,' 'died,' 'lost,' or 'gone.' Explain your choice."
        ),
        judge_criteria=(
            "Emotional impact, constraint adherence, originality, "
            "quality of the explanation."
        ),
    ),
    EvalPrompt(
        id="creative-writing-002",
        task_type="creative",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Rewrite this bland sentence to be vivid and engaging: "
            "'The old man walked slowly down the street on a cold morning.'"
        ),
        judge_criteria=(
            "Sensory detail, originality of word choice, preservation "
            "of core meaning, improvement over original."
        ),
    ),
    EvalPrompt(
        id="creative-writing-003",
        task_type="creative",
        domain="creative_writing",
        quality_speed="balanced",
        prompt=(
            "Write a metaphor that explains recursion to a non-programmer. "
            "Make it memorable and accurate."
        ),
        judge_criteria=(
            "Metaphor clarity, technical accuracy of the recursion concept, "
            "memorability, accessibility to non-programmers."
        ),
    ),
    EvalPrompt(
        id="creative-writing-004",
        task_type="creative",
        domain="creative_writing",
        quality_speed="balanced",
        prompt=(
            "Write two contrasting taglines for a meditation app: one "
            "targeting stressed executives, one targeting college students."
        ),
        judge_criteria=(
            "Audience targeting accuracy, contrast between taglines, "
            "conciseness, persuasive quality."
        ),
    ),
    EvalPrompt(
        id="creative-writing-005",
        task_type="creative",
        domain="creative_writing",
        quality_speed="speed",
        prompt=(
            "Generate five creative names for a coffee shop that also "
            "serves as a coworking space."
        ),
        judge_criteria=(
            "Creativity, memorability, relevance to dual concept "
            "(coffee + coworking), variety."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Reasoning prompts (code, technical, legal) — 6 total
# ---------------------------------------------------------------------------

_REASONING_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="reasoning-code-001",
        task_type="reasoning",
        domain="code",
        quality_speed="quality",
        prompt=(
            "What is the time complexity of this function? Explain your "
            "reasoning step by step.\n\n"
            "def mystery(arr):\n"
            "    n = len(arr)\n"
            "    result = 0\n"
            "    i = 1\n"
            "    while i < n:\n"
            "        for j in range(n):\n"
            "            result += arr[j]\n"
            "        i *= 2\n"
            "    return result"
        ),
        judge_criteria=(
            "Correct identification of O(n log n), clear step-by-step "
            "reasoning about the outer loop (log n iterations due to "
            "doubling) and inner loop (n iterations)."
        ),
    ),
    EvalPrompt(
        id="reasoning-code-002",
        task_type="reasoning",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Why might this code deadlock?\n\n"
            "lock_a = Lock()\n"
            "lock_b = Lock()\n\n"
            "# Thread 1\n"
            "def task1():\n"
            "    with lock_a:\n"
            "        time.sleep(0.1)\n"
            "        with lock_b:\n"
            "            do_work()\n\n"
            "# Thread 2\n"
            "def task2():\n"
            "    with lock_b:\n"
            "        time.sleep(0.1)\n"
            "        with lock_a:\n"
            "            do_work()"
        ),
        judge_criteria=(
            "Identification of circular wait condition, clear explanation "
            "of the deadlock sequence, suggestion of lock ordering fix."
        ),
    ),
    EvalPrompt(
        id="reasoning-technical-001",
        task_type="reasoning",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "A distributed system has three services: A, B, C. A calls B "
            "and C in parallel. B has P99 latency of 50ms, C has P99 of "
            "200ms. What is the expected P99 of A's end-to-end response? "
            "Explain your reasoning about parallel vs. sequential latency."
        ),
        judge_criteria=(
            "Correct reasoning that parallel P99 is dominated by the "
            "slowest call (~200ms plus overhead), not additive. "
            "Discussion of tail latency amplification."
        ),
    ),
    EvalPrompt(
        id="reasoning-technical-002",
        task_type="reasoning",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "Why is eventual consistency sufficient for a social media "
            "likes counter but not for a bank account balance? Explain "
            "the consistency requirements of each."
        ),
        judge_criteria=(
            "Clear distinction between use cases, correct application "
            "of consistency models, practical reasoning about user impact."
        ),
    ),
    EvalPrompt(
        id="reasoning-legal-001",
        task_type="reasoning",
        domain="legal",
        quality_speed="quality",
        prompt=(
            "A company trains an AI model on publicly available web data "
            "that includes copyrighted articles. The model can generate "
            "text similar to those articles but never reproduces them "
            "verbatim. Reason through the fair use analysis (four factors)."
        ),
        judge_criteria=(
            "Correct identification and application of the four fair use "
            "factors, balanced analysis, acknowledgment of legal "
            "uncertainty in this area."
        ),
    ),
    EvalPrompt(
        id="reasoning-legal-002",
        task_type="reasoning",
        domain="legal",
        quality_speed="balanced",
        prompt=(
            "If a self-driving car must choose between two harmful "
            "outcomes, who bears legal liability: the manufacturer, the "
            "software developer, or the car owner? Reason through the "
            "product liability framework."
        ),
        judge_criteria=(
            "Application of product liability principles, distinction "
            "between strict liability and negligence, consideration of "
            "multiple parties, practical reasoning."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Lookup prompts (general, technical) — 4 total
# ---------------------------------------------------------------------------

_LOOKUP_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="lookup-general-001",
        task_type="lookup",
        domain="general",
        quality_speed="speed",
        prompt=(
            "What are the HTTP status codes in the 4xx range and what "
            "does each category represent? List the five most commonly "
            "used ones."
        ),
        judge_criteria=(
            "Accuracy of status code descriptions, correct identification "
            "of the most common ones (400, 401, 403, 404, 429), conciseness."
        ),
    ),
    EvalPrompt(
        id="lookup-general-002",
        task_type="lookup",
        domain="general",
        quality_speed="speed",
        prompt=(
            "What is the difference between UTC and GMT? Are they "
            "interchangeable?"
        ),
        judge_criteria=(
            "Accuracy of the distinction (UTC is atomic, GMT is "
            "astronomical), practical interchangeability note, conciseness."
        ),
    ),
    EvalPrompt(
        id="lookup-technical-001",
        task_type="lookup",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "What is the maximum size of a single PostgreSQL row? What "
            "is the TOAST mechanism and when does it activate?"
        ),
        judge_criteria=(
            "Correct row size limit (~1.6GB page limit context), "
            "accurate TOAST explanation, activation threshold (~2KB)."
        ),
    ),
    EvalPrompt(
        id="lookup-technical-002",
        task_type="lookup",
        domain="technical",
        quality_speed="speed",
        prompt=(
            "What are the default port numbers for: PostgreSQL, Redis, "
            "MongoDB, MySQL, and Elasticsearch?"
        ),
        judge_criteria=(
            "Correct port numbers (5432, 6379, 27017, 3306, 9200), "
            "completeness, no incorrect values."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Translation prompts (general) — 4 total
# ---------------------------------------------------------------------------

_TRANSLATION_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="translation-general-001",
        task_type="translation",
        domain="general",
        quality_speed="quality",
        prompt=(
            "Translate the following technical error message into "
            "user-friendly language that a non-technical person can "
            "understand: 'ConnectionRefusedError: [Errno 111] Connection "
            "refused - socket.connect() failed for host=db.internal:5432'"
        ),
        judge_criteria=(
            "Accuracy of the simplified explanation, accessibility for "
            "non-technical users, preservation of actionable information."
        ),
    ),
    EvalPrompt(
        id="translation-general-002",
        task_type="translation",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "Convert this academic abstract into a tweet thread (3 tweets "
            "max): 'This paper presents a novel approach to federated "
            "learning that reduces communication overhead by 73% through "
            "gradient compression and selective synchronization, while "
            "maintaining model accuracy within 0.2% of centralized training "
            "baselines across four benchmark datasets.'"
        ),
        judge_criteria=(
            "Accuracy of content translation, appropriate simplification, "
            "tweet thread format adherence, engagement quality."
        ),
    ),
    EvalPrompt(
        id="translation-general-003",
        task_type="translation",
        domain="general",
        quality_speed="speed",
        prompt=(
            "Rewrite this legal clause in plain English: 'The indemnifying "
            "party shall hold harmless and indemnify the indemnified party "
            "against any and all claims, damages, losses, costs, and "
            "expenses arising out of or in connection with any breach of "
            "representations or warranties made herein.'"
        ),
        judge_criteria=(
            "Accuracy of the plain-English version, preservation of "
            "legal meaning, readability improvement."
        ),
    ),
    EvalPrompt(
        id="translation-general-004",
        task_type="translation",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "Convert this JSON API response into a human-readable "
            "summary:\n"
            '{"status": "error", "code": 429, "message": "Rate limit '
            'exceeded", "retry_after": 30, "limit": {"requests": 100, '
            '"window": "1m"}, "usage": {"current": 103, "remaining": 0}}'
        ),
        judge_criteria=(
            "Accurate interpretation of all fields, natural language "
            "quality, actionable summary."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Mixed domain coverage — 6 total
# ---------------------------------------------------------------------------

_MIXED_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="mixed-generation-business-001",
        task_type="generation",
        domain="business",
        quality_speed="quality",
        prompt=(
            "Draft a one-page executive summary for a Series A pitch deck "
            "for an AI-powered code review tool. Include: problem, solution, "
            "market size, traction, and ask."
        ),
        judge_criteria=(
            "Business writing quality, completeness of requested sections, "
            "persuasiveness, appropriate detail level for executive summary."
        ),
    ),
    EvalPrompt(
        id="mixed-analysis-legal-001",
        task_type="analysis",
        domain="legal",
        quality_speed="quality",
        prompt=(
            "Analyze the GDPR implications of storing user IP addresses "
            "in server logs for 90 days. What legal basis could apply, and "
            "what obligations does this create?"
        ),
        judge_criteria=(
            "Correct identification of IP addresses as personal data under "
            "GDPR, analysis of legitimate interest basis, data retention "
            "obligations, practical recommendations."
        ),
    ),
    EvalPrompt(
        id="mixed-summarization-general-001",
        task_type="summarization",
        domain="general",
        quality_speed="speed",
        prompt=(
            "In one sentence each, summarize what these three design "
            "patterns solve: Observer, Strategy, Factory Method."
        ),
        judge_criteria=(
            "Accuracy of pattern descriptions, one-sentence constraint, "
            "clarity of the 'what it solves' framing."
        ),
    ),
    EvalPrompt(
        id="mixed-reasoning-business-001",
        task_type="reasoning",
        domain="business",
        quality_speed="balanced",
        prompt=(
            "A marketplace startup has strong supply (sellers) but weak "
            "demand (buyers). The team is debating whether to spend their "
            "limited budget on buyer acquisition or seller quality "
            "improvement. Reason through the tradeoffs."
        ),
        judge_criteria=(
            "Correct identification of the chicken-and-egg marketplace "
            "dynamic, balanced tradeoff reasoning, consideration of "
            "marketplace liquidity."
        ),
    ),
    EvalPrompt(
        id="mixed-creative-general-001",
        task_type="creative",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "Create an analogy that explains microservices architecture "
            "to a restaurant owner. Make it map to at least three "
            "architectural concepts."
        ),
        judge_criteria=(
            "Analogy clarity, mapping accuracy (at least 3 concepts), "
            "accessibility for non-technical audience, memorability."
        ),
    ),
    EvalPrompt(
        id="mixed-lookup-general-001",
        task_type="lookup",
        domain="general",
        quality_speed="speed",
        prompt=(
            "What is the CAP theorem? State the three properties and "
            "which pairs are achievable in practice."
        ),
        judge_criteria=(
            "Correct statement of Consistency, Availability, Partition "
            "tolerance. Correct identification of achievable pairs "
            "(CP, AP, CA in theory). Conciseness."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Aggregated prompt bank
# ---------------------------------------------------------------------------

def get_all_prompts() -> list[EvalPrompt]:
    """Return the complete eval prompt bank, validated against IBR dimensions.

    Returns a list of ~50 EvalPrompt instances covering all 8 task_types
    and 6 domains. Each prompt is validated on first access.
    """
    all_prompts = (
        _GENERATION_PROMPTS
        + _ANALYSIS_PROMPTS
        + _REFACTORING_PROMPTS
        + _SUMMARIZATION_PROMPTS
        + _CREATIVE_PROMPTS
        + _REASONING_PROMPTS
        + _LOOKUP_PROMPTS
        + _TRANSLATION_PROMPTS
        + _MIXED_PROMPTS
    )

    for prompt in all_prompts:
        _validate_prompt(prompt)

    # Verify uniqueness of IDs
    ids = [p.id for p in all_prompts]
    assert len(ids) == len(set(ids)), (
        f"Duplicate prompt IDs found: "
        f"{[pid for pid in ids if ids.count(pid) > 1]}"
    )

    assert len(all_prompts) >= 48, (
        f"Expected at least 48 prompts, got {len(all_prompts)}"
    )
    return all_prompts


def get_prompts_by_task_type(task_type: str) -> list[EvalPrompt]:
    """Return prompts filtered by task_type."""
    assert task_type in IBR_TASK_TYPES, f"Invalid task_type: {task_type}"
    return [p for p in get_all_prompts() if p.task_type == task_type]


def get_prompts_by_domain(domain: str) -> list[EvalPrompt]:
    """Return prompts filtered by domain."""
    assert domain in IBR_DOMAINS, f"Invalid domain: {domain}"
    return [p for p in get_all_prompts() if p.domain == domain]
