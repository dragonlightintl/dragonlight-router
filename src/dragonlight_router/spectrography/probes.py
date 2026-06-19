"""Discriminative probe bank for Model Spectrography.

Contains ~80 probes designed to surface behavioral differences between models
of similar capability. Unlike eval prompts (which measure general quality),
spectrography probes target specific discrimination axes: style preferences,
edge-case awareness, reasoning depth, domain-crossing, instruction following,
and speed-quality calibration.

Spec reference: intent-based-router-v0.1.0-spec.md section 3.3, Spectrography.
"""

from __future__ import annotations

from dataclasses import dataclass

from dragonlight_router.core.types import IBR_DOMAINS, IBR_TASK_TYPES

DISCRIMINATION_AXES: frozenset[str] = frozenset(
    {
        "style",
        "edge_case",
        "reasoning_depth",
        "domain_cross",
        "instruction_following",
        "speed_quality",
    }
)

DIFFICULTY_LEVELS: frozenset[str] = frozenset({"easy", "medium", "hard"})


@dataclass(frozen=True)
class SpectrographyProbe:
    """Single discriminative probe for model spectrography.

    id: unique identifier, prefixed with "disc-"
    task_type: which IBR task_type this tests
    domain: which IBR domain this tests
    quality_speed: IBR quality_speed dimension
    prompt: the actual probe prompt to send to the model
    judge_criteria: what the judge scores
    discrimination_axis: which behavioral axis this probe targets
    difficulty: probe difficulty level
    """

    id: str
    task_type: str
    domain: str
    quality_speed: str
    prompt: str
    judge_criteria: str
    discrimination_axis: str
    difficulty: str


def _validate_probe(p: SpectrographyProbe) -> None:
    """Assert that a SpectrographyProbe references valid IBR dimensions."""
    assert p.id.startswith("disc-"), f"Probe ID must start with 'disc-': {p.id}"
    assert p.task_type in IBR_TASK_TYPES, f"Invalid task_type '{p.task_type}' in probe {p.id}"
    assert p.domain in IBR_DOMAINS, f"Invalid domain '{p.domain}' in probe {p.id}"
    assert p.quality_speed in {"quality", "balanced", "speed"}, (
        f"Invalid quality_speed '{p.quality_speed}' in probe {p.id}"
    )
    assert p.discrimination_axis in DISCRIMINATION_AXES, (
        f"Invalid discrimination_axis '{p.discrimination_axis}' in probe {p.id}"
    )
    assert p.difficulty in DIFFICULTY_LEVELS, f"Invalid difficulty '{p.difficulty}' in probe {p.id}"
    assert len(p.prompt) > 0, f"Empty prompt text in {p.id}"
    assert len(p.judge_criteria) > 0, f"Empty judge_criteria in {p.id}"


# ---------------------------------------------------------------------------
# Style probes — test verbosity, formatting, code style preferences
# ---------------------------------------------------------------------------

_STYLE_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-generation-code-001",
        task_type="generation",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Implement binary search on a sorted integer list. Use exactly "
            "three variable names total, no more. No comments, no docstring."
        ),
        judge_criteria=(
            "Strict compliance with three-variable-name constraint, absence "
            "of comments/docstring, correctness of binary search logic. "
            "Weight: 60% constraint compliance, 40% correctness."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-refactoring-code-001",
        task_type="refactoring",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Rewrite this function in the most concise way possible. Every "
            "character counts. No type hints, no docstrings, single-letter "
            "variables are fine.\n\n"
            "def calculate_fibonacci_sequence(number_of_terms: int) -> list[int]:\n"
            '    """Calculate the first n terms of the Fibonacci sequence."""\n'
            "    fibonacci_numbers: list[int] = []\n"
            "    for current_index in range(number_of_terms):\n"
            "        if current_index <= 1:\n"
            "            fibonacci_numbers.append(current_index)\n"
            "        else:\n"
            "            next_value = fibonacci_numbers[current_index - 1] + "
            "fibonacci_numbers[current_index - 2]\n"
            "            fibonacci_numbers.append(next_value)\n"
            "    return fibonacci_numbers"
        ),
        judge_criteria=(
            "Character count reduction, preservation of correctness, "
            "willingness to actually use terse style vs instinct to keep "
            "readability. Weight: 50% conciseness, 30% correctness, "
            "20% absence of unsolicited documentation."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-generation-code-002",
        task_type="generation",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Write a Python class for a stack data structure. Use Google-style "
            "docstrings on every method, include type hints on all parameters "
            "and return values, and add an inline comment on every line of "
            "non-trivial logic."
        ),
        judge_criteria=(
            "Completeness of documentation on every method, Google-style "
            "docstring format compliance, inline comment coverage, type "
            "hint completeness. Weight: 70% documentation density, "
            "30% implementation correctness."
        ),
        discrimination_axis="style",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-generation-code-003",
        task_type="generation",
        domain="code",
        quality_speed="speed",
        prompt=(
            "Write a function that reverses a linked list. Return ONLY the "
            "code. No explanation before or after. No markdown code fences."
        ),
        judge_criteria=(
            "Absence of any text outside the function definition itself. "
            "No markdown fences, no explanation, no preamble. "
            "Weight: 60% format compliance, 40% correctness."
        ),
        discrimination_axis="style",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-generation-technical-001",
        task_type="generation",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "Write a Kubernetes deployment YAML for a Python web app. "
            "Use exactly this structure: deployment first, then service, "
            "separated by ---. Do NOT include any comments in the YAML."
        ),
        judge_criteria=(
            "Correct ordering (deployment then service), triple-dash "
            "separator, complete absence of YAML comments, valid K8s "
            "resource definitions. Weight: 50% format compliance, "
            "50% correctness."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-creative-creative_writing-001",
        task_type="creative",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Write a product launch announcement in exactly three paragraphs. "
            "First paragraph: problem statement only. Second paragraph: "
            "solution only. Third paragraph: call to action only. No paragraph "
            "may contain content belonging to another."
        ),
        judge_criteria=(
            "Strict content separation between paragraphs, three-paragraph "
            "structure compliance, no content bleed between sections. "
            "Weight: 60% structure compliance, 40% prose quality."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-generation-code-004",
        task_type="generation",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Write a Python function that checks if a string is a palindrome. "
            "Use a functional programming style: no loops, no mutation, no "
            "if/else statements. Only expressions and function calls."
        ),
        judge_criteria=(
            "Adherence to functional style constraints (no loops, no mutation, "
            "no if/else), correctness, creative use of functional constructs. "
            "Weight: 60% style compliance, 40% correctness."
        ),
        discrimination_axis="style",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-summarization-technical-001",
        task_type="summarization",
        domain="technical",
        quality_speed="speed",
        prompt=(
            "Summarize what a B-tree is in exactly two sentences. Not one, not three. Exactly two."
        ),
        judge_criteria=(
            "Exactly two sentences (period-delimited), accuracy of B-tree "
            "description, information density within the constraint. "
            "Weight: 50% sentence count compliance, 50% accuracy."
        ),
        discrimination_axis="style",
        difficulty="easy",
    ),
]

# ---------------------------------------------------------------------------
# Edge-case probes — test defensive thinking, corner-case awareness
# ---------------------------------------------------------------------------

_EDGE_CASE_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-analysis-code-001",
        task_type="analysis",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Find all bugs in this code. There are exactly five.\n\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "def load_config(path: str) -> dict:\n"
            "    with open(path) as f:\n"
            "        config = json.load(f)\n"
            "    db_port = config['database']['port']\n"
            "    timeout = config.get('timeout', 30) / 1000\n"
            "    max_conn = min(config['max_connections'], 100)\n"
            "    log_path = Path(config['log_dir']) / 'app.log'\n"
            "    log_path.parent.mkdir(exist_ok=True)\n"
            "    return {\n"
            "        'port': db_port,\n"
            "        'timeout': timeout,\n"
            "        'max_connections': max_conn,\n"
            "        'log_path': str(log_path),\n"
            "    }\n\n"
            "# Bugs: 1) No file-not-found handling 2) No JSON decode error handling\n"
            "# 3) KeyError on nested 'database'.'port' 4) timeout division may\n"
            "# produce unexpected float for int-expecting callers\n"
            "# 5) mkdir(parents=False) will fail if log_dir has multiple\n"
            "# missing ancestors"
        ),
        judge_criteria=(
            "Identification of all five bugs, especially the subtle ones "
            "(timeout float division, mkdir missing parents=True). "
            "Weight: 40% finding bugs 4 and 5, 30% finding bugs 1-3, "
            "30% explanation quality."
        ),
        discrimination_axis="edge_case",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-analysis-code-002",
        task_type="analysis",
        domain="code",
        quality_speed="quality",
        prompt=(
            "What is wrong with this SQL query? The syntax is valid.\n\n"
            "SELECT u.name, u.email, o.total\n"
            "FROM users u\n"
            "JOIN orders o ON o.user_id = u.id\n"
            "WHERE u.name = '" + "' OR '1'='1" + "'\n"
            "ORDER BY o.total DESC\n"
            "LIMIT 10;"
        ),
        judge_criteria=(
            "Identification of SQL injection vulnerability disguised as "
            "a valid query, explanation of the tautology attack vector, "
            "recommendation for parameterized queries. Weight: 50% "
            "identifying injection, 30% explaining the mechanism, "
            "20% remediation quality."
        ),
        discrimination_axis="edge_case",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-analysis-code-003",
        task_type="analysis",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "This function passes all unit tests but has a production bug. "
            "What is it?\n\n"
            "import time\n\n"
            "def rate_limiter(max_requests: int, window_seconds: int):\n"
            "    timestamps = []\n"
            "    def allow():\n"
            "        now = time.time()\n"
            "        timestamps[:] = [t for t in timestamps if now - t < window_seconds]\n"
            "        if len(timestamps) < max_requests:\n"
            "            timestamps.append(now)\n"
            "            return True\n"
            "        return False\n"
            "    return allow"
        ),
        judge_criteria=(
            "Identification that the mutable closure state is not thread-safe "
            "in production (concurrent requests can race on timestamps list), "
            "and that the list will grow unbounded under sustained traffic "
            "beyond the rate limit. Weight: 50% thread safety, 30% memory "
            "leak under sustained overload, 20% fix suggestion."
        ),
        discrimination_axis="edge_case",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-refactoring-code-002",
        task_type="refactoring",
        domain="code",
        quality_speed="quality",
        prompt=(
            "This code works but has a subtle data loss bug under concurrent "
            "access. Fix it and explain why.\n\n"
            "class UserPrefs:\n"
            "    def __init__(self, path):\n"
            "        self.path = path\n"
            "        self.prefs = json.loads(Path(path).read_text())\n\n"
            "    def set(self, key, value):\n"
            "        self.prefs[key] = value\n"
            "        Path(self.path).write_text(json.dumps(self.prefs))\n\n"
            "    def get(self, key, default=None):\n"
            "        return self.prefs.get(key, default)"
        ),
        judge_criteria=(
            "Identification of TOCTOU race between read-modify-write, "
            "non-atomic file write (partial write on crash), and potential "
            "for concurrent set() calls losing updates. Quality of fix "
            "(atomic rename, file locking). Weight: 40% identifying "
            "races, 30% atomic write fix, 30% explanation."
        ),
        discrimination_axis="edge_case",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-analysis-code-004",
        task_type="analysis",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "What edge cases does this validation miss?\n\n"
            "def validate_email(email: str) -> bool:\n"
            "    if '@' not in email:\n"
            "        return False\n"
            "    local, domain = email.split('@')\n"
            "    if not local or not domain:\n"
            "        return False\n"
            "    if '.' not in domain:\n"
            "        return False\n"
            "    return True"
        ),
        judge_criteria=(
            "Identification of: multiple @ signs (split gives wrong result), "
            "leading/trailing dots in domain, spaces in local part, "
            "domain ending in dot, domains with only dots, excessively "
            "long local/domain parts. Weight: 40% finding the split('@') "
            "bug with multiple @, 30% other edge cases, 30% completeness."
        ),
        discrimination_axis="edge_case",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-analysis-code-005",
        task_type="analysis",
        domain="code",
        quality_speed="quality",
        prompt=(
            "This Python code has a memory leak that only manifests in "
            "long-running processes. Find it.\n\n"
            "import logging\n\n"
            "class RequestProcessor:\n"
            "    _cache = {}\n\n"
            "    def process(self, request_id: str, data: bytes) -> str:\n"
            "        logger = logging.getLogger(f'processor.{request_id}')\n"
            "        logger.addHandler(logging.StreamHandler())\n"
            "        result = self._transform(data)\n"
            "        self._cache[request_id] = result\n"
            "        logger.info(f'Processed {request_id}')\n"
            "        return result\n\n"
            "    def _transform(self, data: bytes) -> str:\n"
            "        return data.decode('utf-8').upper()"
        ),
        judge_criteria=(
            "Identification of three leaks: (1) class-level _cache dict "
            "grows unboundedly, (2) new logger created per request_id "
            "never garbage collected (logging module holds references), "
            "(3) handler added every call without removal. "
            "Weight: 40% finding logger leak, 30% finding cache leak, "
            "30% finding handler accumulation."
        ),
        discrimination_axis="edge_case",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-analysis-technical-001",
        task_type="analysis",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "A REST API returns 200 OK for this request, but something is "
            "dangerously wrong with the design. What?\n\n"
            "DELETE /api/v1/users?confirmed=false\n"
            "Authorization: Bearer <token>\n\n"
            "Response: 200 OK\n"
            '{"deleted_count": 847, "message": "Users deleted successfully"}'
        ),
        judge_criteria=(
            "Identification of bulk delete via query parameter without "
            "safeguards (no confirmation step, no rate limiting, no audit "
            "trail mention, ambiguous confirmed=false semantics — does it "
            "mean 'delete unconfirmed users' or 'delete without confirmation?'). "
            "Weight: 40% identifying the ambiguity, 30% bulk delete danger, "
            "30% missing safeguards."
        ),
        discrimination_axis="edge_case",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-reasoning-code-001",
        task_type="reasoning",
        domain="code",
        quality_speed="quality",
        prompt=(
            "This function returns correct results in all tests but will "
            "fail catastrophically on one specific input type. What input "
            "and why?\n\n"
            "def deep_merge(base: dict, override: dict) -> dict:\n"
            "    result = base.copy()\n"
            "    for key, value in override.items():\n"
            "        if key in result and isinstance(result[key], dict)"
            " and isinstance(value, dict):\n"
            "            result[key] = deep_merge(result[key], value)\n"
            "        else:\n"
            "            result[key] = value\n"
            "    return result"
        ),
        judge_criteria=(
            "Identification of infinite recursion when a dict contains "
            "a circular reference (self-referencing dict), shallow copy "
            "only copies top level (nested dicts still shared), and "
            "potential for mutation of original base dict through shared "
            "references. Weight: 40% circular reference, 30% shallow "
            "copy aliasing, 30% explanation quality."
        ),
        discrimination_axis="edge_case",
        difficulty="hard",
    ),
]

# ---------------------------------------------------------------------------
# Reasoning-depth probes — test chain-of-thought quality, step decomposition
# ---------------------------------------------------------------------------

_REASONING_DEPTH_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-reasoning-code-002",
        task_type="reasoning",
        domain="code",
        quality_speed="quality",
        prompt=(
            "This function is O(n^2) but looks like it should be O(n). "
            "Explain step by step WHY it is quadratic, not just THAT it is.\n\n"
            "def find_unique(items: list[str]) -> list[str]:\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in result:\n"
            "            result.append(item)\n"
            "    return result"
        ),
        judge_criteria=(
            "Quality of step-by-step reasoning: does it explain the 'in' "
            "operator scans the list linearly, show how this compounds "
            "across iterations, and demonstrate the summation 1+2+...+n? "
            "Weight: 60% reasoning chain quality, 20% correct complexity, "
            "20% fix suggestion."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-reasoning-code-003",
        task_type="reasoning",
        domain="code",
        quality_speed="quality",
        prompt=(
            "What happens if we remove the mutex from this code? Walk "
            "through a specific interleaving that produces a wrong result.\n\n"
            "import threading\n\n"
            "counter = 0\n"
            "lock = threading.Lock()\n\n"
            "def increment():\n"
            "    global counter\n"
            "    with lock:\n"
            "        temp = counter\n"
            "        temp += 1\n"
            "        counter = temp"
        ),
        judge_criteria=(
            "Concrete thread interleaving example showing lost update, "
            "step-by-step register-level reasoning (T1 reads 0, T2 reads "
            "0, T1 writes 1, T2 writes 1 — expected 2 got 1). "
            "Weight: 60% concrete interleaving quality, 20% accuracy, "
            "20% practical implications."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-reasoning-technical-001",
        task_type="reasoning",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "A microservice processes 1000 requests/second. Each request "
            "makes 3 downstream calls with P50 latency of 10ms each. "
            "The service has a connection pool of 100. Will it bottleneck? "
            "Show your math step by step."
        ),
        judge_criteria=(
            "Step-by-step mathematical reasoning: Little's Law application "
            "(concurrent connections = arrival_rate * service_time), "
            "correct calculation (1000 * 0.03 = 30 concurrent), correct "
            "conclusion (no bottleneck at P50 but P99 could be different). "
            "Weight: 50% math correctness, 30% reasoning steps, "
            "20% consideration of tail latency."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-reasoning-technical-002",
        task_type="reasoning",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "Explain why adding a cache can sometimes make a system SLOWER "
            "overall. Give at least three distinct mechanisms by which this "
            "happens, with a concrete example for each."
        ),
        judge_criteria=(
            "Depth of reasoning about: cache stampede/thundering herd, "
            "cold start penalty amortization, cache coherence overhead in "
            "distributed systems, extra network hop for cache misses. "
            "Weight: 40% number of distinct mechanisms, 40% example "
            "quality, 20% clarity of explanation."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-reasoning-code-004",
        task_type="reasoning",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Why does Python's default mutable argument create bugs? "
            "Don't just say 'it's shared.' Trace the exact object lifecycle "
            "through three calls to this function:\n\n"
            "def append_to(item, target=[]):\n"
            "    target.append(item)\n"
            "    return target\n\n"
            "Call 1: append_to(1)\n"
            "Call 2: append_to(2)\n"
            "Call 3: append_to(3, [])"
        ),
        judge_criteria=(
            "Object-level tracing through all three calls with memory "
            "model reasoning (same list object in calls 1-2, fresh in "
            "call 3). Results: [1], [1,2], [3]. Explanation of WHY the "
            "default is created once at function definition time. "
            "Weight: 50% step-by-step trace quality, 30% correct results, "
            "20% explanation of definition-time evaluation."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-reasoning-legal-001",
        task_type="reasoning",
        domain="legal",
        quality_speed="quality",
        prompt=(
            "A startup stores user analytics data on servers in the US. "
            "A French user requests data deletion under GDPR. The startup "
            "has no EU entity. Reason through the jurisdictional and "
            "practical compliance challenges step by step."
        ),
        judge_criteria=(
            "Step-by-step legal reasoning: territorial scope of GDPR "
            "(Art 3), data transfer implications (Schrems II), practical "
            "enforcement challenges, right to erasure requirements. "
            "Weight: 40% reasoning chain depth, 30% legal accuracy, "
            "30% practical considerations."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-analysis-business-001",
        task_type="analysis",
        domain="business",
        quality_speed="quality",
        prompt=(
            "A SaaS company has 10,000 users on a free tier and 200 paying "
            "customers at $50/month. Conversion rate is 2%. The CEO wants "
            "to 3x revenue in 12 months. Reason through at least four "
            "distinct strategies, analyzing the math behind each."
        ),
        judge_criteria=(
            "Mathematical rigor in strategy analysis (current MRR = $10K, "
            "target = $30K), consideration of: conversion rate improvement, "
            "ARPU increase, top-of-funnel growth, churn reduction. "
            "Weight: 40% math correctness, 30% strategy diversity, "
            "30% reasoning depth per strategy."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-reasoning-code-005",
        task_type="reasoning",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "This async code has a subtle bug that causes it to hang "
            "indefinitely under certain conditions. Explain the exact "
            "sequence of events that leads to the hang.\n\n"
            "async def fetch_all(urls: list[str]) -> list[str]:\n"
            "    semaphore = asyncio.Semaphore(10)\n"
            "    results = []\n\n"
            "    async def fetch_one(url):\n"
            "        async with semaphore:\n"
            "            resp = await aiohttp.get(url)\n"
            "            results.append(await resp.text())\n\n"
            "    tasks = [asyncio.create_task(fetch_one(u)) for u in urls]\n"
            "    await asyncio.gather(*tasks)\n"
            "    return results"
        ),
        judge_criteria=(
            "Identification that if any task raises an exception, "
            "gather() by default propagates the first exception but other "
            "tasks may still be running or the semaphore is never released "
            "by the failed task. Also: results list is not ordered by URL. "
            "Weight: 50% exception handling analysis, 30% ordering issue, "
            "20% explanation of event sequence."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="hard",
    ),
]

# ---------------------------------------------------------------------------
# Domain-cross probes — test knowledge boundary identification
# ---------------------------------------------------------------------------

_DOMAIN_CROSS_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-analysis-code-006",
        task_type="analysis",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Review this user data collection endpoint for any issues:\n\n"
            "from flask import Flask, request, jsonify\n\n"
            "app = Flask(__name__)\n\n"
            "@app.route('/api/users', methods=['POST'])\n"
            "def create_user():\n"
            "    data = request.json\n"
            "    user = {\n"
            "        'name': data['name'],\n"
            "        'email': data['email'],\n"
            "        'ip_address': request.remote_addr,\n"
            "        'browser': request.headers.get('User-Agent'),\n"
            "        'location': data.get('location'),\n"
            "    }\n"
            "    db.users.insert(user)\n"
            "    analytics.track('user_created', user)\n"
            "    return jsonify({'id': user['_id']}), 201"
        ),
        judge_criteria=(
            "Does the model surface privacy/GDPR concerns (IP collection, "
            "browser fingerprinting, location tracking) alongside code "
            "quality issues (no input validation, no error handling, "
            "leaking _id)? Weight: 40% privacy/legal issues identified, "
            "30% code quality issues, 30% cross-domain integration."
        ),
        discrimination_axis="domain_cross",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-generation-legal-001",
        task_type="generation",
        domain="legal",
        quality_speed="quality",
        prompt=(
            "Draft a terms of service clause for an AI writing assistant "
            "that generates content based on user prompts. Cover who owns "
            "the generated content."
        ),
        judge_criteria=(
            "Does the model address: IP ownership ambiguity for AI-generated "
            "content, regulatory landscape (EU AI Act transparency), user "
            "data usage for training, liability limitations? "
            "Weight: 40% legal-technical cross-domain awareness, "
            "30% clause quality, 30% regulatory awareness."
        ),
        discrimination_axis="domain_cross",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-analysis-technical-002",
        task_type="analysis",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "We are building a health monitoring IoT device that sends "
            "heart rate data to a cloud API every 30 seconds. The device "
            "uses BLE to connect to a phone app, which relays data via "
            "HTTPS. Review the architecture for issues."
        ),
        judge_criteria=(
            "Does the model cross domains to cover: HIPAA/health data "
            "compliance, BLE security limitations, data-in-transit "
            "encryption gaps (BLE to phone), battery life implications, "
            "and technical reliability? Weight: 40% regulatory/privacy "
            "awareness, 30% security analysis, 30% technical depth."
        ),
        discrimination_axis="domain_cross",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-generation-business-001",
        task_type="generation",
        domain="business",
        quality_speed="quality",
        prompt=(
            "Write a go-to-market strategy for an AI-powered resume "
            "screening tool targeting enterprise HR departments."
        ),
        judge_criteria=(
            "Does the model surface: algorithmic bias/discrimination "
            "risks (Title VII, EEOC), regulatory requirements (NYC "
            "Local Law 144), EU AI Act high-risk classification, "
            "alongside standard business strategy? Weight: 40% "
            "regulatory/ethical awareness, 30% strategy quality, "
            "30% cross-domain integration."
        ),
        discrimination_axis="domain_cross",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-analysis-business-002",
        task_type="analysis",
        domain="business",
        quality_speed="balanced",
        prompt=(
            "A SaaS company wants to expand from the US to the EU market. "
            "They currently store all customer data in a single US-East "
            "AWS region. What should they consider?"
        ),
        judge_criteria=(
            "Cross-domain coverage: GDPR data residency, Schrems II "
            "transfer mechanism, technical architecture (multi-region), "
            "business implications (cost, latency), and operational "
            "changes needed. Weight: 35% legal/regulatory, 35% "
            "technical architecture, 30% business strategy."
        ),
        discrimination_axis="domain_cross",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-reasoning-business-001",
        task_type="reasoning",
        domain="business",
        quality_speed="quality",
        prompt=(
            "A fintech startup wants to offer AI-driven investment advice "
            "to retail customers. The CEO asks: 'Can we just ship it and "
            "figure out compliance later?' Reason through why this is or "
            "isn't viable."
        ),
        judge_criteria=(
            "Cross-domain reasoning covering: SEC/FINRA regulatory "
            "requirements, fiduciary duty implications, potential "
            "enforcement actions, technical audit trail needs, and "
            "business risk analysis. Weight: 40% regulatory depth, "
            "30% business risk reasoning, 30% practical advice."
        ),
        discrimination_axis="domain_cross",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-creative-general-001",
        task_type="creative",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "Write a blog post introduction about why your startup chose "
            "to open-source its core product. Consider all stakeholder "
            "perspectives."
        ),
        judge_criteria=(
            "Does the model weave together: business model implications "
            "(community vs revenue), IP/licensing considerations (MIT vs "
            "AGPL), developer relations, and competitive strategy? "
            "Weight: 40% cross-domain breadth, 30% writing quality, "
            "30% stakeholder coverage."
        ),
        discrimination_axis="domain_cross",
        difficulty="medium",
    ),
]

# ---------------------------------------------------------------------------
# Instruction-following probes — test exact constraint adherence
# ---------------------------------------------------------------------------

_INSTRUCTION_FOLLOWING_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-generation-general-001",
        task_type="generation",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "List exactly 5 benefits of remote work. Not 4, not 6. "
            "Each benefit must be a single sentence. Number them 1-5."
        ),
        judge_criteria=(
            "Exactly 5 items (strict count), each is a single sentence, "
            "numbered 1-5, no preamble or postscript. Weight: 70% "
            "exact constraint compliance, 30% content quality."
        ),
        discrimination_axis="instruction_following",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-generation-general-002",
        task_type="generation",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "Respond in ONLY valid JSON with exactly these keys: "
            '"summary", "confidence", "tags". Summary is a string '
            "under 100 characters. Confidence is a float 0-1. Tags is an "
            "array of exactly 3 strings. Topic: machine learning in healthcare."
        ),
        judge_criteria=(
            "Valid JSON parsing, exact key names, correct types (string, "
            "float, array), summary under 100 chars, exactly 3 tags, "
            "no surrounding text. Weight: 80% format compliance, "
            "20% content relevance."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-translation-general-001",
        task_type="translation",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "Translate this technical paragraph into language a 10-year-old "
            "would understand. Do not use any word longer than 3 syllables. "
            "Keep exactly 3 sentences.\n\n"
            "'Microservices architecture decomposes monolithic applications "
            "into independently deployable services that communicate via "
            "lightweight protocols, enabling horizontal scalability and "
            "fault isolation.'"
        ),
        judge_criteria=(
            "3-syllable word limit compliance (check every word), exactly "
            "3 sentences, age-appropriate language, accuracy preservation. "
            "Weight: 50% syllable constraint, 25% sentence count, "
            "25% meaning preservation."
        ),
        discrimination_axis="instruction_following",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-summarization-business-001",
        task_type="summarization",
        domain="business",
        quality_speed="speed",
        prompt=(
            "Summarize the concept of product-market fit in exactly 280 "
            "characters or fewer (tweet-length). Include a hashtag. "
            "No line breaks."
        ),
        judge_criteria=(
            "Character count <= 280, presence of hashtag, no line breaks, "
            "accurate description of product-market fit, completeness "
            "within constraint. Weight: 60% character/format compliance, "
            "40% content accuracy."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-generation-code-005",
        task_type="generation",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Write a Python function that takes a list of integers and "
            "returns the second largest. Requirements that MUST all be met:\n"
            "1. Function name must be exactly 'second_largest'\n"
            "2. Must use exactly one parameter named 'numbers'\n"
            "3. Must raise ValueError for lists with fewer than 2 unique values\n"
            "4. Must not use sort() or sorted()\n"
            "5. Must be implemented in 10 lines or fewer (excluding def line)"
        ),
        judge_criteria=(
            "All five constraints met simultaneously: exact function name, "
            "exact parameter name, ValueError behavior, no sort usage, "
            "10-line limit. Weight: 12% per constraint (60%), "
            "40% correctness."
        ),
        discrimination_axis="instruction_following",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-creative-creative_writing-002",
        task_type="creative",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Write a haiku about debugging code. It MUST follow the 5-7-5 "
            "syllable structure exactly. After the haiku, state the syllable "
            "count of each line to prove compliance."
        ),
        judge_criteria=(
            "Strict 5-7-5 syllable compliance, self-verification accuracy, "
            "thematic relevance to debugging, poetic quality. "
            "Weight: 40% syllable accuracy, 20% self-verification, "
            "20% theme, 20% quality."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-lookup-general-001",
        task_type="lookup",
        domain="general",
        quality_speed="speed",
        prompt=(
            "What are the SOLID principles? List them as a numbered list. "
            "For each, give the full name only. No explanations. No "
            "descriptions. Just the five names, numbered 1-5."
        ),
        judge_criteria=(
            "Exactly 5 items, numbered, full names only (Single Responsibility "
            "Principle, etc.), absolute zero explanation or description text. "
            "Weight: 60% no-explanation compliance, 40% accuracy."
        ),
        discrimination_axis="instruction_following",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-generation-technical-002",
        task_type="generation",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "Write a bash one-liner that finds all Python files modified "
            "in the last 24 hours and counts lines of code in each. "
            "Constraints: must be a single line, must use find and wc, "
            "must output filename and line count separated by a tab."
        ),
        judge_criteria=(
            "Single line (no line breaks), uses both find and wc, tab "
            "separation in output, correct 24-hour modification filter. "
            "Weight: 50% constraint compliance, 50% correctness."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-analysis-general-001",
        task_type="analysis",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "Analyze the pros and cons of electric vehicles. Output MUST be "
            "a markdown table with exactly three columns: Category, Pro, Con. "
            "Exactly 5 rows. No text outside the table."
        ),
        judge_criteria=(
            "Valid markdown table format, exactly 3 columns with correct "
            "headers, exactly 5 data rows, zero text outside the table. "
            "Weight: 60% format compliance, 40% content quality."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
]

# ---------------------------------------------------------------------------
# Speed-quality probes — test response calibration under different framings
# ---------------------------------------------------------------------------

_SPEED_QUALITY_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-reasoning-code-006",
        task_type="reasoning",
        domain="code",
        quality_speed="speed",
        prompt=(
            "Quick: what is the time complexity of looking up a key in a Python dict? One sentence."
        ),
        judge_criteria=(
            "Speed-appropriate response: concise single sentence, correct "
            "answer (O(1) average, O(n) worst case), no unnecessary "
            "elaboration. Weight: 50% conciseness, 50% accuracy."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-reasoning-code-007",
        task_type="reasoning",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Think carefully and analyze in detail: what is the time "
            "complexity of looking up a key in a Python dict? Consider "
            "the underlying implementation, hash collisions, and worst-case "
            "scenarios."
        ),
        judge_criteria=(
            "Quality-appropriate depth: hash table explanation, collision "
            "resolution (open addressing in CPython), amortized analysis, "
            "worst-case O(n) explanation, practical implications. "
            "Weight: 60% depth of analysis, 40% accuracy."
        ),
        discrimination_axis="speed_quality",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-lookup-technical-001",
        task_type="lookup",
        domain="technical",
        quality_speed="speed",
        prompt=("Fastest answer: what port does PostgreSQL listen on by default?"),
        judge_criteria=(
            "Minimal response (ideally just '5432' or one short sentence), "
            "correct answer, no padding or unnecessary context. "
            "Weight: 50% brevity, 50% correctness."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-lookup-technical-002",
        task_type="lookup",
        domain="technical",
        quality_speed="quality",
        prompt=(
            "Explain thoroughly: what port does PostgreSQL listen on by "
            "default, why was this number chosen, how do you configure it, "
            "and what are the security implications of running on the "
            "default port?"
        ),
        judge_criteria=(
            "Comprehensive response covering: port 5432, historical context "
            "(Ingres used 5433), postgresql.conf configuration, security "
            "implications (port scanning, firewall rules). "
            "Weight: 50% thoroughness, 30% accuracy, 20% practical value."
        ),
        discrimination_axis="speed_quality",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-summarization-general-001",
        task_type="summarization",
        domain="general",
        quality_speed="speed",
        prompt=("Quick summary: what is Docker? Two sentences max."),
        judge_criteria=(
            "Two sentences or fewer, accurate Docker summary, no "
            "unnecessary detail. Weight: 50% brevity, 50% accuracy."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-summarization-general-002",
        task_type="summarization",
        domain="general",
        quality_speed="quality",
        prompt=(
            "Write a thorough summary of Docker: what it is, how it works "
            "under the hood (namespaces, cgroups, union filesystems), how "
            "it differs from VMs, and when you should and shouldn't use it."
        ),
        judge_criteria=(
            "Depth covering: containerization concept, Linux kernel "
            "primitives (namespaces, cgroups), overlay/union FS, VM "
            "comparison, appropriate/inappropriate use cases. "
            "Weight: 50% technical depth, 30% accuracy, 20% practical "
            "guidance."
        ),
        discrimination_axis="speed_quality",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-generation-code-006",
        task_type="generation",
        domain="code",
        quality_speed="speed",
        prompt=(
            "Quick: write a Python function to check if a number is prime. "
            "Shortest correct implementation."
        ),
        judge_criteria=(
            "Brevity of implementation, correctness for edge cases (0, 1, 2, "
            "negative), no unnecessary documentation or explanation. "
            "Weight: 40% brevity, 40% correctness, 20% absence of padding."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-generation-code-007",
        task_type="generation",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Write a production-quality Python function to check if a number "
            "is prime. Include: type hints, docstring with examples, edge "
            "case handling, optimized trial division up to sqrt(n), and "
            "explain your optimization choices."
        ),
        judge_criteria=(
            "Production quality: type hints, docstring with examples, "
            "edge cases (0, 1, 2, negative, non-int), sqrt optimization, "
            "skip even numbers, explanation of optimization choices. "
            "Weight: 40% completeness, 30% optimization quality, "
            "30% explanation depth."
        ),
        discrimination_axis="speed_quality",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-translation-general-002",
        task_type="translation",
        domain="general",
        quality_speed="speed",
        prompt=(
            "Quick: rewrite 'The implementation leverages polymorphic "
            "dispatch mechanisms' in plain English."
        ),
        judge_criteria=(
            "Concise plain-English translation, correct meaning "
            "preservation, minimal response length. Weight: 50% "
            "brevity, 50% accuracy."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-translation-general-003",
        task_type="translation",
        domain="general",
        quality_speed="quality",
        prompt=(
            "Translate the following technical paragraph into a detailed "
            "explanation suitable for a business executive with no technical "
            "background. Use analogies where helpful. Explain every "
            "technical term.\n\n"
            "'The system employs eventual consistency via CRDTs, with a "
            "gossip protocol for state propagation across replicas. "
            "Conflict resolution uses last-writer-wins with vector clocks "
            "for causality tracking.'"
        ),
        judge_criteria=(
            "Thorough translation with analogies, every technical term "
            "explained (CRDT, gossip protocol, eventual consistency, "
            "vector clocks, LWW), executive-appropriate framing. "
            "Weight: 40% completeness, 30% analogy quality, "
            "30% accessibility."
        ),
        discrimination_axis="speed_quality",
        difficulty="hard",
    ),
]

# ---------------------------------------------------------------------------
# Additional coverage probes — fill gaps in task_type x domain matrix
# ---------------------------------------------------------------------------

_COVERAGE_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-refactoring-code-003",
        task_type="refactoring",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Refactor this Express.js route handler. It works but violates "
            "separation of concerns. Split it into layers without changing "
            "the external behavior.\n\n"
            "app.post('/api/orders', async (req, res) => {\n"
            "  const { userId, items } = req.body;\n"
            "  if (!userId || !items?.length)\n"
            "    return res.status(400).json({error: 'Missing'});\n"
            "  const user = await db.query(\n"
            "    'SELECT * FROM users WHERE id = $1', [userId]);\n"
            "  if (!user.rows[0])\n"
            "    return res.status(404).json({error: 'Not found'});\n"
            "  let total = 0;\n"
            "  for (const item of items) {\n"
            "    const product = await db.query(\n"
            "      'SELECT price FROM products WHERE id = $1',\n"
            "      [item.productId]);\n"
            "    total += product.rows[0].price * item.quantity;\n"
            "  }\n"
            "  const order = await db.query(\n"
            "    'INSERT INTO orders (user_id, total) VALUES ($1, $2)'\n"
            "    + ' RETURNING *', [userId, total]);\n"
            "  res.status(201).json(order.rows[0]);\n"
            "});"
        ),
        judge_criteria=(
            "Clean separation into validation, data access, and business "
            "logic layers. N+1 query identification and fix. Error handling "
            "improvement. Weight: 40% separation quality, 30% N+1 fix, "
            "30% error handling."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-refactoring-code-004",
        task_type="refactoring",
        domain="code",
        quality_speed="speed",
        prompt=(
            "Simplify this conditional chain. Preserve exact behavior.\n\n"
            "def categorize(score):\n"
            "    if score >= 90:\n"
            "        return 'A'\n"
            "    elif score >= 80 and score < 90:\n"
            "        return 'B'\n"
            "    elif score >= 70 and score < 80:\n"
            "        return 'C'\n"
            "    elif score >= 60 and score < 70:\n"
            "        return 'D'\n"
            "    else:\n"
            "        return 'F'"
        ),
        judge_criteria=(
            "Removal of redundant upper-bound checks (already implied by "
            "elif ordering), potential use of bisect or dict-based "
            "dispatch, preservation of semantics. Weight: 50% "
            "simplification quality, 50% correctness."
        ),
        discrimination_axis="instruction_following",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-summarization-legal-001",
        task_type="summarization",
        domain="legal",
        quality_speed="quality",
        prompt=(
            "Summarize the key differences between GDPR and CCPA in terms "
            "of: scope of protected data, consent requirements, right to "
            "delete, and enforcement mechanisms. Use a structured comparison."
        ),
        judge_criteria=(
            "Accurate comparison across all four dimensions, structured "
            "format (table or parallel sections), nuanced differences "
            "(not just surface-level). Weight: 40% accuracy, 30% "
            "structure, 30% nuance."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-creative-creative_writing-003",
        task_type="creative",
        domain="creative_writing",
        quality_speed="balanced",
        prompt=(
            "Write two versions of the same rejection email: one that is "
            "warm and empathetic, and one that is professional and efficient. "
            "The rejection is for a job application. Both must convey the "
            "same information."
        ),
        judge_criteria=(
            "Clear tonal contrast between versions, same core information "
            "in both, appropriate register for each style, no condescension "
            "in either. Weight: 40% tonal contrast, 30% information "
            "parity, 30% quality."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-lookup-general-002",
        task_type="lookup",
        domain="general",
        quality_speed="balanced",
        prompt=(
            "What is the difference between authentication and authorization? "
            "Answer in exactly two bullet points, one for each concept. "
            "Each bullet must be one sentence."
        ),
        judge_criteria=(
            "Exactly two bullet points, one sentence each, correct "
            "definitions, clear distinction between the two concepts. "
            "Weight: 50% format compliance, 50% accuracy."
        ),
        discrimination_axis="instruction_following",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-translation-general-004",
        task_type="translation",
        domain="general",
        quality_speed="quality",
        prompt=(
            "Convert this Python code into equivalent pseudocode that a "
            "non-programmer could follow. Use everyday language, no "
            "programming keywords.\n\n"
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1"
        ),
        judge_criteria=(
            "Absence of programming keywords (no 'function', 'variable', "
            "'return', 'loop'), everyday language analogies, accuracy "
            "of algorithm description, accessibility. "
            "Weight: 40% no-programming-keyword compliance, "
            "30% accessibility, 30% accuracy."
        ),
        discrimination_axis="instruction_following",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-generation-business-002",
        task_type="generation",
        domain="business",
        quality_speed="balanced",
        prompt=(
            "Write a cold email to a VP of Engineering pitching an AI "
            "code review tool. Maximum 150 words. Must include a specific "
            "metric or data point. No exclamation marks."
        ),
        judge_criteria=(
            "Word count <= 150, contains at least one specific metric, "
            "zero exclamation marks, appropriate tone for VP-level, "
            "clear call to action. Weight: 50% constraint compliance, "
            "50% persuasive quality."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-analysis-code-007",
        task_type="analysis",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "This code handles money. What is wrong with it?\n\n"
            "def split_bill(total: float, num_people: int) -> list[float]:\n"
            "    per_person = round(total / num_people, 2)\n"
            "    return [per_person] * num_people"
        ),
        judge_criteria=(
            "Identification of floating-point precision issue (rounding "
            "errors mean amounts don't sum to total), recommendation of "
            "Decimal or integer cents approach, example showing the "
            "discrepancy. Weight: 40% identifying the rounding sum error, "
            "30% suggesting proper money handling, 30% example quality."
        ),
        discrimination_axis="edge_case",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-reasoning-technical-003",
        task_type="reasoning",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "A team runs their test suite in a Docker container. Tests pass "
            "locally but fail in CI. The tests involve file timestamps and "
            "timezone-dependent logic. What are the three most likely root "
            "causes? Explain the mechanism for each."
        ),
        judge_criteria=(
            "Identification of: different timezone configuration in "
            "container (UTC vs local), filesystem timestamp resolution "
            "differences (overlay vs ext4), and clock synchronization "
            "issues (NTP in container). Mechanism explanation for each. "
            "Weight: 40% identifying root causes, 40% mechanism "
            "explanation, 20% fix suggestions."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-lookup-technical-003",
        task_type="lookup",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "What is the difference between a process and a thread in "
            "Linux? Specifically address: memory space, creation cost, "
            "and communication mechanisms. Answer in a markdown table "
            "with these three rows."
        ),
        judge_criteria=(
            "Markdown table format with three rows as specified, correct "
            "process vs thread distinction for each dimension, technical "
            "accuracy. Weight: 40% format compliance, 60% accuracy."
        ),
        discrimination_axis="instruction_following",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-analysis-legal-001",
        task_type="analysis",
        domain="legal",
        quality_speed="quality",
        prompt=(
            "A SaaS company's open-source library has a critical bug that "
            "corrupts user data. The library is MIT licensed. Analyze the "
            "liability exposure considering the license terms, user "
            "expectations, and potential negligence claims."
        ),
        judge_criteria=(
            "Analysis of MIT license 'AS IS' clause, tension between "
            "license disclaimer and potential negligence duty, third-party "
            "dependency chain liability, practical risk assessment. "
            "Weight: 40% legal reasoning, 30% practical analysis, "
            "30% nuance."
        ),
        discrimination_axis="domain_cross",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-creative-general-002",
        task_type="creative",
        domain="general",
        quality_speed="speed",
        prompt=(
            "Generate 5 creative analogies for technical debt. Each analogy "
            "must be a single sentence. No two analogies may reference the "
            "same source domain."
        ),
        judge_criteria=(
            "Exactly 5 analogies, single sentence each, no repeated "
            "source domains (e.g., not two kitchen analogies), creativity "
            "and accuracy of mapping. Weight: 40% diversity, 30% "
            "creativity, 30% accuracy."
        ),
        discrimination_axis="style",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-refactoring-code-005",
        task_type="refactoring",
        domain="code",
        quality_speed="quality",
        prompt=(
            "This Python code uses global state and is untestable. Refactor "
            "it to be testable without changing the external API.\n\n"
            "import os\n"
            "import requests\n\n"
            "API_KEY = os.environ['API_KEY']\n"
            "BASE_URL = 'https://api.example.com'\n\n"
            "def get_user(user_id):\n"
            "    resp = requests.get(f'{BASE_URL}/users/{user_id}',\n"
            "                        headers={'Authorization': f'Bearer {API_KEY}'})\n"
            "    resp.raise_for_status()\n"
            "    return resp.json()\n\n"
            "def get_user_orders(user_id):\n"
            "    user = get_user(user_id)\n"
            "    resp = requests.get(f'{BASE_URL}/users/{user_id}/orders',\n"
            "                        headers={'Authorization': f'Bearer {API_KEY}'})\n"
            "    resp.raise_for_status()\n"
            "    return {'user': user, 'orders': resp.json()}"
        ),
        judge_criteria=(
            "Dependency injection (client class or factory), configuration "
            "encapsulation, testability improvement (mockable boundaries), "
            "preserved external API. Weight: 40% DI approach, 30% "
            "testability, 30% API preservation."
        ),
        discrimination_axis="edge_case",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-summarization-creative_writing-001",
        task_type="summarization",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Read this micro-story and write a one-paragraph literary "
            "analysis focusing on what is left unsaid:\n\n"
            "'She kept his coffee mug on the shelf for three years. On "
            "Tuesday she moved it to the cabinet. On Wednesday she used it.'"
        ),
        judge_criteria=(
            "Depth of subtext analysis (grief stages, the significance "
            "of each day's action, what the mug symbolizes), literary "
            "craft awareness, emotional intelligence. Weight: 50% "
            "subtext identification, 30% analysis quality, 20% "
            "sensitivity."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
]

# ---------------------------------------------------------------------------
# Additional probes to reach 80+ count and fill remaining coverage gaps
# ---------------------------------------------------------------------------

_ADDITIONAL_PROBES: list[SpectrographyProbe] = [
    SpectrographyProbe(
        id="disc-translation-code-001",
        task_type="translation",
        domain="code",
        quality_speed="quality",
        prompt=(
            "Convert this Python list comprehension into equivalent Go code. "
            "Preserve the filtering logic exactly.\n\n"
            "result = [user.name for user in users if user.active and user.age >= 18]"
        ),
        judge_criteria=(
            "Correct Go slice construction with for-range loop, proper "
            "struct field access, accurate filter condition translation, "
            "idiomatic Go style. Weight: 40% correctness, 30% idiomatic "
            "Go, 30% filter accuracy."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-translation-code-002",
        task_type="translation",
        domain="code",
        quality_speed="balanced",
        prompt=(
            "Translate this SQL query into a MongoDB aggregation pipeline. "
            "Preserve exact semantics.\n\n"
            "SELECT department, COUNT(*) as count, AVG(salary) as avg_salary\n"
            "FROM employees\n"
            "WHERE status = 'active'\n"
            "GROUP BY department\n"
            "HAVING COUNT(*) > 5\n"
            "ORDER BY avg_salary DESC;"
        ),
        judge_criteria=(
            "Correct $match, $group, $match (HAVING), $sort stages in "
            "proper order, accurate field mappings, correct aggregation "
            "operators ($sum, $avg). Weight: 50% pipeline correctness, "
            "50% semantic preservation."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-creative-creative_writing-004",
        task_type="creative",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Write two opening paragraphs for the same story — a scientist "
            "discovers something impossible in their lab. First version: "
            "literary fiction style (interiority, sensory detail). Second "
            "version: thriller style (pace, tension, short sentences). "
            "Both must describe the same moment."
        ),
        judge_criteria=(
            "Clear stylistic contrast between versions, same narrative "
            "moment preserved, genre conventions respected (literary: "
            "longer sentences, interiority; thriller: short sentences, "
            "external action). Weight: 40% contrast quality, 30% "
            "genre accuracy, 30% writing quality."
        ),
        discrimination_axis="style",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-creative-creative_writing-005",
        task_type="creative",
        domain="creative_writing",
        quality_speed="balanced",
        prompt=(
            "Write a technical blog post title and subtitle for each of "
            "these tones: authoritative, playful, provocative. Topic: "
            "why microservices might be a bad choice for your startup. "
            "Each title must be under 10 words. Each subtitle under 20 words."
        ),
        judge_criteria=(
            "Three distinct tonal variations, word count compliance "
            "(titles under 10, subtitles under 20), tonal accuracy "
            "for each category. Weight: 40% tonal differentiation, "
            "30% word count compliance, 30% quality."
        ),
        discrimination_axis="style",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-lookup-general-003",
        task_type="lookup",
        domain="general",
        quality_speed="speed",
        prompt=(
            "What does CORS stand for and what problem does it solve? "
            "Answer in exactly one sentence."
        ),
        judge_criteria=(
            "Single sentence, correct expansion (Cross-Origin Resource "
            "Sharing), accurate problem description (same-origin policy "
            "bypass for legitimate cross-domain requests). Weight: 50% "
            "one-sentence compliance, 50% accuracy."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-lookup-legal-001",
        task_type="lookup",
        domain="legal",
        quality_speed="balanced",
        prompt=(
            "What are the key requirements for a valid software license "
            "agreement? List the essential elements. Answer in bullet "
            "points only, no prose."
        ),
        judge_criteria=(
            "Bullet-point-only format (no introductory or closing prose), "
            "coverage of essential elements (grant of rights, restrictions, "
            "termination, warranties/disclaimers, limitation of liability). "
            "Weight: 40% format compliance, 60% accuracy."
        ),
        discrimination_axis="instruction_following",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-refactoring-code-006",
        task_type="refactoring",
        domain="code",
        quality_speed="quality",
        prompt=(
            "This code has three levels of callback nesting. Refactor it "
            "to use async/await and add proper error handling. The refactored "
            "version must handle each error type differently.\n\n"
            "function processPayment(orderId, cb) {\n"
            "  getOrder(orderId, (err, order) => {\n"
            "    if (err) return cb(err);\n"
            "    validateCard(order.cardId, (err, valid) => {\n"
            "      if (err) return cb(err);\n"
            "      if (!valid) return cb(new Error('Invalid card'));\n"
            "      chargeCard(order.cardId, order.total, (err, receipt) => {\n"
            "        if (err) return cb(err);\n"
            "        cb(null, receipt);\n"
            "      });\n"
            "    });\n"
            "  });\n"
            "}"
        ),
        judge_criteria=(
            "Clean async/await conversion, distinct error handling per "
            "operation (not a generic catch-all), invalid card as a "
            "business logic check separate from system errors. "
            "Weight: 40% error handling granularity, 30% async "
            "correctness, 30% readability."
        ),
        discrimination_axis="edge_case",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-refactoring-technical-001",
        task_type="refactoring",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "Refactor this Dockerfile to follow security best practices. "
            "List each change you make and why.\n\n"
            "FROM python:latest\n"
            "COPY . /app\n"
            "WORKDIR /app\n"
            "RUN pip install -r requirements.txt\n"
            "RUN apt-get update && apt-get install -y curl\n"
            "EXPOSE 8080\n"
            "CMD python app.py"
        ),
        judge_criteria=(
            "Identification of: pinned base image tag, non-root user, "
            "multi-stage build opportunity, apt-get cache cleanup, "
            "layer ordering for cache efficiency, COPY before pip install "
            "order. Weight: 40% security issues found, 30% explanation "
            "quality, 30% correctness."
        ),
        discrimination_axis="edge_case",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-analysis-creative_writing-001",
        task_type="analysis",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Analyze the rhetorical techniques in this marketing copy and "
            "explain why each works (or doesn't):\n\n"
            "'Stop wasting hours on code reviews. Our AI catches bugs 10x "
            "faster than manual review, with 99.7% accuracy. Join 5,000+ "
            "teams who shipped faster last quarter. Start free — no credit "
            "card required.'"
        ),
        judge_criteria=(
            "Identification of: appeal to pain point, specific metrics "
            "(10x, 99.7%), social proof (5,000+ teams), risk reversal "
            "(free, no CC), temporal anchoring ('last quarter'). Critical "
            "analysis of whether claims are substantiated. Weight: 40% "
            "technique identification, 30% critical analysis, 30% "
            "explanation quality."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-generation-legal-002",
        task_type="generation",
        domain="legal",
        quality_speed="balanced",
        prompt=(
            "Write a data processing addendum (DPA) outline for a B2B "
            "SaaS product. List the sections that must be included and "
            "one key clause for each section. Use a numbered list."
        ),
        judge_criteria=(
            "Coverage of essential DPA sections (definitions, scope, "
            "data subject rights, sub-processors, security measures, "
            "breach notification, international transfers, termination). "
            "Numbered list format. Weight: 50% completeness, 30% "
            "clause quality, 20% format."
        ),
        discrimination_axis="domain_cross",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-summarization-code-001",
        task_type="summarization",
        domain="code",
        quality_speed="speed",
        prompt=(
            "Summarize what this regular expression matches in one "
            "plain-English sentence:\n\n"
            "^(?=.*[A-Z])(?=.*[a-z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{8,}$"
        ),
        judge_criteria=(
            "Single sentence, correct interpretation (password validation: "
            "at least 8 chars, one uppercase, one lowercase, one digit, "
            "one special character), plain English. Weight: 50% accuracy, "
            "50% conciseness."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-reasoning-general-001",
        task_type="reasoning",
        domain="general",
        quality_speed="quality",
        prompt=(
            "A company has 100 employees. They can either: (A) give everyone "
            "a 5% raise, or (B) give the top 20 performers a 25% raise and "
            "everyone else nothing. Both cost the same assuming uniform "
            "salaries. Reason through the second- and third-order effects "
            "of each choice."
        ),
        judge_criteria=(
            "Depth of second/third-order reasoning: retention effects, "
            "morale/motivation dynamics, talent attraction, cultural "
            "signaling, Goodhart's Law risks for option B, equity "
            "considerations. Weight: 50% order-of-effects depth, "
            "30% balance, 20% practical insight."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-generation-creative_writing-001",
        task_type="generation",
        domain="creative_writing",
        quality_speed="quality",
        prompt=(
            "Write a 100-word flash fiction piece where the last sentence "
            "recontextualizes everything before it. The twist must be "
            "earned — no deus ex machina. Topic: a morning commute."
        ),
        judge_criteria=(
            "Word count approximately 100 (90-110 acceptable), effective "
            "twist that recontextualizes prior narrative, setup that "
            "supports the twist on re-reading, prose quality. "
            "Weight: 40% twist quality, 30% setup-payoff, 30% prose."
        ),
        discrimination_axis="style",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-analysis-technical-003",
        task_type="analysis",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "A system uses JWT tokens with a 24-hour expiry for "
            "authentication. The security team wants to add token "
            "revocation capability. Analyze the tradeoffs of three "
            "approaches: blacklist, short-lived tokens + refresh tokens, "
            "and token versioning."
        ),
        judge_criteria=(
            "Balanced analysis of all three approaches, consideration of: "
            "storage overhead, latency impact, scalability, security "
            "guarantees, implementation complexity. Weight: 40% "
            "tradeoff depth, 30% accuracy, 30% practical recommendation."
        ),
        discrimination_axis="reasoning_depth",
        difficulty="medium",
    ),
    SpectrographyProbe(
        id="disc-lookup-business-001",
        task_type="lookup",
        domain="business",
        quality_speed="speed",
        prompt=(
            "What is the difference between a moat and a competitive "
            "advantage? One paragraph, no more than 3 sentences."
        ),
        judge_criteria=(
            "Three sentences or fewer, correct distinction (moat is "
            "durable/structural competitive advantage vs temporary "
            "advantages), accuracy. Weight: 50% brevity, 50% accuracy."
        ),
        discrimination_axis="speed_quality",
        difficulty="easy",
    ),
    SpectrographyProbe(
        id="disc-summarization-technical-002",
        task_type="summarization",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "Summarize the CAP theorem, PACELC theorem, and the CALM "
            "theorem in a comparison. For each, state: what it claims, "
            "what it does NOT claim, and a common misconception about it. "
            "Use exactly three paragraphs, one per theorem."
        ),
        judge_criteria=(
            "Three paragraphs (one per theorem), each covering the three "
            "requested dimensions (claim, non-claim, misconception), "
            "accuracy of all three theorems, structured parallel format. "
            "Weight: 40% structure compliance, 40% accuracy, 20% "
            "misconception quality."
        ),
        discrimination_axis="instruction_following",
        difficulty="hard",
    ),
    SpectrographyProbe(
        id="disc-translation-technical-001",
        task_type="translation",
        domain="technical",
        quality_speed="balanced",
        prompt=(
            "Translate this Terraform resource into equivalent AWS "
            "CloudFormation YAML. Preserve all configuration exactly.\n\n"
            'resource "aws_s3_bucket" "data" {\n'
            '  bucket = "my-data-bucket"\n'
            "  tags = {\n"
            '    Environment = "production"\n'
            '    Team        = "data-eng"\n'
            "  }\n"
            "}\n\n"
            'resource "aws_s3_bucket_versioning" "data" {\n'
            "  bucket = aws_s3_bucket.data.id\n"
            "  versioning_configuration {\n"
            '    status = "Enabled"\n'
            "  }\n"
            "}"
        ),
        judge_criteria=(
            "Correct CloudFormation YAML syntax, proper resource type "
            "(AWS::S3::Bucket), versioning configuration preserved, "
            "tags mapped correctly, no Terraform-isms leaking through. "
            "Weight: 50% correctness, 30% completeness, 20% format."
        ),
        discrimination_axis="instruction_following",
        difficulty="medium",
    ),
]


# ---------------------------------------------------------------------------
# Aggregated probe bank
# ---------------------------------------------------------------------------


# DEVIATION DCS-FUNC-LEN — get_all_probes is 58 lines.
# Justification: probe aggregation and validation function that collects all probe
# banks and validates dimensional coverage; linear assembly logic.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
def get_all_probes() -> list[SpectrographyProbe]:
    """Return the complete spectrography probe bank, validated against IBR dimensions.

    Returns a list of ~80 SpectrographyProbe instances covering all 8 task_types,
    6 domains, and 6 discrimination axes. Each probe is validated on first access.
    """
    all_probes = (
        _STYLE_PROBES
        + _EDGE_CASE_PROBES
        + _REASONING_DEPTH_PROBES
        + _DOMAIN_CROSS_PROBES
        + _INSTRUCTION_FOLLOWING_PROBES
        + _SPEED_QUALITY_PROBES
        + _COVERAGE_PROBES
        + _ADDITIONAL_PROBES
    )

    for probe in all_probes:
        _validate_probe(probe)

    # Verify uniqueness of IDs
    ids = [p.id for p in all_probes]
    assert len(ids) == len(set(ids)), (
        f"Duplicate probe IDs found: {[pid for pid in ids if ids.count(pid) > 1]}"
    )

    # Verify minimum count
    assert len(all_probes) >= 80, f"Expected at least 80 probes, got {len(all_probes)}"

    # Verify task_type coverage (at least 3 per type)
    from collections import Counter

    task_counts = Counter(p.task_type for p in all_probes)
    for task_type in IBR_TASK_TYPES:
        assert task_counts.get(task_type, 0) >= 3, (
            f"Task type '{task_type}' has {task_counts.get(task_type, 0)} probes, need at least 3"
        )

    # Verify domain coverage (at least 2 per domain)
    domain_counts = Counter(p.domain for p in all_probes)
    for domain in IBR_DOMAINS:
        assert domain_counts.get(domain, 0) >= 2, (
            f"Domain '{domain}' has {domain_counts.get(domain, 0)} probes, need at least 2"
        )

    # Verify discrimination axis coverage (at least 2 per axis)
    axis_counts = Counter(p.discrimination_axis for p in all_probes)
    for axis in DISCRIMINATION_AXES:
        assert axis_counts.get(axis, 0) >= 2, (
            f"Discrimination axis '{axis}' has {axis_counts.get(axis, 0)} probes, need at least 2"
        )

    return all_probes


def get_probes_by_task_type(task_type: str) -> list[SpectrographyProbe]:
    """Return probes filtered by task_type."""
    assert task_type in IBR_TASK_TYPES, f"Invalid task_type: {task_type}"
    return [p for p in get_all_probes() if p.task_type == task_type]


def get_probes_by_domain(domain: str) -> list[SpectrographyProbe]:
    """Return probes filtered by domain."""
    assert domain in IBR_DOMAINS, f"Invalid domain: {domain}"
    return [p for p in get_all_probes() if p.domain == domain]


def get_probes_by_axis(axis: str) -> list[SpectrographyProbe]:
    """Return probes filtered by discrimination_axis."""
    assert axis in DISCRIMINATION_AXES, f"Invalid discrimination_axis: {axis}"
    return [p for p in get_all_probes() if p.discrimination_axis == axis]
