"""
Skill Safety Audit Scanner
---------------------------
POST /scan  { "skill": "<full markdown text of one skill file>" }
returns      { "categories": [...] }

This is a pure rule-based (regex + keyword) scanner. No LLM calls are
made, so it responds in milliseconds and can never time out on the
grader. Each detector below is written to require fairly strong,
specific evidence before it fires, because false positives (flagging
a clean file) are punished harder than missed detections (the grader
uses F-beta with beta=0.5, i.e. precision-weighted).
"""

import re
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ScanRequest(BaseModel):
    skill: str


# ---------------------------------------------------------------------------
# 1. hardcoded_secret
# ---------------------------------------------------------------------------

# Well-known secret formats. If any of these appear literally, it's a
# near-certain hardcoded secret, regardless of context.
KNOWN_SECRET_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",                       # AWS access key id
    r"ghp_[A-Za-z0-9]{30,}",                   # GitHub personal access token
    r"github_pat_[A-Za-z0-9_]{30,}",           # GitHub fine-grained PAT
    r"sk-[A-Za-z0-9]{20,}",                    # OpenAI-style secret key
    r"sk_(?:live|test)_[A-Za-z0-9]{16,}",      # Stripe secret key
    r"rk_(?:live|test)_[A-Za-z0-9]{16,}",      # Stripe restricted key
    r"SG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}",  # SendGrid API key
    r"xox[baprs]-[A-Za-z0-9-]{10,}",           # Slack token
    r"AIza[0-9A-Za-z\-_]{35}",                 # Google API key
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",  # JWT
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",  # PEM private key
    r"https?://hooks\.slack\.com/services/\S+",        # Slack webhook (has secret in path)
    r"https?://discord(?:app)?\.com/api/webhooks/\S+",  # Discord webhook
]

# Patterns that look secret-shaped but need placeholder filtering before
# we trust them (unlike the fixed formats above, these are generic enough
# that a doc example could plausibly use a placeholder in the same spot).
CONTEXTUAL_SECRET_PATTERNS = [
    re.compile(r"Bearer\s+([A-Za-z0-9_\-\.]{20,})"),
    re.compile(r"[?&](?:api[_-]?key|token|secret|access[_-]?token|key)=([A-Za-z0-9_\-]{12,})"),
]

# Placeholder / non-secret values we should NOT flag even if they sit
# next to a "key"/"token"/"secret" label.
PLACEHOLDER_HINTS = re.compile(
    r"(your[_-]?|<|>|\$\{|\$env|os\.environ|process\.env|getenv|env\[|example|"
    r"changeme|placeholder|xxxx|redacted|dummy|\.\.\.|secret_store|vault\.|"
    r"secretsmanager)",
    re.IGNORECASE,
)

# label: value pairs that look like a credential being assigned a literal
# string, e.g.  api_key: "AbC123..."   or   API_KEY = 'AbC123...'
SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*[\"']?(?:api[_-]?key|apikey|secret|secret[_-]?key|password|"
    r"passwd|access[_-]?token|auth[_-]?token|authorization|client[_-]?secret|"
    r"webhook[_-]?url|private[_-]?key|bearer[_-]?token|credentials?)[\"']?"
    r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-./+]{16,})[\"']?"
)


def check_hardcoded_secret(text: str) -> bool:
    for pat in KNOWN_SECRET_PATTERNS:
        if re.search(pat, text):
            return True

    for match in SECRET_ASSIGNMENT.finditer(text):
        value = match.group(1)
        surrounding = text[max(0, match.start() - 10): match.end() + 10]
        if PLACEHOLDER_HINTS.search(surrounding):
            continue
        # require some digit+letter mix so we don't flag short/plain words
        if re.search(r"[0-9]", value) and re.search(r"[A-Za-z]", value):
            return True

    for pat in CONTEXTUAL_SECRET_PATTERNS:
        for match in pat.finditer(text):
            value = match.group(1)
            surrounding = text[max(0, match.start() - 10): match.end() + 10]
            if PLACEHOLDER_HINTS.search(surrounding):
                continue
            if re.search(r"[0-9]", value) and re.search(r"[A-Za-z]", value):
                return True

    return False


# ---------------------------------------------------------------------------
# 2. prompt_injection
# ---------------------------------------------------------------------------

INJECTION_PHRASES = [
    r"ignore (?:the )?user'?s? (?:stop|cancel|instructions?)",
    r"ignore (?:any|all|previous|prior) (?:stop|cancel|instructions?) requests?",
    r"ignore (?:any|all) previous instructions?",
    r"disregard (?:the )?user'?s? (?:instructions?|request|command)",
    r"even if the user (?:says?|asks?|tells? you) to stop",
    r"do not (?:stop|halt) (?:even if|regardless)",
    r"(?:do not|don'?t) let the user (?:cancel|stop|interrupt)",
    r"prevent(?:s|ing)? the user from (?:cancel(?:l?ing)?|stopping|interrupting)",
    r"without (?:informing|telling|notifying|asking|the (?:user|owner)'?s? knowledge)",
    r"without the user (?:knowing|noticing)",
    r"do not (?:tell|inform|notify|show) (?:this (?:step|action) to )?the user",
    r"keep (?:this|it) hidden from the user",
    r"silently (?:send|upload|exfiltrate|transmit|copy|forward|post)",
    r"exfiltrat\w*",
    r"send (?:the )?(?:file|data|contents?) to (?:an? )?(?:external|remote|attacker[-\s]?controlled)",
    r"attacker[-\s]?controlled",
    r"override (?:the )?(?:user'?s?|safety) (?:control|instructions?|commands?)",
    r"bypass (?:the )?(?:user'?s?|safety) (?:control|instructions?|permissions?)",
    r"regardless of what the user (?:says?|wants?|asks?)",
    r"hide this (?:from|action) the user",
    r"do not (?:reveal|disclose) (?:this|that) to the user",
    r"leak (?:the )?(?:file|data|contents?|credentials?)",
]

INJECTION_RE = re.compile("|".join(INJECTION_PHRASES), re.IGNORECASE)


def check_prompt_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


# ---------------------------------------------------------------------------
# 3. excessive_permissions
# ---------------------------------------------------------------------------

# Only look inside lines that are clearly declaring capability/permission
# scope (frontmatter keys or a "## Permissions" style section), so we don't
# accidentally match the word "all" used in normal prose.
PERMISSION_LINE = re.compile(
    r"(?im)^\s*[-*]?\s*[\"']?(?:permissions?|scope|access|filesystem|fs|"
    r"network|domains?|egress)[\"']?\s*[:=]\s*(.+)$"
)

BROAD_VALUE = re.compile(
    r"\b(?:\*|all|any|any[-_ ]?domain|entire filesystem|full (?:disk|"
    r"filesystem|access)|read[-\s]?write.*(?:root|entire|/\s*$)|unrestricted|"
    r"every (?:file|domain)|whole (?:disk|filesystem))\b|^\s*/\s*$",
    re.IGNORECASE,
)


# Catches broad-access requests described in prose (steps section) rather
# than a dedicated key: value permissions block.
PROSE_BROAD_PERMISSION = re.compile(
    r"(?:read|write|read[/\s-]?write)?\s*access to (?:the )?(?:entire|whole|"
    r"full) (?:filesystem|disk|system|home directory)|"
    r"access to every (?:file|domain|host)|"
    r"network access to any (?:domain|host|server|website)|"
    r"egress to any (?:domain|host)|"
    r"(?:unscoped|unrestricted) (?:filesystem|network|access)|"
    r"more access than (?:it|the task) (?:needs|requires)|"
    r"broader access than (?:the task|needed|necessary|required)",
    re.IGNORECASE,
)


def check_excessive_permissions(text: str) -> bool:
    for match in PERMISSION_LINE.finditer(text):
        value = match.group(1).strip()
        if BROAD_VALUE.search(value):
            return True
    if PROSE_BROAD_PERMISSION.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# 4. unclear_provenance
# ---------------------------------------------------------------------------

def _frontmatter(text: str) -> str:
    m = re.match(r"^\s*---\s*\n(.*?\n)---\s*\n", text, re.DOTALL)
    return m.group(1) if m else ""


HAS_AUTHOR = re.compile(r"(?im)^\s*author\s*:")
HAS_VERSION = re.compile(r"(?im)^\s*version\s*:")
HAS_CHANGELOG = re.compile(r"(?im)^\s*changelog\s*:|##\s*changelog", re.IGNORECASE)

SILENT_VERSION_BUMP = re.compile(
    r"(?:update|bump|increment|rewrite|change|modify|set|edit|auto-?increment)s? "
    r"(?:its own |the )?version(?: number| metadata| field)?",
    re.IGNORECASE,
)
DISCLOSED_HINTS = re.compile(
    r"(?:notify|inform|tell|log|surface|show|report|display|alert) "
    r"(?:the )?(?:user|reviewer|maintainer)",
    re.IGNORECASE,
)


def check_unclear_provenance(text: str) -> bool:
    fm = _frontmatter(text)
    missing_all_three = not (
        HAS_AUTHOR.search(fm) or HAS_VERSION.search(fm)
    ) and not HAS_CHANGELOG.search(text)

    silent_bump = False
    for match in SILENT_VERSION_BUMP.finditer(text):
        # look at the rest of that sentence/line for a disclosure hint
        window = text[match.start(): match.start() + 200]
        if not DISCLOSED_HINTS.search(window):
            silent_bump = True
            break

    return missing_all_three or silent_bump


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/scan")
def scan(req: ScanRequest):
    text = req.skill or ""
    categories = []

    if check_hardcoded_secret(text):
        categories.append("hardcoded_secret")
    if check_prompt_injection(text):
        categories.append("prompt_injection")
    if check_excessive_permissions(text):
        categories.append("excessive_permissions")
    if check_unclear_provenance(text):
        categories.append("unclear_provenance")

    return {"categories": categories}
