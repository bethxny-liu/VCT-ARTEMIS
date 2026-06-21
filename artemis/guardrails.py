"""Pre-flight checks for RAG chat: topic scope and player coverage."""

import re
from functools import lru_cache

from artemis.team.builder import ORG_ALIASES, ROLE_AGENTS, detect_org, load_players

# --- On-topic signals (Valorant / VCT / GC) ---

TOPIC_PATTERNS = [
    re.compile(r"\b(valorant|vct|game changers?|champions tour|esports|vlr\.gg)\b", re.I),
    re.compile(r"\b(vct americas|vct emea|vct pacific|vct china)\b", re.I),
    re.compile(
        r"\b(duelist|initiator|controller|sentinel|flex|entry|igl|"
        r"rating|acs|k/d|kd|clutch|fragger|lineup|roster|agent pool)\b",
        re.I,
    ),
    re.compile(
        r"\b(jett|sova|omen|viper|killjoy|cypher|chamber|raze|fade|breach|"
        r"skye|kayo|gekko|neon|iso|yoru|phoenix|astra|harbor|clove|brimstone|"
        r"deadlock|sage|reyna)\b",
        re.I,
    ),
]

PLAYER_QUESTION = re.compile(
    r"\b(is|how good|how\s+is|tell me about|stats for|who is|compare|"
    r"better than|best player|best duelist|best initiator|top player)\b",
    re.I,
)

NAME_PATTERNS = [
    re.compile(r"\bis\s+([A-Za-z0-9_\-\.]{2,20})\s+(?:the|a|an|still|really)\b", re.I),
    re.compile(r"\babout\s+([A-Za-z0-9_\-\.]{2,20})\b", re.I),
    re.compile(r"\b(?:stats for|compare)\s+([A-Za-z0-9_\-\.]{2,20})\b", re.I),
    re.compile(r"\bwho is\s+([A-Za-z0-9_\-\.]{2,20})\b", re.I),
    re.compile(r"\bhow good is\s+([A-Za-z0-9_\-\.]{2,20})\b", re.I),
]

STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "who", "what", "which", "how",
        "best", "good", "better", "top", "player", "players", "team", "teams",
        "duelist", "initiator", "controller", "sentinel", "flex", "valorant",
        "vct", "game", "meta", "stats", "rating", "compare", "than", "still",
        "really", "most", "goated", "goat", "ever", "right", "now", "today",
    }
)

OFF_TOPIC_MESSAGE = (
    "Artemis only covers Valorant Champions Tour and Game Changers stats — "
    "players, teams, agents, and lineups. Try something like "
    "\"Who's the best duelist in VCT Americas?\" or \"Build a balanced VCT team.\""
)

UNKNOWN_PLAYER_MESSAGE = (
    "I don't have current VCT stats for {name} in this dataset. "
    "They may be inactive or not in the latest scrape. "
    "Ask about an active pro from the pool, or run "
    "`python scripts/refresh_data.py` to refresh."
)

LOW_CONFIDENCE_MESSAGE = (
    "I couldn't find relevant VCT stats for that question in the indexed player pool. "
    "Try naming a specific player or team, or ask about building a lineup."
)


def normalize_handle(name: str) -> str:
    """Loose match for handles (f0rsakeN == FORSAKEN)."""
    return re.sub(r"[^a-z0-9]", "", name.lower()).replace("0", "o")


@lru_cache(maxsize=1)
def _player_names() -> dict[str, str]:
    """Normalized handle -> canonical display name."""
    out: dict[str, str] = {}
    for p in load_players():
        out[normalize_handle(p.name)] = p.name
    return out


@lru_cache(maxsize=1)
def _player_by_canonical() -> dict[str, object]:
    return {p.name: p for p in load_players()}


@lru_cache(maxsize=1)
def _org_tokens() -> set[str]:
    tokens = set(ORG_ALIASES.keys()) | {v.lower() for v in ORG_ALIASES.values()}
    tokens.update(
        {
            "sentinels", "fnatic", "loud", "paper rex", "navi", "drx",
            "100 thieves", "cloud9", "team liquid", "karmine corp",
        }
    )
    return tokens


def _agent_tokens() -> set[str]:
    agents: set[str] = set()
    for pool in ROLE_AGENTS.values():
        agents.update(pool)
    return agents


def players_mentioned(prompt: str) -> list[str]:
    norm_prompt = normalize_handle(prompt)
    tokens = {normalize_handle(t) for t in re.findall(r"[A-Za-z0-9]+", prompt)}
    found: list[str] = []
    seen: set[str] = set()
    for norm_key, canonical in sorted(_player_names().items(), key=lambda x: -len(x[0])):
        if len(norm_key) < 3:
            continue
        matched = norm_key in tokens or (
            len(norm_key) >= 7 and norm_key in norm_prompt
        )
        if matched and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    return found


def resolve_player(prompt: str):
    """First player matched in prompt, or None."""
    mentioned = players_mentioned(prompt)
    if not mentioned:
        return None
    return _player_by_canonical().get(mentioned[0])


def _extract_unknown_handle(prompt: str) -> str | None:
    """Likely player handle in a player-specific question, not in our stats."""
    if not PLAYER_QUESTION.search(prompt):
        return None

    known = _player_names()
    agents = _agent_tokens()
    orgs = _org_tokens()

    for pattern in NAME_PATTERNS:
        for match in pattern.finditer(prompt):
            raw = match.group(1)
            key = normalize_handle(raw)
            if key in {normalize_handle(w) for w in STOPWORDS}:
                continue
            if key in known or key in {normalize_handle(a) for a in agents}:
                continue
            if raw.lower() in orgs or any(raw.lower() in org for org in orgs):
                continue
            return raw
    return None


def is_on_topic(prompt: str) -> bool:
    if players_mentioned(prompt):
        return True
    if any(p.search(prompt) for p in TOPIC_PATTERNS):
        return True
    lower = prompt.lower()
    if any(org in lower for org in _org_tokens()):
        return True
    if detect_org(prompt):
        return True
    return False


GREETING = re.compile(r"^(hi|hello|hey|yo|sup|howdy)[\s!.?]*$", re.I)


def check_rag_guardrails(prompt: str) -> str | None:
    """
    Return a user-facing refusal message, or None if RAG should proceed.
    Team-builder and eval routes should call this only for RAG paths.
    """
    stripped = prompt.strip()
    if GREETING.match(stripped):
        return (
            "Hey — I'm Artemis, your VCT analyst. Ask me to build a team, "
            "rate a roster, or dig into player stats."
        )

    unknown = _extract_unknown_handle(prompt)
    if unknown and not players_mentioned(prompt):
        return UNKNOWN_PLAYER_MESSAGE.format(name=unknown)

    if not is_on_topic(prompt):
        return OFF_TOPIC_MESSAGE

    return None


def retrieval_is_weak(prompt: str, nodes: list, min_score: float = 0.35) -> bool:
    """True when retrieval likely won't support a grounded answer."""
    if not nodes:
        return True
    if players_mentioned(prompt):
        return False
    if detect_org(prompt):
        return False
    best = max(getattr(n, "score", 0) or 0 for n in nodes)
    return best < min_score
