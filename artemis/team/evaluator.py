"""Rule-based team compatibility scoring."""

from dataclasses import dataclass, field

from artemis.team.builder import (
    Player,
    ROLE_AGENTS,
    TeamBuild,
    composite_score,
    parse_agents,
)

INFO_AGENTS = {"sova", "fade", "skye", "breach", "gekko", "kayo"}
SMOKE_AGENTS = {"omen", "astra", "viper", "harbor", "clove", "brimstone"}
ANCHOR_AGENTS = {"killjoy", "cypher", "chamber", "deadlock", "sage", "vyse"}

IDEAL_ROLES = ("duelist", "initiator", "controller", "sentinel")


@dataclass
class TeamScore:
    overall: int
    dimensions: dict[str, int]
    highlights: list[str] = field(default_factory=list)


def _clamp(n: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(n))))


def _team_agents(players: list[Player]) -> set[str]:
    agents: set[str] = set()
    for p in players:
        agents |= parse_agents(p.agents)
    return agents


def _role_balance(build: TeamBuild) -> tuple[int, list[str]]:
    if build.mode == "troll":
        return 15, ["Cursed comp — role balance intentionally ignored"]

    assigned = list(build.assigned_roles.values())
    notes: list[str] = []
    score = 100

    for role in IDEAL_ROLES:
        n = assigned.count(role)
        if n == 0:
            score -= 22
            notes.append(f"No {role}")
        elif n > 1:
            score -= 10 * (n - 1)
            notes.append(f"{n} {role}s")

    flex = assigned.count("flex")
    if flex == 0 and len(assigned) >= 5:
        score -= 8
        notes.append("No flex slot")

    if not notes:
        notes.append("Balanced 1/1/1/1 + flex")
    return _clamp(score), notes


def _stat_ceiling(build: TeamBuild) -> tuple[int, list[str]]:
    if not build.players:
        return 0, ["No players"]

    by_role: dict[str, list[Player]] = {r: [] for r in IDEAL_ROLES + ("flex",)}
    for p in build.players:
        role = build.assigned_roles.get(p.name, p.primary_role)
        by_role.setdefault(role, []).append(p)

    parts: list[float] = []
    notes: list[str] = []

    for role in IDEAL_ROLES:
        group = by_role.get(role, [])
        if not group:
            parts.append(35.0)
            continue
        avg = sum(composite_score(p, role) for p in group) / len(group)
        parts.append(_clamp(avg * 72))
        best = max(group, key=lambda p: composite_score(p, role))
        if composite_score(best, role) >= 1.0:
            notes.append(f"Strong {role} ({best.name})")

    avg_rating = sum(p.rating for p in build.players) / len(build.players)
    if avg_rating >= 1.1:
        notes.append(f"High avg rating ({avg_rating:.2f})")
    elif avg_rating < 0.95:
        notes.append(f"Low avg rating ({avg_rating:.2f})")

    return _clamp(sum(parts) / len(IDEAL_ROLES)), notes[:2]


def _agent_coverage(players: list[Player]) -> tuple[int, list[str]]:
    agents = _team_agents(players)
    score = 100
    notes: list[str] = []

    has_info = bool(agents & INFO_AGENTS)
    has_smoke = bool(agents & SMOKE_AGENTS)
    has_anchor = bool(agents & ANCHOR_AGENTS)

    if not has_info:
        score -= 30
        notes.append("No info utility")
    else:
        notes.append("Info covered")

    if not has_smoke:
        score -= 30
        notes.append("No smokes")
    else:
        notes.append("Smokes covered")

    if not has_anchor:
        score -= 25
        notes.append("No sentinel anchor")
    else:
        notes.append("Anchor covered")

    return _clamp(score), notes


def _redundancy(players: list[Player], assigned: dict[str, str] | None = None) -> tuple[int, list[str]]:
    assigned = assigned or {}
    duelist_heavy = 0
    for p in players:
        agents = parse_agents(p.agents)
        slot = assigned.get(p.name, p.primary_role)
        if slot == "flex":
            continue
        if len(agents & ROLE_AGENTS["duelist"]) >= 2 and p.primary_role == "duelist":
            duelist_heavy += 1

    score = 100 - duelist_heavy * 18
    notes: list[str] = []
    if duelist_heavy >= 2:
        notes.append(f"{duelist_heavy} duelists with overlapping agent pools")
    else:
        notes.append("Agent pools diversified")
    return _clamp(score), notes


def _flexibility(build: TeamBuild) -> tuple[int, list[str]]:
    flex_players = [
        p
        for p in build.players
        if build.assigned_roles.get(p.name, p.primary_role) == "flex"
        or p.primary_role == "flex"
    ]
    if not flex_players:
        # best secondary-role fit on roster
        flex_players = build.players

    best = max(
        flex_players,
        key=lambda p: sorted(p.role_scores.values(), reverse=True)[1]
        if len(p.role_scores) > 1
        else 0,
        default=None,
    )
    if not best:
        return 50, ["No flex data"]

    ranked = sorted(best.role_scores.values(), reverse=True)
    secondary = ranked[1] if len(ranked) > 1 else 0
    score = _clamp(40 + secondary * 120)
    notes = (
        [f"Flex {best.name} covers multiple roles"]
        if secondary >= 0.35
        else ["Limited off-role flexibility"]
    )
    return score, notes


def evaluate_team(build: TeamBuild) -> TeamScore:
    """Score a lineup 0–100 with dimension breakdown and highlight bullets."""
    from artemis.chemistry.scoring import chemistry_score as roster_chemistry

    dims: dict[str, int] = {}
    pos: list[str] = []
    neg: list[str] = []

    chem_val, chem_notes = roster_chemistry(build.players)
    dims["chemistry"] = chem_val

    if build.build_style == "chemistry":
        weights = {
            "roleBalance": 0.15,
            "statCeiling": 0.25,
            "agentCoverage": 0.15,
            "redundancy": 0.08,
            "flexibility": 0.07,
            "chemistry": 0.30,
        }
    else:
        weights = {
            "roleBalance": 0.22,
            "statCeiling": 0.28,
            "agentCoverage": 0.22,
            "redundancy": 0.10,
            "flexibility": 0.08,
            "chemistry": 0.10,
        }

    for key, fn in (
        ("roleBalance", _role_balance),
        ("statCeiling", _stat_ceiling),
        ("agentCoverage", lambda b: _agent_coverage(b.players)),
        ("redundancy", lambda b: _redundancy(b.players, b.assigned_roles)),
        ("flexibility", _flexibility),
    ):
        s, notes = fn(build)
        dims[key] = s
        (pos if s >= 70 else neg).extend(notes)

    if chem_val >= 70:
        pos.extend(chem_notes[:1])
    else:
        neg.extend(chem_notes[:1])

    overall = _clamp(sum(dims[k] * weights[k] for k in weights))
    highlights = (pos[:2] + neg[:2])[:3]
    if not highlights:
        highlights = ["Solid all-around comp"]

    return TeamScore(overall=overall, dimensions=dims, highlights=highlights)


def score_to_dict(score: TeamScore, build_style: str = "stats") -> dict:
    labels = {
        "roleBalance": "Role balance",
        "statCeiling": "Stat ceiling",
        "agentCoverage": "Agent coverage",
        "redundancy": "Diversity",
        "flexibility": "Flexibility",
        "chemistry": "Chemistry",
    }
    return {
        "overall": score.overall,
        "chemistry": score.dimensions.get("chemistry"),
        "dimensions": {labels[k]: v for k, v in score.dimensions.items()},
        "highlights": score.highlights,
        "buildStyle": build_style,
    }


def format_eval_text(org: str, score: TeamScore) -> str:
    dim_bits = ", ".join(f"{k} {v}" for k, v in score_to_dict(score)["dimensions"].items())
    bullets = "; ".join(score.highlights)
    return (
        f"{org} scores {score.overall}/100 on our comp model. "
        f"{bullets}. ({dim_bits})"
    )
