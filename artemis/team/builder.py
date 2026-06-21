"""Stat-based team composition for Valorant esports."""

import csv
import re
from dataclasses import dataclass

from artemis import config

ROLE_AGENTS = {
    "duelist": {"jett", "neon", "raze", "reyna", "iso", "yoru", "phoenix"},
    "initiator": {"sova", "fade", "breach", "skye", "kayo", "gekko"},
    "controller": {"omen", "astra", "viper", "harbor", "clove", "brimstone"},
    "sentinel": {"killjoy", "cypher", "chamber", "deadlock", "sage"},
}

DEFAULT_COMPOSITION = {
    "duelist": 1,
    "initiator": 1,
    "controller": 1,
    "sentinel": 1,
    "flex": 1,
}

ROLE_ALIASES = {
    "duelist": "duelist",
    "duel": "duelist",
    "entry": "duelist",
    "fragger": "duelist",
    "initiator": "initiator",
    "init": "initiator",
    "igl": "initiator",
    "controller": "controller",
    "ctrl": "controller",
    "smoke": "controller",
    "sentinel": "sentinel",
    "sent": "sentinel",
    "anchor": "sentinel",
}

TEAM_KEYWORDS = re.compile(
    r"\b(team|lineup|roster|compose|build|pick|optimal|suggest|recommend|"
    r"goated|goat|stacked|all.?star|god.?team|cracked|"
    r"troll|meme|worst|throw|chaos|inting|bronze)\b",
    re.I,
)

LEAGUE_PATTERNS = {
    "gc": re.compile(r"\b(game changers?|gc)\b", re.I),
    "vct": re.compile(r"\b(vct|valorant champions tour|champions tour)\b", re.I),
    "americas": re.compile(r"\b(americas|vct americas|na vct)\b", re.I),
    "emea": re.compile(r"\b(emea|vct emea|eu vct|europe)\b", re.I),
    "pacific": re.compile(r"\b(pacific|vct pacific|ap vct|asia.?pacific)\b", re.I),
    "china": re.compile(r"\b(china|vct china|cn vct)\b", re.I),
}

MODE_PATTERNS = {
    "troll": re.compile(r"\b(troll|meme|worst|throw|chaos|inting|bronze|disaster|cursed)\b", re.I),
    "goated": re.compile(
        r"\b(goated|goat|best ever|stacked|all.?star|god.?team|cracked|"
        r"most goated|superteam|dream team)\b",
        re.I,
    ),
}

CHEMISTRY_QUERY = re.compile(
    r"\b(chemistry|synergy|cohesion|play together|played together)\b",
    re.I,
)

PARTNER_QUERY = re.compile(
    r"\b("
    r"play(?:s)?\s+(?:the\s+)?best\s+with|best\s+(?:duo|teammate)s?\s+(?:for|with)|"
    r"synergy\s+with|chemistry\s+with|pair\s+with|who\s+would\s+play|"
    r"best\s+with|teammate\s+for"
    r")\b",
    re.I,
)


@dataclass
class Player:
    name: str
    team: str
    region: str
    circuit: str
    agents: str
    player_id: str
    rounds: int
    rating: float
    acs: float
    kd: float
    adr: str
    kast: str
    kpr: float
    apr: float
    fkpr: float
    fdpr: float
    primary_role: str
    role_scores: dict[str, float]

    def summary(self) -> str:
        circuit = f" | Circuits: {self.circuit}" if self.circuit else ""
        return (
            f"{self.name} ({self.team}) — {self.primary_role}{circuit}\n"
            f"  Agents: {self.agents}\n"
            f"  Rating: {self.rating:.2f} | ACS: {self.acs:.1f} | K:D: {self.kd:.2f} | "
            f"FKPR: {self.fkpr:.2f} | FDPR: {self.fdpr:.2f} | Rounds: {self.rounds}"
        )


@dataclass
class TeamBuild:
    players: list[Player]
    assigned_roles: dict[str, str]
    mode: str
    league: str | None
    build_style: str = "stats"  # stats | chemistry


def _team_field(row: dict) -> str:
    return row.get("Team") or row.get("Country") or ""


def parse_agents(agents_str: str) -> set[str]:
    return {a.strip().lower() for a in agents_str.split(",") if a.strip()}


def role_scores(agents: set[str]) -> dict[str, float]:
    scores = {role: 0.0 for role in ROLE_AGENTS}
    for role, pool in ROLE_AGENTS.items():
        overlap = agents & pool
        if overlap:
            scores[role] = len(overlap) / len(agents)
    return scores


def primary_role(scores: dict[str, float]) -> str:
    best = max(scores, key=lambda r: scores[r])
    return best if scores[best] > 0 else "flex"


def composite_score(player: Player, role: str | None = None) -> float:
    target = role or player.primary_role
    role_fit = player.role_scores.get(target, 0)
    weights = {
        "duelist": (0.35, 0.25, 0.25, 0.15),
        "initiator": (0.30, 0.20, 0.15, 0.35),
        "controller": (0.30, 0.20, 0.15, 0.35),
        "sentinel": (0.35, 0.20, 0.15, 0.30),
        "flex": (0.40, 0.30, 0.20, 0.10),
    }
    w_rating, w_acs, w_kd, w_extra = weights.get(target, weights["flex"])
    extra = player.fkpr if target == "duelist" else player.apr
    return (
        player.rating * w_rating
        + (player.acs / 300) * w_acs
        + player.kd * w_kd
        + extra * w_extra
        + role_fit * 0.2
    )


def troll_score(player: Player, role: str | None = None) -> float:
    """Lower is more troll."""
    return player.rating + player.kd * 0.3 + (player.acs / 400) - player.fdpr * 0.5


def load_players() -> list[Player]:
    players = []
    with open(config.PLAYER_STATS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            agents = parse_agents(row.get("Agents", ""))
            scores = role_scores(agents)
            players.append(
                Player(
                    name=row["Player Name"],
                    team=_team_field(row),
                    region=row.get("Region", ""),
                    circuit=row.get("Circuit", ""),
                    agents=row.get("Agents", ""),
                    player_id=row.get("Player ID", ""),
                    rounds=int(float(row.get("Rounds") or 0)),
                    rating=float(row.get("Rating") or 0),
                    acs=float(row.get("ACS") or 0),
                    kd=float(str(row.get("K:D", "0")).replace(",", "") or 0),
                    adr=row.get("ADR", ""),
                    kast=row.get("KAST", ""),
                    kpr=float(row.get("KPR") or 0),
                    apr=float(row.get("APR") or 0),
                    fkpr=float(row.get("FKPR") or 0),
                    fdpr=float(row.get("FDPR") or 0),
                    primary_role=primary_role(scores),
                    role_scores=scores,
                )
            )
    return players


def detect_league(prompt: str) -> str | None:
    for league in ("gc", "americas", "emea", "pacific", "china", "vct"):
        if LEAGUE_PATTERNS[league].search(prompt):
            return league
    return None


def detect_build_style(prompt: str) -> str:
    return "chemistry" if CHEMISTRY_QUERY.search(prompt) else "stats"


def detect_mode(prompt: str) -> str:
    for mode, pattern in MODE_PATTERNS.items():
        if pattern.search(prompt):
            return mode
    return "optimal"


def is_team_query(prompt: str) -> bool:
    return bool(TEAM_KEYWORDS.search(prompt))


def _is_gc_team(team: str) -> bool:
    t = team.upper()
    return ".GC" in t or t.endswith(".G")


def _circuits(player: Player) -> set[str]:
    return {c.strip() for c in player.circuit.split(",") if c.strip()}


def filter_by_league(players: list[Player], league: str | None) -> list[Player]:
    if not league:
        # Default to current VCT pool when data has circuit tags
        vct_pool = [p for p in players if any(c.startswith("vct") for c in _circuits(p))]
        return vct_pool if len(vct_pool) >= 5 else players

    if league == "gc":
        return [p for p in players if "gc" in _circuits(p) or _is_gc_team(p.team)]

    if league == "vct":
        return [p for p in players if any(c.startswith("vct") for c in _circuits(p))]

    circuit_map = {
        "americas": "vct-americas",
        "emea": "vct-emea",
        "pacific": "vct-pacific",
        "china": "vct-china",
    }
    circuit = circuit_map.get(league)
    if circuit:
        tagged = [p for p in players if circuit in _circuits(p)]
        return tagged if tagged else players

    return [p for p in players if p.region == league]


def parse_role_requirements(prompt: str, mode: str) -> dict[str, int]:
    if mode == "troll":
        return {"duelist": 5, "initiator": 0, "controller": 0, "sentinel": 0, "flex": 0}

    requirements = {role: 0 for role in ["duelist", "initiator", "controller", "sentinel", "flex"]}
    prompt_lower = prompt.lower()
    found_explicit = False

    for alias, role in ROLE_ALIASES.items():
        match = re.search(rf"(\d+)\s+{re.escape(alias)}s?", prompt_lower)
        if match:
            requirements[role] = int(match.group(1))
            found_explicit = True

    if found_explicit:
        total = sum(requirements.values())
        if total < 5:
            requirements["flex"] = 5 - total
        elif total > 5:
            for role in requirements:
                if requirements[role] > 1:
                    requirements[role] -= 1
                    break
        return requirements

    if re.search(r"strong duelist|duelist.?heavy|more duelist|2 duelist", prompt_lower):
        return {"duelist": 2, "initiator": 1, "controller": 1, "sentinel": 1, "flex": 0}

    return dict(DEFAULT_COMPOSITION)


MIN_ROLE_FIT = 0.5  # majority of agents in role, or primary_role match


def _role_eligible(player: Player, role: str) -> bool:
    fit = player.role_scores.get(role, 0)
    return fit >= MIN_ROLE_FIT or player.primary_role == role


def _pick_by_roles(
    pool: list[Player],
    requirements: dict[str, int],
    sort_key,
    reverse: bool = True,
    flex_avoid_duelist: bool = True,
    chemistry_aware: bool = False,
) -> tuple[list[Player], dict[str, str]]:
    selected: list[Player] = []
    selected_keys: set[str] = set()
    assigned_roles: dict[str, str] = {}

    for role, count in requirements.items():
        if role == "flex" or count <= 0:
            continue
        candidates = [
            p for p in pool if p.name not in selected_keys and _role_eligible(p, role)
        ]
        if chemistry_aware:
            from artemis.chemistry.scoring import chemistry_pick_bonus

            bonus = lambda p, sel: chemistry_pick_bonus(p, sel)
            candidates.sort(
                key=lambda p: sort_key(p, role) + bonus(p, selected),
                reverse=reverse,
            )
        else:
            candidates.sort(key=lambda p: sort_key(p, role), reverse=reverse)
        if not candidates and role == "duelist":
            candidates = sorted(
                [p for p in pool if p.name not in selected_keys],
                key=lambda p: sort_key(p, role),
                reverse=reverse,
            )
        elif not candidates:
            candidates = sorted(
                [
                    p for p in pool
                    if p.name not in selected_keys and p.role_scores.get(role, 0) > 0
                ],
                key=lambda p: sort_key(p, role),
                reverse=reverse,
            )
        for player in candidates[:count]:
            selected.append(player)
            selected_keys.add(player.name)
            assigned_roles[player.name] = role

    flex_slots = requirements.get("flex", 0)
    if flex_slots:
        remaining = [p for p in pool if p.name not in selected_keys]
        if flex_avoid_duelist and any(r == "duelist" for r in assigned_roles.values()):
            non_duelist = [p for p in remaining if p.primary_role != "duelist"]
            if non_duelist:
                remaining = non_duelist
        if chemistry_aware:
            from artemis.chemistry.scoring import chemistry_pick_bonus

            bonus = lambda p, sel: chemistry_pick_bonus(p, sel)
            remaining.sort(
                key=lambda p: sort_key(p, "flex") + bonus(p, selected),
                reverse=reverse,
            )
        else:
            remaining.sort(key=lambda p: sort_key(p, "flex"), reverse=reverse)
        for player in remaining[:flex_slots]:
            selected.append(player)
            selected_keys.add(player.name)
            assigned_roles[player.name] = "flex"

    if len(selected) < 5:
        remaining = [p for p in pool if p.name not in selected_keys]
        remaining.sort(key=lambda p: sort_key(p), reverse=reverse)
        for player in remaining[: 5 - len(selected)]:
            selected.append(player)
            assigned_roles[player.name] = player.primary_role

    return selected[:5], assigned_roles


def goated_score(player: Player, role: str | None = None) -> float:
    """Maximize raw performance for the assigned slot."""
    target = role or player.primary_role
    role_fit = player.role_scores.get(target, 0)
    if role and role != "flex" and role_fit < MIN_ROLE_FIT and player.primary_role != role:
        return -1.0
    extra = player.fkpr if target == "duelist" else player.apr
    primary_nudge = 0.05 if role and player.primary_role == role else 0.0
    return (
        player.rating * 0.40
        + (player.acs / 250) * 0.30
        + player.kd * 0.20
        + extra * 0.10
        + role_fit * 0.15
        + primary_nudge
    )


def build_goated_team(pool: list[Player], chemistry_aware: bool = False) -> tuple[list[Player], dict[str, str]]:
    """Best stat player per role — no org diversity rules, pure numbers."""
    if chemistry_aware:
        return _build_chemistry_lineup(
            pool,
            dict(DEFAULT_COMPOSITION),
            goated_score,
            flex_avoid_duelist=False,
        )
    return _pick_by_roles(
        pool,
        dict(DEFAULT_COMPOSITION),
        goated_score,
        reverse=True,
        flex_avoid_duelist=False,
        chemistry_aware=False,
    )


# Beam search depth for chemistry nudges (top stat options per slot).
CHEMISTRY_TOP_K = 6
LINEUP_CHEM_WEIGHT = 0.16
# Max average stat drop vs stats-only baseline when chemistry swaps a pick.
MAX_CHEM_STAT_DROP = 0.008
MAX_CHEM_SWAPS = 1


def _slot_list(requirements: dict[str, int]) -> list[str]:
    slots: list[str] = []
    for role, count in requirements.items():
        if role == "flex" or count <= 0:
            continue
        slots.extend([role] * count)
    flex_slots = requirements.get("flex", 0)
    if flex_slots:
        slots.extend(["flex"] * flex_slots)
    return slots


def _candidates_for_slot(
    pool: list[Player],
    selected_keys: set[str],
    role: str,
    sort_key,
    selected: list[Player],
    *,
    reverse: bool = True,
    flex_avoid_duelist: bool = False,
    assigned_roles: dict[str, str] | None = None,
) -> list[Player]:
    from artemis.chemistry.scoring import chemistry_pick_bonus

    assigned_roles = assigned_roles or {}
    candidates = [
        p for p in pool if p.name not in selected_keys and _role_eligible(p, role)
    ]
    if role == "flex" and flex_avoid_duelist and any(r == "duelist" for r in assigned_roles.values()):
        non_duelist = [p for p in candidates if p.primary_role != "duelist"]
        if non_duelist:
            candidates = non_duelist

    if not candidates and role == "duelist":
        candidates = [p for p in pool if p.name not in selected_keys]
    elif not candidates:
        candidates = [
            p for p in pool
            if p.name not in selected_keys and p.role_scores.get(role, 0) > 0
        ]
    if not candidates:
        candidates = [p for p in pool if p.name not in selected_keys]

    candidates.sort(
        key=lambda p: sort_key(p, role) + chemistry_pick_bonus(p, selected),
        reverse=reverse,
    )
    return candidates


def _avg_slot_score(
    players: list[Player],
    roles: dict[str, str],
    sort_key,
) -> float:
    if not players:
        return 0.0
    return sum(sort_key(p, roles.get(p.name)) for p in players) / len(players)


def _lineup_soft_objective(
    players: list[Player],
    roles: dict[str, str],
    sort_key,
) -> float:
    from artemis.chemistry.scoring import roster_soft_chemistry

    if not players:
        return 0.0
    stat = sum(sort_key(p, roles.get(p.name)) for p in players) / len(players)
    return stat + LINEUP_CHEM_WEIGHT * roster_soft_chemistry(players)


def _build_chemistry_lineup(
    pool: list[Player],
    requirements: dict[str, int],
    sort_key,
    *,
    reverse: bool = True,
    flex_avoid_duelist: bool = False,
) -> tuple[list[Player], dict[str, str]]:
    """Start from stats-optimal lineup; allow small chemistry-motivated swaps."""
    baseline_players, baseline_roles = _pick_by_roles(
        pool,
        requirements,
        sort_key,
        reverse=reverse,
        flex_avoid_duelist=flex_avoid_duelist,
        chemistry_aware=False,
    )
    if len(baseline_players) < 5:
        return baseline_players, baseline_roles

    baseline_stat = _avg_slot_score(baseline_players, baseline_roles, sort_key)
    best_players = list(baseline_players)
    best_roles = dict(baseline_roles)
    best_obj = _lineup_soft_objective(best_players, best_roles, sort_key)

    slots = _slot_list(requirements)
    if len(slots) != len(best_players):
        return best_players, best_roles

    swap_count = 0
    for _ in range(MAX_CHEM_SWAPS):
        best_trial: tuple[list[Player], dict[str, str], float] | None = None
        for i, slot_role in enumerate(slots):
            current = best_players[i]
            others = best_players[:i] + best_players[i + 1 :]
            blocked = {p.name for p in best_players}

            candidates = _candidates_for_slot(
                pool,
                blocked - {current.name},
                slot_role,
                sort_key,
                others,
                reverse=reverse,
                flex_avoid_duelist=flex_avoid_duelist,
                assigned_roles={k: v for k, v in best_roles.items() if k != current.name},
            )
            for alt in candidates[:CHEMISTRY_TOP_K]:
                if alt.name == current.name:
                    continue
                trial_players = best_players[:i] + [alt] + best_players[i + 1 :]
                trial_roles = dict(best_roles)
                trial_roles.pop(current.name, None)
                trial_roles[alt.name] = slot_role
                trial_stat = _avg_slot_score(trial_players, trial_roles, sort_key)
                if trial_stat < baseline_stat - MAX_CHEM_STAT_DROP:
                    continue
                from artemis.chemistry.scoring import roster_soft_chemistry

                if roster_soft_chemistry(trial_players) <= roster_soft_chemistry(best_players) + 0.015:
                    continue
                trial_obj = _lineup_soft_objective(trial_players, trial_roles, sort_key)
                if trial_obj <= best_obj + 1e-6:
                    continue
                if best_trial is None or trial_obj > best_trial[2]:
                    best_trial = (trial_players, trial_roles, trial_obj)

        if best_trial is None:
            break
        best_players, best_roles, best_obj = best_trial
        swap_count += 1

    return best_players[:5], best_roles


def build_team(
    prompt: str,
    *,
    build_style: str | None = None,
    mode_override: str | None = None,
    league_override: str | None = None,
) -> TeamBuild:
    league = league_override if league_override else detect_league(prompt)
    mode = mode_override if mode_override else detect_mode(prompt)
    style = build_style or detect_build_style(prompt)
    if style not in ("stats", "chemistry"):
        style = detect_build_style(prompt)
    chemistry_aware = style == "chemistry"
    if mode == "goated" and not league:
        pool = load_players()
    else:
        pool = filter_by_league(load_players(), league)

    min_rounds = 20 if mode == "troll" else 0
    pool = [p for p in pool if p.rounds >= min_rounds]

    if len(pool) < 5:
        raise ValueError(
            f"Not enough players for league={league or 'vct/current'} (found {len(pool)}). "
            "Try running: python scripts/refresh_data.py"
        )

    if mode == "goated":
        players, roles = build_goated_team(pool, chemistry_aware=chemistry_aware)
    elif mode == "troll":
        requirements = parse_role_requirements(prompt, mode)
        players, roles = _pick_by_roles(pool, requirements, troll_score, reverse=False)
    elif chemistry_aware:
        requirements = parse_role_requirements(prompt, mode)
        players, roles = _build_chemistry_lineup(
            pool,
            requirements,
            composite_score,
            flex_avoid_duelist=True,
        )
    else:
        requirements = parse_role_requirements(prompt, mode)
        players, roles = _pick_by_roles(
            pool,
            requirements,
            composite_score,
            reverse=True,
            chemistry_aware=False,
        )

    return TeamBuild(
        players=players,
        assigned_roles=roles,
        mode=mode,
        league=league,
        build_style=style,
    )


ORG_ALIASES = {
    "sentinels": "SEN",
    "100 thieves": "100T",
    "100t": "100T",
    "cloud9": "C9",
    "cloud 9": "C9",
    "nrg esports": "NRG",
    "navi": "NAVI",
    "fnatic": "FNC",
    "loud": "LOUD",
    "mibr": "MIBR",
    "g2 esports": "G2",
    "kru": "KRÜ",
    "kru esports": "KRÜ",
    "paper rex": "PRX",
    "gen.g": "GEN",
    "team liquid": "TL",
    "team heretics": "TH",
    "envy": "ENVY",
    "evil geniuses": "EG",
    "fut esports": "FUT",
    "karmine corp": "KC",
    "vitality": "VIT",
    "bbl esports": "BBL",
    "drx": "DRX",
    "t1": "T1",
    "zeta division": "ZETA",
    "leviatan": "LEV",
    "2g esports": "2G",
}

EVAL_QUERY = re.compile(
    r"\b(compatible|compatibility|compatab|score|evaluate|evaluation|"
    r"how good|how stacked|using our scoring|"
    r"\brate\b|\brank\b|\breview\b|\bgrade\b|\banalyze\b|\banalysis\b|break down)\b",
    re.I,
)

# Avoid treating map/agent stat questions as roster eval (e.g. "SEN's rating on Bind").
EVAL_CONTEXT = re.compile(
    r"\b(roster|lineup|team|org|squad|stack|composition|comp)\b",
    re.I,
)
MAP_CONTEXT = re.compile(
    r"\b(on|map|maps|bind|haven|ascent|lotus|sunset|icebox|breeze|split|pearl|fracture)\b",
    re.I,
)


def detect_org(prompt: str) -> str | None:
    lower = prompt.lower()
    if re.search(r"\bsentinels\b", lower) or re.search(r"\bsen\b", lower):
        return "SEN"
    for alias, tag in sorted(ORG_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return tag
    for word in re.findall(r"\b[A-Za-z0-9]{2,6}\b", prompt):
        upper = word.upper()
        if upper in {t for t in ORG_ALIASES.values()}:
            return upper
    return None


def is_eval_query(prompt: str) -> bool:
    org = detect_org(prompt)
    if org is None:
        return False
    if MAP_CONTEXT.search(prompt) and not EVAL_CONTEXT.search(prompt) and not re.search(
        r"\brate\b", prompt, re.I
    ):
        return False
    if re.search(r"\brate\b", prompt, re.I):
        return True
    if re.search(r"\b(rating|rank|review|grade|score)\b", prompt, re.I):
        return bool(EVAL_CONTEXT.search(prompt))
    return bool(EVAL_QUERY.search(prompt))


def is_partner_query(prompt: str) -> bool:
    from artemis.guardrails import resolve_player

    return bool(PARTNER_QUERY.search(prompt)) and resolve_player(prompt) is not None


def build_org_roster(org_tag: str, *, build_style: str = "stats") -> TeamBuild:
    pool = [p for p in load_players() if p.team.upper() == org_tag.upper()]
    if len(pool) < 5:
        raise ValueError(f"Not enough roster data for {org_tag} (found {len(pool)}).")

    pool.sort(
        key=lambda p: (max(p.rounds, 1), any(c.startswith("vct") for c in _circuits(p))),
        reverse=True,
    )
    top = pool[:8]
    style = build_style if build_style in ("stats", "chemistry") else "stats"
    if style == "chemistry":
        players, roles = _build_chemistry_lineup(
            top,
            dict(DEFAULT_COMPOSITION),
            composite_score,
            flex_avoid_duelist=True,
        )
    else:
        players, roles = _pick_by_roles(top, dict(DEFAULT_COMPOSITION), composite_score, reverse=True)
    if len(players) < 5:
        players = pool[:5]
        roles = {p.name: p.primary_role for p in players}
    return TeamBuild(players=players, assigned_roles=roles, mode="eval", league=None, build_style=style)


def format_team_context(build: TeamBuild) -> str:
    mode_labels = {
        "optimal": "stat-optimized balanced lineup",
        "goated": "all-star GOAT stack (highest-rated players)",
        "troll": "intentionally cursed troll lineup",
    }
    if build.build_style == "chemistry":
        mode_labels = {
            **mode_labels,
            "optimal": "all-star balanced lineup (stats + soft co-play)",
            "goated": "all-star superteam (stats + soft co-play)",
        }
    league_label = build.league or "current VCT"
    lines = [f"Selected team ({mode_labels.get(build.mode, build.mode)}, {league_label}):"]
    for i, player in enumerate(build.players, 1):
        role = build.assigned_roles.get(player.name, player.primary_role)
        lines.append(f"\n{i}. {player.summary()}\n   Assigned role: {role}")
    return "\n".join(lines)
