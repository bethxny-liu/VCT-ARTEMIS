"""Co-play chemistry from scraped pair data and optional player tags."""

import csv
import json
import math
from functools import lru_cache

from artemis import config

PAIR_CSV = config.ROOT_DIR / "player-data" / "player_pairs.csv"
TAGS_JSON = config.ROOT_DIR / "player-data" / "player_tags.json"

# maps_together at which pair synergy saturates
MAPS_SATURATION = 15

# Soft nudge applied during lineup search (stat scores are ~0.8–1.3).
CHEMISTRY_WEIGHT = 0.12
CHEM_MAP_WEIGHT = 0.12
CHEM_LANG_MATCH = 0.045
CHEM_LANG_MISMATCH = -0.018
CHEM_IGL_FIRST = 0.03
CHEM_IGL_DUPLICATE = -0.05


@lru_cache(maxsize=1)
def load_pairs() -> dict[tuple[str, str], int]:
    if not PAIR_CSV.exists():
        return {}
    pairs: dict[tuple[str, str], int] = {}
    with open(PAIR_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b = row["Player A ID"], row["Player B ID"]
            key = (min(a, b), max(a, b))
            pairs[key] = pairs.get(key, 0) + int(row.get("Maps Together") or 0)
    return pairs


@lru_cache(maxsize=1)
def load_tags() -> dict[str, dict]:
    if not TAGS_JSON.exists():
        return {}
    with open(TAGS_JSON, encoding="utf-8") as f:
        return json.load(f)


def player_tag(player_id: str) -> dict:
    if not player_id:
        return {}
    return load_tags().get(player_id, {})


def pair_maps_together(player_a_id: str, player_b_id: str) -> int:
    if not player_a_id or not player_b_id:
        return 0
    key = (min(player_a_id, player_b_id), max(player_a_id, player_b_id))
    return load_pairs().get(key, 0)


def pair_synergy(player_a_id: str, player_b_id: str) -> float:
    """0–1 synergy from shared maps."""
    maps = pair_maps_together(player_a_id, player_b_id)
    if maps <= 0:
        return 0.0
    return min(1.0, math.log1p(maps) / math.log1p(MAPS_SATURATION))


def effective_pair_synergy(
    player_a_id: str,
    player_b_id: str,
    *,
    team_a: str = "",
    team_b: str = "",
) -> float:
    """Pair synergy with same-org proxy when map data is missing (display scoring)."""
    syn = pair_synergy(player_a_id, player_b_id)
    if syn == 0 and team_a and team_b and team_a.upper() == team_b.upper():
        return 0.2
    return syn


def roster_map_synergy(players) -> float:
    """Average shared-map synergy only (no org proxy)."""
    if len(players) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i, a in enumerate(players):
        for b in players[i + 1 :]:
            if a.player_id and b.player_id:
                total += pair_synergy(a.player_id, b.player_id)
                n += 1
    return total / n if n else 0.0


def roster_pair_synergy(players) -> float:
    """Average pairwise synergy for display (includes same-org proxy)."""
    if len(players) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i, a in enumerate(players):
        for b in players[i + 1 :]:
            total += effective_pair_synergy(
                a.player_id,
                b.player_id,
                team_a=a.team,
                team_b=b.team,
            )
            n += 1
    return total / n


def roster_language_cohesion(players) -> float:
    """1.0 when comms align; lower when languages mix."""
    langs = [player_tag(p.player_id).get("language") for p in players]
    langs = [lang for lang in langs if lang]
    if not langs:
        return 0.55
    top = max(langs.count(lang) for lang in set(langs))
    return top / len(langs)


def roster_igl_balance(players) -> float:
    """Prefer exactly one tagged IGL on the roster."""
    igls = sum(1 for p in players if player_tag(p.player_id).get("igl"))
    if igls == 0:
        return 0.65
    if igls == 1:
        return 1.0
    if igls == 2:
        return 0.45
    return 0.2


def roster_soft_chemistry(players) -> float:
    """0–1 blend for soft lineup objective: maps, language, IGL."""
    if len(players) < 2:
        return 0.0
    maps = roster_map_synergy(players)
    # Dampen when 3+ from one org — keep all-star cross-region viable.
    org_counts: dict[str, int] = {}
    for p in players:
        if p.team:
            key = p.team.upper()
            org_counts[key] = org_counts.get(key, 0) + 1
    if org_counts and max(org_counts.values()) >= 3:
        maps *= 0.45
    lang = roster_language_cohesion(players)
    igl = roster_igl_balance(players)
    return 0.50 * maps + 0.35 * lang + 0.15 * igl


def _language_pick_nudge(player, selected) -> float:
    lang = player_tag(player.player_id).get("language")
    if not lang:
        return 0.0
    sel_langs = [
        player_tag(s.player_id).get("language")
        for s in selected
        if player_tag(s.player_id).get("language")
    ]
    if not sel_langs:
        return 0.0
    if lang in sel_langs:
        return CHEM_LANG_MATCH
    return CHEM_LANG_MISMATCH


def _igl_pick_nudge(player, selected) -> float:
    if not player_tag(player.player_id).get("igl"):
        return 0.0
    selected_igls = sum(
        1 for s in selected if player_tag(s.player_id).get("igl")
    )
    if selected_igls == 0:
        return CHEM_IGL_FIRST
    return CHEM_IGL_DUPLICATE * min(selected_igls, 2)


def chemistry_pick_bonus(player, selected, weight: float = CHEMISTRY_WEIGHT) -> float:
    """Soft stat nudge from shared maps, language match, and IGL fit."""
    if not selected:
        return 0.0

    map_part = 0.0
    n = 0
    for s in selected:
        if not player.player_id or not s.player_id:
            continue
        map_part += pair_synergy(player.player_id, s.player_id)
        n += 1
    map_bonus = (map_part / n * CHEM_MAP_WEIGHT) if n else 0.0

    tag_bonus = _language_pick_nudge(player, selected) + _igl_pick_nudge(player, selected)
    return (map_bonus + tag_bonus) * weight / CHEMISTRY_WEIGHT


def rank_teammates_for(anchor, *, limit: int = 5) -> list[tuple[object, float, dict]]:
    """Best co-play fits for one player from the full pool."""
    from artemis.team.builder import load_players

    scored: list[tuple[object, float, dict]] = []
    for other in load_players():
        if other.name == anchor.name:
            continue
        detail = pair_link_detail(anchor, other)
        score = detail["strength"]
        if other.team and anchor.team and other.team.upper() == anchor.team.upper():
            score += 0.08
        scored.append((other, score, detail))
    scored.sort(key=lambda row: (row[1], row[0].rating), reverse=True)
    return scored[:limit]


def format_partner_answer(anchor, ranked: list[tuple[object, float, dict]]) -> str:
    """Text answer for 'who plays best with X' queries."""
    lines = [f"Best co-play fits for {anchor.name} ({anchor.team}) from our VCT pool:\n"]
    has_maps = False
    for i, (player, _score, detail) in enumerate(ranked, 1):
        bits: list[str] = []
        if detail["maps"] > 0:
            has_maps = True
            bits.append(f"{detail['maps']} shared maps")
        if detail["langMatch"]:
            bits.append("aligned comms")
        if player.team == anchor.team:
            bits.append("same org")
        extra = ", ".join(bits) if bits else "similar profile"
        lines.append(f"{i}. {player.name} ({player.team}) — {extra} · {player.rating:.2f} rating")
    if not has_maps:
        lines.append(
            "\nNote: map co-play data is sparse for this region — "
            "rankings lean on org roster and comms tags where available."
        )
    return "\n".join(lines)


def language_penalty(players) -> tuple[float, list[str]]:
    """Small penalty when roster mixes many primary languages (from manual tags)."""
    langs = set()
    for p in players:
        lang = player_tag(p.player_id).get("language")
        if lang:
            langs.add(lang)
    notes: list[str] = []
    if len(langs) <= 1:
        return 1.0, notes
    if len(langs) == 2:
        notes.append("Mixed comms (2 languages)")
        return 0.92, notes
    notes.append(f"Mixed comms ({len(langs)} languages)")
    return 0.85, notes


def igl_notes(players) -> list[str]:
    igls = [p.name for p in players if player_tag(p.player_id).get("igl")]
    if len(igls) == 1:
        return [f"IGL: {igls[0]}"]
    if len(igls) > 1:
        return [f"Multiple IGL tags ({', '.join(igls)})"]
    return []


def pair_link_detail(a, b) -> dict:
    """Edge metadata for chemistry viz: maps, language, org."""
    maps = pair_maps_together(a.player_id, b.player_id)
    map_syn = pair_synergy(a.player_id, b.player_id)
    tag_a = player_tag(a.player_id)
    tag_b = player_tag(b.player_id)
    lang_a = tag_a.get("language")
    lang_b = tag_b.get("language")
    lang_match = bool(lang_a and lang_b and lang_a == lang_b)
    same_org = (
        bool(a.team and b.team and a.team.upper() == b.team.upper())
    )

    if maps > 0 and lang_match:
        link_type = "maps_lang"
    elif maps > 0:
        link_type = "maps"
    elif lang_match:
        link_type = "language"
    elif same_org:
        link_type = "org"
    else:
        link_type = "none"

    strength = (
        map_syn * 0.55
        + (0.30 if lang_match else 0.0)
        + (0.15 if same_org and maps == 0 else 0.0)
    )

    return {
        "maps": maps,
        "synergy": round(map_syn, 3),
        "langMatch": lang_match,
        "linkType": link_type,
        "strength": round(strength, 3),
    }


def player_viz_meta(player) -> dict:
    """Tag metadata attached to cluster plot points."""
    tag = player_tag(player.player_id)
    return {
        "language": tag.get("language"),
        "igl": bool(tag.get("igl")),
    }


def chemistry_score(players) -> tuple[int, list[str]]:
    """0–100 chemistry score + highlight bullets."""
    pairs = load_pairs()
    if not pairs:
        return 50, ["No co-play data — run scrape_chemistry"]

    synergy = roster_pair_synergy(players)
    soft = roster_soft_chemistry(players)
    lang_factor, lang_notes = language_penalty(players)

    score = _clamp(40 + synergy * 45 * lang_factor + soft * 15)
    notes: list[str] = []

    if synergy >= 0.45:
        notes.append("Strong shared match history")
    elif synergy >= 0.2:
        notes.append("Some co-play history")
    elif roster_language_cohesion(players) >= 0.8:
        notes.append("Cross-org picks, aligned comms")
    else:
        notes.append("Mostly solo stat picks — low shared reps")

    best_pair = (0.0, "", "")
    best_maps = 0
    ids = [(p.player_id, p.name) for p in players if p.player_id]
    for i, (ida, na) in enumerate(ids):
        for idb, nb in ids[i + 1 :]:
            syn = pair_synergy(ida, idb)
            if syn > best_pair[0]:
                best_pair = (syn, na, nb)
                best_maps = pair_maps_together(ida, idb)
    if best_pair[0] >= 0.25:
        notes.append(f"Top duo: {best_pair[1]} + {best_pair[2]} ({best_maps} maps)")

    notes.extend(lang_notes)
    notes.extend(igl_notes(players))
    return score, notes[:4]


def _clamp(n: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(n))))
