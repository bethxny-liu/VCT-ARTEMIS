"""PCA + KMeans player map for chemistry visualization (no LLM)."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from artemis.chemistry.scoring import pair_link_detail, player_tag, player_viz_meta
from artemis.team.builder import Player, load_players

# Human-readable names aligned with _feature_matrix column order.
FEATURE_LABELS = (
    "Rating",
    "ACS",
    "K/D",
    "First kills",
    "Assists",
    "Experience",
    "Comms language",
    "IGL",
)


def _feature_matrix(players: list[Player], *, chemistry_mode: bool = False) -> np.ndarray:
    """Stat space + optional comms/IGL dimensions for chemistry-aware layout."""
    chem_scale = 2.0 if chemistry_mode else 0.75
    rows = []
    for p in players:
        tag = player_tag(p.player_id)
        lang_code = {"en": 0.15, "pt": 0.35, "id": 0.55, "zh": 0.75, "th": 0.95, "ko": 0.85}.get(
            tag.get("language") or "", 0.0
        )
        igl_flag = 1.0 if tag.get("igl") else 0.0
        rows.append(
            [
                p.rating,
                p.acs / 300.0,
                p.kd,
                p.fkpr,
                p.apr,
                p.rounds / 600.0,
                lang_code * chem_scale,
                igl_flag * chem_scale,
            ]
        )
    return np.array(rows, dtype=float)


def _axis_label(component: np.ndarray) -> str:
    """Plain-language axis name from the strongest PCA loadings."""
    ranked = sorted(
        enumerate(component),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    primary = FEATURE_LABELS[ranked[0][0]]
    if len(ranked) < 2:
        return primary

    secondary = FEATURE_LABELS[ranked[1][0]]
    primary_weight = abs(float(ranked[0][1]))
    secondary_weight = abs(float(ranked[1][1]))
    if secondary_weight >= primary_weight * 0.45 and secondary != primary:
        return f"{primary} & {secondary}"
    return primary


def _vct_pool() -> list[Player]:
    pool = []
    for p in load_players():
        if any(c.startswith("vct") for c in p.circuit.split(",")):
            pool.append(p)
    return pool if len(pool) >= 20 else load_players()


def build_chemistry_plot(
    selected: list[Player],
    background_limit: int = 120,
    *,
    build_style: str = "stats",
    assigned_roles: dict[str, str] | None = None,
) -> dict:
    """
    2D player map: PCA layout with KMeans background groups;
    edges show co-play links within the selected lineup.
    """
    _ = assigned_roles
    chemistry_mode = build_style == "chemistry"
    pool = _vct_pool()
    selected_ids = {p.player_id for p in selected if p.player_id}

    by_id = {p.player_id: p for p in pool if p.player_id}
    for p in selected:
        if p.player_id:
            by_id[p.player_id] = p
    pool = list(by_id.values())

    background = [p for p in pool if p.player_id not in selected_ids]
    background.sort(key=lambda p: p.rating, reverse=True)
    plot_players = selected + background[:background_limit]

    X = _feature_matrix(plot_players, chemistry_mode=chemistry_mode)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    components = pca.components_

    n_clusters = min(5, max(2, len(plot_players) // 15))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)

    points = []
    for i, p in enumerate(plot_players):
        meta = player_viz_meta(p)
        points.append(
            {
                "id": p.player_id,
                "name": p.name,
                "team": p.team,
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "cluster": int(labels[i]),
                "selected": p.player_id in selected_ids,
                "rating": round(p.rating, 2),
                "language": meta.get("language"),
                "igl": meta.get("igl"),
            }
        )

    edges = []
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            if not a.player_id or not b.player_id:
                continue
            detail = pair_link_detail(a, b)
            edges.append({"a": a.player_id, "b": b.player_id, **detail})

    x_label = _axis_label(components[0])
    y_label = _axis_label(components[1])
    variance_pct = round(sum(pca.explained_variance_ratio_) * 100)

    if chemistry_mode:
        subtitle = (
            "Your lineup (large dots) plotted against other VCT pros by stats and comms fit. "
            "Lines show who has shared maps or aligned comms."
        )
    else:
        subtitle = (
            "Your lineup (large dots) plotted against other VCT pros by performance stats. "
            "Lines show shared map history between your picks."
        )

    return {
        "type": "player_map",
        "points": points,
        "edges": edges,
        "clusters": n_clusters,
        "axisLabels": {"x": x_label, "y": y_label},
        "varianceExplainedPct": variance_pct,
        "chemistryMode": chemistry_mode,
        "title": "Player fit map",
        "subtitle": subtitle,
        "note": (
            "← → horizontal axis: "
            f"{x_label} · ↑ ↓ vertical axis: {y_label} · "
            "Drag to pan · scroll to zoom"
        ),
    }
