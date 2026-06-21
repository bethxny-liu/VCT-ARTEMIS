"""Scrape player stats from vlr.gg and write player-data/player_stats.csv."""

import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = ROOT_DIR / "player-data" / "player_stats.csv"

REGIONS = {
    "na": "North America",
    "eu": "Europe",
    "ap": "Asia-Pacific",
    "jp": "Japan",
    "sa": "Latin America",
    "oce": "Oceania",
    "mn": "MENA",
    "gc": "Game Changers",
    "cg": "Collegiate",
}

CSV_COLUMNS = [
    "Player Name",
    "Team",
    "Region",
    "Circuit",
    "Agents",
    "Rounds",
    "Rating",
    "ACS",
    "K:D",
    "ADR",
    "KAST",
    "KPR",
    "APR",
    "FKPR",
    "FDPR",
    "Player ID",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

GLOBAL_URL = (
    "https://www.vlr.gg/stats/"
    "?event_group_id=all&event_id=all&region={region}"
    "&min_rounds=0&min_rating=900&agent=all&map_id=all&timespan=90d"
)

EVENT_STATS_URL = (
    "https://www.vlr.gg/event/stats/{event_id}/{slug}"
    "?event_group_id=all&region=all&min_rounds=0&min_rating=900"
    "&agent=all&map_id=all&timespan=all"
)


def parse_agents(row) -> str:
    agents = []
    for img in row.find_all("img"):
        src = img.get("src", "")
        if "/game/agents/" in src:
            agents.append(src.split("/")[-1].replace(".png", ""))
    return ", ".join(agents)


def parse_player_cell(row) -> tuple[str, str, str]:
    player_td = row.find_all("td")[0]
    name = player_td.find("div", class_="text-of").text.strip()
    team_el = player_td.find("div", class_="stats-player-country")
    team = team_el.text.strip() if team_el else ""
    href = player_td.find("a")["href"]
    player_id = href.split("/")[2]
    return name, team, player_id


def fetch_url(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def parse_stats_table(html: str, circuit: str, region: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "wf-table mod-stats mod-scroll"})
    if not table:
        return []

    columns = [th.text.strip() for th in table.find("thead").find_all("th")]
    col_index = {name: i for i, name in enumerate(columns)}
    players = []

    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue

        name, team, player_id = parse_player_cell(row)
        values = [td.text.strip() for td in cells]

        def get_col(label: str) -> str:
            idx = col_index.get(label)
            return values[idx] if idx is not None and idx < len(values) else ""

        players.append(
            {
                "Player Name": name,
                "Team": team,
                "Region": region,
                "Circuit": circuit,
                "Agents": parse_agents(row),
                "Rounds": get_col("Rnd"),
                "Rating": get_col("R2.0"),
                "ACS": get_col("ACS"),
                "K:D": get_col("K:D"),
                "ADR": get_col("ADR"),
                "KAST": get_col("KAST"),
                "KPR": get_col("KPR"),
                "APR": get_col("APR"),
                "FKPR": get_col("FKPR"),
                "FDPR": get_col("FDPR"),
                "Player ID": player_id,
            }
        )
    return players


def discover_vct_events(max_events: int = 8) -> list[tuple[str, str, str]]:
    """Return (event_id, slug, circuit_label) for recent VCT events with stats."""
    html = fetch_url("https://www.vlr.gg/events")
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.match(r"/event/(\d+)/([^/?#]+)", href)
        if not match:
            continue
        event_id, slug = match.groups()
        if event_id in seen:
            continue
        if "vct-" not in slug.lower():
            continue
        if any(skip in slug for skip in ("game-changers", "gc-", "challengers")):
            continue

        if "americas" in slug:
            circuit = "vct-americas"
        elif "emea" in slug:
            circuit = "vct-emea"
        elif "pacific" in slug:
            circuit = "vct-pacific"
        elif "china" in slug:
            circuit = "vct-china"
        else:
            circuit = "vct"

        candidates.append((event_id, slug, circuit))
        seen.add(event_id)

    # Prefer completed stage-1 events (stage-2 upcoming often has no stats yet)
    def sort_key(item: tuple[str, str, str]) -> tuple[int, int]:
        _id, slug, _circuit = item
        stage1 = 0 if "stage-1" in slug else 1
        stage2 = 1 if "stage-2" in slug else 0
        return (stage1, stage2)

    candidates.sort(key=sort_key)
    return candidates[:max_events]


def scrape_event(event_id: str, slug: str, circuit: str) -> list[dict]:
    url = EVENT_STATS_URL.format(event_id=event_id, slug=slug)
    print(f"Fetching {circuit} — {slug}...")
    players = parse_stats_table(fetch_url(url), circuit=circuit, region="vct")
    print(f"  Found {len(players)} players")
    return players


def scrape_global(region: str = "all") -> list[dict]:
    url = GLOBAL_URL.format(region=region)
    print(f"Fetching global 90-day stats ({region})...")
    players = parse_stats_table(fetch_url(url), circuit="global", region=region)
    print(f"  Found {len(players)} players")
    return players


def infer_region_tag(team: str, region: str) -> str:
    if _is_gc_team(team):
        return "gc"
    return region


def _is_gc_team(team: str) -> bool:
    t = team.upper()
    return ".GC" in t or t.endswith(".G")


def normalize_circuits(player: dict) -> str:
    circuits = set(player.get("_circuits", set()))
    if _is_gc_team(player["Team"]):
        circuits.add("gc")
    else:
        circuits.discard("gc")
    return ",".join(sorted(c for c in circuits if c))


def merge_players(all_players: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}

    for player in all_players:
        key = (player["Player Name"].lower(), player["Team"].lower())
        rating = float(player["Rating"] or 0)
        existing = merged.get(key)

        if not existing:
            merged[key] = dict(player)
            merged[key]["_circuits"] = {player["Circuit"]}
            continue

        existing["_circuits"].add(player["Circuit"])
        if rating > float(existing["Rating"] or 0):
            for field in (
                "Agents", "Rounds", "Rating", "ACS", "K:D", "ADR",
                "KAST", "KPR", "APR", "FKPR", "FDPR", "Region",
            ):
                existing[field] = player[field]

    result = []
    for player in merged.values():
        circuits = player.pop("_circuits")
        player["Circuit"] = normalize_circuits({"Team": player["Team"], "_circuits": circuits})
        player["Region"] = infer_region_tag(player["Team"], player["Region"])
        result.append(player)

    return sorted(result, key=lambda p: float(p["Rating"] or 0), reverse=True)


def scrape_all(mode: str = "full") -> list[dict]:
    all_players: list[dict] = []

    if mode in ("full", "vct"):
        for event_id, slug, circuit in discover_vct_events():
            try:
                batch = scrape_event(event_id, slug, circuit)
                if batch:
                    all_players.extend(batch)
            except requests.RequestException as e:
                print(f"  Error scraping event {event_id}: {e}")
            time.sleep(0.4)

    if mode in ("full", "global"):
        try:
            all_players.extend(scrape_global("all"))
        except requests.RequestException as e:
            print(f"  Error scraping global stats: {e}")

    return merge_players(all_players)


def write_csv(players: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(players)
    print(f"Wrote {len(players)} players to {OUTPUT_CSV}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode in REGIONS:
        players = merge_players(parse_stats_table(fetch_url(GLOBAL_URL.format(region=mode)), "global", mode))
    elif mode not in ("full", "vct", "global", "gc"):
        print(f"Unknown mode '{mode}'. Use: full, vct, global, gc, or a region code.")
        sys.exit(1)
    else:
        players = scrape_all(mode)

    if not players:
        print("No players scraped.")
        sys.exit(1)
    write_csv(players)


if __name__ == "__main__":
    main()
