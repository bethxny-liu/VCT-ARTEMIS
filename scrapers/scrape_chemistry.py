"""Scrape VCT match lineups from vlr.gg and build player co-play pairs."""

import csv
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = ROOT_DIR / "player-data" / "player_pairs.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

CSV_COLUMNS = [
    "Player A ID",
    "Player B ID",
    "Player A Name",
    "Player B Name",
    "Team",
    "Maps Together",
]

MAX_MATCHES_PER_EVENT = 40


def fetch_url(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def discover_vct_events(max_events: int = 6) -> list[tuple[str, str]]:
    html = fetch_url("https://www.vlr.gg/events")
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    for link in soup.find_all("a", href=True):
        match = re.match(r"/event/(\d+)/([^/?#]+)", link["href"])
        if not match:
            continue
        event_id, slug = match.groups()
        if event_id in seen or "vct-" not in slug.lower():
            continue
        if any(skip in slug for skip in ("game-changers", "gc-", "challengers")):
            continue
        candidates.append((event_id, slug))
        seen.add(event_id)

    def sort_key(item: tuple[str, str]) -> tuple[int, int]:
        _id, slug = item
        return (0 if "stage-1" in slug else 1, 1 if "stage-2" in slug else 0)

    candidates.sort(key=sort_key)
    return candidates[:max_events]


def discover_event_matches(event_id: str, slug: str, limit: int) -> list[str]:
    url = f"https://www.vlr.gg/event/matches/{event_id}/{slug}"
    html = fetch_url(url)
    soup = BeautifulSoup(html, "html.parser")
    match_paths: list[str] = []

    for link in soup.find_all("a", href=True):
        m = re.match(r"/(\d+)/([^/?#]+)", link["href"])
        if not m:
            continue
        match_id, match_slug = m.groups()
        if match_id == event_id:
            continue
        path = f"/{match_id}/{match_slug}"
        if path not in match_paths:
            match_paths.append(path)
        if len(match_paths) >= limit:
            break

    return match_paths


def parse_match_lineups(html: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return [(team_label, [(player_id, name), ...]), ...] per map played."""
    soup = BeautifulSoup(html, "html.parser")
    map_lineups: list[tuple[str, list[tuple[str, str]]]] = []

    for game in soup.select(".vm-stats-game"):
        score_el = game.select_one(".score")
        if not score_el or score_el.get_text(strip=True) in ("", "–", "-"):
            continue

        teams: list[tuple[str, list[tuple[str, str]]]] = []
        for team_block in game.select(".team"):
            team_name = team_block.get_text(strip=True)[:20] or "unknown"
            players: list[tuple[str, str]] = []
            for row in team_block.select("tr"):
                link = row.select_one('td.mod-player a[href*="/player/"]')
                if not link:
                    continue
                player_id = link["href"].split("/")[2]
                name_el = link.select_one(".text-of")
                name = name_el.get_text(strip=True) if name_el else link.get_text(strip=True)
                if player_id and name:
                    players.append((player_id, name))
            if len(players) >= 4:
                teams.append((team_name, players[:5]))

        if len(teams) == 2:
            all_players = [p for _, ps in teams for p in ps]
            map_lineups.append(("map", all_players))

    if map_lineups:
        return map_lineups

    # Fallback: two tables inside each game block
    for game in soup.select(".vm-stats-game"):
        score_el = game.select_one(".score")
        if not score_el or score_el.get_text(strip=True) in ("", "–", "-"):
            continue
        tables = game.select("table")
        roster: list[tuple[str, str]] = []
        for table in tables:
            side: list[tuple[str, str]] = []
            for link in table.select('td.mod-player a[href*="/player/"]'):
                player_id = link["href"].split("/")[2]
                name_el = link.select_one(".text-of")
                name = name_el.get_text(strip=True) if name_el else ""
                if player_id and name:
                    side.append((player_id, name))
            if len(side) >= 4:
                roster.extend(side[:5])
        if len(roster) >= 8:
            map_lineups.append(("map", roster))

    return map_lineups


def scrape_match(match_path: str) -> list[list[tuple[str, str]]]:
    url = f"https://www.vlr.gg{match_path}"
    html = fetch_url(url)
    lineups = parse_match_lineups(html)
    return [players for _, players in lineups]


def accumulate_pairs(
    pair_maps: dict[tuple[str, str], dict],
    roster: list[tuple[str, str]],
    team_hint: str = "",
) -> None:
    by_id = {pid: name for pid, name in roster}
    ids = list(by_id.keys())
    for a, b in combinations(sorted(ids), 2):
        key = (a, b)
        if key not in pair_maps:
            pair_maps[key] = {
                "Player A ID": a,
                "Player B ID": b,
                "Player A Name": by_id[a],
                "Player B Name": by_id[b],
                "Team": team_hint,
                "Maps Together": 0,
            }
        pair_maps[key]["Maps Together"] += 1


def scrape_chemistry(max_events: int = 6, max_matches: int = MAX_MATCHES_PER_EVENT) -> list[dict]:
    pair_maps: dict[tuple[str, str], dict] = {}
    events = discover_vct_events(max_events)
    print(f"Scanning {len(events)} VCT events for co-play data...")

    for event_id, slug in events:
        try:
            match_paths = discover_event_matches(event_id, slug, max_matches)
        except requests.RequestException as e:
            print(f"  Skip event {slug}: {e}")
            continue

        print(f"  {slug}: {len(match_paths)} matches")
        for path in match_paths:
            try:
                for roster in scrape_match(path):
                    # Split into two teams of 5 when roster has 10 players
                    if len(roster) == 10:
                        accumulate_pairs(pair_maps, roster[:5])
                        accumulate_pairs(pair_maps, roster[5:])
                    elif len(roster) >= 5:
                        accumulate_pairs(pair_maps, roster[:5])
            except requests.RequestException as e:
                print(f"    Skip {path}: {e}")
            time.sleep(0.25)

    rows = sorted(pair_maps.values(), key=lambda r: r["Maps Together"], reverse=True)
    print(f"Found {len(rows)} player pairs")
    return rows


def write_csv(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pairs to {OUTPUT_CSV}")


def main():
    max_events = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = scrape_chemistry(max_events=max_events)
    if not rows:
        print("No pairs scraped.")
        sys.exit(1)
    write_csv(rows)


if __name__ == "__main__":
    main()
