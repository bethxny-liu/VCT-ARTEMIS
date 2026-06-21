import json
import os
import re

import chromadb
import requests
from bs4 import BeautifulSoup
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from artemis import config
from artemis.chemistry.viz import build_chemistry_plot
from artemis.guardrails import (
    LOW_CONFIDENCE_MESSAGE,
    check_rag_guardrails,
    retrieval_is_weak,
)
from artemis.team.builder import (
    build_org_roster,
    build_team,
    detect_org,
    format_team_context,
    is_eval_query,
    is_partner_query,
    is_team_query,
)
from artemis.chemistry.scoring import format_partner_answer, rank_teammates_for
from artemis.guardrails import resolve_player
from artemis.team.evaluator import evaluate_team, format_eval_text, score_to_dict

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

Settings.llm = OpenAI(model=config.LLM_MODEL, temperature=0.2, api_key=config.OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(
    model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY
)

llm = Settings.llm
app = Flask(__name__, static_folder=str(config.ROOT_DIR / "static"))
CORS(app)

RAG_SYSTEM_PROMPT = (
    "You are Artemis, a Valorant esports analyst for VCT and Game Changers.\n"
    "Use ONLY the retrieved stats below. Be concise — 2-4 short paragraphs max.\n"
    "Cite specific stats when relevant.\n"
    "If the question is not about Valorant esports, refuse briefly.\n"
    "If a named player is not in the retrieved context, say you do not have "
    "their stats — do NOT use outside knowledge or invent numbers.\n\n"
    "{context}"
)

TEAM_EXPLAIN_SYSTEM = """You are Artemis, a sharp Valorant esports analyst.

A stats engine has ALREADY locked in the 5-player lineup below. Your job is ONLY to explain it.

STRICT RULES:
- NEVER say you cannot build, change, swap, or pick players. The roster is final.
- NEVER list full stat blocks — the UI already shows Rating, ACS, and agents.
- NEVER use markdown headers or numbered essays.
- Total output under 100 words across all fields.
- Match player "name" exactly to the handles provided.

Return ONLY valid JSON (no markdown fences):
{
  "summary": "1-2 sentences: comp identity + how they win rounds",
  "players": [
    {"name": "exact handle", "note": "max 12 words on this player's job"}
  ]
}

Lineup data:
{context}"""

CONDENSE_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

ROLE_COLORS = {
    "duelist": "#FF4655",
    "initiator": "#FFC84B",
    "controller": "#7B61FF",
    "sentinel": "#00D8D8",
    "flex": "#8B978F",
}


def get_chat_history(history="[]"):
    history = json.loads(history) if isinstance(history, str) else history
    roles = {"left_bubble": MessageRole.ASSISTANT, "right_bubble": MessageRole.USER}
    return [
        ChatMessage(role=roles[chat["position"]], content=chat["message"])
        for chat in history
    ]


def get_index() -> VectorStoreIndex:
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No index at {config.CHROMA_DIR}. Run: python scripts/refresh_data.py"
        )
    db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = db.get_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def ask_rag(prompt: str, chat_history=None) -> str:
    index = get_index()
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    nodes = retriever.retrieve(prompt)

    if retrieval_is_weak(prompt, nodes):
        return LOW_CONFIDENCE_MESSAGE

    memory = ChatMemoryBuffer.from_defaults(token_limit=10000)
    chat = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        system_prompt=RAG_SYSTEM_PROMPT,
        condense_prompt=CONDENSE_PROMPT,
    )
    response = chat.chat(prompt, chat_history or [])
    return str(response.response)


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def explain_team(prompt: str, build) -> dict:
    context = format_team_context(build)
    temperature = 0.55 if build.mode == "troll" else 0.25
    if build.mode == "goated":
        temperature = 0.45

    tone = {
        "optimal": "Professional and confident.",
        "goated": "Hype but brief — this is an all-star stack.",
        "troll": "Funny and roast-y, but still cite why they're bad.",
    }.get(build.mode, "")

    explainer = OpenAI(model=config.LLM_MODEL, temperature=temperature, api_key=config.OPENAI_API_KEY)
    response = explainer.chat(
        [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=TEAM_EXPLAIN_SYSTEM.replace("{context}", context) + f"\nTone: {tone}",
            ),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
    )
    raw = str(response.message.content)
    try:
        return _parse_json_response(raw)
    except json.JSONDecodeError:
        return {"summary": raw[:280], "players": []}


def _agent_list(agents_str: str) -> list[str]:
    return [a.strip().lower() for a in agents_str.split(",") if a.strip()]


def team_payload(build, explanation: dict | None = None) -> list[dict]:
    notes: dict[str, str] = {}
    if explanation:
        notes = {
            p["name"].lower(): p.get("note", "")
            for p in explanation.get("players", [])
            if isinstance(p, dict) and "name" in p
        }
    cards = []
    for p in build.players:
        role = build.assigned_roles.get(p.name, p.primary_role)
        agents = _agent_list(p.agents)
        cards.append(
            {
                "name": p.name,
                "team": p.team,
                "role": role,
                "roleColor": ROLE_COLORS.get(role, ROLE_COLORS["flex"]),
                "agents": agents,
                "primaryAgent": agents[0] if agents else "jett",
                "rating": round(p.rating, 2),
                "acs": round(p.acs, 1),
                "kd": round(p.kd, 2),
                "fkpr": round(p.fkpr, 2),
                "playerId": p.player_id,
                "photoUrl": f"/img/players/{p.player_id}.png" if p.player_id else None,
                "vlrUrl": f"https://www.vlr.gg/player/{p.player_id}" if p.player_id else None,
                "note": notes.get(p.name.lower(), ""),
                "mode": build.mode,
            }
        )
    return cards


def ask(
    prompt: str,
    chat_history=None,
    settings: dict | None = None,
) -> tuple[str, list[dict] | None, str | None, dict | None, dict | None]:
    settings = settings or {}
    team_opts = {
        "build_style": settings.get("buildStyle"),
        "mode_override": settings.get("teamMode") or None,
        "league_override": settings.get("league") or None,
    }
    if team_opts["mode_override"] == "auto":
        team_opts["mode_override"] = None
    if team_opts["league_override"] == "auto":
        team_opts["league_override"] = None

    if is_eval_query(prompt):
        org = detect_org(prompt)
        if not org:
            return LOW_CONFIDENCE_MESSAGE, None, None, None, None
        eval_style = team_opts.get("build_style") or "stats"
        if eval_style not in ("stats", "chemistry"):
            eval_style = "stats"
        build = build_org_roster(org, build_style=eval_style)
        ev = evaluate_team(build)
        return (
            format_eval_text(org, ev),
            team_payload(build),
            "eval",
            score_to_dict(ev, build.build_style),
            build_chemistry_plot(
                build.players,
                build_style=build.build_style,
                assigned_roles=build.assigned_roles,
            ),
        )

    if re.search(r"\brate\b", prompt, re.I) and detect_org(prompt) is None:
        return LOW_CONFIDENCE_MESSAGE, None, None, None, None

    if is_partner_query(prompt):
        anchor = resolve_player(prompt)
        if anchor:
            ranked = rank_teammates_for(anchor)
            return format_partner_answer(anchor, ranked), None, None, None, None

    if is_team_query(prompt):
        build = build_team(prompt, **team_opts)
        ev = evaluate_team(build)
        explanation = explain_team(prompt, build)
        summary = explanation.get("summary", "Here's your lineup.")
        return (
            summary,
            team_payload(build, explanation),
            build.mode,
            score_to_dict(ev, build.build_style),
            build_chemistry_plot(
                build.players,
                build_style=build.build_style,
                assigned_roles=build.assigned_roles,
            ),
        )

    refusal = check_rag_guardrails(prompt)
    if refusal:
        return refusal, None, None, None, None

    return ask_rag(prompt, chat_history), None, None, None, None


@app.route("/")
def index_page():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health", methods=["GET"])
def health():
    indexed = config.CHROMA_DIR.exists()
    pairs = (config.ROOT_DIR / "player-data" / "player_pairs.csv").exists()
    return jsonify({"status": "ok", "indexed": indexed, "chemistry": pairs})


VLR_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Artemis/1.0)"}


@app.route("/img/agents/<name>.png")
def agent_image(name: str):
    slug = re.sub(r"[^a-z0-9]", "", name.lower())
    if not slug:
        return "", 404
    url = f"https://www.vlr.gg/img/vlr/game/agents/{slug}.png"
    try:
        resp = requests.get(url, headers=VLR_HEADERS, timeout=10)
    except requests.RequestException:
        return "", 502
    if resp.status_code != 200:
        return "", resp.status_code
    return Response(
        resp.content,
        mimetype="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _player_photo_url(player_id: str) -> str | None:
    try:
        resp = requests.get(
            f"https://www.vlr.gg/player/{player_id}",
            headers=VLR_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]
    avatar = soup.select_one(".wf-avatar.mod-player img")
    if avatar and avatar.get("src"):
        src = avatar["src"]
        return src if src.startswith("http") else f"https:{src}"
    return None


@app.route("/img/players/<player_id>.png")
def player_image(player_id: str):
    if not re.fullmatch(r"\d+", player_id):
        return "", 404

    config.PHOTO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.PHOTO_CACHE_DIR / f"{player_id}.png"
    if cache_path.exists():
        return send_from_directory(
            config.PHOTO_CACHE_DIR,
            f"{player_id}.png",
            mimetype="image/png",
            max_age=604800,
        )

    photo_url = _player_photo_url(player_id)
    if not photo_url:
        return "", 404

    try:
        img = requests.get(photo_url, headers=VLR_HEADERS, timeout=15)
        img.raise_for_status()
    except requests.RequestException:
        return "", 502

    cache_path.write_bytes(img.content)
    return Response(
        img.content,
        mimetype="image/png",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        chat_history = get_chat_history(data.get("chatHistory", "[]"))
        settings = data.get("settings") or {}
        result, team, mode, evaluation, chemistry_plot = ask(prompt, chat_history, settings)
        payload = {"result": result}
        if team:
            payload["team"] = team
            payload["mode"] = mode
        if evaluation:
            payload["evaluation"] = evaluation
            payload["buildStyle"] = evaluation.get("buildStyle")
        if chemistry_plot:
            payload["chemistryPlot"] = chemistry_plot
        return jsonify(payload)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        print("Artemis CLI — type 'quit' to exit")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in ("", "q", "quit", "exit"):
                break
            try:
                result, team, mode, evaluation, _plot = ask(question)
                if team:
                    print(f"\n--- Selected team ({mode}) ---")
                    for p in team:
                        print(f"  {p['role']:10} {p['name']} ({p['team']}) | {p['rating']:.2f} rating")
                print(f"\nArtemis: {result}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        app.run(host="0.0.0.0", port=8000, debug=True)
