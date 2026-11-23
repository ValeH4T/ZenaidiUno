"""Shadowing v2 — paste a YouTube link, practise it sentence by sentence.

Runs as a free Gradio-SDK Space in Gradio's Server mode (a FastAPI app that
Gradio launches the way Spaces expects). The server only fetches the caption
text, splits it into sentences and translates them; the video itself plays in
the visitor's browser through YouTube's official player. It also checks
pronunciation: the page sends a short recording and Whisper (on the CPU) writes
down what it heard. No GPU is used.
"""
import json
import os
import plistlib
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

try:
    import spaces  # ZeroGPU helper (preinstalled on Spaces) — must be imported before gradio
except ImportError:
    spaces = None

import gradio as gr
import requests
from fastapi import Body, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- helpers ---

ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
URL_PATTERNS = [
    re.compile(r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})"),
]


def video_id(raw: str) -> str | None:
    raw = (raw or "").strip()
    if ID_RE.match(raw):
        return raw
    for p in URL_PATTERNS:
        m = p.search(raw)
        if m:
            return m.group(1)
    return None


def base_lang(code: str | None) -> str:
    return (code or "").split("-")[0].lower()


def proxy_url() -> str | None:
    return os.getenv("PROXY_URL") or None


class Cache:
    def __init__(self, size=200):
        self.size, self.d, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.size:
                self.d.popitem(last=False)


TRANSCRIPTS = Cache(200)
TRANSLATIONS = Cache(20000)

# ------------------------------------------------- caption fetching (A) ----
# Each fetcher returns a list of "cues": {"start", "end", "words": [(text, t|None), ...]}
# t is a per-word time when the source has it (YouTube auto captions via json3).


class FetchError(Exception):
    def __init__(self, code, detail=""):
        super().__init__(code)
        self.code, self.detail = code, detail


def _clean(text: str) -> str:
    text = text.replace("\n", " ").replace("​", "")
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [Musica], [Applausi] …
    text = re.sub(r"&gt;&gt;|>>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _pick(tracks, lang):
    """tracks: list of dicts {code, name, generated}. Pick the best one."""
    if not tracks:
        return None
    want = base_lang(lang)
    manual = [t for t in tracks if not t["generated"]]
    auto = [t for t in tracks if t["generated"]]
    if want:
        for group in (manual, auto):
            for t in group:
                if base_lang(t["code"]) == want:
                    return t
    # The auto-generated track is in the language actually spoken in the video.
    spoken = base_lang(auto[0]["code"]) if auto else None
    if spoken:
        for t in manual:
            if base_lang(t["code"]) == spoken:
                return t
        return auto[0]
    return manual[0]


def _raw_to_cues(raw):
    """raw: [{"text", "start", "dur"}] in seconds -> cues"""
    raw = sorted(raw, key=lambda s: s["start"])
    cues = []
    for i, s in enumerate(raw):
        end = s["start"] + max(s["dur"], 0.2)
        if i + 1 < len(raw) and s["start"] < raw[i + 1]["start"] < end:
            end = raw[i + 1]["start"]  # rolling auto captions overlap
        text = _clean(s["text"])
        if text:
            cues.append({"start": s["start"], "end": end, "words": [(w, None) for w in text.split()]})
    return cues


def fetch_transcript_api(vid, lang):
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api import _errors as E

    kw = {}
    if proxy_url():
        from youtube_transcript_api.proxies import GenericProxyConfig

        kw["proxy_config"] = GenericProxyConfig(http_url=proxy_url(), https_url=proxy_url())
    api = YouTubeTranscriptApi(**kw)
    try:
        tl = api.list(vid)
        objs = list(tl)
        tracks = [
            {"code": t.language_code, "name": t.language, "generated": t.is_generated, "_obj": t}
            for t in objs
        ]
        choice = _pick(tracks, lang)
        if not choice:
            raise FetchError("no_transcript")
        fetched = choice["_obj"].fetch()
    except FetchError:
        raise
    except (E.TranscriptsDisabled, E.NoTranscriptFound) as e:
        raise FetchError("no_transcript", str(e)[:300])
    except (E.VideoUnavailable, E.InvalidVideoId) as e:
        raise FetchError("unavailable", str(e)[:300])
    except E.RequestBlocked as e:
        raise FetchError("blocked", str(e)[:300])
    except Exception as e:  # noqa: BLE001
        raise FetchError("error", f"{type(e).__name__}: {str(e)[:300]}")

    raw = [{"text": s.text, "start": float(s.start), "dur": float(s.duration)} for s in fetched]
    cues = _raw_to_cues(raw)
    meta = {
        "language_code": choice["code"],
        "language": choice["name"],
        "is_generated": choice["generated"],
        "tracks": [{k: t[k] for k in ("code", "name", "generated")} for t in tracks],
    }
    return cues, meta


# ------------------------------------------------- caption fetching (B) ----


def _json3_to_cues(data):
    cues = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        t0 = ev.get("tStartMs", 0) / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        words = []
        for sg in segs:
            txt = sg.get("utf8", "")
            if not txt.strip():
                continue
            has_off = "tOffsetMs" in sg or len(segs) > 1
            t = t0 + sg.get("tOffsetMs", 0) / 1000.0 if has_off else None
            for j, w in enumerate(_clean(txt).split()):
                words.append((w, t if j == 0 else None))
        if words:
            cues.append({"start": t0, "end": t0 + max(dur, 0.2), "words": words})
    for i in range(len(cues) - 1):
        if cues[i]["start"] < cues[i + 1]["start"] < cues[i]["end"]:
            cues[i]["end"] = cues[i + 1]["start"]
    return cues


def fetch_ytdlp(vid, lang):
    import yt_dlp

    opts = {"skip_download": True, "quiet": True, "no_warnings": True, "noplaylist": True}
    if proxy_url():
        opts["proxy"] = proxy_url()
    cookies = os.getenv("YT_COOKIES")
    if cookies:
        path = "/tmp/yt_cookies.txt"
        with open(path, "w") as f:
            f.write(cookies)
        opts["cookiefile"] = path
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            manual = info.get("subtitles") or {}
            auto = info.get("automatic_captions") or {}
            tracks = [{"code": c, "name": c, "generated": False} for c in manual if c != "live_chat"]
            orig = [c for c in auto if c.endswith("-orig")]
            spoken = (orig[0][:-5] if orig else None) or base_lang(info.get("language"))
            if spoken and (spoken in auto or orig):
                tracks.append({"code": orig[0] if orig else spoken, "name": spoken, "generated": True})
            choice = _pick(tracks, lang)
            if not choice:
                raise FetchError("no_transcript")
            pool = auto if choice["generated"] else manual
            fmts = pool.get(choice["code"]) or []
            url = next((f["url"] for f in fmts if f.get("ext") == "json3"), None)
            if not url:
                raise FetchError("no_transcript", "no json3 caption format")
            data = json.loads(ydl.urlopen(url).read().decode("utf-8"))
    except FetchError:
        raise
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        code = "blocked" if ("not a bot" in msg or "429" in msg or "Sign in" in msg) else "error"
        raise FetchError(code, msg[:300])
    cues = _json3_to_cues(data)
    if not cues:
        raise FetchError("no_transcript", "empty captions")
    meta = {
        "language_code": base_lang(choice["code"].replace("-orig", "")),
        "language": choice["name"],
        "is_generated": choice["generated"],
        "tracks": [{k: t[k] for k in ("code", "name", "generated")} for t in tracks],
        "title": info.get("title"),
    }
    return cues, meta


# ------------------------------------------------- caption fetching (C) ----
# Supadata (free plan: 100 transcripts/month). Used only when YouTube blocks the
# Space's own requests. Needs the Space secret SUPADATA_API_KEY.

SUPADATA = "https://api.supadata.ai/v1/transcript"


def fetch_supadata(vid, lang):
    key = os.getenv("SUPADATA_API_KEY", "").strip()
    if not key:
        raise FetchError("no_key", "SUPADATA_API_KEY secret is not set")
    headers = {"x-api-key": key}
    params = {"url": f"https://www.youtube.com/watch?v={vid}", "mode": "native"}
    if base_lang(lang):
        params["lang"] = base_lang(lang)
    try:
        r = requests.get(SUPADATA, params=params, headers=headers, timeout=60)
        if r.status_code == 202:  # async job — poll it
            job = (r.json() or {}).get("jobId")
            for _ in range(40):
                time.sleep(2)
                r = requests.get(f"{SUPADATA}/{job}", headers=headers, timeout=30)
                if r.status_code != 200 or (r.json() or {}).get("status") in ("completed", "failed"):
                    break
        data = r.json() if r.content else {}
    except Exception as e:  # noqa: BLE001
        raise FetchError("error", f"{type(e).__name__}: {str(e)[:200]}")
    if not isinstance(data, dict):
        data = {}
    if r.status_code >= 400 or data.get("error"):
        code = str(data.get("error") or r.status_code)
        msg = f"{r.status_code} {code} {data.get('message') or ''} {data.get('details') or ''}"[:300]
        if r.status_code in (401, 403) or "unauthorized" in code or "key" in code:
            raise FetchError("bad_key", msg)
        if r.status_code in (402, 429) or code in ("limit-exceeded", "upgrade-required"):
            raise FetchError("quota", msg)
        if r.status_code == 404 or code in ("transcript-unavailable", "not-found"):
            raise FetchError("no_transcript", msg)
        raise FetchError("error", msg)
    content = data.get("content") or (data.get("result") or {}).get("content") or []
    if not isinstance(content, list) or not content:
        raise FetchError("no_transcript", "empty transcript")
    raw = [
        {"text": str(c.get("text", "")), "start": float(c.get("offset", 0)) / 1000.0, "dur": float(c.get("duration", 0)) / 1000.0}
        for c in content
        if isinstance(c, dict)
    ]
    code = data.get("lang") or content[0].get("lang") or ""
    langs = [l for l in (data.get("availableLangs") or []) if isinstance(l, str)]
    meta = {
        "language_code": code,
        "language": code,
        "is_generated": None,
        "tracks": [{"code": l, "name": l, "generated": None} for l in langs],
    }
    return _raw_to_cues(raw), meta


# --------------------------------------- caption fetching (D): phone-assisted
# YouTube refuses this server's requests for a video's player data ("confirm
# you're not a bot"), but the signed caption URLs inside that data are not tied
# to an IP. So the user's phone (an iOS Shortcut from the YouTube share menu) or
# browser (a bookmarklet on youtube.com) asks YouTube for the player data and
# posts it here; the server then downloads and splits the captions itself.

IOS_CLIENT = {
    "clientName": "IOS", "clientVersion": "21.46.3", "deviceMake": "Apple", "deviceModel": "iPhone16,2",
    "osName": "iPhone", "osVersion": "18.6.0.22G86", "hl": "it", "gl": "IT",
}
_IOS_VERSION = {"v": IOS_CLIENT["clientVersion"], "checked": 0.0}


def ios_client():
    """YouTube stops answering old app versions after a while, so present the current
    YouTube iPhone app version, read from the App Store once a day."""
    if time.time() - _IOS_VERSION["checked"] > 24 * 3600:
        _IOS_VERSION["checked"] = time.time()
        try:
            r = requests.get("https://itunes.apple.com/lookup", params={"bundleId": "com.google.ios.youtube", "country": "us"}, timeout=8)
            v = str(r.json()["results"][0]["version"])
            if re.fullmatch(r"\d+\.\d+(\.\d+)?", v):
                _IOS_VERSION["v"] = v
        except Exception:  # noqa: BLE001
            pass  # keep the last known version
    return {**IOS_CLIENT, "clientVersion": _IOS_VERSION["v"]}
INGESTED = Cache(300)  # video id -> {"t", "title", "tracks": [... with "url"], "default": lang}
INGEST_TTL = 6 * 3600  # YouTube's caption URLs are valid for ~7 hours


def _text_of(name):
    if isinstance(name, dict):
        return name.get("simpleText") or "".join(r.get("text", "") for r in name.get("runs", []))
    return str(name or "")


def tracks_from_player(pr):
    c = (pr.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}
    tracks = [
        {
            "code": t.get("languageCode", ""),
            "name": _text_of(t.get("name")) or t.get("languageCode", ""),
            "generated": t.get("kind") == "asr",
            "url": t.get("baseUrl", ""),
        }
        for t in (c.get("captionTracks") or [])
        if t.get("baseUrl")
    ]
    default = None  # language of the video's original audio
    aud = c.get("audioTracks") or []
    di = c.get("defaultAudioTrackIndex", 0)
    if aud and isinstance(di, int) and 0 <= di < len(aud):
        aid = aud[di].get("audioTrackId") or ""
        if aid:
            default = base_lang(aid.split(".")[0])
        else:
            idx = aud[di].get("captionTrackIndices") or []
            if idx and isinstance(idx[0], int) and idx[0] < len(tracks):
                default = base_lang(tracks[idx[0]]["code"])
    return tracks, default


def json3_url(url):
    return re.sub(r"([?&])fmt=[^&]*", r"\1", url).rstrip("&") + "&fmt=json3"


def client_fetch_plan(vid, lang):
    """YouTube also throttles this server's caption downloads, so the page downloads the
    caption file itself (the signed URL works from any IP) and posts it to /api/split."""
    item = INGESTED.get(vid)
    if not item or time.time() - item["t"] > INGEST_TTL:
        return None
    choice = _pick(item["tracks"], lang or item["default"])
    if not choice:
        return None
    return {
        "need_captions": True,
        "video_id": vid,
        "url": json3_url(choice["url"]),
        "meta": {
            "language_code": choice["code"],
            "language": choice["name"],
            "is_generated": choice["generated"],
            "tracks": [{k: t[k] for k in ("code", "name", "generated")} for t in item["tracks"]],
            "title": item["title"],
        },
    }


def parse_posted(raw: bytes):
    """The player data as posted by a Shortcut (JSON, or a plist) or a bookmarklet (JSON text)."""
    for load in (lambda b: json.loads(b.decode("utf-8")), plistlib.loads):
        try:
            d = load(raw)
            if isinstance(d, dict):
                return d
        except Exception:  # noqa: BLE001
            pass
    s = raw.decode("utf-8", "ignore")  # e.g. a multipart wrapper around the JSON
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            d = json.loads(s[i : j + 1])
            if isinstance(d, dict):
                return d
        except Exception:  # noqa: BLE001
            pass
    return None


# ------------------------------------------------------ sentence split -----

TERMINAL = re.compile(r"[.!?…。！？]+[\"'»”’)\]]*$")
ABBREV = {"sig.", "sig.ra", "dott.", "dr.", "prof.", "ecc.", "etc.", "mr.", "mrs.", "ms.", "st.", "ing.", "avv.", "pag.", "vs."}
SOFT = (",", ";", ":", "—", "–")


def _timed_words(cues):
    """Flatten cues into words with a start time each (interpolating where unknown)."""
    out = []
    for c in cues:
        ws = c["words"]
        total = sum(len(w) + 1 for w, _ in ws) or 1
        off = 0
        span = max(c["end"] - c["start"], 0.2)
        for k, (w, t) in enumerate(ws):
            est = c["start"] + span * off / total
            out.append({
                "w": w,
                "t": t if t is not None else est,
                "real": t is not None,
                "cue_end": k == len(ws) - 1,
                "cend": c["end"],
            })
            off += len(w) + 1
    # keep times monotonic
    for i in range(1, len(out)):
        if out[i]["t"] < out[i - 1]["t"]:
            out[i]["t"] = out[i - 1]["t"]
    # estimate where each word ends (used to find pauses and sentence ends)
    for i, w in enumerate(out):
        nxt_t = out[i + 1]["t"] if i + 1 < len(out) else None
        limit = w["cend"] if (w["cue_end"] or nxt_t is None) else nxt_t
        if nxt_t is not None and nxt_t > w["t"]:
            limit = min(limit, nxt_t)
        e = min(limit, w["t"] + 0.15 + 0.07 * len(w["w"])) if w["real"] else limit
        w["e"] = max(e, w["t"] + 0.1)
    return out


def _join(ws):
    s = " ".join(w["w"] for w in ws)
    s = re.sub(r"\s+([,.;:!?…»”’)])", r"\1", s)
    s = re.sub(r"([«“(])\s+", r"\1", s)
    return s.strip()


def split_sentences(cues):
    words = _timed_words(cues)
    if not words:
        return [], False
    n_term = sum(1 for w in words if TERMINAL.search(w["w"]))
    punctuated = n_term >= max(2, len(words) / 40)
    has_word_times = sum(1 for c in cues for _, t in c["words"] if t is not None) > len(words) * 0.5

    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        n = len(cur)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["t"] - w["e"]) if nxt else 99
        brk = False
        if punctuated:
            if TERMINAL.search(w["w"]) and w["w"].lower() not in ABBREV and not re.fullmatch(r"\d+\.", w["w"]):
                brk = True
            elif n >= 24 and w["w"].endswith(SOFT):
                brk = True
            elif n >= 40:
                brk = True
        else:
            if has_word_times:
                brk = (gap >= 0.55 and n >= 4) or (gap >= 0.3 and n >= 10) or n >= 16
            else:
                brk = (w["cue_end"] and n >= 6) or (w["cue_end"] and gap >= 0.6 and n >= 3) or n >= 18
        if brk or not nxt:
            groups.append(cur)
            cur = []

    # merge tiny trailing fragments (1 word, no own punctuation) into the previous sentence
    merged = []
    for g in groups:
        if merged and len(g) == 1 and not TERMINAL.search(g[0]["w"]) and g[0]["t"] - merged[-1][-1]["e"] < 0.6:
            merged[-1].extend(g)
        else:
            merged.append(g)

    sents = []
    for g in merged:
        sents.append({"start": round(g[0]["t"], 2), "end": round(g[-1]["e"], 2), "text": _join(g)})
    # make sentences contiguous where the gap is small, so nothing is clipped
    for i in range(len(sents) - 1):
        a, b = sents[i], sents[i + 1]
        if b["start"] - a["end"] < 0.8:
            a["end"] = b["start"]
        else:
            a["end"] = round(a["end"] + 0.35, 2)
        if a["end"] <= a["start"]:
            a["end"] = round(a["start"] + 0.8, 2)
    if sents:
        sents[-1]["end"] = round(max(sents[-1]["end"] + 0.4, sents[-1]["start"] + 0.8), 2)
    return [s for s in sents if s["text"]], punctuated


# --------------------------------------------------------- translation -----

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


def _gtx(text, sl, tl):
    r = _session.post(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": sl or "auto", "tl": tl, "dt": "t"},
        data={"q": text},
        timeout=20,
        proxies={"https": proxy_url(), "http": proxy_url()} if proxy_url() else None,
    )
    r.raise_for_status()
    data = r.json()
    return "".join(part[0] for part in (data[0] or []) if part and part[0])


# Spoken fillers like "eh", "oh" push Google's Arabic into Egyptian dialect
# ("…ده ميلان ايه مش لينس"). They are removed from the text sent to the translator
# only; the sentence shown on screen stays exactly as spoken.
INTJ_RE = re.compile(r"[,;]?\s*\b(e+h+|e+hm+|a+h+|o+h+|u+h+|u+hm+|m+h+|beh|mah|ahah\w*)\b(?=[\s,.!?;:…]|$)", re.I)
# Words that mark Egyptian (or other spoken) Arabic rather than Modern Standard Arabic.
DIALECT_RE = re.compile(
    r"(?:^|(?<=[\s،.!?؟«\"]))(مفيش|ده|إيه|ايه|مش|عشان|علشان|كده|كدا|بتاع|بتاعة|بتاعت|أوي|اوي|إزاي|ازاي|دلوقتي|دلوقت"
    r"|عايز|عايزة|عاوز|فين|إمتى|امتى|بص|بس|اللي|ليه|يا عم|هوه|إحنا|احنا|بقى|كمان|برضه|برضو)(?=$|[\s،.!?؟»\"])"
)


def strip_interjections(s):
    c = INTJ_RE.sub("", s)
    c = re.sub(r"^[\s,;.…]+", "", c)
    c = re.sub(r"(^|[\"«“(])\s*[,;]\s*", r"\1", c)
    c = re.sub(r"\s+([,.!?;:])", r"\1", c)
    c = re.sub(r",(\s*[.!?])", r"\1", c)
    c = re.sub(r"\s{2,}", " ", c).strip()
    return c if re.search(r"[^\W\d_]", c) else s


def dialect_score(s):
    return len(DIALECT_RE.findall(s or ""))


def translate_many(texts, sl, tl):
    """Translate with caching. For Arabic, keep the output in Modern Standard Arabic."""
    arabic = base_lang(tl) == "ar"
    src = [strip_interjections(t) if arabic else t for t in texts]
    out = [None] * len(texts)
    todo = []
    for i, s in enumerate(src):
        hit = TRANSLATIONS.get((sl, tl, s))
        if hit is not None:
            out[i] = hit
        else:
            todo.append(i)
    if not todo:
        return out
    for i, r in zip(todo, _translate_list([src[i] for i in todo], sl, tl)):
        out[i] = r
    if arabic:  # anything still in dialect: re-translate through English, keep the more standard one
        bad = [i for i in todo if out[i] and dialect_score(out[i])]
        if bad:
            en = _translate_list([src[i] for i in bad], sl, "en")
            pairs = [(i, strip_interjections(e)) for i, e in zip(bad, en) if e]
            for (i, _), r in zip(pairs, _translate_list([e for _, e in pairs], "en", tl)):
                if r and dialect_score(r) < dialect_score(out[i]):
                    out[i] = r
    for i in todo:
        if out[i]:
            TRANSLATIONS.put((sl, tl, src[i]), out[i])
    return out


def _translate_list(texts, sl, tl):
    """Uncached batch translation, newline-joined chunks with a per-line fallback."""
    out = [None] * len(texts)
    chunks, cur, size = [], [], 0
    for i in range(len(texts)):
        L = len(texts[i]) + 1
        if cur and size + L > 3500:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += L
    if cur:
        chunks.append(cur)

    def run_chunk(idxs):
        joined = "\n".join(texts[i].replace("\n", " ") for i in idxs)
        try:
            res = _gtx(joined, sl, tl).split("\n")
        except Exception:  # noqa: BLE001
            res = []
        if len(res) == len(idxs):
            for i, r in zip(idxs, res):
                out[i] = r.strip()
        else:  # line count drifted — translate one by one
            for i in idxs:
                try:
                    out[i] = _gtx(texts[i], sl, tl).strip()
                except Exception:  # noqa: BLE001
                    out[i] = None

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(run_chunk, chunks))
    return out


# ----------------------------------------------------------------- app -----

app = gr.Server(docs_url=None, redoc_url=None, openapi_url=None)

MESSAGES = {
    "bad_url": "الرابط مو واضح — الصق رابط فيديو يوتيوب كامل.",
    "no_transcript": "هذا الفيديو ما فيه ترجمة نصية (Captions) على يوتيوب، فما أقدر أقسّمه لجمل.",
    "unavailable": "الفيديو غير متاح (خاص أو محذوف أو مقيّد).",
    "blocked": "يوتيوب حاجب سيرفر التطبيق. افتح الفيديو من تطبيق يوتيوب ← مشاركة ← Shadowing، أو من الكمبيوتر اضغط زر Shadowing في المفضلة.",
    "no_key": "يوتيوب حاجب سيرفر التطبيق. افتح الفيديو من تطبيق يوتيوب ← مشاركة ← Shadowing، أو من الكمبيوتر اضغط زر Shadowing في المفضلة.",
    "bad_key": "مفتاح Supadata غير صحيح — تأكد منه في إعدادات المساحة.",
    "quota": "خلصت حصة Supadata المجانية لهذا الشهر. الفيديوهات اللي فتحتها قبل محفوظة وتشتغل عادي.",
    "error": "صار خطأ وأنا أجيب نص الفيديو.",
}
FETCHERS = (
    ("transcript-api", fetch_transcript_api),
    ("yt-dlp", fetch_ytdlp),
    ("supadata", fetch_supadata),
)
SKIP_UNTIL = {}  # fetcher name -> time until which it is skipped (YouTube blocked it)
PRIORITY = ("no_transcript", "unavailable", "quota", "bad_key", "no_key", "blocked", "error")


CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS"}
# Other pages built on this server (the new page on GitHub Pages, or /v3 here) may open lessons.
PAGES = {"gh": "https://valeh4t.github.io/ZenaidiUno/", "v3": "/v3"}


def err(code, status, detail=""):
    body = {"error": code, "message": MESSAGES.get(code, code), "detail": detail}
    return JSONResponse(body, status_code=status, headers=CORS)


def ok(body):
    return JSONResponse(body, headers=CORS)


def build_result(vid, source, cues, meta, t0):
    sents, punctuated = split_sentences(cues)
    if not sents:
        raise FetchError("no_transcript", "no text")
    return {
        "video_id": vid,
        "source": source,
        "punctuated": punctuated,
        "seconds": round(time.time() - t0, 2),
        **meta,
        "sentences": sents,
    }


def app_base(request: Request) -> str:
    host = os.getenv("SPACE_HOST") or request.headers.get("x-forwarded-host") or request.url.netloc
    return f"https://{host}"


@app.get("/api/req")
def api_req(u: str = ""):
    """Step 1 of the Shortcut/bookmarklet: the request body to send to YouTube for this video."""
    body = {"context": {"client": ios_client()}, "videoId": video_id(u) or "", "contentCheckOk": True, "racyCheckOk": True}
    return PlainTextResponse(json.dumps(body), headers=CORS)


@app.options("/api/ingest")
@app.options("/api/req")
@app.options("/api/asr")
def api_preflight():
    return Response(status_code=204, headers=CORS)


@app.post("/api/ingest")
async def api_ingest(request: Request):
    """Step 3: receives YouTube's player data, prepares the lesson, answers with the page to open."""
    base = app_base(request)
    page = PAGES.get(request.query_params.get("to", ""), "/")  # which page should open the lesson
    page = page if page.startswith("https://") else base + page

    def reply(target):
        return PlainTextResponse(target, headers=CORS)

    def fail(msg):
        return reply(f"{page}#msg=" + quote(msg))

    pr = parse_posted(await request.body())
    if not pr:
        return fail("ما وصلت بيانات الفيديو من يوتيوب. راجع خطوات الاختصار.")
    vd = pr.get("videoDetails") or {}
    vid = vd.get("videoId")
    if not vid:
        ps = pr.get("playabilityStatus") or {}
        return fail("يوتيوب ما رجّع بيانات الفيديو: " + str(ps.get("reason") or ps.get("status") or "")[:150])
    tracks, default = tracks_from_player(pr)
    if not tracks:
        return fail(MESSAGES["no_transcript"])
    INGESTED.put(vid, {"t": time.time(), "title": vd.get("title") or "", "tracks": tracks, "default": default})
    for k in [k for k in TRANSCRIPTS.d if k[0] == vid]:  # fresh signed URLs → rebuild on next open
        TRANSCRIPTS.d.pop(k, None)
    return reply(f"{page}#v={vid}")


@app.post("/api/split")
def api_split(p: dict = Body(...)):
    """The page posts the caption file it downloaded from YouTube; answer with the sentences."""
    vid = video_id(str(p.get("video_id", "")))
    cap = p.get("captions")
    meta = p.get("meta") or {}
    if not vid or not cap:
        return err("error", 400, "missing captions")
    try:
        data = json.loads(cap) if isinstance(cap, str) else cap
        cues = _json3_to_cues(data)
        if not cues:
            raise FetchError("no_transcript", "empty caption file")
        keep = ("language_code", "language", "is_generated", "tracks", "title")
        res = build_result(vid, "phone", cues, {k: meta.get(k) for k in keep}, time.time())
    except FetchError as e:
        return err(e.code, 404, e.detail)
    except Exception as e:  # noqa: BLE001
        return err("error", 502, f"{type(e).__name__}: {str(e)[:200]}")
    TRANSCRIPTS.put((vid, base_lang(str(p.get("lang") or ""))), res)
    return res


@app.get("/api/transcript")
def api_transcript(url: str = Query(...), lang: str = Query(""), fresh: int = Query(0), raw: int = Query(0)):
    """raw=1: the page wants YouTube's caption file itself (it splits the sentences on the device)."""
    vid = video_id(url)
    if not vid:
        return err("bad_url", 400)
    key = (vid, base_lang(lang))
    plan = client_fetch_plan(vid, lang)
    if raw and plan:
        return ok(plan)
    if not fresh:
        hit = TRANSCRIPTS.get(key)
        if hit:
            return ok(hit)
    if plan:
        return ok(plan)
    errors = []
    for name, fn in FETCHERS:
        if SKIP_UNTIL.get(name, 0) > time.time():
            errors.append((name, "blocked", "skipped (blocked recently)"))
            continue
        try:
            t0 = time.time()
            res = build_result(vid, name, *fn(vid, lang), t0)
            TRANSCRIPTS.put(key, res)
            return ok(res)
        except FetchError as e:
            errors.append((name, e.code, e.detail))
            if e.code == "blocked" and name != "supadata":
                SKIP_UNTIL[name] = time.time() + 30 * 60
            if e.code in ("unavailable", "no_transcript"):
                break  # a definite answer about the video — don't spend Supadata credits
    codes = [c for _, c, _ in errors]
    code = next((c for c in PRIORITY if c in codes), "error")
    status = {"no_transcript": 404, "unavailable": 404, "blocked": 503, "no_key": 503, "quota": 429, "bad_key": 502}.get(code, 502)
    return err(code, status, " | ".join(f"{n}: {c} {d}" for n, c, d in errors)[:900])


@app.post("/api/translate")
def api_translate(payload: dict = Body(...)):
    texts = [str(t)[:1000] for t in (payload.get("texts") or [])][:200]
    tl = str(payload.get("target") or "ar")[:10]
    sl = base_lang(str(payload.get("source") or "auto")) or "auto"
    if not texts:
        return {"translations": []}
    return {"translations": translate_many(texts, sl, tl)}


# ------------------------------------------------------ pronunciation ---
# The page records the sentence with the microphone and sends it here; Whisper writes down what it heard
# and the page compares it with the sentence. (Safari's own speech recognition breaks on iPhone after the
# page plays the video, so recording + transcribing here is the reliable path.)

ASR_MODEL = os.getenv("ASR_MODEL", "small")
ASR_MODELS = ("tiny", "base", "small")
_ASR = {}
_ASR_LOAD = threading.Lock()
_ASR_RUN = threading.Lock()
_ASR_STATE = {"ready": False, "error": "", "load_secs": 0.0}


def asr_model(name):
    with _ASR_LOAD:
        m = _ASR.get(name)
        if m is None:
            from faster_whisper import WhisperModel

            t0 = time.time()
            m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=max(1, os.cpu_count() or 1))
            _ASR[name] = m
            if name == ASR_MODEL:
                _ASR_STATE["load_secs"] = round(time.time() - t0, 1)
        return m


def _warm_asr():
    try:
        import numpy as np

        m = asr_model(ASR_MODEL)
        list(m.transcribe(np.zeros(16000, dtype=np.float32), language="it", beam_size=1, without_timestamps=True)[0])
        _ASR_STATE["ready"] = True
    except Exception as e:  # noqa: BLE001
        _ASR_STATE["error"] = f"{type(e).__name__}: {e}"[:300]


threading.Thread(target=_warm_asr, daemon=True).start()


def audio_from_bytes(raw: bytes):
    """16 kHz mono float32 from a WAV (what the page sends) or any other audio file."""
    import io
    import wave

    import numpy as np

    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        with wave.open(io.BytesIO(raw)) as w:
            ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            frames = w.readframes(n)
        if sw != 2:
            raise ValueError("16-bit WAV only")
        a = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        if sr != 16000 and len(a):
            a = np.interp(np.arange(0, len(a) * 16000 / sr) * sr / 16000, np.arange(len(a)), a).astype(np.float32)
        return a
    from faster_whisper import decode_audio

    return decode_audio(io.BytesIO(raw), sampling_rate=16000)


def transcribe(raw: bytes, lang: str, name: str):
    audio = audio_from_bytes(raw)
    if len(audio) < 16000 * 0.25:
        return {"text": "", "secs": 0, "model": name, "short": True}
    lang = base_lang(lang)
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES as codes
    except Exception:  # noqa: BLE001
        codes = ()
    t0 = time.time()
    with _ASR_RUN:  # one at a time: the CPU is shared
        segs, _info = asr_model(name).transcribe(
            audio, language=lang if lang in codes else None, beam_size=1, best_of=1, temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
    return {"text": text, "secs": round(time.time() - t0, 2), "model": name, "audio_secs": round(len(audio) / 16000, 2)}


@app.post("/api/asr")
async def api_asr(request: Request, lang: str = Query(""), model: str = Query("")):
    from starlette.concurrency import run_in_threadpool

    raw = await request.body()
    if not raw:
        return err("no_audio", 400)
    if len(raw) > 3_000_000:
        return err("too_long", 413)
    name = model if model in ASR_MODELS else ASR_MODEL
    try:
        return ok(await run_in_threadpool(transcribe, raw, lang, name))
    except Exception as e:  # noqa: BLE001
        return err("asr_failed", 500, f"{type(e).__name__}: {e}"[:300])


def _deployed_sha():
    try:
        with open(os.path.join(HERE, "DEPLOYED_FROM_GITHUB")) as f:
            return f.read().strip()[:12] or "github"
    except OSError:
        return ""


@app.get("/healthz")
def health():
    return {
        "ok": True,
        "proxy": bool(proxy_url()),
        "supadata_key": bool(os.getenv("SUPADATA_API_KEY", "").strip()),
        "skipped": [n for n, t in SKIP_UNTIL.items() if t > time.time()],
        "ingested": len(INGESTED.d),
        "youtube_ios_version": _IOS_VERSION["v"],
        "v3": os.path.exists(os.path.join(HERE, "v3.html")),
        "deployed_from": os.getenv("DEPLOYED_FROM", "") or _deployed_sha(),
        "cpus": os.cpu_count(),
        "asr_model": ASR_MODEL,
        "asr": _ASR_STATE,
    }


def _icon_png(size):
    path = f"/tmp/icon2-{size}.png"
    if not os.path.exists(path):
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (size, size), (6, 37, 157))
        d = ImageDraw.Draw(im)
        s = size
        d.polygon([(s * 0.38, s * 0.28), (s * 0.38, s * 0.72), (s * 0.74, s * 0.5)], fill=(255, 255, 255))
        im.save(path)
    return path


@app.get("/icon-{size}.png")
def icon(size: int):
    size = 180 if size not in (180, 192, 512) else size
    return FileResponse(_icon_png(size), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/manifest.json")
def manifest():
    m = {
        "name": "Shadowing",
        "short_name": "Shadowing",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f8f6f2",
        "theme_color": "#f8f6f2",
        "icons": [{"src": f"/icon-{s}.png", "sizes": f"{s}x{s}", "type": "image/png"} for s in (192, 512)],
    }
    return Response(json.dumps(m), media_type="application/manifest+json")


@app.get("/v3")
def page_v3():
    path = os.path.join(HERE, "v3.html")
    if not os.path.exists(path):
        return Response("Not deployed yet", status_code=404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "index.html"), headers={"Cache-Control": "no-cache"})


# ZeroGPU Spaces only finish starting once the app declares a GPU function.
# This app never calls it, so it uses no GPU time.
if spaces is not None:

    @spaces.GPU
    def _zerogpu_placeholder():
        return None


if __name__ == "__main__":
    app.launch(ssr_mode=False, footer_links=[])
