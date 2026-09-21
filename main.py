import os
import re
import secrets
import asyncio
import logging
import requests
import edge_tts

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
from typing import Awaitable, Callable

from telegram import (
    Update,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()

# --- YouTube proxy config (optional, but usually required on a VPS) ----
# YouTube blocks most cloud/VPS IP ranges from the transcript endpoint.
# Configurable two ways:
#   1. Env vars at startup (WEBSHARE_PROXY_USERNAME/PASSWORD or
#      PROXY_HTTP_URL/PROXY_HTTPS_URL) — persists across restarts.
#   2. /setproxy in Telegram — changes it live, no restart needed, but
#      resets to the env-var default if the bot restarts.
# OWNER_ID locks /setproxy, /clearproxy and /checkproxy to one Telegram
# user ID — this is server-wide config, not a per-chat preference, so
# without a lock ANY user of the bot could hijack or disable it.
OWNER_ID_RAW = os.environ.get("OWNER_ID", "").strip()
OWNER_ID = int(OWNER_ID_RAW) if OWNER_ID_RAW.isdigit() else None

# Long-stable, always-captioned video used to smoke-test a proxy.
TEST_VIDEO_ID = "dQw4w9WgXcQ"

# Protocols tried, in order, when the input doesn't specify one explicitly.
PROXY_SCHEMES_TO_TRY = ["http", "socks5", "socks4"]

# ---- Bulk-checker tuning (all env-tunable) ----------------------------
# Concurrency: how many proxies are checked in parallel. Above ~100 your
# proxy provider usually becomes the bottleneck, not the bot.
MAX_CONCURRENT_PROXY_CHECKS = int(os.environ.get("PROXY_CHECK_CONCURRENCY", "50"))

# Timeouts for the LIGHTWEIGHT bulk check (a single HTTP request per proxy).
# Dead proxies used to hang for 60s on the transcript API's default timeout;
# 5s connect / 10s read makes them fail fast and multiplies throughput.
PROXY_CHECK_CONNECT_TIMEOUT = float(os.environ.get("PROXY_CHECK_CONNECT_TIMEOUT", "5"))
PROXY_CHECK_READ_TIMEOUT = float(os.environ.get("PROXY_CHECK_READ_TIMEOUT", "10"))

# Delay before the single retry per scheme on transient (non-block) errors.
PROXY_CHECK_RETRY_DELAY = float(os.environ.get("PROXY_CHECK_RETRY_DELAY", "0.5"))

# Size of the thread pool used to run blocking checks. Must be >= concurrency,
# otherwise tasks queue behind idle threads and your "50 parallel" runs at 32.
# Default = 2x concurrency, floored at 64, so there's headroom for retries too.
PROXY_CHECK_EXECUTOR_WORKERS = int(
    os.environ.get("PROXY_CHECK_EXECUTOR_WORKERS", str(max(64, MAX_CONCURRENT_PROXY_CHECKS * 2)))
)

# Minimum seconds between Telegram progress-message edits. Telegram rate-limits
# edits to roughly 1/sec per chat; 1.5s is safe and still feels live.
PROGRESS_EDIT_INTERVAL = float(os.environ.get("PROGRESS_EDIT_INTERVAL", "1.5"))

# Guardrails for the .txt upload path.
MAX_PROXY_FILE_BYTES = 10 * 1024 * 1024
ALLOWED_PROXY_FILE_EXTS = (".txt", ".csv", ".list", ".proxies")

# Live, mutable proxy state — starts from env vars, changeable via /setproxy.
current_proxy = {
    "type": None,  # None | "webshare" | "generic"
    "webshare_username": os.environ.get("WEBSHARE_PROXY_USERNAME", ""),
    "webshare_password": os.environ.get("WEBSHARE_PROXY_PASSWORD", ""),
    "http_url": os.environ.get("PROXY_HTTP_URL", ""),
    "https_url": os.environ.get("PROXY_HTTPS_URL", ""),
}
if current_proxy["webshare_username"] and current_proxy["webshare_password"]:
    current_proxy["type"] = "webshare"
elif current_proxy["http_url"] or current_proxy["https_url"]:
    current_proxy["type"] = "generic"


def is_owner(update: Update) -> bool:
    """No OWNER_ID configured = single-user/personal bot, anyone can manage the proxy."""
    if OWNER_ID is None:
        return True
    return bool(update.effective_user) and update.effective_user.id == OWNER_ID


def describe_current_proxy(reveal: bool = False) -> str:
    if current_proxy["type"] == "webshare":
        user = current_proxy["webshare_username"] or "(unset)"
        if not reveal and len(user) > 4:
            user = user[:2] + "…" + user[-2:]
        return f"Webshare ({user})"
    if current_proxy["type"] == "generic":
        url = current_proxy["http_url"] or current_proxy["https_url"] or "(unset)"
        if not reveal:
            url = re.sub(r"//[^@]+@", "//***:***@", url)  # mask user:pass@ in the URL
        return f"Generic ({url})"
    return "None — connecting directly"


def build_youtube_api() -> YouTubeTranscriptApi:
    """Instance YouTubeTranscriptApi, proxied per the current live config."""
    if current_proxy["type"] == "webshare":
        from youtube_transcript_api.proxies import WebshareProxyConfig
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=current_proxy["webshare_username"],
                proxy_password=current_proxy["webshare_password"],
            )
        )
    if current_proxy["type"] == "generic":
        from youtube_transcript_api.proxies import GenericProxyConfig
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=current_proxy["http_url"] or None,
                https_url=current_proxy["https_url"] or None,
            )
        )
    return YouTubeTranscriptApi()


def test_proxy_against_youtube() -> None:
    """FULL smoke test: fetch a real transcript for a known-good video through
    whatever proxy is currently ACTIVE. Slow (2-3 requests through the
    transcript library, 30-60s timeouts) — used only for a single proxy, by
    /checkproxy and /setproxy verification. Raises on failure; run via executor."""
    ytt_api = build_youtube_api()
    transcript_list = ytt_api.list(TEST_VIDEO_ID)
    try:
        transcript = next(iter(transcript_list))
    except StopIteration as exc:
        raise RuntimeError("Test video returned no transcript tracks.") from exc
    transcript.fetch()


def test_proxy_via_urls(http_url: str, https_url: str) -> None:
    """
    LIGHTWEIGHT bulk check: a single HTTP request to YouTube's oembed endpoint
    through the candidate proxy, with short timeouts.

    Why not use youtube_transcript_api here? Because a full transcript fetch
    makes 2-3 round trips with 30-60s default timeouts — a dead proxy would
    hang for a full minute before failing, capping bulk throughput at ~1/s
    regardless of concurrency. A single short-timeout request fails dead
    proxies in ~5s and multiplies throughput 10-20x.

    This checks REACHABILITY, not that YouTube will serve a transcript
    through that particular IP. The winning proxy is re-verified with the
    full test (test_proxy_against_youtube) before it's activated.
    """
    proxies = {}
    if http_url:
        proxies["http"] = http_url
    if https_url:
        proxies["https"] = https_url

    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://youtu.be/{TEST_VIDEO_ID}", "format": "json"},
            proxies=proxies or None,
            timeout=(PROXY_CHECK_CONNECT_TIMEOUT, PROXY_CHECK_READ_TIMEOUT),
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            allow_redirects=True,
        )
    except requests.exceptions.ProxyError as exc:
        raise RuntimeError(f"proxy connect failed: {exc}") from exc
    except requests.exceptions.ConnectTimeout as exc:
        raise RuntimeError(f"connect timeout ({PROXY_CHECK_CONNECT_TIMEOUT}s)") from exc
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(f"read timeout ({PROXY_CHECK_READ_TIMEOUT}s)") from exc
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(f"SSL error: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"connection error: {exc}") from exc

    if resp.status_code == 429:
        raise RuntimeError("rate-limited by YouTube (proxy reachable but throttled)")
    if resp.status_code >= 500:
        raise RuntimeError(f"upstream {resp.status_code}")
    resp.raise_for_status()


def build_generic_proxy_urls(
    host: str, port: str, user: str | None, password: str | None, scheme: str
) -> tuple[str, str]:
    auth = f"{user}:{password}@" if user and password else ""
    url = f"{scheme}://{auth}{host}:{port}"
    return url, url


# ----------------------------------------------------------------------
# Proxy string parsing
# ----------------------------------------------------------------------

_SCHEME_PREFIX_RE = re.compile(
    r"^(?P<scheme>https?|socks5h?|socks4a?|socks4)://(?P<rest>.+)$",
    re.IGNORECASE,
)
_USERPASS_AT_RE = re.compile(
    r"^(?P<user>[^:@\s]+):(?P<pw>[^:@\s]+)@(?P<host>[\w.\-]+):(?P<port>\d{2,5})$"
)
_INLINE_SEP_RE = re.compile(r"[,;\t]")


def _parse_hostport_pair(raw: str) -> tuple[str, str, str | None, str | None] | None:
    """(host, port, user|None, pass|None) or None. No scheme handling here."""
    raw = raw.strip().strip("'\"")
    m = _USERPASS_AT_RE.match(raw)
    if m:
        return m.group("host"), m.group("port"), m.group("user"), m.group("pw")

    parts = raw.split(":")
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1], None, None            # host:port
    if len(parts) == 4:
        a, b, c, d = parts
        if b.isdigit():
            return a, b, c, d                            # host:port:user:pass
        if d.isdigit():
            return c, d, a, b                            # user:pass:host:port
    return None


def parse_proxy_input(
    raw: str,
) -> tuple[str, str, str | None, str | None, str | None] | None:
    """
    (host, port, user|None, pass|None, forced_scheme|None) or None.
    forced_scheme is set only when the input explicitly had one — otherwise
    every scheme in PROXY_SCHEMES_TO_TRY is worth trying.
    """
    raw = raw.strip().strip("'\"")
    if not raw:
        return None

    m = _SCHEME_PREFIX_RE.match(raw)
    if m:
        inner = _parse_hostport_pair(m.group("rest"))
        if inner is None:
            return None
        host, port, user, pw = inner
        return host, port, user, pw, m.group("scheme").lower()

    inner = _parse_hostport_pair(raw)
    if inner is None:
        return None
    host, port, user, pw = inner
    return host, port, user, pw, None


def parse_proxy_lines(raw_text: str) -> list[str]:
    """
    Turn a pasted / uploaded blob into a clean list of candidate strings.

    Contract:
      * one proxy per line is the primary format
      * blank lines, full-line comments (#, //) and trailing # comments are ignored
      * if a line has commas/semicolons/tabs and doesn't parse as-is, split it
        (covers CSV exports and "one line, many proxies" pastes)
      * duplicates are dropped, order preserved
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("//"):
            continue

        candidates = [line]
        if parse_proxy_input(line) is None and _INLINE_SEP_RE.search(line):
            candidates = [tok.strip() for tok in _INLINE_SEP_RE.split(line) if tok.strip()]

        for cand in candidates:
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


# ----------------------------------------------------------------------
# Proxy auto-detection (single proxy, from /setproxy)
# ----------------------------------------------------------------------

async def auto_configure_proxy(
    chat,
    host: str,
    port: str,
    user: str | None,
    password: str | None,
    forced_scheme: str | None = None,
):
    """
    Try each candidate protocol (or just the one the input specified) against
    a live YouTube fetch, and activate the first one that actually works.
    Global state is untouched until a working scheme is confirmed.
    """
    schemes = [forced_scheme] if forced_scheme else PROXY_SCHEMES_TO_TRY
    status_msg = await chat.send_message(f"Auto-detecting proxy protocol for {host}:{port}…")

    loop = asyncio.get_running_loop()
    attempts = []
    for scheme in schemes:
        http_url, https_url = build_generic_proxy_urls(host, port, user, password, scheme)
        try:
            # Use the full test — this is a single proxy, correctness matters.
            await loop.run_in_executor(None, test_proxy_via_urls, http_url, https_url)
        except (RequestBlocked, IpBlocked):
            attempts.append(f"{scheme}:// → blocked by YouTube")
            continue
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{scheme}:// → {type(exc).__name__}: {exc}")
            continue

        current_proxy.update(
            type="generic", http_url=http_url, https_url=https_url,
            webshare_username="", webshare_password="",
        )
        await status_msg.edit_text(
            f"✅ Working via {scheme}:// — proxy set: {describe_current_proxy()}"
        )
        return

    report = "\n".join(attempts)
    await status_msg.edit_text(
        f"❌ {host}:{port} didn't work on {'/'.join(schemes)}:\n\n{report}\n\n"
        "Try a different proxy, or double-check the credentials."
    )


# --- NVIDIA NIM (translation) config -----------------------------------
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
# Just a starting default — /model (or the buttons) let a chat switch to any
# model ID at all, nothing is hardcoded or restricted.
DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"

# --- Edge TTS (voice generation) config ---------------------------------
# No API key needed. /setvoice accepts any edge-tts voice ID — nothing
# hardcoded or restricted; browse the real list with /voices <filter>.
DEFAULT_EDGE_VOICE = "en-US-AriaNeural"

YOUTUBE_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be|m\.youtube\.com)/[\w\-./?=&%]+",
    re.IGNORECASE,
)

# Quick-pick shortcuts shown as buttons — these never limit what you can
# type manually via /model or /setvoice, they're just one-tap defaults.
QUICK_LANGUAGES = [
    "Hindi", "English", "Urdu", "Spanish",
    "French", "Arabic", "Bengali", "Chinese",
    "Japanese", "German", "Russian", "Portuguese",
]

QUICK_MODELS = [
    ("Llama 3.1 70B", "meta/llama-3.1-70b-instruct"),
    ("Llama 3.1 8B", "meta/llama-3.1-8b-instruct"),
    ("Nemotron 70B", "nvidia/llama-3.1-nemotron-70b-instruct"),
    ("Mixtral 8x22B", "mistralai/mixtral-8x22b-instruct-v0.1"),
    ("Gemma 2 27B", "google/gemma-2-27b-it"),
]

QUICK_VOICES = [
    ("English (US, F)", "en-US-AriaNeural"),
    ("English (UK, M)", "en-GB-RyanNeural"),
    ("Hindi (F)", "hi-IN-SwaraNeural"),
    ("Hindi (M)", "hi-IN-MadhurNeural"),
    ("Urdu (M)", "ur-PK-AsadNeural"),
    ("Spanish (F)", "es-ES-ElviraNeural"),
    ("Arabic (M)", "ar-SA-HamedNeural"),
]


# --- Keyboards -------------------------------------------------------------

def build_main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Get a transcript", callback_data="menu:transcript")],
        [InlineKeyboardButton("🌐 Translate last transcript", callback_data="menu:translate")],
        [InlineKeyboardButton("🔊 Narrate last transcript", callback_data="menu:voice")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="menu:settings")],
    ])


def build_post_transcript_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🌐 Translate", callback_data="menu:translate"),
        InlineKeyboardButton("🔊 Narrate", callback_data="narrate:original"),
    ]])


def build_post_translate_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔊 Narrate this", callback_data="narrate:translated"),
        InlineKeyboardButton("🌐 Another language", callback_data="menu:translate"),
    ]])


def build_language_keyboard() -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(lang, callback_data=f"lang:{lang}") for lang in QUICK_LANGUAGES]
    rows = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    rows.append([InlineKeyboardButton("✍️ Type a different language", callback_data="lang:custom")])
    return InlineKeyboardMarkup(rows)


def build_model_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(label, callback_data=f"setmodel:{model_id}")] for label, model_id in QUICK_MODELS]
    rows.append([InlineKeyboardButton("📜 Browse every model", callback_data="models:list")])
    return InlineKeyboardMarkup(rows)


def build_voice_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(label, callback_data=f"setvoice:{voice_id}")] for label, voice_id in QUICK_VOICES]
    rows.append([InlineKeyboardButton("📜 Browse every voice", callback_data="voices:list")])
    return InlineKeyboardMarkup(rows)


def build_settings_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Change model", callback_data="changemodel")],
        [InlineKeyboardButton("🎙 Change voice", callback_data="changevoice")],
    ])


def build_stop_keyboard(run_id: str) -> InlineKeyboardMarkup:
    """Inline keyboard attached to the bulk-check progress message so a long
    run can be cancelled with one tap. run_id makes sure a stale button from
    an earlier (already-finished) run can't stop the current one."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🛑 Stop", callback_data=f"stopbulk:{run_id}")
    ]])


# --- Text chunking for Telegram's 4096-char message cap -----------------

def chunk_lines(lines: list[str], max_chars: int = 3800) -> list[str]:
    """Split a list of lines into chunks that never cut a line in half."""
    chunks, current = [], ""
    for line in lines:
        if current and len(current) + 1 + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


# --- Progress bar for the bulk proxy check ------------------------------

def format_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class ProgressTracker:
    """Live progress bar for the bulk proxy check.

    Edits the status message at most once every PROGRESS_EDIT_INTERVAL seconds
    so a large run doesn't trip Telegram's edit rate limit but still looks
    live. The final tick always forces an edit so the bar lands on 100%.
    When a stop_event is supplied and fires, the very next tick bypasses the
    throttle so the message immediately reflects "stopping".
    """

    def __init__(
        self,
        total: int,
        status_msg,
        header: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        stop_event: asyncio.Event | None = None,
    ):
        self.total = total
        self.status_msg = status_msg
        self.header = header
        self.reply_markup = reply_markup
        self.stop_event = stop_event
        self.done = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.started = asyncio.get_running_loop().time()
        self._lock = asyncio.Lock()
        self._last_edit = 0.0
        self._showed_stopping = False

    async def tick(self, ok: bool, skipped: bool = False) -> None:
        async with self._lock:
            self.done += 1
            if skipped:
                self.skipped += 1
            elif ok:
                self.passed += 1
            else:
                self.failed += 1

            now = asyncio.get_running_loop().time()
            is_last = self.done >= self.total
            stopping_now = self.stop_event is not None and self.stop_event.is_set()
            just_started_stopping = stopping_now and not self._showed_stopping
            if just_started_stopping:
                self._showed_stopping = True

            if (
                not is_last
                and not just_started_stopping
                and (now - self._last_edit) < PROGRESS_EDIT_INTERVAL
            ):
                return
            self._last_edit = now

        try:
            await self.status_msg.edit_text(self._render(), reply_markup=self.reply_markup)
        except Exception:
            pass  # message edited/deleted elsewhere — never fatal

    def _render(self) -> str:
        width = 18
        ratio = self.done / self.total if self.total else 0.0
        filled = int(width * ratio)
        bar = "█" * filled + "░" * (width - filled)
        pct = ratio * 100

        elapsed = asyncio.get_running_loop().time() - self.started
        rate = self.done / elapsed if elapsed > 0 and self.done > 0 else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 else 0.0

        lines = [
            self.header,
            "",
            f"`[{bar}]` {pct:5.1f}%",
            f"✅ {self.passed} working   ❌ {self.failed} failed",
            f"⚡ {rate:0.1f}/s   ⏳ ETA {format_duration(eta)}",
        ]
        if self.skipped:
            lines.append(f"⏭ {self.skipped} skipped")
        if self.stop_event is not None and self.stop_event.is_set():
            lines.append("🛑 Stopping…")
        return "\n".join(lines)


# --- YouTube transcript helpers ---------------------------------------

def extract_video_id(url: str) -> str | None:
    """Pull the 11-character video ID out of any common YouTube URL shape."""
    url = url.strip()
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.netloc or "").lower()

    if "youtu.be" in host:
        video_id = parsed.path.lstrip("/")
        return video_id.split("/")[0] if video_id else None

    if "youtube.com" in host:
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            return qs.get("v", [None])[0]
        # /shorts/<id>, /embed/<id>, /live/<id>
        for prefix in ("/shorts/", "/embed/", "/live/"):
            if parsed.path.startswith(prefix):
                return parsed.path[len(prefix):].split("/")[0]

    return None


def fetch_transcript_text(video_id: str, preferred_langs=("en",)) -> tuple[str, str]:
    """
    Returns (clean_text, language_used). Tries manually-created transcripts
    first, then falls back to auto-generated ones, in the preferred
    languages, then finally whatever is available.
    """
    ytt_api = build_youtube_api()
    transcript_list = ytt_api.list(video_id)

    transcript = None
    try:
        transcript = transcript_list.find_manually_created_transcript(preferred_langs)
    except NoTranscriptFound:
        pass

    if transcript is None:
        try:
            transcript = transcript_list.find_generated_transcript(preferred_langs)
        except NoTranscriptFound:
            pass

    if transcript is None:
        # last resort: grab the first transcript available, in any language
        try:
            transcript = next(iter(transcript_list))
        except StopIteration as exc:
            raise NoTranscriptFound(video_id, preferred_langs, transcript_list) from exc

    raw_entries = transcript.fetch()
    language_used = transcript.language

    return clean_transcript(raw_entries), language_used


def clean_transcript(entries) -> str:
    """
    Turn the list of {text, start, duration} chunks into readable
    paragraphs: strip timestamps, fix spacing/line-break artifacts, and
    start a new paragraph whenever there's a natural pause (>2.5s gap)
    in the speech.

    Some caption tracks (especially community-uploaded / lyric-style ones)
    embed a literal timestamp label like "0:05" or "[1:23:45]" at the START
    of each caption line, separate from the real start/duration timing
    metadata. Left in, these get mistranslated and read aloud as digits by
    TTS, so a *leading* timestamp is stripped here. Only the leading
    position is targeted — a time mentioned naturally mid-sentence (e.g.
    "the meeting is at 10:30") is left alone since it's real spoken
    content, not an artifact. The real timing used for paragraph breaks
    below always comes from entry.start/entry.duration, never the text.
    """
    leading_timestamp = re.compile(
        r"^[\[\(]?\d{1,2}(?::\d{2}){1,2}\b[\]\)]?\s*[-–—:]?\s*"
    )

    paragraphs = []
    current = []
    last_end = 0.0

    for entry in entries:
        text = entry.text.replace("\n", " ").strip()
        text = leading_timestamp.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        gap = entry.start - last_end
        if current and gap > 2.5:
            paragraphs.append(" ".join(current))
            current = []

        current.append(text)
        last_end = entry.start + entry.duration

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


async def fetch_transcript_or_report(
    chat, status_msg, url: str
) -> tuple[str, str, str] | None:
    """Shared fetch+error-reporting used by both /transcript and /translate flows.
    Returns (video_id, text, language) on success, None on error (already reported)."""
    video_id = extract_video_id(url)
    if not video_id:
        await status_msg.edit_text(
            "That doesn't look like a valid YouTube URL. Try a link like "
            "https://youtube.com/watch?v=... or https://youtu.be/..."
        )
        return None
    try:
        text, language = fetch_transcript_text(video_id)
    except TranscriptsDisabled:
        await status_msg.edit_text("Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        await status_msg.edit_text("No transcript is available for this video.")
        return None
    except VideoUnavailable:
        await status_msg.edit_text("That video is unavailable (private, deleted, or region-locked).")
        return None
    except (RequestBlocked, IpBlocked):
        await status_msg.edit_text(
            "YouTube is blocking this server's IP address — very common when a bot "
            "runs on a VPS/cloud host. This isn't a one-off, it'll keep happening "
            "until a proxy is set.\n\n"
            "Fix: upload a .txt with working proxies and I'll test them and set "
            "the first working one, or /setproxy <proxy> directly."
        )
        return None
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors to the user
        logger.exception("Transcript fetch failed for %s", video_id)
        await status_msg.edit_text(f"Couldn't fetch that transcript: {exc}")
        return None
    return video_id, text, language


# --- NVIDIA NIM helpers ---------------------------------------------------

def list_nvidia_models() -> list[str]:
    """Live list of every model this NVIDIA API key can call — no whitelist."""
    resp = requests.get(
        NVIDIA_MODELS_URL,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return sorted(m["id"] for m in data.get("data", []))


def call_nvidia_chat(model: str, messages: list) -> str:
    """One blocking call to a NIM chat-completions model. Run via executor."""
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set on the server.")

    resp = requests.post(
        NVIDIA_CHAT_URL,
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        timeout=180,
    )
    if not resp.ok:
        raise RuntimeError(f"NVIDIA API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    """Split on paragraph breaks so no single request gets too large for the model."""
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks or [text]


async def translate_text(
    text: str,
    target_language: str,
    model: str,
    progress_cb: Callable[[int, int], Awaitable[None]] | None = None,
) -> str:
    """Translate arbitrarily long text via NVIDIA NIM, chunk by chunk, in order.
    Optionally calls progress_cb(done, total) after each chunk."""
    loop = asyncio.get_running_loop()
    chunks = chunk_text(text)
    total = len(chunks)
    translated_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise professional translator. Translate the "
                    "user's text faithfully into the requested language. Preserve "
                    "paragraph breaks and tone. Output ONLY the translated text, "
                    "no notes, no explanations, no original text."
                ),
            },
            {
                "role": "user",
                "content": f"Translate the following text to {target_language}:\n\n{chunk}",
            },
        ]
        result = await loop.run_in_executor(None, call_nvidia_chat, model, messages)
        translated_chunks.append(result)
        if progress_cb is not None:
            try:
                await progress_cb(idx, total)
            except Exception:
                pass  # a failed progress ping must never kill the translation
    return "\n\n".join(translated_chunks)


# --- Edge TTS helpers ------------------------------------------------------

async def list_edge_voices(filter_str: str | None = None) -> list[str]:
    """Live list of every edge-tts voice, optionally filtered — no whitelist."""
    voices = await edge_tts.list_voices()
    names = sorted(v["ShortName"] for v in voices)
    if filter_str:
        f = filter_str.lower()
        names = [n for n in names if f in n.lower()]
    return names


async def generate_speech(text: str, voice: str, output_path: str) -> None:
    """Render text to an mp3 with edge-tts. Handles arbitrarily long text."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


# --- Shared action flows (used by both commands and inline buttons) -------

async def run_transcript_fetch(
    update: Update, context: ContextTypes.DEFAULT_TYPE, url: str
) -> None:
    chat = update.effective_chat
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    status_msg = await chat.send_message("Fetching transcript…")

    result = await fetch_transcript_or_report(chat, status_msg, url)
    if result is None:
        return
    video_id, text, language = result

    if not text.strip():
        await status_msg.edit_text("The transcript came back empty.")
        return

    # cache so /translate and /voice can reuse it without re-fetching
    context.chat_data["last_transcript"] = text
    context.chat_data["last_video_id"] = video_id
    context.chat_data.pop("last_translation", None)

    file_path = f"/tmp/transcript_{video_id}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Transcript for https://youtu.be/{video_id} (language: {language})\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

    try:
        await status_msg.delete()
        with open(file_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat.id,
                document=f,
                filename=f"transcript_{video_id}.txt",
                caption=f"Transcript ready ({language}). What next?",
                reply_markup=build_post_transcript_keyboard(),
            )
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


async def run_translation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target_language: str,
    url: str | None,
) -> None:
    chat = update.effective_chat
    if not NVIDIA_API_KEY:
        await chat.send_message("NVIDIA_API_KEY is not set on the server.")
        return

    if url:
        status_msg = await chat.send_message("Fetching transcript…")
        result = await fetch_transcript_or_report(chat, status_msg, url)
        if result is None:
            return
        video_id, text, _ = result
        context.chat_data["last_transcript"] = text
        context.chat_data["last_video_id"] = video_id
    else:
        text = context.chat_data.get("last_transcript")
        video_id = context.chat_data.get("last_video_id", "transcript")
        if not text:
            await chat.send_message(
                "No transcript on file yet. Send a YouTube link first, or run "
                "/translate <language> <url> directly."
            )
            return
        status_msg = await chat.send_message("Translating…")

    model = context.chat_data.get("nvidia_model", DEFAULT_NVIDIA_MODEL)
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    await status_msg.edit_text(f"Translating with {model}…")

    async def on_chunk_progress(done: int, total: int) -> None:
        try:
            await status_msg.edit_text(f"Translating with {model}… ({done}/{total})")
        except Exception:
            pass  # edit raced with something else; not fatal

    try:
        translated = await translate_text(text, target_language, model, on_chunk_progress)
    except Exception as exc:
        logger.exception("Translation failed")
        await status_msg.edit_text(f"Translation failed: {exc}")
        return

    safe_lang = re.sub(r"[^A-Za-z0-9]+", "_", target_language).strip("_") or "translated"
    context.chat_data["last_translation"] = translated
    file_path = f"/tmp/transcript_{video_id}_{safe_lang}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Translated transcript — target language: {target_language} — model: {model}\n")
        f.write("=" * 60 + "\n\n")
        f.write(translated)

    try:
        await status_msg.delete()
        with open(file_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat.id,
                document=f,
                filename=os.path.basename(file_path),
                caption=f"Translated to {target_language} (model: {model}).",
                reply_markup=build_post_translate_keyboard(),
            )
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


async def run_voice_generation(
    update: Update, context: ContextTypes.DEFAULT_TYPE, source: str | None
) -> None:
    chat = update.effective_chat

    if source == "original":
        text = context.chat_data.get("last_transcript")
    elif source == "translated":
        text = context.chat_data.get("last_translation")
    else:
        # default: narrate the translation if one exists, else the raw transcript
        text = context.chat_data.get("last_translation") or context.chat_data.get("last_transcript")

    if not text:
        await chat.send_message(
            "No transcript on file yet. Fetch one first (send a YouTube link, "
            "or /transcript <url>), then run /voice."
        )
        return

    voice = context.chat_data.get("edge_voice", DEFAULT_EDGE_VOICE)
    video_id = context.chat_data.get("last_video_id", "audio")

    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.UPLOAD_VOICE)
    status_msg = await chat.send_message(f"Generating speech with {voice}…")

    file_path = f"/tmp/speech_{video_id}.mp3"
    try:
        await generate_speech(text, voice, file_path)
    except Exception as exc:
        logger.exception("TTS generation failed")
        await status_msg.edit_text(f"Speech generation failed: {exc}")
        try:
            os.remove(file_path)
        except OSError:
            pass
        return

    try:
        await status_msg.delete()
        with open(file_path, "rb") as f:
            await context.bot.send_audio(
                chat_id=chat.id,
                audio=f,
                title=f"{video_id} narration",
                performer=voice,
                caption=f"Voice: {voice}",
            )
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


async def send_all_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not NVIDIA_API_KEY:
        await chat.send_message("NVIDIA_API_KEY is not set on the server.")
        return

    loop = asyncio.get_running_loop()
    try:
        models = await loop.run_in_executor(None, list_nvidia_models)
    except Exception as exc:
        await chat.send_message(f"Couldn't fetch model list: {exc}")
        return

    if not models:
        await chat.send_message("NVIDIA returned no models for this key.")
        return

    header = "Models available to your key (pick any with /model <id>):"
    for chunk in chunk_lines([header, ""] + models):
        await chat.send_message(chunk)


async def send_all_voices(
    update: Update, context: ContextTypes.DEFAULT_TYPE, filter_str: str | None
) -> None:
    chat = update.effective_chat
    try:
        names = await list_edge_voices(filter_str)
    except Exception as exc:
        await chat.send_message(f"Couldn't fetch voice list: {exc}")
        return

    if not names:
        await chat.send_message("No voices matched that filter.")
        return

    header = (
        f"Voices matching '{filter_str}' (pick one with /setvoice <id>):"
        if filter_str
        else "All edge-tts voices — filter it, e.g. /voices en-US or /voices Hindi:"
    )
    for chunk in chunk_lines([header, ""] + names):
        await chat.send_message(chunk)


async def send_settings_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    model = context.chat_data.get("nvidia_model", DEFAULT_NVIDIA_MODEL)
    voice = context.chat_data.get("edge_voice", DEFAULT_EDGE_VOICE)
    text = f"⚙️ Current settings\n\nTranslation model: {model}\nNarration voice: {voice}"
    await chat.send_message(text, reply_markup=build_settings_keyboard())


# --- Commands ---------------------------------------------------------

HELP_TEXT = (
    "Send me a YouTube link (or /transcript <url>) and I'll send back "
    "a clean .txt transcript of the video.\n\n"
    "Translation (NVIDIA NIM):\n"
    "/translate <language> — translate the last transcript fetched here\n"
    "/translate <language> <url> — fetch + translate in one go\n"
    "/models — list every model your NVIDIA API key can use\n"
    "/model <model_id> — pick which one to translate with\n\n"
    "Voice (Edge TTS, free):\n"
    "/voice [original|translated] — narrate the last transcript/translation as .mp3\n"
    "/voices <filter> — browse voices, e.g. /voices en-US\n"
    "/setvoice <voice_id> — pick which voice to narrate with\n\n"
    "/settings — see and change your current model & voice\n\n"
    "Proxy (only needed if YouTube blocks this server's IP):\n"
    "The reliable way: just upload a .txt file with one proxy per line. "
    "I'll test them all (with a live progress bar and a 🛑 Stop button), "
    "activate the first working one, and send back:\n"
    "  • clean_proxies.txt — only the working ones\n"
    "  • proxy_check_report.txt — full pass/fail with reasons\n\n"
    "/setproxy <anything> — auto-detect a single proxy and set it\n"
    "/checkproxies <list> — test a short inline list\n"
    "/proxystatus — show the current proxy\n"
    "/checkproxy — re-verify the current proxy\n"
    "/clearproxy — go back to a direct connection"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, reply_markup=build_main_menu_keyboard())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, reply_markup=build_main_menu_keyboard())


async def transcript_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /transcript <youtube_url>")
        return
    await run_transcript_fetch(update, context, context.args[0])


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_all_models(update, context)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        model_id = " ".join(context.args).strip()
        context.chat_data["nvidia_model"] = model_id
        await update.message.reply_text(f"Translation model set to: {model_id}")
        return

    current = context.chat_data.get("nvidia_model", DEFAULT_NVIDIA_MODEL)
    await update.message.reply_text(
        f"Current translation model: {current}\n\n"
        "Pick a shortcut below, or set any other model your key supports "
        "with /model <model_id>.",
        reply_markup=build_model_keyboard(),
    )


async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not NVIDIA_API_KEY:
        await update.message.reply_text("NVIDIA_API_KEY is not set on the server.")
        return
    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "/translate <language> — translate the last transcript fetched here\n"
            "/translate <language> <youtube_url> — fetch + translate together\n"
            "Example: /translate Hindi\n"
            "Example: /translate Spanish https://youtu.be/dQw4w9WgXcQ"
        )
        return

    args = list(context.args)
    url = args[-1] if args and YOUTUBE_URL_PATTERN.search(args[-1]) else None
    if url:
        args = args[:-1]
    target_language = " ".join(args).strip()

    if not target_language:
        await update.message.reply_text("Please specify a target language, e.g. /translate French")
        return

    await run_translation(update, context, target_language, url)


async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    filter_str = " ".join(context.args) if context.args else None
    await send_all_voices(update, context, filter_str)


async def setvoice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        voice_id = context.args[0].strip()
        context.chat_data["edge_voice"] = voice_id
        await update.message.reply_text(f"Voice set to: {voice_id}")
        return

    current = context.chat_data.get("edge_voice", DEFAULT_EDGE_VOICE)
    await update.message.reply_text(
        f"Current voice: {current}\n\n"
        "Pick a shortcut below, or set any edge-tts voice with /setvoice <voice_id>.",
        reply_markup=build_voice_keyboard(),
    )


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    source = context.args[0].lower().strip() if context.args else None
    if source is not None and source not in ("original", "translated"):
        await update.message.reply_text(
            "Usage: /voice — narrate the latest text\n"
            "/voice original — force the original transcript\n"
            "/voice translated — force the translated version"
        )
        return
    await run_voice_generation(update, context, source)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_settings_message(update, context)


async def setproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update):
        await update.message.reply_text("Only the bot owner can change the proxy.")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Paste a proxy in ANY common shape and I'll auto-detect the format "
            "and protocol, then verify it against YouTube before activating it:\n"
            "• host:port:username:password\n"
            "• username:password@host:port\n"
            "• host:port (no auth)\n"
            "• scheme://user:pass@host:port (http/https/socks5/socks4)\n\n"
            "Or be explicit:\n"
            "/setproxy webshare <username> <password>\n"
            "/setproxy generic <http_url> [https_url]\n\n"
            "For a LIST of proxies, just upload a .txt with one per line — that's "
            "the reliable path, and you'll get back a clean file.\n\n"
            f"Current: {describe_current_proxy()}"
        )
        return

    mode = args[0].lower()
    chat = update.effective_chat

    if mode == "webshare":
        if len(args) < 3:
            await update.message.reply_text("Usage: /setproxy webshare <username> <password>")
            return
        user, password = args[1], args[2]
        previous = dict(current_proxy)
        current_proxy.update(
            type="webshare", webshare_username=user, webshare_password=password,
            http_url="", https_url="",
        )
        status_msg = await chat.send_message(f"Testing Webshare ({user}) against YouTube…")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, test_proxy_against_youtube)
        except Exception as exc:  # noqa: BLE001
            current_proxy.clear()
            current_proxy.update(previous)
            await status_msg.edit_text(f"❌ Didn't work: {type(exc).__name__}: {exc}\nReverted.")
            return
        await status_msg.edit_text(f"✅ Working — proxy set: {describe_current_proxy()}")
        return

    if mode == "generic":
        if len(args) < 2:
            await update.message.reply_text("Usage: /setproxy generic <http_url> [https_url]")
            return
        http_url = args[1]
        https_url = args[2] if len(args) > 2 else args[1]
        previous = dict(current_proxy)
        current_proxy.update(
            type="generic", http_url=http_url, https_url=https_url,
            webshare_username="", webshare_password="",
        )
        status_msg = await chat.send_message("Testing generic proxy against YouTube…")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, test_proxy_against_youtube)
        except Exception as exc:  # noqa: BLE001
            current_proxy.clear()
            current_proxy.update(previous)
            await status_msg.edit_text(f"❌ Didn't work: {type(exc).__name__}: {exc}\nReverted.")
            return
        await status_msg.edit_text(f"✅ Working — proxy set: {describe_current_proxy()}")
        return

    # Anything else: auto-detect. Accept one token, or several space-separated
    # fields (host port user pass) that some providers export instead of colons.
    candidate = args[0] if len(args) == 1 else ":".join(args)
    parsed = parse_proxy_input(candidate)
    if parsed is None:
        await update.message.reply_text(
            "Couldn't recognize that format. Paste it exactly as your provider gave "
            "it to you, or be explicit with /setproxy webshare <user> <pass> or "
            "/setproxy generic <http_url>."
        )
        return

    host, port, user, password, forced_scheme = parsed
    await auto_configure_proxy(chat, host, port, user, password, forced_scheme)


# ----------------------------------------------------------------------
# Bulk proxy check — the reliable file-in / file-out path
# ----------------------------------------------------------------------

async def check_proxy_candidate(
    loop,
    semaphore: asyncio.Semaphore,
    raw: str,
    parsed,
    stop_event: asyncio.Event | None = None,
) -> dict:
    """
    Test one proxy candidate reliably: try every viable protocol (or just the
    one an explicit scheme:// URL specified), with one retry per protocol to
    smooth over transient network blips. A definitive YouTube block is NOT
    retried (retrying won't change a hard block) — everything else gets a
    second attempt before being marked failed.

    If stop_event is set at any checkpoint, returns immediately with
    skipped=True so a stopped run drains its queue in near-zero time.
    """
    def _skipped() -> dict:
        return {
            "raw": raw, "ok": False, "skipped": True, "scheme": None,
            "http_url": None, "https_url": None, "note": "stopped",
        }

    host, port, user, password, forced_scheme = parsed
    schemes = [forced_scheme] if forced_scheme else PROXY_SCHEMES_TO_TRY
    attempts = []
    async with semaphore:
        if stop_event is not None and stop_event.is_set():
            return _skipped()
        for scheme in schemes:
            if stop_event is not None and stop_event.is_set():
                return _skipped()
            http_url, https_url = build_generic_proxy_urls(host, port, user, password, scheme)
            for attempt_num in (1, 2):
                if stop_event is not None and stop_event.is_set():
                    return _skipped()
                try:
                    await loop.run_in_executor(None, test_proxy_via_urls, http_url, https_url)
                    return {
                        "raw": raw, "ok": True, "skipped": False, "scheme": scheme,
                        "http_url": http_url, "https_url": https_url,
                        "note": f"working via {scheme}://",
                    }
                except (RequestBlocked, IpBlocked):
                    attempts.append(f"{scheme}:// → blocked by YouTube")
                    break  # deterministic block — retrying this scheme won't help
                except Exception as exc:  # noqa: BLE001
                    if attempt_num == 1:
                        await asyncio.sleep(PROXY_CHECK_RETRY_DELAY)
                        continue
                    attempts.append(f"{scheme}:// → {type(exc).__name__}: {exc}")
    return {
        "raw": raw, "ok": False, "skipped": False, "scheme": None,
        "http_url": None, "https_url": None,
        "note": "; ".join(attempts) or "no working protocol",
    }


async def run_bulk_proxy_check(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    raw_text: str,
    status_msg=None,
    source_label: str = "your list",
) -> None:
    """
    The one engine behind /checkproxies and .txt-file uploads.

    Input:  raw text (one proxy per line, blank lines / # comments allowed)
    Output: clean_proxies.txt (working only) + proxy_check_report.txt (full pass/fail),
            and the first working proxy is activated (after full verification).
    The progress message carries a 🛑 Stop button; pressing it sets a stop
    event, short-circuits every queued check, and still produces both files
    from whatever was already tested.
    """
    chat = update.effective_chat

    raw_candidates = parse_proxy_lines(raw_text)
    if not raw_candidates:
        msg = "No proxy-looking lines found in that input."
        if status_msg:
            await status_msg.edit_text(msg)
        else:
            await chat.send_message(msg)
        return

    parsed_list: list[tuple[str, tuple]] = []
    unparsed: list[str] = []
    for raw in raw_candidates:
        parsed = parse_proxy_input(raw)
        if parsed is None:
            unparsed.append(raw)
        else:
            parsed_list.append((raw, parsed))

    if not parsed_list:
        sample = "\n".join(unparsed[:5])
        msg = (
            f"None of the {len(raw_candidates)} lines looked like a proxy I could parse.\n"
            f"First few:\n{sample}"
        )
        if status_msg:
            await status_msg.edit_text(msg)
        else:
            await chat.send_message(msg)
        return

    total = len(parsed_list)
    header = (
        f"Testing {total} prox{'y' if total == 1 else 'ies'} from {source_label} "
        f"({MAX_CONCURRENT_PROXY_CHECKS} parallel, "
        f"{PROXY_CHECK_CONNECT_TIMEOUT:.0f}s/{PROXY_CHECK_READ_TIMEOUT:.0f}s timeouts)"
    )

    # Register the run so the Stop button has something to flip. A fresh run_id
    # makes sure a stale button from an earlier run can't cancel this one.
    run_id = secrets.token_hex(4)
    stop_event = asyncio.Event()
    context.chat_data["bulk_proxy_run"] = {"id": run_id, "stop": stop_event}
    stop_kb = build_stop_keyboard(run_id)

    if status_msg:
        await status_msg.edit_text(header, reply_markup=stop_kb)
    else:
        status_msg = await chat.send_message(header, reply_markup=stop_kb)

    tracker = ProgressTracker(
        total, status_msg, header,
        reply_markup=stop_kb, stop_event=stop_event,
    )

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(MAX_CONCURRENT_PROXY_CHECKS)

    async def one(raw: str, parsed: tuple) -> dict:
        result = await check_proxy_candidate(loop, sem, raw, parsed, stop_event)
        await tracker.tick(result["ok"], skipped=result.get("skipped", False))
        return result

    try:
        results = await asyncio.gather(*(one(raw, p) for raw, p in parsed_list))
    finally:
        # Whether we finished or got cancelled, clear the run registration so
        # the stop button (which is about to disappear anyway) has nothing to
        # flip, and so a new run can start cleanly.
        context.chat_data.pop("bulk_proxy_run", None)

    was_stopped = stop_event.is_set()
    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"] and not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]

    # ---- build the two output files -----------------------------------
    clean_body = "\n".join(r["raw"] for r in passed)
    if clean_body:
        clean_body += "\n"

    report_lines = [
        f"Proxy check report — {len(passed)}/{len(results)} working",
        f"Source: {source_label}",
    ]
    if was_stopped:
        report_lines.append("Run was stopped early by the user.")
    report_lines.append("")
    for r in results:
        if r.get("skipped"):
            tag = "SKIP"
        elif r["ok"]:
            tag = "PASS"
        else:
            tag = "FAIL"
        report_lines.append(f"[{tag}] {r['raw']} — {r['note']}")
    if unparsed:
        report_lines.append("")
        report_lines.append(f"Couldn't parse ({len(unparsed)} lines, skipped):")
        report_lines.extend(unparsed)

    clean_path = "/tmp/clean_proxies.txt"
    report_path = "/tmp/proxy_check_report.txt"
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_body)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # Remove the stop button by deleting the progress message before we post
    # the result files. (Deleting the message removes its inline keyboard.)
    try:
        await status_msg.delete()
    except Exception:
        pass

    summary_parts = [f"{len(passed)}/{len(results)} working"]
    if was_stopped:
        summary_parts.append(f"stopped early ({len(skipped)} skipped)")
    if unparsed:
        summary_parts.append(f"{len(unparsed)} unparseable line(s) skipped")
    summary = ", ".join(summary_parts)

    try:
        if passed:
            with open(clean_path, "rb") as f:
                await context.bot.send_document(
                    chat_id=chat.id,
                    document=f,
                    filename="clean_proxies.txt",
                    caption=f"✅ {summary}",
                )
        else:
            await chat.send_message(f"❌ {summary} — no working proxies, nothing clean to send.")

        with open(report_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat.id,
                document=f,
                filename="proxy_check_report.txt",
                caption="Full pass/fail report.",
            )
    finally:
        for p in (clean_path, report_path):
            try:
                os.remove(p)
            except OSError:
                pass

    if passed:
        top = passed[0]
        # The bulk check was a fast reachability screen; before activating,
        # re-verify the winner with the FULL transcript test so we don't set
        # a proxy that reaches YouTube but can't actually fetch captions.
        verify_msg = await chat.send_message(
            f"Verifying the winner with a full transcript fetch: {top['raw']}…"
        )
        try:
            prev = dict(current_proxy)
            current_proxy.update(
                type="generic",
                http_url=top["http_url"],
                https_url=top["https_url"],
                webshare_username="",
                webshare_password="",
            )
            await loop.run_in_executor(None, test_proxy_against_youtube)
        except Exception as exc:  # noqa: BLE001
            current_proxy.clear()
            current_proxy.update(prev)
            await verify_msg.edit_text(
                f"⚠️ {top['raw']} passed the reachability screen but failed the full "
                f"transcript test ({type(exc).__name__}: {exc}).\n"
                "Try another line from clean_proxies.txt with /setproxy."
            )
        else:
            await verify_msg.edit_text(
                f"🏆 Activated and fully verified: {top['raw']} (via {top['scheme']}://)\n"
                "Run /proxystatus to confirm."
            )


async def checkproxies_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update):
        await update.message.reply_text("Only the bot owner can run this.")
        return

    if not context.args:
        await update.message.reply_text(
            "Easiest way: just upload a .txt file with one proxy per line — "
            "you'll get back clean_proxies.txt containing only the working ones, "
            "plus a full proxy_check_report.txt. The progress message has a "
            "🛑 Stop button to cancel mid-run.\n\n"
            "Or paste a short list inline (comma, semicolon, or newline separated):\n"
            "/checkproxies 31.59.20.176:6754:user:pass, 45.12.13.14:8080:u2:p2"
        )
        return

    # context.args already split on whitespace; join back so the line-parser
    # can also handle comma/semicolon splits.
    raw_text = " ".join(context.args)
    await run_bulk_proxy_check(update, context, raw_text, source_label="your message")


async def proxy_file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Primary interface for bulk proxy checking: upload a .txt (one proxy per
    line) and get back clean_proxies.txt + proxy_check_report.txt.
    """
    if not is_owner(update):
        await update.message.reply_text("Only the bot owner can check proxy lists.")
        return

    document = update.message.document
    filename = document.file_name or "upload"
    lower = filename.lower()

    if not lower.endswith(ALLOWED_PROXY_FILE_EXTS):
        await update.message.reply_text(
            "Send your proxy list as a .txt file (also accepted: .csv, .list, .proxies), "
            "one proxy per line."
        )
        return

    if document.file_size and document.file_size > MAX_PROXY_FILE_BYTES:
        await update.message.reply_text("That file's too large — keep proxy lists under 10 MB.")
        return

    status_msg = await update.message.reply_text(f"Reading {filename}…")
    try:
        tg_file = await context.bot.get_file(document.file_id)
        raw_bytes = await tg_file.download_as_bytearray()
    except Exception as exc:
        await status_msg.edit_text(f"Couldn't download that file: {exc}")
        return

    text = bytes(raw_bytes).decode("utf-8", errors="ignore")
    if not text.strip():
        await status_msg.edit_text("That file looked empty.")
        return

    await run_bulk_proxy_check(
        update, context, text, status_msg=status_msg, source_label=filename
    )


async def clearproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update):
        await update.message.reply_text("Only the bot owner can change the proxy.")
        return
    current_proxy.update(
        type=None, webshare_username="", webshare_password="", http_url="", https_url=""
    )
    await update.message.reply_text("Proxy cleared — requests will connect to YouTube directly.")


async def proxystatus_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Current proxy: {describe_current_proxy()}")


async def checkproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update):
        await update.message.reply_text("Only the bot owner can run this.")
        return

    chat = update.effective_chat
    proxy_desc = describe_current_proxy()
    status_msg = await chat.send_message(f"Testing against YouTube via: {proxy_desc}…")

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, test_proxy_against_youtube)
    except (RequestBlocked, IpBlocked) as exc:
        await status_msg.edit_text(
            f"❌ Still blocked using {proxy_desc}.\n\n{exc}\n\n"
            "This proxy/IP doesn't work for YouTube — try a different provider "
            "or a fresh rotating-residential plan."
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Proxy check failed")
        await status_msg.edit_text(f"❌ Test failed via {proxy_desc}:\n{exc}")
        return

    await status_msg.edit_text(
        f"✅ Working — successfully fetched a test transcript via: {proxy_desc}"
    )


# --- Callback router ---------------------------------------------------

async def _reply_from_callback(query, context: ContextTypes.DEFAULT_TYPE, text: str,
                               reply_markup: InlineKeyboardMarkup | None = None) -> None:
    """Reply to a callback safely: prefer the original message, fall back to a
    direct message to the user if the original message is unreachable."""
    if query.message is not None:
        await query.message.reply_text(text, reply_markup=reply_markup)
    elif query.from_user is not None:
        await context.bot.send_message(
            chat_id=query.from_user.id, text=text, reply_markup=reply_markup
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data.startswith("stopbulk:"):
        # Set the stop event for the matching run. If the run already finished
        # (or is a stale button from an earlier run), do nothing — the button
        # will be gone in a moment anyway, since the progress message is
        # deleted once the run ends.
        run_id = data.split(":", 1)[1]
        run = context.chat_data.get("bulk_proxy_run")
        if run and run["id"] == run_id and not run["stop"].is_set():
            run["stop"].set()
        return

    if data == "menu:transcript":
        await _reply_from_callback(query, context, "Send me a YouTube link and I'll fetch the transcript.")

    elif data == "menu:translate":
        if not NVIDIA_API_KEY:
            await _reply_from_callback(query, context, "NVIDIA_API_KEY is not set on the server.")
            return
        if not context.chat_data.get("last_transcript"):
            await _reply_from_callback(
                query, context, "No transcript on file yet — send a YouTube link first."
            )
            return
        await _reply_from_callback(
            query, context, "Pick a language:", reply_markup=build_language_keyboard()
        )

    elif data == "menu:voice":
        await run_voice_generation(update, context, source=None)

    elif data == "menu:settings":
        await send_settings_message(update, context)

    elif data == "changemodel":
        await _reply_from_callback(
            query, context,
            "Pick a shortcut, or use /model <model_id> for any other:",
            reply_markup=build_model_keyboard(),
        )

    elif data == "changevoice":
        await _reply_from_callback(
            query, context,
            "Pick a shortcut, or use /setvoice <voice_id> for any other:",
            reply_markup=build_voice_keyboard(),
        )

    elif data == "models:list":
        await send_all_models(update, context)

    elif data == "voices:list":
        await send_all_voices(update, context, filter_str=None)

    elif data.startswith("setmodel:"):
        model_id = data.split(":", 1)[1]
        context.chat_data["nvidia_model"] = model_id
        await _reply_from_callback(query, context, f"Translation model set to: {model_id}")

    elif data.startswith("setvoice:"):
        voice_id = data.split(":", 1)[1]
        context.chat_data["edge_voice"] = voice_id
        await _reply_from_callback(query, context, f"Voice set to: {voice_id}")

    elif data.startswith("lang:"):
        lang = data.split(":", 1)[1]
        if lang == "custom":
            await _reply_from_callback(
                query, context,
                "Type it as: /translate <language>\nExample: /translate Bengali",
            )
            return
        await run_translation(update, context, target_language=lang, url=None)

    elif data.startswith("narrate:"):
        source = data.split(":", 1)[1]
        await run_voice_generation(update, context, source=source)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    match = YOUTUBE_URL_PATTERN.search(text)
    if match:
        await run_transcript_fetch(update, context, match.group(0))
    else:
        await update.message.reply_text(
            "Send a YouTube link and I'll pull the transcript for you, or /help "
            "for everything I can do."
        )


async def post_init(application: Application) -> None:
    """Registers the native Telegram '/' command menu, resizes the default
    executor so blocking proxy checks actually run in parallel, and logs
    startup warnings."""
    await application.bot.set_my_commands([
        BotCommand("start", "Welcome menu with buttons"),
        BotCommand("help", "List everything the bot can do"),
        BotCommand("transcript", "Get a transcript: /transcript <url>"),
        BotCommand("translate", "Translate the transcript"),
        BotCommand("models", "List available NVIDIA NIM models"),
        BotCommand("model", "Set the translation model"),
        BotCommand("voice", "Narrate the transcript as speech"),
        BotCommand("voices", "Browse edge-tts voices"),
        BotCommand("setvoice", "Set the narration voice"),
        BotCommand("settings", "View/change model & voice"),
        BotCommand("setproxy", "Configure a proxy for YouTube (owner only)"),
        BotCommand("clearproxy", "Disable the proxy (owner only)"),
        BotCommand("proxystatus", "Show the current proxy"),
        BotCommand("checkproxy", "Test the proxy against YouTube (owner only)"),
        BotCommand("checkproxies", "Test a whole list of proxies at once (owner only)"),
    ])

    # Resize the default ThreadPoolExecutor. Without this, loop.run_in_executor(None, ...)
    # is capped at min(32, cpu+4) threads, so "50 parallel" would still run at ~32.
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=PROXY_CHECK_EXECUTOR_WORKERS,
        thread_name_prefix="proxy-check",
    )
    loop.set_default_executor(executor)

    if OWNER_ID is None:
        logger.warning(
            "OWNER_ID is not set — /setproxy, /clearproxy and /checkproxies are open "
            "to ANY user of this bot. Set OWNER_ID to your numeric Telegram user ID "
            "to lock them down."
        )
    if not NVIDIA_API_KEY:
        logger.info("NVIDIA_API_KEY not set — /translate and /models are disabled.")
    logger.info(
        "Bulk proxy checker: concurrency=%d, executor_workers=%d, "
        "timeouts=%.1fs/%.1fs, retry_delay=%.1fs, progress_edit=%.1fs",
        MAX_CONCURRENT_PROXY_CHECKS, PROXY_CHECK_EXECUTOR_WORKERS,
        PROXY_CHECK_CONNECT_TIMEOUT, PROXY_CHECK_READ_TIMEOUT,
        PROXY_CHECK_RETRY_DELAY, PROGRESS_EDIT_INTERVAL,
    )


def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "PUT-YOUR-TOKEN-HERE":
        raise SystemExit(
            "Set the BOT_TOKEN environment variable to your Telegram bot token first."
        )

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("transcript", transcript_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("voices", voices_command))
    app.add_handler(CommandHandler("setvoice", setvoice_command))
    app.add_handler(CommandHandler("voice", voice_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("setproxy", setproxy_command))
    app.add_handler(CommandHandler("clearproxy", clearproxy_command))
    app.add_handler(CommandHandler("proxystatus", proxystatus_command))
    app.add_handler(CommandHandler("checkproxy", checkproxy_command))
    app.add_handler(CommandHandler("checkproxies", checkproxies_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, proxy_file_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Bot starting…")
    app.run_polling()


if __name__ == "__main__":
    main()
