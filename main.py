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


# ==========================================================================
# LOGGING
# ==========================================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ==========================================================================
# ENVIRONMENT
# ==========================================================================

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()

NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "",
).strip()

OWNER_ID_RAW = os.environ.get(
    "OWNER_ID",
    "",
).strip()

OWNER_ID = (
    int(OWNER_ID_RAW)
    if OWNER_ID_RAW.isdigit()
    else None
)


# ==========================================================================
# NVIDIA TRANSLATION CONFIG
# ==========================================================================

NVIDIA_CHAT_URL = (
    "https://integrate.api.nvidia.com/v1/chat/completions"
)

NVIDIA_MODELS_URL = (
    "https://integrate.api.nvidia.com/v1/models"
)

# Current NVIDIA NIM model ID.
DEFAULT_NVIDIA_MODEL = (
    "nvidia/nemotron-3-ultra-550b-a55b"
)

# --------------------------------------------------------------------------
# IMPORTANT:
# User requested a 30 RPM limit.
#
# 60 / 30 = 2 seconds exactly.
#
# We intentionally use 2.2 seconds between request starts so the bot
# stays below the limit instead of sitting exactly on the boundary.
# --------------------------------------------------------------------------

NVIDIA_RPM_LIMIT = 30

NVIDIA_MIN_REQUEST_INTERVAL = (
    60.0 / NVIDIA_RPM_LIMIT
) + 0.2

# Translation chunk size.
#
# We deliberately don't use the model's huge context window for one giant
# translation request. Smaller chunks give much more reliable output.
TRANSLATION_MAX_CHARS = int(
    os.environ.get(
        "TRANSLATION_MAX_CHARS",
        "5000",
    )
)

# Maximum generated tokens for one translation chunk.
TRANSLATION_MAX_TOKENS = int(
    os.environ.get(
        "TRANSLATION_MAX_TOKENS",
        "8192",
    )
)

# Maximum retry attempts for temporary NVIDIA errors.
NVIDIA_MAX_RETRIES = int(
    os.environ.get(
        "NVIDIA_MAX_RETRIES",
        "4",
    )
)


# ==========================================================================
# EDGE TTS CONFIG
# ==========================================================================

DEFAULT_EDGE_VOICE = (
    "en-US-AriaNeural"
)

# English narration is slightly slower for more natural pacing.
ENGLISH_TTS_RATE = "-8%"

# Other languages remain at normal speed.
DEFAULT_TTS_RATE = "+0%"

# --------------------------------------------------------------------------
# BUGFIX / FEATURE:
#
# Edge TTS has no official length limit, but in practice a single very
# long request over its websocket API is prone to simply hanging or
# dying silently instead of raising a clean error - this is the
# "not responding on long requests" problem.
#
# The fix: split long narration text into modest-sized chunks (same
# safe splitter used for translation), synthesize each chunk on its
# own with a hard timeout + retries, then stitch the resulting mp3
# segments back together into one file.
# --------------------------------------------------------------------------

TTS_MAX_CHARS = int(
    os.environ.get(
        "TTS_MAX_CHARS",
        "1800",
    )
)

TTS_CHUNK_TIMEOUT = float(
    os.environ.get(
        "TTS_CHUNK_TIMEOUT",
        "45",
    )
)

TTS_MAX_RETRIES = int(
    os.environ.get(
        "TTS_MAX_RETRIES",
        "3",
    )
)

TTS_CHUNK_DELAY = float(
    os.environ.get(
        "TTS_CHUNK_DELAY",
        "0.3",
    )
)


# ==========================================================================
# YOUTUBE / PROXY CONFIG
# ==========================================================================

TEST_VIDEO_ID = "dQw4w9WgXcQ"

PROXY_SCHEMES_TO_TRY = [
    "http",
    "socks5",
    "socks4",
]

MAX_CONCURRENT_PROXY_CHECKS = int(
    os.environ.get(
        "PROXY_CHECK_CONCURRENCY",
        "50",
    )
)

PROXY_CHECK_CONNECT_TIMEOUT = float(
    os.environ.get(
        "PROXY_CHECK_CONNECT_TIMEOUT",
        "5",
    )
)

PROXY_CHECK_READ_TIMEOUT = float(
    os.environ.get(
        "PROXY_CHECK_READ_TIMEOUT",
        "10",
    )
)

PROXY_CHECK_RETRY_DELAY = float(
    os.environ.get(
        "PROXY_CHECK_RETRY_DELAY",
        "0.5",
    )
)

PROGRESS_EDIT_INTERVAL = float(
    os.environ.get(
        "PROGRESS_EDIT_INTERVAL",
        "1.5",
    )
)

PROXY_CHECK_EXECUTOR_WORKERS = int(
    os.environ.get(
        "PROXY_CHECK_EXECUTOR_WORKERS",
        str(
            max(
                64,
                MAX_CONCURRENT_PROXY_CHECKS * 2,
            )
        ),
    )
)

MAX_PROXY_FILE_BYTES = (
    10 * 1024 * 1024
)

ALLOWED_PROXY_FILE_EXTS = (
    ".txt",
    ".csv",
    ".list",
    ".proxies",
)


# ==========================================================================
# CURRENT PROXY
# ==========================================================================

current_proxy = {
    "type": None,

    "webshare_username": os.environ.get(
        "WEBSHARE_PROXY_USERNAME",
        "",
    ).strip(),

    "webshare_password": os.environ.get(
        "WEBSHARE_PROXY_PASSWORD",
        "",
    ).strip(),

    "http_url": os.environ.get(
        "PROXY_HTTP_URL",
        "",
    ).strip(),

    "https_url": os.environ.get(
        "PROXY_HTTPS_URL",
        "",
    ).strip(),
}


if (
    current_proxy["webshare_username"]
    and current_proxy["webshare_password"]
):

    current_proxy["type"] = "webshare"

elif (
    current_proxy["http_url"]
    or current_proxy["https_url"]
):

    current_proxy["type"] = "generic"


# ==========================================================================
# QUICK MODEL OPTIONS
# ==========================================================================

QUICK_MODELS = [
    (
        "Nemotron 3 Ultra",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ),
    (
        "Llama 3.1 70B",
        "meta/llama-3.1-70b-instruct",
    ),
    (
        "Llama 3.1 8B",
        "meta/llama-3.1-8b-instruct",
    ),
    (
        "Mixtral 8x22B",
        "mistralai/mixtral-8x22b-instruct-v0.1",
    ),
]


# ==========================================================================
# QUICK LANGUAGES
# ==========================================================================

QUICK_LANGUAGES = [
    "Hindi",
    "English",
    "Urdu",
    "Spanish",
    "French",
    "Arabic",
    "Bengali",
    "Chinese",
    "Japanese",
    "German",
    "Russian",
    "Portuguese",
]


# ==========================================================================
# QUICK VOICES
# ==========================================================================

QUICK_VOICES = [
    (
        "English (US, F)",
        "en-US-AriaNeural",
    ),
    (
        "English (US, M)",
        "en-US-GuyNeural",
    ),
    (
        "English (UK, M)",
        "en-GB-RyanNeural",
    ),
    (
        "Hindi (F)",
        "hi-IN-SwaraNeural",
    ),
    (
        "Hindi (M)",
        "hi-IN-MadhurNeural",
    ),
    (
        "Urdu (M)",
        "ur-PK-AsadNeural",
    ),
    (
        "Spanish (F)",
        "es-ES-ElviraNeural",
    ),
    (
        "Arabic (M)",
        "ar-SA-HamedNeural",
    ),
]


# ==========================================================================
# YOUTUBE URL
# ==========================================================================

YOUTUBE_URL_PATTERN = re.compile(
    r"(?:https?://)?"
    r"(?:www\.)?"
    r"(?:youtube\.com|youtu\.be|m\.youtube\.com)"
    r"/[\w\-./?=&%]+",
    re.IGNORECASE,
)


# ==========================================================================
# PROXY STATE
# ==========================================================================

def is_owner(
    update: Update,
) -> bool:

    if OWNER_ID is None:
        return True

    user = update.effective_user

    return bool(
        user
        and user.id == OWNER_ID
    )


def describe_current_proxy(
    reveal: bool = False,
) -> str:

    if current_proxy["type"] == "webshare":

        user = (
            current_proxy[
                "webshare_username"
            ]
            or "(unset)"
        )

        if (
            not reveal
            and len(user) > 4
        ):

            user = (
                user[:2]
                + "…"
                + user[-2:]
            )

        return (
            f"Webshare ({user})"
        )

    if current_proxy["type"] == "generic":

        url = (
            current_proxy["http_url"]
            or current_proxy["https_url"]
            or "(unset)"
        )

        if not reveal:

            url = re.sub(
                r"//[^@]+@",
                "//***:***@",
                url,
            )

        return (
            f"Generic ({url})"
        )

    return (
        "None — connecting directly"
    )


# ==========================================================================
# YOUTUBE API
# ==========================================================================

def build_youtube_api():

    if (
        current_proxy["type"]
        == "webshare"
    ):

        from youtube_transcript_api.proxies import (
            WebshareProxyConfig,
        )

        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=(
                    current_proxy[
                        "webshare_username"
                    ]
                ),
                proxy_password=(
                    current_proxy[
                        "webshare_password"
                    ]
                ),
            )
        )

    if (
        current_proxy["type"]
        == "generic"
    ):

        from youtube_transcript_api.proxies import (
            GenericProxyConfig,
        )

        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=(
                    current_proxy[
                        "http_url"
                    ]
                    or None
                ),
                https_url=(
                    current_proxy[
                        "https_url"
                    ]
                    or None
                ),
            )
        )

    return YouTubeTranscriptApi()


# ==========================================================================
# FULL PROXY TEST
# ==========================================================================

def test_proxy_against_youtube():

    api = build_youtube_api()

    transcript_list = api.list(
        TEST_VIDEO_ID
    )

    try:

        transcript = next(
            iter(transcript_list)
        )

    except StopIteration as exc:

        raise RuntimeError(
            "Test video returned no "
            "transcript tracks."
        ) from exc

    transcript.fetch()


# ==========================================================================
# LIGHT PROXY TEST
# ==========================================================================

def test_proxy_via_urls(
    http_url: str,
    https_url: str,
):

    proxies = {}

    if http_url:
        proxies["http"] = http_url

    if https_url:
        proxies["https"] = https_url

    try:

        response = requests.get(
            "https://www.youtube.com/oembed",
            params={
                "url": (
                    "https://youtu.be/"
                    + TEST_VIDEO_ID
                ),
                "format": "json",
            },
            proxies=(
                proxies
                if proxies
                else None
            ),
            timeout=(
                PROXY_CHECK_CONNECT_TIMEOUT,
                PROXY_CHECK_READ_TIMEOUT,
            ),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(X11; Linux x86_64) "
                    "AppleWebKit/537.36"
                )
            },
            allow_redirects=True,
        )

    except requests.exceptions.ProxyError as exc:

        raise RuntimeError(
            f"proxy connect failed: {exc}"
        ) from exc

    except requests.exceptions.ConnectTimeout as exc:

        raise RuntimeError(
            "connect timeout"
        ) from exc

    except requests.exceptions.ReadTimeout as exc:

        raise RuntimeError(
            "read timeout"
        ) from exc

    except requests.exceptions.ConnectionError as exc:

        raise RuntimeError(
            f"connection error: {exc}"
        ) from exc

    if response.status_code == 429:

        raise RuntimeError(
            "rate-limited by YouTube"
        )

    if response.status_code >= 500:

        raise RuntimeError(
            f"upstream {response.status_code}"
        )

    response.raise_for_status()


# ==========================================================================
# PROXY URL
# ==========================================================================

def build_generic_proxy_urls(
    host: str,
    port: str,
    user: str | None,
    password: str | None,
    scheme: str,
):

    auth = ""

    if user and password:

        auth = (
            f"{user}:{password}@"
        )

    url = (
        f"{scheme}://"
        f"{auth}"
        f"{host}:{port}"
    )

    return url, url


# ==========================================================================
# PROXY PARSING
# ==========================================================================

_SCHEME_PREFIX_RE = re.compile(
    r"^(?P<scheme>"
    r"https?|"
    r"socks5h?|"
    r"socks4a?|"
    r"socks4"
    r")://"
    r"(?P<rest>.+)$",
    re.IGNORECASE,
)

_USERPASS_AT_RE = re.compile(
    r"^(?P<user>[^:@\s]+):"
    r"(?P<pw>[^:@\s]+)@"
    r"(?P<host>[\w.\-]+):"
    r"(?P<port>\d{2,5})$"
)

_INLINE_SEP_RE = re.compile(
    r"[,;\t]"
)


def _parse_hostport_pair(
    raw: str,
):

    raw = (
        raw
        .strip()
        .strip("'\"")
    )

    match = _USERPASS_AT_RE.match(
        raw
    )

    if match:

        return (
            match.group("host"),
            match.group("port"),
            match.group("user"),
            match.group("pw"),
        )

    parts = raw.split(":")

    if (
        len(parts) == 2
        and parts[1].isdigit()
    ):

        return (
            parts[0],
            parts[1],
            None,
            None,
        )

    if len(parts) == 4:

        a, b, c, d = parts

        if b.isdigit():

            return (
                a,
                b,
                c,
                d,
            )

        if d.isdigit():

            return (
                c,
                d,
                a,
                b,
            )

    return None


def parse_proxy_input(
    raw: str,
):

    raw = (
        raw
        .strip()
        .strip("'\"")
    )

    if not raw:
        return None

    match = _SCHEME_PREFIX_RE.match(
        raw
    )

    if match:

        inner = _parse_hostport_pair(
            match.group("rest")
        )

        if inner is None:
            return None

        host, port, user, pw = inner

        return (
            host,
            port,
            user,
            pw,
            match.group("scheme").lower(),
        )

    inner = _parse_hostport_pair(
        raw
    )

    if inner is None:
        return None

    host, port, user, pw = inner

    return (
        host,
        port,
        user,
        pw,
        None,
    )


def parse_proxy_lines(
    raw_text: str,
):

    seen = set()
    output = []

    for raw_line in raw_text.splitlines():

        line = raw_line.split(
            "#",
            1,
        )[0].strip()

        if not line:
            continue

        if line.startswith("//"):
            continue

        candidates = [line]

        if (
            parse_proxy_input(line)
            is None
            and _INLINE_SEP_RE.search(line)
        ):

            candidates = [
                token.strip()
                for token
                in _INLINE_SEP_RE.split(line)
                if token.strip()
            ]

        for candidate in candidates:

            if (
                candidate
                and candidate not in seen
            ):

                seen.add(candidate)
                output.append(candidate)

    return output


# ==========================================================================
# VIDEO ID
# ==========================================================================

def extract_video_id(
    url: str,
):

    url = url.strip()

    parsed = urlparse(
        url
        if "://" in url
        else f"https://{url}"
    )

    host = (
        parsed.netloc
        or ""
    ).lower()

    if "youtu.be" in host:

        video_id = (
            parsed.path
            .lstrip("/")
        )

        return (
            video_id.split("/")[0]
            if video_id
            else None
        )

    if "youtube.com" in host:

        if parsed.path == "/watch":

            query = parse_qs(
                parsed.query
            )

            return query.get(
                "v",
                [None],
            )[0]

        for prefix in (
            "/shorts/",
            "/embed/",
            "/live/",
        ):

            if parsed.path.startswith(
                prefix
            ):

                return (
                    parsed.path[
                        len(prefix):
                    ]
                    .split("/")[0]
                )

    return None


# ==========================================================================
# TRANSCRIPT CLEANING
# ==========================================================================

def clean_transcript(
    entries,
):

    leading_timestamp = re.compile(
        r"^[\[\(]?"
        r"\d{1,2}"
        r"(?::\d{2}){1,2}"
        r"\b"
        r"[\]\)]?"
        r"\s*[-–—:]?\s*"
    )

    paragraphs = []
    current = []

    last_end = 0.0

    for entry in entries:

        text = (
            entry.text
            .replace("\n", " ")
            .strip()
        )

        text = leading_timestamp.sub(
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            continue

        gap = (
            entry.start
            - last_end
        )

        if (
            current
            and gap > 2.5
        ):

            paragraphs.append(
                " ".join(current)
            )

            current = []

        current.append(text)

        last_end = (
            entry.start
            + entry.duration
        )

    if current:

        paragraphs.append(
            " ".join(current)
        )

    return "\n\n".join(
        paragraphs
    )


# ==========================================================================
# FETCH TRANSCRIPT
# ==========================================================================

def fetch_transcript_text(
    video_id: str,
    preferred_langs=("en",),
):

    api = build_youtube_api()

    transcript_list = api.list(
        video_id
    )

    transcript = None

    try:

        transcript = (
            transcript_list
            .find_manually_created_transcript(
                preferred_langs
            )
        )

    except NoTranscriptFound:
        pass

    if transcript is None:

        try:

            transcript = (
                transcript_list
                .find_generated_transcript(
                    preferred_langs
                )
            )

        except NoTranscriptFound:
            pass

    if transcript is None:

        try:

            transcript = next(
                iter(transcript_list)
            )

        except StopIteration as exc:

            raise NoTranscriptFound(
                video_id,
                preferred_langs,
                transcript_list,
            ) from exc

    raw_entries = transcript.fetch()

    return (
        clean_transcript(raw_entries),
        transcript.language,
    )


# ==========================================================================
# TRANSCRIPT ERROR HANDLING
# ==========================================================================

async def fetch_transcript_or_report(
    chat,
    status_msg,
    url: str,
):

    video_id = extract_video_id(
        url
    )

    if not video_id:

        await status_msg.edit_text(
            "That doesn't look like a valid "
            "YouTube URL."
        )

        return None

    try:

        text, language = (
            fetch_transcript_text(
                video_id
            )
        )

    except TranscriptsDisabled:

        await status_msg.edit_text(
            "Transcripts are disabled for "
            "this video."
        )

        return None

    except NoTranscriptFound:

        await status_msg.edit_text(
            "No transcript is available "
            "for this video."
        )

        return None

    except VideoUnavailable:

        await status_msg.edit_text(
            "That video is unavailable."
        )

        return None

    except (
        RequestBlocked,
        IpBlocked,
    ):

        await status_msg.edit_text(
            "YouTube is blocking this "
            "server's IP.\n\n"
            "Upload a proxy list or use "
            "/setproxy."
        )

        return None

    except Exception as exc:

        logger.exception(
            "Transcript fetch failed"
        )

        await status_msg.edit_text(
            f"Couldn't fetch transcript:\n"
            f"{exc}"
        )

        return None

    return (
        video_id,
        text,
        language,
    )


# ==========================================================================
# SMART TRANSLATION CHUNKER
# ==========================================================================

def split_oversized_paragraph(
    paragraph: str,
    max_chars: int,
):

    paragraph = paragraph.strip()

    if not paragraph:
        return []

    if len(paragraph) <= max_chars:
        return [paragraph]

    # First split on sentence endings.
    sentences = re.split(
        r"(?<=[.!?。！？])\s+",
        paragraph,
    )

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # If a single sentence itself is huge,
        # split it safely by whitespace.
        if len(sentence) > max_chars:

            if current:
                chunks.append(
                    current.strip()
                )
                current = ""

            words = sentence.split()
            word_chunk = ""

            for word in words:

                candidate = (
                    f"{word_chunk} {word}"
                    if word_chunk
                    else word
                )

                if (
                    len(candidate)
                    > max_chars
                ):

                    if word_chunk:
                        chunks.append(
                            word_chunk.strip()
                        )

                    word_chunk = word

                else:

                    word_chunk = candidate

            if word_chunk:
                chunks.append(
                    word_chunk.strip()
                )

            continue

        candidate = (
            f"{current} {sentence}"
            if current
            else sentence
        )

        if (
            len(candidate)
            > max_chars
        ):

            if current:
                chunks.append(
                    current.strip()
                )

            current = sentence

        else:

            current = candidate

    if current:
        chunks.append(
            current.strip()
        )

    return chunks


def chunk_text_for_translation(
    text: str,
    max_chars: int = TRANSLATION_MAX_CHARS,
):

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_parts = (
            split_oversized_paragraph(
                paragraph,
                max_chars,
            )
        )

        for part in paragraph_parts:

            candidate = (
                f"{current}\n\n{part}"
                if current
                else part
            )

            if (
                len(candidate)
                > max_chars
            ):

                if current:
                    chunks.append(
                        current.strip()
                    )

                current = part

            else:

                current = candidate

    if current:
        chunks.append(
            current.strip()
        )

    return chunks or [text]


# ==========================================================================
# NVIDIA RATE LIMITER
# ==========================================================================

class NvidiaRateLimiter:

    def __init__(
        self,
        interval: float,
    ):

        self.interval = interval

        self.lock = asyncio.Lock()

        self.last_request = 0.0

    async def wait(self):

        async with self.lock:

            loop = asyncio.get_running_loop()

            now = loop.time()

            wait_time = (
                self.interval
                - (
                    now
                    - self.last_request
                )
            )

            if wait_time > 0:

                await asyncio.sleep(
                    wait_time
                )

            self.last_request = (
                loop.time()
            )


nvidia_rate_limiter = (
    NvidiaRateLimiter(
        NVIDIA_MIN_REQUEST_INTERVAL
    )
)


# ==========================================================================
# NVIDIA API RESPONSE
# ==========================================================================

def call_nvidia_chat(
    model: str,
    messages: list,
):

    if not NVIDIA_API_KEY:

        raise RuntimeError(
            "NVIDIA_API_KEY is not set."
        )

    response = requests.post(
        NVIDIA_CHAT_URL,
        headers={
            "Authorization":
                f"Bearer {NVIDIA_API_KEY}",
            "Content-Type":
                "application/json",
            "Accept":
                "application/json",
        },
        json={
            "model": model,
            "messages": messages,

            # Translation doesn't need random output.
            "temperature": 0.15,

            "top_p": 0.95,

            "max_tokens":
                TRANSLATION_MAX_TOKENS,

            # Nemotron 3 Ultra supports this.
            # We disable reasoning because this is
            # a translation task.
            "reasoning_effort": "none",

            "stream": False,
        },
        timeout=300,
    )

    if response.status_code == 429:

        retry_after = (
            response.headers.get(
                "Retry-After"
            )
        )

        error = RuntimeError(
            "NVIDIA API rate limit "
            f"(429)"
        )

        if retry_after:

            setattr(
                error,
                "retry_after",
                retry_after,
            )

        raise error

    if (
        response.status_code
        >= 500
    ):

        raise RuntimeError(
            "NVIDIA server error "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    if not response.ok:

        raise RuntimeError(
            "NVIDIA API error "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    try:

        content = (
            data["choices"][0]
            ["message"]["content"]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:

        raise RuntimeError(
            "Unexpected NVIDIA response:\n"
            f"{data}"
        ) from exc

    if content is None:
        return ""

    return content.strip()


# ==========================================================================
# NVIDIA TRANSLATION ONE CHUNK
# ==========================================================================

async def translate_single_chunk(
    chunk: str,
    target_language: str,
    model: str,
):

    loop = asyncio.get_running_loop()

    for attempt in range(
        NVIDIA_MAX_RETRIES
    ):

        # --------------------------------------------------------------
        # GLOBAL 30 RPM PACING
        # --------------------------------------------------------------

        await nvidia_rate_limiter.wait()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional "
                    "translation engine.\n\n"

                    "Translate the user's text "
                    "faithfully into the requested "
                    "target language.\n\n"

                    "Rules:\n"
                    "1. Output ONLY the translation.\n"
                    "2. Do not explain anything.\n"
                    "3. Do not summarize.\n"
                    "4. Do not add missing information.\n"
                    "5. Preserve paragraph breaks.\n"
                    "6. Preserve names, numbers, "
                    "URLs and technical terms.\n"
                    "7. Keep the meaning and tone "
                    "of the original.\n"
                    "8. Do not add headings unless "
                    "they exist in the source."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target language: "
                    f"{target_language}\n\n"
                    "Translate this text:\n\n"
                    f"{chunk}"
                ),
            },
        ]

        try:

            result = (
                await loop.run_in_executor(
                    None,
                    call_nvidia_chat,
                    model,
                    messages,
                )
            )

            if not result:

                raise RuntimeError(
                    "NVIDIA returned an empty "
                    "translation."
                )

            return result

        except Exception as exc:

            error_text = str(exc)

            is_rate_limit = (
                "429" in error_text
                or "rate limit"
                in error_text.lower()
            )

            is_temporary = (
                is_rate_limit
                or "500" in error_text
                or "502" in error_text
                or "503" in error_text
                or "504" in error_text
                or "timeout"
                in error_text.lower()
                or "temporarily"
                in error_text.lower()
            )

            if (
                not is_temporary
                or attempt
                >= NVIDIA_MAX_RETRIES - 1
            ):

                raise

            retry_after = getattr(
                exc,
                "retry_after",
                None,
            )

            if retry_after:

                try:

                    delay = max(
                        float(retry_after),
                        NVIDIA_MIN_REQUEST_INTERVAL,
                    )

                except ValueError:

                    delay = (
                        3.0
                        * (
                            2 ** attempt
                        )
                    )

            else:

                delay = (
                    3.0
                    * (
                        2 ** attempt
                    )
                )

            # Add small jitter so repeated failures
            # don't synchronize.
            delay += (
                secrets.randbelow(500)
                / 1000.0
            )

            logger.warning(
                "NVIDIA temporary error "
                "(attempt %d/%d): %s. "
                "Retrying in %.2fs",
                attempt + 1,
                NVIDIA_MAX_RETRIES,
                exc,
                delay,
            )

            await asyncio.sleep(
                delay
            )

    raise RuntimeError(
        "Translation failed after "
        "maximum retries."
    )


# ==========================================================================
# FULL TRANSLATION
# ==========================================================================

async def translate_text(
    text: str,
    target_language: str,
    model: str,
    progress_cb: Callable[
        [int, int],
        Awaitable[None],
    ] | None = None,
):

    chunks = (
        chunk_text_for_translation(
            text,
            TRANSLATION_MAX_CHARS,
        )
    )

    total = len(chunks)

    logger.info(
        "Translation split into %d chunks "
        "(max %d chars/chunk)",
        total,
        TRANSLATION_MAX_CHARS,
    )

    translated_chunks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        logger.info(
            "Translating chunk %d/%d "
            "(%d chars)",
            index,
            total,
            len(chunk),
        )

        result = (
            await translate_single_chunk(
                chunk,
                target_language,
                model,
            )
        )

        translated_chunks.append(
            result
        )

        if progress_cb:

            try:

                await progress_cb(
                    index,
                    total,
                )

            except Exception:
                pass

    return "\n\n".join(
        translated_chunks
    )


# ==========================================================================
# NVIDIA MODELS
# ==========================================================================

def list_nvidia_models():

    if not NVIDIA_API_KEY:

        raise RuntimeError(
            "NVIDIA_API_KEY is not configured."
        )

    response = requests.get(
        NVIDIA_MODELS_URL,
        headers={
            "Authorization":
                f"Bearer {NVIDIA_API_KEY}"
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return sorted(
        model["id"]
        for model in data.get(
            "data",
            [],
        )
        if "id" in model
    )


# ==========================================================================
# EDGE TTS
# ==========================================================================

async def list_edge_voices(
    filter_str=None,
):

    voices = await edge_tts.list_voices()

    names = sorted(
        voice["ShortName"]
        for voice in voices
    )

    if filter_str:

        needle = filter_str.lower()

        names = [
            name
            for name in names
            if needle in name.lower()
        ]

    return names


def get_tts_rate(
    voice: str,
    source_language: str | None = None,
):

    voice_lower = (
        voice.lower()
    )

    # English voices get slightly slower pacing.
    if (
        voice_lower.startswith("en-")
        or (
            source_language
            and source_language.lower()
            in (
                "english",
                "en",
            )
        )
    ):

        return ENGLISH_TTS_RATE

    return DEFAULT_TTS_RATE


def chunk_text_for_tts(
    text: str,
    max_chars: int = TTS_MAX_CHARS,
):
    """
    Splits narration text into safe-sized pieces for Edge TTS.

    Reuses the same paragraph -> sentence -> word safe splitter that
    powers translation chunking. It never cuts a word in half and
    guarantees every chunk stays under max_chars.
    """

    return chunk_text_for_translation(
        text,
        max_chars,
    )


async def generate_speech_chunk(
    text: str,
    voice: str,
    rate: str,
    output_path: str,
):
    """
    Synthesizes ONE chunk of text with a hard timeout and retries.

    This is the actual fix for "Edge TTS not responding on long
    requests": instead of one unbounded call that can hang forever,
    each chunk gets a bounded time budget, and a failed/hung attempt
    is retried with backoff instead of freezing the whole bot.
    """

    last_exc = None

    for attempt in range(
        TTS_MAX_RETRIES
    ):

        try:

            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=rate,
            )

            await asyncio.wait_for(
                communicate.save(
                    output_path
                ),
                timeout=TTS_CHUNK_TIMEOUT,
            )

            if (
                os.path.exists(output_path)
                and os.path.getsize(output_path) > 0
            ):

                return

            raise RuntimeError(
                "Edge TTS produced an "
                "empty audio file."
            )

        except Exception as exc:

            last_exc = exc

            logger.warning(
                "Edge TTS chunk failed "
                "(attempt %d/%d): %s",
                attempt + 1,
                TTS_MAX_RETRIES,
                exc,
            )

            try:
                os.remove(output_path)
            except OSError:
                pass

            await asyncio.sleep(
                1.5 * (attempt + 1)
            )

    raise RuntimeError(
        "Edge TTS failed after "
        f"{TTS_MAX_RETRIES} attempts: "
        f"{last_exc}"
    )


async def generate_speech(
    text: str,
    voice: str,
    output_path: str,
    source_language: str | None = None,
    progress_cb: Callable[
        [int, int],
        Awaitable[None],
    ] | None = None,
):
    """
    Chunk-based narration generator.

    Long text is split into TTS_MAX_CHARS-sized pieces, each piece is
    synthesized separately (with its own timeout/retry budget), and
    the resulting mp3 segments are stitched into one final file.
    Edge TTS emits plain MPEG audio frames with no ID3 container, so a
    raw byte-level concatenation of the parts plays back correctly
    end-to-end without needing ffmpeg on the host.
    """

    rate = get_tts_rate(
        voice,
        source_language,
    )

    logger.info(
        "Generating TTS with voice=%s rate=%s",
        voice,
        rate,
    )

    chunks = chunk_text_for_tts(
        text,
        TTS_MAX_CHARS,
    )

    total = len(chunks)

    if total == 1:

        await generate_speech_chunk(
            chunks[0],
            voice,
            rate,
            output_path,
        )

        if progress_cb:

            try:
                await progress_cb(1, 1)
            except Exception:
                pass

        return

    logger.info(
        "TTS split into %d chunks "
        "(max %d chars/chunk)",
        total,
        TTS_MAX_CHARS,
    )

    part_paths = []

    try:

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            part_path = (
                f"{output_path}."
                f"part{index}.mp3"
            )

            await generate_speech_chunk(
                chunk,
                voice,
                rate,
                part_path,
            )

            part_paths.append(
                part_path
            )

            if progress_cb:

                try:

                    await progress_cb(
                        index,
                        total,
                    )

                except Exception:
                    pass

            if index < total:

                await asyncio.sleep(
                    TTS_CHUNK_DELAY
                )

        with open(
            output_path,
            "wb",
        ) as outfile:

            for part_path in part_paths:

                with open(
                    part_path,
                    "rb",
                ) as infile:

                    outfile.write(
                        infile.read()
                    )

    finally:

        for part_path in part_paths:

            try:
                os.remove(part_path)
            except OSError:
                pass


# ==========================================================================
# MAIN TRANSCRIPT FLOW
# ==========================================================================

async def run_transcript_fetch(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    url: str,
):

    chat = update.effective_chat

    await context.bot.send_chat_action(
        chat_id=chat.id,
        action=ChatAction.TYPING,
    )

    status_msg = await chat.send_message(
        "Fetching transcript…"
    )

    result = await fetch_transcript_or_report(
        chat,
        status_msg,
        url,
    )

    if result is None:
        return

    (
        video_id,
        text,
        language,
    ) = result

    if not text.strip():

        await status_msg.edit_text(
            "The transcript came back empty."
        )

        return

    context.chat_data[
        "last_transcript"
    ] = text

    context.chat_data[
        "last_video_id"
    ] = video_id

    context.chat_data[
        "last_source_language"
    ] = language

    context.chat_data.pop(
        "last_translation",
        None,
    )

    file_path = (
        f"/tmp/transcript_"
        f"{video_id}.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Transcript for "
            f"https://youtu.be/{video_id} "
            f"(language: {language})\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(text)

    try:

        await status_msg.delete()

        with open(
            file_path,
            "rb",
        ) as file:

            await context.bot.send_document(
                chat_id=chat.id,
                document=file,
                filename=(
                    f"transcript_{video_id}.txt"
                ),
                caption=(
                    f"Transcript ready "
                    f"({language})."
                ),
                reply_markup=(
                    build_post_transcript_keyboard()
                ),
            )

    finally:

        try:
            os.remove(file_path)
        except OSError:
            pass


# ==========================================================================
# TRANSLATION FLOW
# ==========================================================================

async def run_translation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target_language: str,
    url: str | None,
):

    chat = update.effective_chat

    if not NVIDIA_API_KEY:

        await chat.send_message(
            "NVIDIA_API_KEY is not set."
        )

        return

    if url:

        status_msg = await chat.send_message(
            "Fetching transcript…"
        )

        result = await fetch_transcript_or_report(
            chat,
            status_msg,
            url,
        )

        if result is None:
            return

        (
            video_id,
            text,
            source_language,
        ) = result

        context.chat_data[
            "last_transcript"
        ] = text

        context.chat_data[
            "last_video_id"
        ] = video_id

        context.chat_data[
            "last_source_language"
        ] = source_language

    else:

        text = context.chat_data.get(
            "last_transcript"
        )

        video_id = context.chat_data.get(
            "last_video_id",
            "transcript",
        )

        source_language = (
            context.chat_data.get(
                "last_source_language"
            )
        )

        if not text:

            await chat.send_message(
                "No transcript on file yet. "
                "Send a YouTube link first."
            )

            return

        status_msg = await chat.send_message(
            "Preparing translation…"
        )

    model = context.chat_data.get(
        "nvidia_model",
        DEFAULT_NVIDIA_MODEL,
    )

    chunks = (
        chunk_text_for_translation(
            text,
            TRANSLATION_MAX_CHARS,
        )
    )

    total_chunks = len(chunks)

    estimated_seconds = (
        max(
            0,
            total_chunks - 1,
        )
        * NVIDIA_MIN_REQUEST_INTERVAL
    )

    estimated_minutes = (
        estimated_seconds / 60
    )

    await status_msg.edit_text(
        "Translation started.\n\n"
        f"Model: {model}\n"
        f"Chunks: {total_chunks}\n"
        f"Chunk size: ~{TRANSLATION_MAX_CHARS} "
        "characters\n"
        f"Rate protection: "
        f"~{NVIDIA_MIN_REQUEST_INTERVAL:.1f}s "
        "between requests\n"
        f"Minimum pacing estimate: "
        f"~{estimated_minutes:.1f} min"
    )

    async def progress(
        done: int,
        total: int,
    ):

        remaining = max(
            0,
            total - done,
        )

        remaining_seconds = (
            remaining
            * NVIDIA_MIN_REQUEST_INTERVAL
        )

        remaining_str = format_duration(
            remaining_seconds
        )

        try:

            await status_msg.edit_text(
                "Translating long transcript…\n\n"
                f"Model: {model}\n"
                f"Progress: {done}/{total} chunks\n"
                f"Remaining: {remaining_str}\n"
                f"Rate: "
                f"~{60 / NVIDIA_MIN_REQUEST_INTERVAL:.1f}"
                " requests/min"
            )

        except Exception:
            pass

    try:

        translated = (
            await translate_text(
                text,
                target_language,
                model,
                progress,
            )
        )

    except Exception as exc:

        logger.exception(
            "Translation failed"
        )

        await status_msg.edit_text(
            "Translation failed:\n\n"
            f"{exc}"
        )

        return

    context.chat_data[
        "last_translation"
    ] = translated

    context.chat_data[
        "last_translation_language"
    ] = target_language

    safe_language = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        target_language,
    ).strip("_")

    if not safe_language:
        safe_language = "translated"

    file_path = (
        f"/tmp/transcript_"
        f"{video_id}_"
        f"{safe_language}.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Translated transcript\n"
        )

        file.write(
            f"Target language: "
            f"{target_language}\n"
        )

        file.write(
            f"Source language: "
            f"{source_language or 'unknown'}\n"
        )

        file.write(
            f"Model: {model}\n"
        )

        file.write(
            f"Chunks: {total_chunks}\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            translated
        )

    try:

        await status_msg.delete()

        with open(
            file_path,
            "rb",
        ) as file:

            await context.bot.send_document(
                chat_id=chat.id,
                document=file,
                filename=os.path.basename(
                    file_path
                ),
                caption=(
                    f"✅ Translation complete\n"
                    f"Language: "
                    f"{target_language}\n"
                    f"Chunks: "
                    f"{total_chunks}\n"
                    f"Model: "
                    f"{model}"
                ),
                reply_markup=(
                    build_post_translate_keyboard()
                ),
            )

    finally:

        try:
            os.remove(file_path)
        except OSError:
            pass


# ==========================================================================
# TTS FLOW
# ==========================================================================

async def run_voice_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    source: str | None,
):

    chat = update.effective_chat

    if source == "original":

        text = context.chat_data.get(
            "last_transcript"
        )

        source_language = (
            context.chat_data.get(
                "last_source_language"
            )
        )

    elif source == "translated":

        text = context.chat_data.get(
            "last_translation"
        )

        source_language = (
            context.chat_data.get(
                "last_translation_language"
            )
        )

    else:

        if context.chat_data.get(
            "last_translation"
        ):

            text = (
                context.chat_data.get(
                    "last_translation"
                )
            )

            source_language = (
                context.chat_data.get(
                    "last_translation_language"
                )
            )

        else:

            text = (
                context.chat_data.get(
                    "last_transcript"
                )
            )

            source_language = (
                context.chat_data.get(
                    "last_source_language"
                )
            )

    if not text:

        await chat.send_message(
            "No transcript on file yet."
        )

        return

    voice = context.chat_data.get(
        "edge_voice",
        DEFAULT_EDGE_VOICE,
    )

    video_id = context.chat_data.get(
        "last_video_id",
        "audio",
    )

    rate = get_tts_rate(
        voice,
        source_language,
    )

    await context.bot.send_chat_action(
        chat_id=chat.id,
        action=ChatAction.UPLOAD_VOICE,
    )

    status_msg = await chat.send_message(
        "Generating narration…\n\n"
        f"Voice: {voice}\n"
        f"Rate: {rate}"
    )

    file_path = (
        f"/tmp/speech_"
        f"{video_id}.mp3"
    )

    async def progress(
        done: int,
        total: int,
    ):

        # Only worth reporting once text was actually split into
        # multiple chunks; a single-chunk narration finishes fast.
        if total <= 1:
            return

        try:

            await context.bot.send_chat_action(
                chat_id=chat.id,
                action=ChatAction.UPLOAD_VOICE,
            )

        except Exception:
            pass

        try:

            await status_msg.edit_text(
                "Generating narration…\n\n"
                f"Voice: {voice}\n"
                f"Rate: {rate}\n"
                f"Progress: "
                f"{done}/{total} chunks"
            )

        except Exception:
            pass

    try:

        await generate_speech(
            text,
            voice,
            file_path,
            source_language,
            progress,
        )

    except Exception as exc:

        logger.exception(
            "TTS generation failed"
        )

        await status_msg.edit_text(
            "Speech generation failed:\n"
            f"{exc}"
        )

        try:
            os.remove(file_path)
        except OSError:
            pass

        return

    try:

        await status_msg.delete()

        with open(
            file_path,
            "rb",
        ) as file:

            await context.bot.send_audio(
                chat_id=chat.id,
                audio=file,
                title=(
                    f"{video_id} narration"
                ),
                performer=voice,
                caption=(
                    f"Voice: {voice}\n"
                    f"Rate: {rate}"
                ),
            )

    finally:

        try:
            os.remove(file_path)
        except OSError:
            pass


# ==========================================================================
# HELPERS
# ==========================================================================

def chunk_lines(
    lines,
    max_chars=3800,
):

    chunks = []
    current = ""

    for line in lines:

        if (
            current
            and len(current)
            + len(line)
            + 1
            > max_chars
        ):

            chunks.append(
                current
            )

            current = line

        else:

            current = (
                f"{current}\n{line}"
                if current
                else line
            )

    if current:
        chunks.append(current)

    return chunks


def format_duration(
    seconds: float,
):

    seconds = int(
        max(
            0,
            seconds,
        )
    )

    if seconds < 60:

        return f"{seconds}s"

    minutes, seconds = divmod(
        seconds,
        60,
    )

    if minutes < 60:

        return (
            f"{minutes}m"
            f"{seconds:02d}s"
        )

    hours, minutes = divmod(
        minutes,
        60,
    )

    return (
        f"{hours}h"
        f"{minutes:02d}m"
    )


# ==========================================================================
# KEYBOARDS
# ==========================================================================

def build_main_menu_keyboard():

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "📄 Get transcript",
                callback_data=(
                    "menu:transcript"
                ),
            )
        ],
        [
            InlineKeyboardButton(
                "🌐 Translate",
                callback_data=(
                    "menu:translate"
                ),
            ),
            InlineKeyboardButton(
                "🔊 Narrate",
                callback_data=(
                    "menu:voice"
                ),
            ),
        ],
        [
            InlineKeyboardButton(
                "⚙️ Settings",
                callback_data=(
                    "menu:settings"
                ),
            )
        ],
    ])


def build_post_transcript_keyboard():

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🌐 Translate",
                callback_data=(
                    "menu:translate"
                ),
            ),
            InlineKeyboardButton(
                "🔊 Narrate",
                callback_data=(
                    "narrate:original"
                ),
            ),
        ]
    ])


def build_post_translate_keyboard():

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🔊 Narrate",
                callback_data=(
                    "narrate:translated"
                ),
            ),
            InlineKeyboardButton(
                "🌐 Another language",
                callback_data=(
                    "menu:translate"
                ),
            ),
        ]
    ])


def build_language_keyboard():

    buttons = [
        InlineKeyboardButton(
            language,
            callback_data=(
                f"lang:{language}"
            ),
        )
        for language
        in QUICK_LANGUAGES
    ]

    rows = [
        buttons[i:i + 3]
        for i in range(
            0,
            len(buttons),
            3,
        )
    ]

    rows.append([
        InlineKeyboardButton(
            "✍️ Custom language",
            callback_data=(
                "lang:custom"
            ),
        )
    ])

    return InlineKeyboardMarkup(
        rows
    )


def build_model_keyboard():

    rows = [
        [
            InlineKeyboardButton(
                label,
                callback_data=(
                    f"setmodel:{model_id}"
                ),
            )
        ]
        for label, model_id
        in QUICK_MODELS
    ]

    rows.append([
        InlineKeyboardButton(
            "📜 Browse models",
            callback_data=(
                "models:list"
            ),
        )
    ])

    return InlineKeyboardMarkup(
        rows
    )


def build_voice_keyboard():

    rows = [
        [
            InlineKeyboardButton(
                label,
                callback_data=(
                    f"setvoice:{voice_id}"
                ),
            )
        ]
        for label, voice_id
        in QUICK_VOICES
    ]

    rows.append([
        InlineKeyboardButton(
            "📜 Browse voices",
            callback_data=(
                "voices:list"
            ),
        )
    ])

    return InlineKeyboardMarkup(
        rows
    )


def build_settings_keyboard():

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🧠 Change model",
                callback_data=(
                    "changemodel"
                ),
            )
        ],
        [
            InlineKeyboardButton(
                "🎙 Change voice",
                callback_data=(
                    "changevoice"
                ),
            )
        ],
    ])


def build_stop_keyboard(
    run_id: str,
):

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🛑 Stop",
                callback_data=(
                    f"stopbulk:{run_id}"
                ),
            )
        ]
    ])


# ==========================================================================
# HELP
# ==========================================================================

HELP_TEXT = """
Send me a YouTube link and I'll fetch its transcript.

TRANSCRIPT
/transcript <url>

TRANSLATION
/translate <language>
/translate <language> <youtube_url>

/models
/model <model_id>

VOICE
/voice
/voice original
/voice translated

/voices <filter>
/setvoice <voice_id>

/settings

PROXY
/setproxy <proxy>
/clearproxy
/proxystatus
/checkproxy
/checkproxies <list>

/stop

Long translations are automatically split into
smaller chunks and paced to stay under the
NVIDIA 30 RPM limit.

Long narrations are automatically split into
smaller chunks too, so Edge TTS doesn't stall
or time out on big transcripts.
""".strip()


# ==========================================================================
# COMMANDS
# ==========================================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        HELP_TEXT,
        reply_markup=(
            build_main_menu_keyboard()
        ),
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await start(
        update,
        context,
    )


async def transcript_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not context.args:

        await update.message.reply_text(
            "Usage: "
            "/transcript <youtube_url>"
        )

        return

    await run_transcript_fetch(
        update,
        context,
        context.args[0],
    )


async def models_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not NVIDIA_API_KEY:

        await update.message.reply_text(
            "NVIDIA_API_KEY is not set."
        )

        return

    loop = asyncio.get_running_loop()

    try:

        models = (
            await loop.run_in_executor(
                None,
                list_nvidia_models,
            )
        )

    except Exception as exc:

        await update.message.reply_text(
            f"Couldn't fetch models:\n"
            f"{exc}"
        )

        return

    lines = [
        "NVIDIA models available:",
        "",
    ]

    lines.extend(models)

    for chunk in chunk_lines(
        lines
    ):

        await update.message.reply_text(
            chunk
        )


async def model_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if context.args:

        model_id = (
            " ".join(
                context.args
            )
            .strip()
        )

        context.chat_data[
            "nvidia_model"
        ] = model_id

        await update.message.reply_text(
            "Translation model set to:\n"
            f"{model_id}"
        )

        return

    current = context.chat_data.get(
        "nvidia_model",
        DEFAULT_NVIDIA_MODEL,
    )

    await update.message.reply_text(
        "Current model:\n"
        f"{current}\n\n"
        "Choose another:",
        reply_markup=(
            build_model_keyboard()
        ),
    )


async def translate_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not NVIDIA_API_KEY:

        await update.message.reply_text(
            "NVIDIA_API_KEY is not set."
        )

        return

    if not context.args:

        await update.message.reply_text(
            "Usage:\n"
            "/translate Hindi\n"
            "/translate Spanish <youtube_url>"
        )

        return

    args = list(
        context.args
    )

    url = None

    if (
        args
        and YOUTUBE_URL_PATTERN.search(
            args[-1]
        )
    ):

        url = args[-1]
        args = args[:-1]

    target_language = (
        " ".join(args).strip()
    )

    if not target_language:

        await update.message.reply_text(
            "Please specify a language."
        )

        return

    await run_translation(
        update,
        context,
        target_language,
        url,
    )


async def voices_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    filter_str = (
        " ".join(
            context.args
        )
        if context.args
        else None
    )

    try:

        names = await list_edge_voices(
            filter_str
        )

    except Exception as exc:

        await update.message.reply_text(
            f"Couldn't fetch voices:\n"
            f"{exc}"
        )

        return

    if not names:

        await update.message.reply_text(
            "No voices matched."
        )

        return

    lines = [
        (
            f"Voices matching "
            f"'{filter_str}':"
            if filter_str
            else "Edge TTS voices:"
        ),
        "",
    ]

    lines.extend(names)

    for chunk in chunk_lines(
        lines
    ):

        await update.message.reply_text(
            chunk
        )


async def setvoice_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if context.args:

        voice_id = (
            context.args[0].strip()
        )

        context.chat_data[
            "edge_voice"
        ] = voice_id

        await update.message.reply_text(
            "Voice set to:\n"
            f"{voice_id}"
        )

        return

    current = context.chat_data.get(
        "edge_voice",
        DEFAULT_EDGE_VOICE,
    )

    await update.message.reply_text(
        "Current voice:\n"
        f"{current}",
        reply_markup=(
            build_voice_keyboard()
        ),
    )


async def voice_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    source = (
        context.args[0].lower().strip()
        if context.args
        else None
    )

    if source not in (
        None,
        "original",
        "translated",
    ):

        await update.message.reply_text(
            "Usage:\n"
            "/voice\n"
            "/voice original\n"
            "/voice translated"
        )

        return

    await run_voice_generation(
        update,
        context,
        source,
    )


async def settings_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    model = context.chat_data.get(
        "nvidia_model",
        DEFAULT_NVIDIA_MODEL,
    )

    voice = context.chat_data.get(
        "edge_voice",
        DEFAULT_EDGE_VOICE,
    )

    await update.message.reply_text(
        "⚙️ Current settings\n\n"
        f"Translation model:\n"
        f"{model}\n\n"
        f"Narration voice:\n"
        f"{voice}\n\n"
        f"NVIDIA pacing:\n"
        f"{NVIDIA_MIN_REQUEST_INTERVAL:.1f}s "
        "between requests\n"
        f"≈ "
        f"{60 / NVIDIA_MIN_REQUEST_INTERVAL:.1f} "
        "requests/min",
        reply_markup=(
            build_settings_keyboard()
        ),
    )


# ==========================================================================
# STOP
# ==========================================================================

async def stop_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):

        await update.message.reply_text(
            "Only the bot owner can stop a run."
        )

        return

    run = context.chat_data.get(
        "bulk_proxy_run"
    )

    if not run:

        await update.message.reply_text(
            "No bulk proxy check is running."
        )

        return

    if run["stop"].is_set():

        await update.message.reply_text(
            "Already stopping."
        )

        return

    run["stop"].set()

    await update.message.reply_text(
        "🛑 Stopping the proxy check…"
    )


# ==========================================================================
# SET PROXY
# ==========================================================================

async def setproxy_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):

        await update.message.reply_text(
            "Only the bot owner can change "
            "the proxy."
        )

        return

    if not context.args:

        await update.message.reply_text(
            "Usage:\n"
            "/setproxy host:port\n"
            "/setproxy host:port:user:pass\n"
            "/setproxy user:pass@host:port\n"
            "/setproxy socks5://host:port"
        )

        return

    args = context.args

    mode = args[0].lower()

    chat = update.effective_chat

    if mode == "webshare":

        if len(args) < 3:

            await update.message.reply_text(
                "Usage: "
                "/setproxy webshare "
                "<username> <password>"
            )

            return

        username = args[1]
        password = args[2]

        previous = dict(
            current_proxy
        )

        current_proxy.update(
            type="webshare",
            webshare_username=username,
            webshare_password=password,
            http_url="",
            https_url="",
        )

        status = await chat.send_message(
            "Testing Webshare proxy…"
        )

        loop = asyncio.get_running_loop()

        try:

            await loop.run_in_executor(
                None,
                test_proxy_against_youtube,
            )

        except Exception as exc:

            current_proxy.clear()
            current_proxy.update(
                previous
            )

            await status.edit_text(
                "❌ Proxy failed:\n"
                f"{exc}\n\n"
                "Previous proxy restored."
            )

            return

        await status.edit_text(
            "✅ Webshare proxy works."
        )

        return

    if mode == "generic":

        if len(args) < 2:

            await update.message.reply_text(
                "Usage:\n"
                "/setproxy generic "
                "<http_url> [https_url]"
            )

            return

        http_url = args[1]

        https_url = (
            args[2]
            if len(args) > 2
            else args[1]
        )

        previous = dict(
            current_proxy
        )

        current_proxy.update(
            type="generic",
            http_url=http_url,
            https_url=https_url,
            webshare_username="",
            webshare_password="",
        )

        status = await chat.send_message(
            "Testing generic proxy…"
        )

        loop = asyncio.get_running_loop()

        try:

            await loop.run_in_executor(
                None,
                test_proxy_against_youtube,
            )

        except Exception as exc:

            current_proxy.clear()
            current_proxy.update(
                previous
            )

            await status.edit_text(
                "❌ Proxy failed:\n"
                f"{exc}\n\n"
                "Previous proxy restored."
            )

            return

        await status.edit_text(
            "✅ Generic proxy works."
        )

        return

    candidate = (
        args[0]
        if len(args) == 1
        else ":".join(args)
    )

    parsed = parse_proxy_input(
        candidate
    )

    if parsed is None:

        await update.message.reply_text(
            "Couldn't recognize that "
            "proxy format."
        )

        return

    (
        host,
        port,
        user,
        password,
        forced_scheme,
    ) = parsed

    await auto_configure_proxy(
        chat,
        host,
        port,
        user,
        password,
        forced_scheme,
    )


# ==========================================================================
# AUTO PROXY
# ==========================================================================

async def auto_configure_proxy(
    chat,
    host,
    port,
    user,
    password,
    forced_scheme=None,
):

    schemes = (
        [forced_scheme]
        if forced_scheme
        else PROXY_SCHEMES_TO_TRY
    )

    status_msg = await chat.send_message(
        "Auto-detecting proxy protocol for "
        f"{host}:{port}…"
    )

    loop = asyncio.get_running_loop()

    attempts = []

    for scheme in schemes:

        (
            http_url,
            https_url,
        ) = build_generic_proxy_urls(
            host,
            port,
            user,
            password,
            scheme,
        )

        try:

            await loop.run_in_executor(
                None,
                test_proxy_via_urls,
                http_url,
                https_url,
            )

        except Exception as exc:

            attempts.append(
                f"{scheme}:// → {exc}"
            )

            continue

        current_proxy.update(
            type="generic",
            http_url=http_url,
            https_url=https_url,
            webshare_username="",
            webshare_password="",
        )

        await status_msg.edit_text(
            "✅ Working proxy found.\n\n"
            f"Protocol: {scheme}://\n"
            f"Proxy: {host}:{port}"
        )

        return

    await status_msg.edit_text(
        "❌ No working protocol found.\n\n"
        + "\n".join(attempts)
    )


# ==========================================================================
# SINGLE PROXY CHECK
# ==========================================================================

async def checkproxy_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):

        await update.message.reply_text(
            "Only the bot owner can run this."
        )

        return

    status = await update.effective_chat.send_message(
        "Testing current proxy…"
    )

    loop = asyncio.get_running_loop()

    try:

        await loop.run_in_executor(
            None,
            test_proxy_against_youtube,
        )

    except Exception as exc:

        await status.edit_text(
            "❌ Proxy test failed:\n"
            f"{exc}"
        )

        return

    await status.edit_text(
        "✅ Proxy works and the full "
        "YouTube transcript test passed."
    )


# ==========================================================================
# PROXY STATUS
# ==========================================================================

async def proxystatus_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "Current proxy:\n"
        + describe_current_proxy()
    )


# ==========================================================================
# CLEAR PROXY
# ==========================================================================

async def clearproxy_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):

        await update.message.reply_text(
            "Only the bot owner can change "
            "the proxy."
        )

        return

    current_proxy.update(
        type=None,
        webshare_username="",
        webshare_password="",
        http_url="",
        https_url="",
    )

    await update.message.reply_text(
        "Proxy cleared."
    )


# ==========================================================================
# BULK PROXY CHECK
# ==========================================================================

class ProgressTracker:

    def __init__(
        self,
        total,
        status_msg,
        header,
        reply_markup=None,
        stop_event=None,
    ):

        self.total = total

        self.status_msg = (
            status_msg
        )

        self.header = header

        self.reply_markup = (
            reply_markup
        )

        self.stop_event = (
            stop_event
        )

        self.done = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0

        self.started = (
            asyncio.get_running_loop()
            .time()
        )

        self.lock = asyncio.Lock()

        self.last_edit = 0.0

    async def tick(
        self,
        ok,
        skipped=False,
    ):

        async with self.lock:

            self.done += 1

            if skipped:

                self.skipped += 1

            elif ok:

                self.passed += 1

            else:

                self.failed += 1

            now = (
                asyncio.get_running_loop()
                .time()
            )

            should_edit = (
                self.done
                >= self.total
                or (
                    now
                    - self.last_edit
                    >= PROGRESS_EDIT_INTERVAL
                )
            )

            if not should_edit:
                return

            self.last_edit = now

        try:

            await self.status_msg.edit_text(
                self.render(),
                reply_markup=(
                    self.reply_markup
                ),
            )

        except Exception:
            pass

    def render(self):

        width = 18

        ratio = (
            self.done / self.total
            if self.total
            else 0
        )

        filled = int(
            width * ratio
        )

        bar = (
            "█" * filled
            + "░" * (
                width - filled
            )
        )

        percent = ratio * 100

        elapsed = (
            asyncio.get_running_loop()
            .time()
            - self.started
        )

        rate = (
            self.done / elapsed
            if elapsed > 0
            and self.done
            else 0
        )

        remaining = (
            self.total - self.done
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        text = (
            f"{self.header}\n\n"
            f"`[{bar}]` "
            f"{percent:5.1f}%\n\n"
            f"✅ {self.passed} working\n"
            f"❌ {self.failed} failed\n"
            f"⚡ {rate:.1f}/s\n"
            f"⏳ ETA "
            f"{format_duration(eta)}"
        )

        if self.skipped:

            text += (
                f"\n⏭ {self.skipped} skipped"
            )

        if (
            self.stop_event
            and self.stop_event.is_set()
        ):

            text += (
                "\n\n🛑 Stopping…"
            )

        return text


async def check_proxy_candidate(
    loop,
    semaphore,
    raw,
    parsed,
    stop_event,
):

    if stop_event.is_set():

        return {
            "raw": raw,
            "ok": False,
            "skipped": True,
            "scheme": None,
            "http_url": None,
            "https_url": None,
            "note": "stopped",
        }

    (
        host,
        port,
        user,
        password,
        forced_scheme,
    ) = parsed

    schemes = (
        [forced_scheme]
        if forced_scheme
        else PROXY_SCHEMES_TO_TRY
    )

    attempts = []

    async with semaphore:

        if stop_event.is_set():

            return {
                "raw": raw,
                "ok": False,
                "skipped": True,
                "scheme": None,
                "http_url": None,
                "https_url": None,
                "note": "stopped",
            }

        for scheme in schemes:

            if stop_event.is_set():

                return {
                    "raw": raw,
                    "ok": False,
                    "skipped": True,
                    "scheme": None,
                    "http_url": None,
                    "https_url": None,
                    "note": "stopped",
                }

            (
                http_url,
                https_url,
            ) = build_generic_proxy_urls(
                host,
                port,
                user,
                password,
                scheme,
            )

            try:

                await loop.run_in_executor(
                    None,
                    test_proxy_via_urls,
                    http_url,
                    https_url,
                )

                return {
                    "raw": raw,
                    "ok": True,
                    "skipped": False,
                    "scheme": scheme,
                    "http_url": http_url,
                    "https_url": https_url,
                    "note": (
                        f"working via "
                        f"{scheme}://"
                    ),
                }

            except Exception as exc:

                attempts.append(
                    f"{scheme}:// → "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                await asyncio.sleep(
                    PROXY_CHECK_RETRY_DELAY
                )

    return {
        "raw": raw,
        "ok": False,
        "skipped": False,
        "scheme": None,
        "http_url": None,
        "https_url": None,
        "note": (
            "; ".join(attempts)
            or "no working protocol"
        ),
    }


async def run_bulk_proxy_check(
    update,
    context,
    raw_text,
    status_msg=None,
    source_label="your list",
):

    chat = update.effective_chat

    raw_candidates = (
        parse_proxy_lines(
            raw_text
        )
    )

    if not raw_candidates:

        text = (
            "No proxy-looking lines found."
        )

        if status_msg:

            await status_msg.edit_text(
                text
            )

        else:

            await chat.send_message(
                text
            )

        return

    parsed_list = []
    unparsed = []

    for raw in raw_candidates:

        parsed = parse_proxy_input(
            raw
        )

        if parsed is None:

            unparsed.append(raw)

        else:

            parsed_list.append(
                (
                    raw,
                    parsed,
                )
            )

    if not parsed_list:

        await status_msg.edit_text(
            "No valid proxy entries "
            "could be parsed."
        )

        return

    total = len(
        parsed_list
    )

    run_id = secrets.token_hex(
        4
    )

    stop_event = asyncio.Event()

    context.chat_data[
        "bulk_proxy_run"
    ] = {
        "id": run_id,
        "stop": stop_event,
    }

    header = (
        f"Testing {total} proxies\n\n"
        f"Parallel checks: "
        f"{MAX_CONCURRENT_PROXY_CHECKS}\n"
        f"Source: {source_label}"
    )

    keyboard = (
        build_stop_keyboard(
            run_id
        )
    )

    if status_msg:

        await status_msg.edit_text(
            header,
            reply_markup=keyboard,
        )

    else:

        status_msg = await chat.send_message(
            header,
            reply_markup=keyboard,
        )

    tracker = ProgressTracker(
        total,
        status_msg,
        header,
        keyboard,
        stop_event,
    )

    loop = asyncio.get_running_loop()

    semaphore = asyncio.Semaphore(
        MAX_CONCURRENT_PROXY_CHECKS
    )

    async def one(
        raw,
        parsed,
    ):

        result = (
            await check_proxy_candidate(
                loop,
                semaphore,
                raw,
                parsed,
                stop_event,
            )
        )

        await tracker.tick(
            result["ok"],
            result.get(
                "skipped",
                False,
            ),
        )

        return result

    tasks = [
        asyncio.create_task(
            one(
                raw,
                parsed,
            )
        )
        for raw, parsed
        in parsed_list
    ]

    try:

        results = await asyncio.gather(
            *tasks
        )

    finally:

        context.chat_data.pop(
            "bulk_proxy_run",
            None,
        )

    passed = [
        result
        for result in results
        if result["ok"]
    ]

    skipped = [
        result
        for result in results
        if result.get("skipped")
    ]

    was_stopped = (
        stop_event.is_set()
    )

    clean_path = (
        "/tmp/clean_proxies.txt"
    )

    report_path = (
        "/tmp/proxy_check_report.txt"
    )

    with open(
        clean_path,
        "w",
        encoding="utf-8",
    ) as file:

        for result in passed:

            file.write(
                result["raw"]
                + "\n"
            )

    report_lines = [
        "Proxy check report",
        "",
        f"Total: {len(results)}",
        f"Working: {len(passed)}",
        f"Skipped: {len(skipped)}",
        f"Unparsed: {len(unparsed)}",
        "",
    ]

    for result in results:

        if result.get("skipped"):

            tag = "SKIP"

        elif result["ok"]:

            tag = "PASS"

        else:

            tag = "FAIL"

        report_lines.append(
            f"[{tag}] "
            f"{result['raw']} — "
            f"{result['note']}"
        )

    if unparsed:

        report_lines.extend([
            "",
            "Unparsed:",
        ])

        report_lines.extend(
            unparsed
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

    try:

        await status_msg.delete()

    except Exception:
        pass

    if passed:

        with open(
            clean_path,
            "rb",
        ) as file:

            await context.bot.send_document(
                chat_id=chat.id,
                document=file,
                filename=(
                    "clean_proxies.txt"
                ),
                caption=(
                    f"✅ "
                    f"{len(passed)} working "
                    f"proxy/proxies found."
                ),
            )

    else:

        await chat.send_message(
            "❌ No working proxies found."
        )

    with open(
        report_path,
        "rb",
    ) as file:

        await context.bot.send_document(
            chat_id=chat.id,
            document=file,
            filename=(
                "proxy_check_report.txt"
            ),
        )

    for path in (
        clean_path,
        report_path,
    ):

        try:
            os.remove(path)
        except OSError:
            pass

    # Full verification of first working proxy.
    if not passed:
        return

    winner = passed[0]

    verify_msg = await chat.send_message(
        "Verifying first working proxy "
        "with a full transcript test…"
    )

    previous = dict(
        current_proxy
    )

    try:

        current_proxy.update(
            type="generic",
            http_url=winner["http_url"],
            https_url=winner["https_url"],
            webshare_username="",
            webshare_password="",
        )

        await loop.run_in_executor(
            None,
            test_proxy_against_youtube,
        )

    except Exception as exc:

        current_proxy.clear()
        current_proxy.update(
            previous
        )

        await verify_msg.edit_text(
            "⚠️ First proxy passed the "
            "lightweight test but failed "
            "full transcript verification.\n\n"
            f"{exc}\n\n"
            "Previous proxy restored."
        )

    else:

        await verify_msg.edit_text(
            "🏆 First working proxy activated "
            "and fully verified.\n\n"
            f"{winner['raw']}"
        )


# ==========================================================================
# CHECK PROXIES COMMAND
# ==========================================================================

async def checkproxies_command(
    update,
    context,
):

    if not is_owner(update):

        await update.message.reply_text(
            "Only the bot owner can run this."
        )

        return

    if not context.args:

        await update.message.reply_text(
            "Upload a proxy list file or use:\n"
            "/checkproxies host:port:user:pass"
        )

        return

    # BUGFIX: joining args with a space and relying on a comma/semicolon
    # splitter afterwards meant multiple space-separated proxies passed
    # directly as command arguments (e.g. "/checkproxies a:1 b:2") were
    # silently treated as one unparsable blob. Joining with newlines
    # lets each argument be parsed as its own candidate line.
    await run_bulk_proxy_check(
        update,
        context,
        "\n".join(
            context.args
        ),
        source_label="command",
    )


# ==========================================================================
# PROXY FILE
# ==========================================================================

async def proxy_file_handler(
    update,
    context,
):

    if not is_owner(update):

        await update.message.reply_text(
            "Only the bot owner can check "
            "proxy lists."
        )

        return

    document = (
        update.message.document
    )

    filename = (
        document.file_name
        or "upload"
    )

    if not filename.lower().endswith(
        ALLOWED_PROXY_FILE_EXTS
    ):

        await update.message.reply_text(
            "Upload .txt, .csv, .list "
            "or .proxies file."
        )

        return

    if (
        document.file_size
        and document.file_size
        > MAX_PROXY_FILE_BYTES
    ):

        await update.message.reply_text(
            "File is too large. "
            "Maximum is 10 MB."
        )

        return

    status_msg = (
        await update.message.reply_text(
            f"Reading {filename}…"
        )
    )

    try:

        telegram_file = (
            await context.bot.get_file(
                document.file_id
            )
        )

        raw_bytes = (
            await telegram_file.download_as_bytearray()
        )

    except Exception as exc:

        await status_msg.edit_text(
            "Couldn't download file:\n"
            f"{exc}"
        )

        return

    text = bytes(
        raw_bytes
    ).decode(
        "utf-8",
        errors="ignore",
    )

    if not text.strip():

        await status_msg.edit_text(
            "The file is empty."
        )

        return

    await run_bulk_proxy_check(
        update,
        context,
        text,
        status_msg=status_msg,
        source_label=filename,
    )


# ==========================================================================
# CALLBACKS
# ==========================================================================

async def button_callback(
    update,
    context,
):

    query = update.callback_query

    await query.answer()

    data = query.data or ""

    # ----------------------------------------------------------------------
    # STOP
    # ----------------------------------------------------------------------

    if data.startswith(
        "stopbulk:"
    ):

        if not is_owner(update):

            await query.answer(
                "Only the owner can stop this.",
                show_alert=True,
            )

            return

        run_id = data.split(
            ":",
            1,
        )[1]

        run = context.chat_data.get(
            "bulk_proxy_run"
        )

        if (
            run
            and run["id"] == run_id
        ):

            run["stop"].set()

            await query.answer(
                "Stopping…"
            )

        else:

            await query.answer(
                "This run is no longer active."
            )

        return

    # ----------------------------------------------------------------------
    # MENU
    # ----------------------------------------------------------------------

    if data == "menu:transcript":

        await query.message.reply_text(
            "Send me a YouTube link."
        )

        return

    if data == "menu:translate":

        if not context.chat_data.get(
            "last_transcript"
        ):

            await query.message.reply_text(
                "Send a YouTube link first."
            )

            return

        await query.message.reply_text(
            "Choose target language:",
            reply_markup=(
                build_language_keyboard()
            ),
        )

        return

    if data == "menu:voice":

        await run_voice_generation(
            update,
            context,
            None,
        )

        return

    if data == "menu:settings":

        await settings_command(
            update,
            context,
        )

        return

    # ----------------------------------------------------------------------
    # MODELS
    # ----------------------------------------------------------------------

    if data == "changemodel":

        await query.message.reply_text(
            "Choose model:",
            reply_markup=(
                build_model_keyboard()
            ),
        )

        return

    if data == "models:list":

        await models_command(
            update,
            context,
        )

        return

    if data.startswith(
        "setmodel:"
    ):

        model_id = data.split(
            ":",
            1,
        )[1]

        context.chat_data[
            "nvidia_model"
        ] = model_id

        await query.message.reply_text(
            "Model set to:\n"
            f"{model_id}"
        )

        return

    # ----------------------------------------------------------------------
    # VOICE
    # ----------------------------------------------------------------------

    if data == "changevoice":

        await query.message.reply_text(
            "Choose voice:",
            reply_markup=(
                build_voice_keyboard()
            ),
        )

        return

    if data == "voices:list":

        await voices_command(
            update,
            context,
        )

        return

    if data.startswith(
        "setvoice:"
    ):

        voice_id = data.split(
            ":",
            1,
        )[1]

        context.chat_data[
            "edge_voice"
        ] = voice_id

        await query.message.reply_text(
            "Voice set to:\n"
            f"{voice_id}"
        )

        return

    # ----------------------------------------------------------------------
    # LANGUAGE
    # ----------------------------------------------------------------------

    if data.startswith(
        "lang:"
    ):

        language = data.split(
            ":",
            1,
        )[1]

        if language == "custom":

            await query.message.reply_text(
                "Use:\n"
                "/translate <language>"
            )

            return

        await run_translation(
            update,
            context,
            language,
            None,
        )

        return

    # ----------------------------------------------------------------------
    # NARRATION
    # ----------------------------------------------------------------------

    if data.startswith(
        "narrate:"
    ):

        source = data.split(
            ":",
            1,
        )[1]

        await run_voice_generation(
            update,
            context,
            source,
        )


# ==========================================================================
# NORMAL MESSAGE
# ==========================================================================

async def message_handler(
    update,
    context,
):

    text = (
        update.message.text
        or ""
    )

    match = (
        YOUTUBE_URL_PATTERN.search(
            text
        )
    )

    if match:

        await run_transcript_fetch(
            update,
            context,
            match.group(0),
        )

        return

    await update.message.reply_text(
        "Send a YouTube link or use /help."
    )


# ==========================================================================
# POST INIT
# ==========================================================================

async def post_init(
    application,
):

    await application.bot.set_my_commands([
        BotCommand(
            "start",
            "Welcome menu",
        ),
        BotCommand(
            "help",
            "Show help",
        ),
        BotCommand(
            "transcript",
            "Get transcript",
        ),
        BotCommand(
            "translate",
            "Translate transcript",
        ),
        BotCommand(
            "models",
            "List NVIDIA models",
        ),
        BotCommand(
            "model",
            "Set NVIDIA model",
        ),
        BotCommand(
            "voice",
            "Narrate transcript",
        ),
        BotCommand(
            "voices",
            "List TTS voices",
        ),
        BotCommand(
            "setvoice",
            "Set TTS voice",
        ),
        BotCommand(
            "settings",
            "Show settings",
        ),
        BotCommand(
            "setproxy",
            "Set proxy",
        ),
        BotCommand(
            "clearproxy",
            "Clear proxy",
        ),
        BotCommand(
            "proxystatus",
            "Proxy status",
        ),
        BotCommand(
            "checkproxy",
            "Test proxy",
        ),
        BotCommand(
            "checkproxies",
            "Check proxy list",
        ),
        BotCommand(
            "stop",
            "Stop proxy check",
        ),
    ])

    loop = asyncio.get_running_loop()

    executor = ThreadPoolExecutor(
        max_workers=PROXY_CHECK_EXECUTOR_WORKERS,
        thread_name_prefix="bot-worker",
    )

    loop.set_default_executor(
        executor
    )

    logger.info(
        "Bot initialized."
    )

    logger.info(
        "Default NVIDIA model: %s",
        DEFAULT_NVIDIA_MODEL,
    )

    logger.info(
        "Translation chunk size: %d chars",
        TRANSLATION_MAX_CHARS,
    )

    logger.info(
        "TTS chunk size: %d chars",
        TTS_MAX_CHARS,
    )

    logger.info(
        "NVIDIA request interval: %.2fs "
        "(~%.1f RPM)",
        NVIDIA_MIN_REQUEST_INTERVAL,
        60 / NVIDIA_MIN_REQUEST_INTERVAL,
    )

    if OWNER_ID is None:

        logger.warning(
            "OWNER_ID is not configured."
        )


# ==========================================================================
# MAIN
# ==========================================================================

def main():

    if not BOT_TOKEN:

        raise SystemExit(
            "BOT_TOKEN is not configured."
        )

    application = (
        Application
        .builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # ----------------------------------------------------------------------
    # COMMANDS
    # ----------------------------------------------------------------------

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "transcript",
            transcript_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "translate",
            translate_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "models",
            models_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "model",
            model_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "voices",
            voices_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "setvoice",
            setvoice_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "voice",
            voice_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "settings",
            settings_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "setproxy",
            setproxy_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "clearproxy",
            clearproxy_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "proxystatus",
            proxystatus_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "checkproxy",
            checkproxy_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "checkproxies",
            checkproxies_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "stop",
            stop_command,
        )
    )

    # ----------------------------------------------------------------------
    # CALLBACKS
    # ----------------------------------------------------------------------

    application.add_handler(
        CallbackQueryHandler(
            button_callback
        )
    )

    # ----------------------------------------------------------------------
    # PROXY FILES
    # ----------------------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            proxy_file_handler,
        )
    )

    # ----------------------------------------------------------------------
    # NORMAL TEXT
    # ----------------------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            message_handler,
        )
    )

    logger.info(
        "Starting Telegram bot…"
    )

    application.run_polling()


# ==========================================================================
# ENTRY
# ==========================================================================

if __name__ == "__main__":
    main()
