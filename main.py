import os
import re
import asyncio
import logging
import requests
import edge_tts

from urllib.parse import urlparse, parse_qs

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
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "PUT-YOUR-TOKEN-HERE")

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
    r"(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be|m\.youtube\.com)/\S+",
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
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

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
        transcript = next(iter(transcript_list))

    raw_entries = transcript.fetch()
    language_used = transcript.language

    return clean_transcript(raw_entries), language_used


def clean_transcript(entries) -> str:
    """
    Turn the list of {text, start, duration} chunks into readable
    paragraphs: strip timestamps, fix spacing/line-break artifacts, and
    start a new paragraph whenever there's a natural pause (>2.5s gap)
    in the speech.
    """
    paragraphs = []
    current = []
    last_end = 0.0

    for entry in entries:
        text = entry.text.replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
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


async def fetch_transcript_or_report(chat, status_msg, url: str) -> tuple[str, str] | None:
    """Shared fetch+error-reporting used by both /transcript and /translate flows."""
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


async def translate_text(text: str, target_language: str, model: str) -> str:
    """Translate arbitrarily long text via NVIDIA NIM, chunk by chunk, in order."""
    loop = asyncio.get_running_loop()
    translated_chunks = []
    for chunk in chunk_text(text):
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

async def run_transcript_fetch(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
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

    await status_msg.delete()
    with open(file_path, "rb") as f:
        await context.bot.send_document(
            chat_id=chat.id,
            document=f,
            filename=f"transcript_{video_id}.txt",
            caption=f"Transcript ready ({language}). What next?",
            reply_markup=build_post_transcript_keyboard(),
        )
    os.remove(file_path)


async def run_translation(update: Update, context: ContextTypes.DEFAULT_TYPE, target_language: str, url: str | None) -> None:
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

    try:
        translated = await translate_text(text, target_language, model)
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

    await status_msg.delete()
    with open(file_path, "rb") as f:
        await context.bot.send_document(
            chat_id=chat.id,
            document=f,
            filename=os.path.basename(file_path),
            caption=f"Translated to {target_language} (model: {model}).",
            reply_markup=build_post_translate_keyboard(),
        )
    os.remove(file_path)


async def run_voice_generation(update: Update, context: ContextTypes.DEFAULT_TYPE, source: str | None) -> None:
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
        return

    await status_msg.delete()
    with open(file_path, "rb") as f:
        await context.bot.send_audio(
            chat_id=chat.id,
            audio=f,
            title=f"{video_id} narration",
            performer=voice,
            caption=f"Voice: {voice}",
        )
    os.remove(file_path)


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

    body = "Models available to your key (pick any with /model <id>):\n\n" + "\n".join(models)
    for i in range(0, len(body), 3800):  # stay under Telegram's 4096-char message cap
        await chat.send_message(body[i : i + 3800])


async def send_all_voices(update: Update, context: ContextTypes.DEFAULT_TYPE, filter_str: str | None) -> None:
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
        f"Voices matching '{filter_str}' (pick one with /setvoice <id>):\n\n"
        if filter_str
        else "All edge-tts voices — filter it, e.g. /voices en-US or /voices Hindi:\n\n"
    )
    body = header + "\n".join(names)
    for i in range(0, len(body), 3800):
        await chat.send_message(body[i : i + 3800])


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
    "/voice — narrate the last transcript/translation as .mp3\n"
    "/voices <filter> — browse voices, e.g. /voices en-US\n"
    "/setvoice <voice_id> — pick which voice to narrate with\n\n"
    "/settings — see and change your current model & voice"
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
    await run_voice_generation(update, context, source)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_settings_message(update, context)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data == "menu:transcript":
        await query.message.reply_text("Send me a YouTube link and I'll fetch the transcript.")

    elif data == "menu:translate":
        if not NVIDIA_API_KEY:
            await query.message.reply_text("NVIDIA_API_KEY is not set on the server.")
            return
        if not context.chat_data.get("last_transcript"):
            await query.message.reply_text("No transcript on file yet — send a YouTube link first.")
            return
        await query.message.reply_text("Pick a language:", reply_markup=build_language_keyboard())

    elif data == "menu:voice":
        await run_voice_generation(update, context, source=None)

    elif data == "menu:settings":
        await send_settings_message(update, context)

    elif data == "changemodel":
        await query.message.reply_text(
            "Pick a shortcut, or use /model <model_id> for any other:",
            reply_markup=build_model_keyboard(),
        )

    elif data == "changevoice":
        await query.message.reply_text(
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
        await query.message.reply_text(f"Translation model set to: {model_id}")

    elif data.startswith("setvoice:"):
        voice_id = data.split(":", 1)[1]
        context.chat_data["edge_voice"] = voice_id
        await query.message.reply_text(f"Voice set to: {voice_id}")

    elif data.startswith("lang:"):
        lang = data.split(":", 1)[1]
        if lang == "custom":
            await query.message.reply_text("Type it as: /translate <language>\nExample: /translate Bengali")
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
            "Send a YouTube link and I'll pull the transcript for you, or /help for everything I can do."
        )


async def post_init(application: Application) -> None:
    """Registers the native Telegram '/' command menu."""
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
    ])


def main() -> None:
    if BOT_TOKEN == "PUT-YOUR-TOKEN-HERE":
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
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Bot starting…")
    app.run_polling()


if __name__ == "__main__":
    main()
