# YouTube Transcript, Translate & Voice Telegram Bot

A Telegram bot that pulls a clean transcript from any YouTube video, translates it into any language using an NVIDIA NIM model of your choice, and narrates it back as an MP3 using Microsoft Edge TTS — all with inline buttons, no typing required after the first link.

## Features

- **Transcript extraction** — send a YouTube link (or `/transcript <url>`) and get back a clean `.txt` file. Timestamps stripped, text reflowed into readable paragraphs.
- **AI translation** — translate the transcript into any language via [NVIDIA NIM](https://build.nvidia.com). Model is fully configurable per chat, not hardcoded.
- **Text-to-speech narration** — turn the transcript (original or translated) into a spoken `.mp3` via [edge-tts](https://github.com/rany2/edge-tts), free, no API key, hundreds of voices/languages.
- **Inline keyboards** — every result comes with buttons for the obvious next step (Translate / Narrate / pick a language / pick a model / pick a voice).
- **Native Telegram command menu** — tap the `/` icon in any chat to see every command with a description.
- **Per-chat settings** — each chat remembers its own chosen model, voice, last transcript, and last translation for the session.

## Requirements

- Python 3.10+
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- An [NVIDIA NIM API key](https://build.nvidia.com) (optional — only needed for `/translate`)

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Configuration

Set your bot token (required) and NVIDIA key (optional, enables translation):

```bash
export BOT_TOKEN="123456:ABC-your-telegram-bot-token"
export NVIDIA_API_KEY="nvapi-your-nvidia-nim-key"
```

On Windows (PowerShell):

```powershell
$env:BOT_TOKEN="123456:ABC-your-telegram-bot-token"
$env:NVIDIA_API_KEY="nvapi-your-nvidia-nim-key"
```

## Running

```bash
python telegram_transcript_bot.py
```

The bot starts polling immediately — open it in Telegram and send `/start`.

## Commands

| Command | Description |
|---|---|
| `/start` | Welcome menu with buttons |
| `/help` | List everything the bot can do |
| `/transcript <url>` | Fetch a transcript as a `.txt` file |
| `/translate <language>` | Translate the last transcript fetched in this chat |
| `/translate <language> <url>` | Fetch a video's transcript and translate it in one step |
| `/models` | List every NVIDIA NIM model your key can use |
| `/model <model_id>` | Set which model `/translate` uses (any model ID — no restriction) |
| `/voice` | Narrate the last transcript/translation as an `.mp3` |
| `/voice original\|translated` | Force which text gets narrated |
| `/voices <filter>` | Browse edge-tts voices, e.g. `/voices en-US` or `/voices Hindi` |
| `/setvoice <voice_id>` | Set which voice `/voice` uses (any edge-tts voice ID) |
| `/settings` | View and change your current model & voice with buttons |

You can also just paste a YouTube link with no command — the bot detects it and fetches the transcript automatically.

## Typical flow

1. Send a YouTube link → get the transcript `.txt`, with **Translate** / **Narrate** buttons attached.
2. Tap **Translate** → pick a language from the buttons (or type `/translate <language>`) → get the translated `.txt`, with a **Narrate this** button.
3. Tap **Narrate this** → get an `.mp3` in the current voice.
4. Change the model or voice anytime with `/settings`, `/model`, or `/setvoice`.

## Notes & limitations

- Per-chat state (last transcript, chosen model/voice) is kept in memory only — it resets if the bot restarts.
- Translation requires captions/subtitles to exist on the source video; if YouTube has no transcript at all, there's nothing to translate.
- Telegram bots can send files up to 50 MB; a narrated `.mp3` of an unusually long video could approach that limit.
- This project makes outbound calls to YouTube (via `youtube-transcript-api`), NVIDIA NIM, and Microsoft's edge-tts endpoint — make sure your hosting environment allows outbound HTTPS.

## Project structure

```
.
├── telegram_transcript_bot.py   # bot entry point and all logic
├── requirements.txt             # Python dependencies
└── README.md
```

## License

Add a license of your choice (MIT is a common default for personal bots).
