---
title: Shadowing v2
emoji: 🎧
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: YouTube link → sentence-by-sentence shadowing player
---

# Shadowing v2

Paste a YouTube link. The server fetches the video's captions, splits them into
sentences and translates them; each sentence then plays through YouTube's
official player — loop it, pause after it to repeat it aloud, slow it down.

- Controls: previous · play/pause · repeat sentence · next. Tap any sentence to play it.
- Settings: speed, repeat count, pause after each sentence, translation (Arabic / English), font size, caption track.
- Videos you open are saved in your own browser (nothing is stored on the server).

Personal use only. No audio or video is downloaded. No GPU is used.

Optional Space secrets, only if YouTube ever blocks the server:
`PROXY_URL` (an http proxy URL) and/or `YT_COOKIES` (Netscape-format cookies for yt-dlp).
