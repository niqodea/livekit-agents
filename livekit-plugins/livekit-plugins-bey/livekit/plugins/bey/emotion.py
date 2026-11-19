"""The emotion a Beyond Presence avatar speaks each reply with.

The emotion is attached to the reply's audio frames, and the avatar's audio output tells
the avatar the emotion of each audio stream it opens, so it holds exactly for that reply.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Literal, get_args

from livekit import rtc
from livekit.agents import Agent, ModelSettings
from livekit.agents.tts._provider_format import ExpressiveTag, TranscriptMarkupStripper

Emotion = Literal["happy", "sad"]
"""An emotion the avatar can speak with. Emotions show on Genesis 3 avatars only."""

EmotionIntensity = Literal["low", "medium", "high"]
"""How strongly the avatar shows an emotion."""

# Where `with_emotion` puts a frame's emotion, as "<emotion>:<intensity>", the value the
# avatar reads from its audio stream's `bey.emotion` attribute
EMOTION_USERDATA = "bey.emotion"


def with_emotion(
    audio: AsyncIterable[rtc.AudioFrame],
    emotion: Emotion | None,
    *,
    intensity: EmotionIntensity = "medium",
) -> AsyncIterable[rtc.AudioFrame]:
    """Have the avatar speak ``audio``, one reply's audio, with ``emotion``.

    Return it from your agent's ``tts_node`` or ``realtime_audio_output_node``::

        def realtime_audio_output_node(self, audio, model_settings):
            return bey.with_emotion(audio, self.emotion)

    Args:
        audio: The reply's audio.
        emotion: The emotion to speak it with; None for the avatar's neutral face.
        intensity: How strongly to show it.

    Raises:
        ValueError: If ``emotion`` or ``intensity`` is not supported.
    """
    # TODO: Warn once and ignore the emotion when the avatar does not show emotions,
    # once the sessions API tells the plugin the avatar's model
    if emotion is not None and emotion not in get_args(Emotion):
        raise ValueError(f"Unsupported emotion: {emotion!r}")
    if intensity not in get_args(EmotionIntensity):
        raise ValueError(f"Unsupported emotion intensity: {intensity!r}")
    if emotion is None:
        return audio
    return _mark(audio, f"{emotion}:{intensity}")


def emotion_instructions() -> str:
    """Prompt text that teaches the LLM the marker ``emotion_tts_node`` reads.

    Add it to your agent's instructions. With LiveKit's expressive mode, pass it as
    ``tts_instructions_append`` instead, so one marker steers both the voice and the face.
    The text has no braces, so it is safe inside the expressive template.
    """
    names = ", ".join(get_args(Emotion))
    return (
        "Your face is animated from your expression markers. Start every reply with one "
        '<expr type="expression" label="..."/> marker for the expression you say it with. '
        f"The face shows only these expressions: {names}. Use calm for most replies, and "
        "another expression only in moments that truly warrant it. A marker with any other "
        "label leaves the face neutral."
    )


async def emotion_tts_node(
    agent: Agent, text: AsyncIterable[str], model_settings: ModelSettings
) -> AsyncIterator[rtc.AudioFrame]:
    """The default ``tts_node``, with the reply spoken with the expression its first marker
    names, a ``<expr type="expression" label="..."/>`` marker as ``emotion_instructions``
    asks for. Return it from your agent's ``tts_node``::

        def tts_node(self, text, model_settings):
            return bey.emotion_tts_node(self, text, model_settings)

    With LiveKit's expressive mode the markup goes on to the TTS, so the marker steers the
    voice too; otherwise it is stripped before the TTS. LiveKit keeps it out of the
    transcript either way. The face changes once per reply, so later markers in the reply
    only steer the voice.
    """
    # Private, as Keyframe's plugin reads it: LiveKit has no public way to tell yet
    expressive = agent._get_activity_or_raise()._resolve_expressive_options() is not None
    stripper = TranscriptMarkupStripper()
    chunks = aiter(text)
    head: list[str] = []
    clean_head = ""
    async for chunk in chunks:
        head.append(chunk)
        clean_head += stripper.push(chunk)
        # The reply's expression is the marker before its first spoken word
        if clean_head.strip():
            break
    else:
        clean_head += stripper.flush()

    async def _reply() -> AsyncIterator[str]:
        if expressive:
            for chunk in head:
                yield chunk
            async for chunk in chunks:
                yield chunk
            return
        if clean_head:
            yield clean_head
        async for chunk in chunks:
            if clean := stripper.push(chunk):
                yield clean
        if clean := stripper.flush():
            yield clean

    emotion = _read_label(stripper.tags)
    audio = Agent.default.tts_node(agent, _reply(), model_settings)
    async for frame in with_emotion(audio, emotion):
        yield frame


def _read_label(tags: list[ExpressiveTag]) -> Emotion | None:
    """The emotion the first expression marker names as a whole word, if any."""
    label = next((t["value"] for t in tags if t["type"] in ("expression", "emotion")), None)
    if label is None:
        return None
    found = [
        (m.start(), emotion)
        for emotion in get_args(Emotion)
        if (m := re.search(rf"\b{emotion}\b", label.lower()))
    ]
    return min(found)[1] if found else None


def get_emotion(frame: rtc.AudioFrame) -> str | None:
    """The emotion `with_emotion` gave the reply ``frame`` belongs to, as
    "<emotion>:<intensity>"; None for the avatar's neutral face."""
    emotion: str | None = frame.userdata.get(EMOTION_USERDATA)
    return emotion


async def _mark(audio: AsyncIterable[rtc.AudioFrame], value: str) -> AsyncIterator[rtc.AudioFrame]:
    async for frame in audio:
        frame.userdata[EMOTION_USERDATA] = value
        yield frame
