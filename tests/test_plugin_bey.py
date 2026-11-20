"""Tests for the Beyond Presence plugin: retry classification when starting a session,
and the emotion each reply is spoken with.

Same behaviour as the Anam fix in #7314: a 4xx fails on the first attempt with the
provider's status, a 5xx is retried `max_retry` times, and a 5xx that persists is
raised as the provider's error rather than a generic connection error.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("bey")

_OPTS = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status
        self.ok = status < 400

    async def text(self) -> str:
        return f"status {self.status}"

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _ScriptedSession:
    """Answers each POST with the next status in `script`; an exception instance in
    the script is raised instead. The last entry repeats."""

    def __init__(self, script: list[int | BaseException]) -> None:
        self._script = list(script)
        self.posts = 0

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts += 1
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return _Response(step)


def _avatar(session: _ScriptedSession):
    from livekit.plugins.bey import AvatarSession

    avatar = AvatarSession(api_key="test-key", api_url="http://bey.test", conn_options=_OPTS)
    avatar._http_session = session  # type: ignore[assignment]
    return avatar


async def test_client_error_is_not_retried_and_keeps_its_status():
    """A bad API key fails the same way every time; retrying it only delays the
    error and replaced the 401 with a generic 'after all retries' message."""
    session = _ScriptedSession([401])

    with pytest.raises(APIStatusError) as exc_info:
        await _avatar(session)._start_agent("wss://livekit.test", "token")

    assert session.posts == 1
    assert exc_info.value.status_code == 401


async def test_server_error_is_retried_until_success():
    session = _ScriptedSession([503, 503, 200])

    await _avatar(session)._start_agent("wss://livekit.test", "token")

    assert session.posts == 3


async def test_persistent_server_error_is_raised_after_max_retry_retries():
    """One initial attempt plus `max_retry` retries, like Anam, and the final error
    is the provider's status rather than a generic connection error."""
    session = _ScriptedSession([503])

    with pytest.raises(APIStatusError) as exc_info:
        await _avatar(session)._start_agent("wss://livekit.test", "token")

    assert session.posts == _OPTS.max_retry + 1
    assert exc_info.value.status_code == 503


async def test_network_error_is_retried_and_chained():
    session = _ScriptedSession([aiohttp.ClientConnectionError("connection refused")])

    with pytest.raises(APIConnectionError) as exc_info:
        await _avatar(session)._start_agent("wss://livekit.test", "token")

    assert session.posts == _OPTS.max_retry + 1
    assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectionError)


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(sample_rate=24000, num_channels=1, samples_per_channel=240)


async def _frames(n: int = 2):
    for _ in range(n):
        yield _audio_frame()


async def test_with_emotion_marks_every_frame_of_the_reply():
    from livekit.plugins import bey

    frames = [f async for f in bey.with_emotion(_frames(), "sad", intensity="high")]

    assert [f.userdata["bey.emotion"] for f in frames] == ["sad:high", "sad:high"]


@pytest.mark.parametrize("emotion", [None, "neutral"])
async def test_with_no_emotion_leaves_the_reply_neutral(emotion: str | None):
    from livekit.plugins import bey

    frames = [f async for f in bey.with_emotion(_frames(), emotion)]

    assert all("bey.emotion" not in f.userdata for f in frames)


@pytest.mark.parametrize(("emotion", "intensity"), [("angry", "medium"), ("happy", "extreme")])
def test_with_emotion_refuses_what_it_does_not_support(emotion: str, intensity: str):
    from livekit.plugins import bey

    with pytest.raises(ValueError):
        bey.with_emotion(_frames(), emotion, intensity=intensity)  # type: ignore[arg-type]


async def test_each_reply_stream_carries_its_emotion(
    monkeypatch: pytest.MonkeyPatch,
):
    from livekit.agents import utils
    from livekit.plugins import bey
    from livekit.plugins.bey.avatar import _EmotionAudioOutput

    room = MagicMock()
    room.isconnected.return_value = True
    room.local_participant._rpc_handlers = {}
    room.local_participant.stream_bytes = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(utils, "wait_for_participant", AsyncMock())
    output = _EmotionAudioOutput(room, destination_identity="bey-avatar-agent")
    await output._start_task()

    async def forward(emotion: bey.Emotion | None) -> None:
        async for frame in bey.with_emotion(_frames(1), emotion):
            await output.capture_frame(frame)
        output.flush()
        output.on_playback_finished(playback_position=0.01, interrupted=False)

    await asyncio.create_task(forward("happy"))
    await asyncio.create_task(forward(None))

    opened = [
        call.kwargs["attributes"] for call in room.local_participant.stream_bytes.await_args_list
    ]
    assert [a.get("bey.emotion") for a in opened] == ["happy:medium", None]
    # A sample rate would have LiveKit resample the audio into frames without the emotion
    assert output.sample_rate is None


async def test_reply_audio_reaches_the_avatar_whole_at_16_khz(
    monkeypatch: pytest.MonkeyPatch,
):
    from livekit.agents import utils
    from livekit.plugins.bey.avatar import _EmotionAudioOutput

    room = MagicMock()
    room.isconnected.return_value = True
    room.local_participant._rpc_handlers = {}
    writer = AsyncMock()
    room.local_participant.stream_bytes = AsyncMock(return_value=writer)
    monkeypatch.setattr(utils, "wait_for_participant", AsyncMock())
    output = _EmotionAudioOutput(room, destination_identity="bey-avatar-agent")
    await output._start_task()

    async def forward() -> None:
        # 50 ms at 24 kHz, of which the resampler holds the last samples until the flush
        async for frame in _frames(5):
            await output.capture_frame(frame)
        output.flush()

    await asyncio.create_task(forward())
    await asyncio.gather(*output._tasks)

    attributes = room.local_participant.stream_bytes.await_args.kwargs["attributes"]
    assert attributes["sample_rate"] == "16000"
    written = b"".join(call.args[0] for call in writer.write.await_args_list)
    assert len(written) // 2 == 800
    writer.aclose.assert_awaited_once()


def _agent(*, expressive: bool) -> MagicMock:
    agent = MagicMock()
    activity = agent._get_activity_or_raise.return_value
    activity._resolve_expressive_options.return_value = {} if expressive else None
    return agent


_HAPPY = '<expr type="expression" label="happy"/>'


@pytest.mark.parametrize(
    ("reply", "spoken", "emotion"),
    [
        ([_HAPPY[:20], _HAPPY[20:] + " That's ", "great!"], "That's great!", "happy:medium"),
        (['<expr type="expression" label="sad and quiet"/> Oh no.'], "Oh no.", "sad:medium"),
        (['<expr type="expression" label="neutral"/> Sure.'], "Sure.", None),
        (['<expr type="expression" label="calm"/> Sure.'], "Sure.", None),
        (["Sure, ", "let me check."], "Sure, let me check.", None),
        (["Hi! ", _HAPPY, " Great."], "Hi! Great.", None),
    ],
)
async def test_emotion_tts_node_speaks_the_reply_with_its_first_marker(
    monkeypatch: pytest.MonkeyPatch, reply: list[str], spoken: str, emotion: str | None
):
    from livekit.agents import Agent
    from livekit.plugins import bey

    heard: list[str] = []

    async def tts_node(agent, text, model_settings):
        async for chunk in text:
            heard.append(chunk)
        yield _audio_frame()

    async def text():
        for chunk in reply:
            yield chunk

    monkeypatch.setattr(Agent.default, "tts_node", tts_node)
    agent = _agent(expressive=False)
    frames = [f async for f in bey.emotion_tts_node(agent, text(), MagicMock())]

    assert "".join(heard).strip() == spoken
    assert [f.userdata.get("bey.emotion") for f in frames] == [emotion]


async def test_emotion_tts_node_passes_the_marker_to_an_expressive_tts(
    monkeypatch: pytest.MonkeyPatch,
):
    from livekit.agents import Agent
    from livekit.plugins import bey

    heard: list[str] = []

    async def tts_node(agent, text, model_settings):
        async for chunk in text:
            heard.append(chunk)
        yield _audio_frame()

    async def text():
        yield _HAPPY + " That's great!"

    monkeypatch.setattr(Agent.default, "tts_node", tts_node)
    agent = _agent(expressive=True)
    frames = [f async for f in bey.emotion_tts_node(agent, text(), MagicMock())]

    assert "".join(heard) == _HAPPY + " That's great!"
    assert [f.userdata.get("bey.emotion") for f in frames] == ["happy:medium"]
