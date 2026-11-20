# Beyond Presence plugin for LiveKit Agents

Support for [Beyond Presence](https://docs.bey.dev) virtual avatars.

See [https://docs.livekit.io/agents/integrations/avatar/bey/](https://docs.livekit.io/agents/integrations/avatar/bey/) for more information.

## Installation

```bash
pip install livekit-plugins-bey
```

## Pre-requisites

Create a developer API key from the [creator dashboard](https://app.bey.chat) and set the `BEY_API_KEY` environment variable with it:

```bash
export BEY_API_KEY=<your-bey-api-key>
```

## Emotions

Genesis 3 avatars can speak each reply with an emotion: `happy` or `sad`, or `neutral` for their usual face (`bey.Emotion`). Other avatars ignore it.
The emotion holds for the whole reply and the avatar is neutral again on the next one, so the face never changes mid-sentence.

### Let the LLM choose

Override `tts_node` with `bey.emotion_tts_node` and add `bey.emotion_instructions()` to the prompt.
The LLM starts each reply with LiveKit's `<expr type="expression" label="..."/>` marker, and the avatar speaks the reply with the emotion the label names.

```python
from livekit.agents import Agent
from livekit.plugins import bey


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=f"{MY_INSTRUCTIONS}\n\n{bey.emotion_instructions()}")

    def tts_node(self, text, model_settings):
        return bey.emotion_tts_node(self, text, model_settings)
```

LiveKit keeps the marker out of the transcript.
With [expressive mode](https://docs.livekit.io/agents/models/tts/expressive/), pass `bey.emotion_instructions()` as `tts_instructions_append` instead, so one marker steers both the voice and the face:

```python
session = AgentSession(
    tts=inference.TTS("cartesia/sonic-3.6"),
    expressive={"tts_instructions_append": bey.emotion_instructions()},
)
```

### Choose it yourself

Wrap a reply's audio with `bey.with_emotion`, for example with a realtime model, which speaks without writing text first:

```python
class MyAgent(Agent):
    emotion: bey.Emotion | None = None

    async def on_user_turn_completed(self, turn_ctx, new_message):
        self.emotion = await my_classifier(new_message.text_content)  # "neutral", "happy" or "sad"

    def realtime_audio_output_node(self, audio, model_settings):
        return bey.with_emotion(audio, self.emotion)
```

`with_emotion` also takes an `intensity`: `"low"`, `"medium"` (the default) or `"high"`.
