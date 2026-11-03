# [Audio API](@id audio_api)

Synthesize speech from text, transcribe audio to text, and translate speech into
another language. [`speak`](@ref) runs on OpenAI's Realtime API, on Gemini TTS or on a
local Ollama speech model, and returns 24 kHz, 16-bit mono audio, as a WAV file or as raw
PCM. [`transcribe`](@ref) uploads an audio file to OpenAI and, with a local model that
hears audio, to Ollama ([Audio](@ref ollama_transcription)). [`translate`](@ref) streams a
WAV file through an OpenAI Realtime translation session, or uploads an audio file to
Ollama, and returns the translated text.

## Request Types

```@docs
SpeechRequest
TranscriptionRequest
```

## Result Types

```@docs
SpeechSuccess
TranscriptionSuccess
AudioFailure
AudioCallError
```

## Functions

```@docs
speak
transcribe
translate
```

## Accessors

```@docs
transcript_text
save_audio
```

## Usage

```julia
# Speech from OpenAI (gpt-realtime-2.1-mini, voice "marin"), saved as a WAV file
r = speak("Hello from UniLM.")
r isa SpeechSuccess && save_audio(r, "hello.wav")

# Speech from Gemini TTS (gemini-3.8-flash-tts), another voice and a delivery, as raw PCM
r = speak("Hello from UniLM."; service=GEMINIServiceEndpoint, voice="Puck",
          instructions="warm and enthusiastic", response_format="pcm")

# Transcribe audio to text in its source language
t = transcribe("hello.wav"; languages=["en"], keywords=["UniLM", "Julia"])
t isa TranscriptionSuccess && println(transcript_text(t))

# Translate speech into French text (into English when no language is named)
t = translate("bonjour.wav"; language="fr")

# On your own machine: Ollama, with installed models (ollama pull gemma4:e4b phonon)
t = transcribe("memo.wav"; service=OllamaEndpoint(), model="gemma4:e4b", response_format="text")
r = speak("Hello from UniLM."; service=OllamaEndpoint(), model="phonon", voice="Zoey")
t = translate("memo.wav"; language="fr", service=OllamaEndpoint(), model="gemma4:e4b")
```

## Speech

| `service` | speaks through | default `model` | default `voice` | voices | `speed` |
|:--|:--|:--|:--|:--|:--|
| `OPENAIServiceEndpoint` | a Realtime API session | `gpt-realtime-2.1-mini` | `marin` | `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar` | `0.25` to `1.5` |
| `GEMINIServiceEndpoint` | Gemini TTS (`generateContent`) | `gemini-3.8-flash-tts` | `Kore` | a prebuilt voice (the 30 studio voices or one of the Extended Voice Library), a designed or replicated voice's `voice_…` id, or a replicated voice's `voicekey_…` key, sent as given | none |
| [`OllamaEndpoint`](@ref) | Ollama's `/v1/audio/speech` (Ollama 0.40.9 or later) | none: name an installed speech model, such as `phonon` | the model's own (`voice` is not sent) | one of the model's voices, which [`model_info`](@ref) lists as `voices`, sent as given | `1` only |

Every service produces 16-bit mono PCM at 24 kHz: `response_format = "wav"` (the
default) returns it as a WAV file — on OpenAI and Gemini UniLM writes a 44-byte header, and
Ollama's header, streamed before the length is known, holds the maximum size — and `"pcm"`
returns the samples (`content_type` `"audio/pcm"`). Any other format, a voice the Realtime
session does not have (`nova`, `onyx` and `fable` are not Realtime voices), a blank Gemini or
Ollama voice, a speed out of range, any `speed` on Gemini, and an empty `input` throw
`ArgumentError` before any request, as does `GEMINIOpenAIServiceEndpoint`: Google's
OpenAI-compatible API has no speech route, so the error points to `GEMINIServiceEndpoint`.

On OpenAI, `speak` opens a Realtime WebSocket session, sets the voice, PCM output and
instructions to read the text word for word, adds the text as a user message and
collects the response's audio. Realtime models are conversational, and OpenAI notes
that a model is not guaranteed to follow its instructions. `instructions` add delivery
directions (tone, pace, accent) after UniLM's own. OpenAI's `tts-1`, `tts-1-hd` and
`gpt-4o-mini-tts` shut down on January 6, 2027, and `gpt-realtime-2.1-mini` on the
Realtime API is the replacement OpenAI names
([text to speech](https://developers.openai.com/api/docs/guides/text-to-speech),
[Realtime voices](https://developers.openai.com/api/reference/resources/realtime/client-events)).

On Gemini, `speak` sends one `generateContent` request with `responseModalities`
`["AUDIO"]`, the voice in `speechConfig` and 24 kHz L16 output requested. Gemini 3.8 TTS
reads the text verbatim; `instructions` become the text's speech style, which is also
where the pace goes (`"speaking slowly"`): Gemini TTS has no speed parameter
([speech generation](https://ai.google.dev/gemini-api/docs/speech-generation)).

On Ollama, `speak` sends one POST to `/v1/audio/speech` with the model, the text, the
format, and the voice and speed when set ([speech](https://docs.ollama.com/capabilities/speech));
there is no default model, so name an installed speech model such as `phonon`. Ollama answers
with HTTP 400 what its model does not support, so UniLM refuses, before sending it, a request
without a model, an `input` of more than 4,096 characters, a `speed` other than 1, non-empty
`instructions` and a format other than `"wav"` and `"pcm"`. A voice is each model's
own and is sent as given: Ollama answers one the model does not have with a 400
[`AudioFailure`](@ref) that names the voices it has. The reply's bytes come back as Ollama
streamed them, and a reply under another `Content-Type` is an [`AudioCallError`](@ref);
transient statuses are retried as on Gemini.

An error the provider reports is an [`AudioFailure`](@ref): an HTTP error status (a
WebSocket upgrade the server refuses included: a bad key is a 401), a Gemini reply whose
candidate did not finish (a `finishReason` other than `STOP`, such as `MAX_TOKENS` or
`SAFETY`) or that has no candidate (a blocked prompt), with `status` 200 and the reply as
`response`, or, on a Realtime session, the `error` event or a `response.done` that did not
complete, with `status = nothing` and the event's JSON as `response`. An `error` event the
server sends before it closes the session is that failure even when the close interrupts
UniLM's sending. A session that closes before its response is done (its close status and
reason in `error`, the close itself in `cause`), a timeout, a cancel and a reply without
usable audio are an [`AudioCallError`](@ref). Audio received before a failure is never returned. On
OpenAI, `total_deadline` bounds the whole call whatever the socket does, `connect_timeout`
the WebSocket open, and `stream_idle_timeout` every send and every wait for the next
event. Every service takes `cancel`, a [`CancelToken`](@ref), the ambient one unless
given: a token already cancelled sends nothing, and a cancel ends the call at once, with
an `AudioCallError` whose `cause` is a [`UniLMCancelled`](@ref). Speech is not priced:
[`token_usage`](@ref) and [`estimated_cost`](@ref) throw for a [`SpeechSuccess`](@ref).

## Translation

`translate(path; language="en")` on OpenAI (`model` `gpt-realtime-translate` unless named)
opens a Realtime translation session (`wss://api.openai.com/v1/realtime/translations`), sets the
target language (`audio.output.language`), streams the audio in 200 ms frames, closes
the session and joins the translated transcript into the text of a
[`TranscriptionSuccess`](@ref). The session also returns translated speech, which
`translate` does not keep. Speech already in the target language can come back as empty
text: an English clip translated into English did. A short clip's translation also varies
from call to call: of ten translations into English of a two-second French sentence made
with [`speak`](@ref), one came back empty and six named words the sentence does not have
(October 2026). A translation session takes 24 kHz
mono 16-bit PCM only, so `path` must be a 16-bit PCM WAV file, at any sample rate and
channel count: UniLM mixes the channels down and resamples by linear interpolation,
which suits speech (there is no anti-aliasing filter). Any other file throws
`ArgumentError` before any I/O, naming the conversion; convert it first, for example
with `ffmpeg -i in.m4a -c:a pcm_s16le out.wav`. A language the session does not
translate into is an [`AudioFailure`](@ref) carrying the provider's `error` event; the
bounds and the cancellation are those of speech on OpenAI. Translation replaces
OpenAI's file-upload translation endpoint, which shuts down on February 26, 2027
([Realtime translation](https://developers.openai.com/api/docs/guides/realtime-translation)).

On [`OllamaEndpoint`](@ref) (Ollama 0.40.12 or later), `translate` uploads the file as it is
to Ollama's `/v1/audio/translations`, with the model and the target `language` as
`to_language`, the "ISO 639-1 code of the output language"
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)), and returns the
reply's `text`, its JSON in `raw`. Ollama decodes a WAV file itself and passes any other
container to the model whole. There is no default model: name an installed model that hears
audio, such as `gemma4:e4b`; a call without one, a blank `language` and a missing or empty file
throw `ArgumentError` before the request. A language Ollama does not translate into is its 400,
an [`AudioFailure`](@ref), and transient statuses are retried as a transcription's are.

## Transcription

Transcription takes the service's default model when `model` is left `nothing`:
`gpt-transcribe` on OpenAI. Its `languages` and `keywords` options
are vectors of strings sent as repeated multipart fields. A singular `language`
is translated to a one-element `languages` list for this model; older models keep
their singular field. Setting both forms raises `ArgumentError`.

Ollama has no default: name a model that hears audio, such as `gemma4:e4b`. Its
`/v1/audio/transcriptions` (Ollama 0.40.12 or later) reads one `language`, a `prompt`,
`temperature` (any finite number; its default 0) and a `response_format` of `json`, `text` or `verbose_json`
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)), and UniLM reads
the reply by its `Content-Type`: JSON, or the bare text for `"text"`. A reply labelled
otherwise is an [`AudioCallError`](@ref) naming the label: Ollama before 0.40.12 labelled
its JSON `application/x-ndjson`. `languages`, `keywords` and any other format (`srt`,
`vtt`, `diarized_json`) throw `ArgumentError` before the request, in Ollama's words. The
live suite `test/integration_ollama.jl`, whose 91 tests include a transcription,
passed unchanged on Ollama 0.40.13 (2 November 2026). An endpoint declares transcription with the
capability `:transcription`, which `:audio` implies.

OpenAI deprecated `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
`gpt-4o-transcribe-diarize`, and `whisper-1` on August 26, 2026, with shutdown
scheduled for February 26, 2027. The `srt`, `vtt` and `verbose_json` formats
(subtitles and timestamps) are documented only for `whisper-1`, and `diarized_json`
(speaker labels) only for `gpt-4o-transcribe-diarize`.
[OpenAI transcription guide](https://developers.openai.com/api/docs/guides/transcription),
[deprecation schedule](https://developers.openai.com/api/docs/deprecations).
