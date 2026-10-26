# [Audio API](@id audio_api)

Synthesize speech from text, transcribe audio to text, and translate speech into
another language. [`speak`](@ref) runs on OpenAI's Realtime API or on Gemini TTS and
returns 24 kHz, 16-bit mono audio, as a WAV file or as raw PCM. [`transcribe`](@ref)
uploads an audio file to OpenAI and, with a local model that hears audio, to Ollama
([Audio](@ref ollama_transcription)). [`translate`](@ref) streams a
WAV file through an OpenAI Realtime translation session and returns the translated
text.

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

# Transcribe on your own machine: Ollama, with an installed model that hears audio
t = transcribe("memo.wav"; service=OllamaEndpoint(), model="gemma4:e4b")
```

## Speech

| `service` | speaks through | default `model` | default `voice` | voices | `speed` |
|:--|:--|:--|:--|:--|:--|
| `OPENAIServiceEndpoint` | a Realtime API session | `gpt-realtime-2.1-mini` | `marin` | `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar` | `0.25` to `1.5` |
| `GEMINIServiceEndpoint` | Gemini TTS (`generateContent`) | `gemini-3.8-flash-tts` | `Kore` | the 30 prebuilt voices | none |

Both services produce 16-bit mono PCM at 24 kHz: `response_format = "wav"` (the
default) wraps it in a 44-byte WAV header, `"pcm"` returns the samples
(`content_type` `"audio/pcm"`). Any other format, a voice the service does not have
(`nova`, `onyx` and `fable` are not Realtime voices), a speed out of range, any
`speed` on Gemini, and an empty `input` throw `ArgumentError` before any request, as does
`GEMINIOpenAIServiceEndpoint`: Google's OpenAI-compatible API has no speech route, so the
error points to `GEMINIServiceEndpoint`.

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
event. Both services observe the ambient [`CancelToken`](@ref): a token already
cancelled sends nothing, and a cancel ends the call at once, with an `AudioCallError`
whose `cause` is a [`UniLMCancelled`](@ref). Speech is not priced: [`token_usage`](@ref)
and [`estimated_cost`](@ref) throw for a [`SpeechSuccess`](@ref).

## Translation

`translate(path; language="en", model="gpt-realtime-translate")` opens a Realtime
translation session (`wss://api.openai.com/v1/realtime/translations`), sets the
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

## Transcription

Transcription takes the service's default model when `model` is left `nothing`:
`gpt-transcribe` on OpenAI. Its `languages` and `keywords` options
are vectors of strings sent as repeated multipart fields. A singular `language`
is translated to a one-element `languages` list for this model; older models keep
their singular field. Setting both forms raises `ArgumentError`.

Ollama has no default: name a model that hears audio, such as `gemma4:e4b`. Its
`/v1/audio/transcriptions` transcribes at temperature 0 and reads one `language`
and a `prompt`, and UniLM asks it for JSON only, the one form in which Ollama
reports a failure that arrives after its reply has begun, as an
[`AudioCallError`](@ref) (by Ollama 0.40.6's source; untested here).
`response_format = "text"` throws `ArgumentError` before the request — such a
failure would read as an empty transcript — and so do `temperature`, `languages`,
`keywords` and any other format: leave `response_format` unset and read the text
with [`transcript_text`](@ref). Ollama labels its replies `application/x-ndjson`, so
they are read as JSON whatever the label. An endpoint declares transcription with
the capability `:transcription`, which `:audio` implies.

OpenAI deprecated `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
`gpt-4o-transcribe-diarize`, and `whisper-1` on August 26, 2026, with shutdown
scheduled for February 26, 2027. The `srt`, `vtt` and `verbose_json` formats
(subtitles and timestamps) are documented only for `whisper-1`, and `diarized_json`
(speaker labels) only for `gpt-4o-transcribe-diarize`.
[OpenAI transcription guide](https://developers.openai.com/api/docs/guides/transcription),
[deprecation schedule](https://developers.openai.com/api/docs/deprecations).
