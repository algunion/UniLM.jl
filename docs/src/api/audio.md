# [Audio API](@id audio_api)

Synthesize speech from text, transcribe audio to text, and translate audio into
English. Text-to-speech returns raw audio bytes (`mp3`, `wav`, and friends);
transcription and translation upload an audio file and return text. Speech and
translation are OpenAI's; transcription runs on OpenAI and, with a local model that
hears audio, on Ollama ([Local Models with Ollama](@ref ollama_transcription)).

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
# Text-to-speech: synthesize and save to disk
result = speak("Hello from UniLM.", voice="alloy")
result isa SpeechSuccess && save_audio(result, "hello.mp3")

# Transcribe audio to text in its source language
t = transcribe("hello.mp3"; languages=["en"], keywords=["UniLM", "Julia"])
t isa TranscriptionSuccess && println(transcript_text(t))

# Translate foreign-language audio into English
translate("bonjour.mp3")

# Transcribe on your own machine: Ollama, with an installed model that hears audio
t = transcribe("memo.wav"; service=OllamaEndpoint(), model="gemma4:e4b")
```

Transcription takes the service's default model when `model` is left `nothing`:
`gpt-transcribe` on OpenAI. Its `languages` and `keywords` options
are vectors of strings sent as repeated multipart fields. A singular `language`
is translated to a one-element `languages` list for this model; older models keep
their singular field. Setting both forms raises `ArgumentError`.

Ollama has no default: name a model that hears audio, such as `gemma4:e4b`. Its
`/v1/audio/transcriptions` transcribes at temperature 0 and reads one `language`,
a `prompt` and a `response_format` of `"json"` or `"text"`; `temperature`,
`languages`, `keywords` and any other format throw `ArgumentError` before the
request. Ollama labels its replies `application/x-ndjson`, so they are read by the
format asked for. With `"text"`, a failure after Ollama has begun its reply comes
back as an empty transcript (by Ollama 0.40.6's source; untested here); JSON, the
default, reports it as an [`AudioCallError`](@ref). An endpoint declares
transcription with the capability `:transcription`, which `:audio` implies.

OpenAI deprecated `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
`gpt-4o-transcribe-diarize`, and `whisper-1` on August 26, 2026, with shutdown
scheduled for February 26, 2027. Translation still defaults to `whisper-1` because
it remains the documented model for `/audio/translations`; `gpt-transcribe` is
for transcription. [OpenAI transcription guide](https://developers.openai.com/api/docs/guides/transcription),
[deprecation schedule](https://developers.openai.com/api/docs/deprecations).
