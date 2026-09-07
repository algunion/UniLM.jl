# [Audio API](@id audio_api)

Synthesize speech from text, transcribe audio to text, and translate audio into
English. Text-to-speech returns raw audio bytes (`mp3`, `wav`, and friends);
transcription and translation upload an audio file and return text. OpenAI only.

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
```

Transcription defaults to `gpt-transcribe`. Its `languages` and `keywords` options
are vectors of strings sent as repeated multipart fields. A singular `language`
is translated to a one-element `languages` list for this model; older models keep
their singular field. Setting both forms raises `ArgumentError`.

OpenAI deprecated `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
`gpt-4o-transcribe-diarize`, and `whisper-1` on August 26, 2026, with shutdown
scheduled for February 26, 2027. Translation still defaults to `whisper-1` because
it remains the documented model for `/audio/translations`; `gpt-transcribe` is
for transcription. [OpenAI transcription guide](https://developers.openai.com/api/docs/guides/transcription),
[deprecation schedule](https://developers.openai.com/api/docs/deprecations).
