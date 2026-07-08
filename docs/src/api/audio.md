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
t = transcribe("hello.mp3")
t isa TranscriptionSuccess && println(transcript_text(t))

# Translate foreign-language audio into English
translate("bonjour.mp3")
```
