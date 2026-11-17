# [Audio API](@id audio_api)

Synthesize speech from text, transcribe audio to text, and translate speech into
another language. [`speak`](@ref) runs on OpenAI's Realtime API, on Gemini TTS or on a
local Ollama speech model, and returns 24 kHz, 16-bit mono audio, as a WAV file or as raw
PCM. [`transcribe`](@ref) uploads an audio file to OpenAI, to Gemini Transcribe and, with a
local model that hears audio, to Ollama ([Audio](@ref ollama_transcription)), and can stream the
transcript from OpenAI. [`translate`](@ref) streams a
WAV file through an OpenAI Realtime translation session, or uploads an audio file to
Ollama, and returns the translated text.

## Request Types

```@docs
SpeechRequest
SpeechTurn
TranscriptionRequest
TranscriptionVAD
```

## Result Types

```@docs
SpeechSuccess
TranscriptionSuccess
TranscriptSegment
TranscriptWord
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

# Stream the transcript as it is made (a Task, as a streamed Chat)
t = fetch(transcribe("hello.wav"; stream=true, callback=(chunk, _) -> chunk isa String && print(chunk)))

# Transcribe with Gemini (gemini-3.5-transcribe), its speakers labelled
t = transcribe("call.wav"; service=GEMINIServiceEndpoint, diarization=true)
t isa TranscriptionSuccess && foreach(s -> println(s.speaker, ": ", s.text), t.segments)

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
| `OPENAIServiceEndpoint` | a Realtime API session | `gpt-realtime-2.1-mini` | `marin` | `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar`, or a custom voice's `voice_…` id, sent as `{"id": …}` | `0.25` to `1.5` |
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
`language` sets the BCP-47 tag it reads in (`speechConfig.languageCode`, such as `"fr-FR"`;
Gemini answered `"xx-ZZ"` with HTTP 400 "Invalid language_code", an [`AudioFailure`](@ref)).

A dialogue reads several voices in one request: `input` is a vector of [`SpeechTurn`](@ref)s and
`speakers` gives each speaker a voice, a prebuilt one or any other, as `voice` takes them:

```julia
r = speak([SpeechTurn("Joe", "How's it going today, Jane?"),
           SpeechTurn("Jane", "Not too bad, ready to test these voices?"; style="cheerful")];
          service=GEMINIServiceEndpoint, speakers=["Joe" => "Puck", "Jane" => "Kore"])
```

Each turn goes as a part of its own, its speaker and style in `speechMetadata`, and the speakers
as `multiSpeakerVoiceConfig`. Every turn's speaker must be among `speakers`; a dialogue takes no
`voice` (its speakers have theirs) and no `instructions` (each turn has its style), and a speaker
named twice or a blank one is refused before the request. Google's guide says this "supports up to
2 speakers using prebuilt voices"; three speakers, and a designed voice beside a prebuilt one, were
each spoken on 17 November 2026, so UniLM refuses neither. OpenAI's Realtime session and Ollama read
one voice and take no language: a dialogue, `speakers` or a `language` there throws `ArgumentError`.
The speech's format stays UniLM's: 24 kHz 16-bit mono, as WAV or PCM, whichever `response_format`
asks; Gemini's other output formats (MP3, Opus, A-law, μ-law, URI delivery) are not requested.

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
with `ffmpeg -i in.m4a -c:a pcm_s16le out.wav`. A `source_language` throws `ArgumentError`
before any I/O too: the session sets only the target language and has no field for the
spoken one
([translation client events](https://developers.openai.com/api/reference/resources/realtime/translation-client-events)).
A language the session does not translate into is an [`AudioFailure`](@ref) carrying the
provider's `error` event; the bounds and the cancellation are those of speech on OpenAI.
Translation replaces OpenAI's file-upload translation endpoint, which shuts down on February 26, 2027
([Realtime translation](https://developers.openai.com/api/docs/guides/realtime-translation)).

On [`OllamaEndpoint`](@ref) (Ollama 0.40.12 or later), `translate` uploads the file as it is
to Ollama's `/v1/audio/translations`, with the model, the target `language` as
`to_language`, the "ISO 639-1 code of the output language", and `source_language`, when given,
as Ollama's own `language`, "the spoken language, if known"
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)), and returns the
reply's `text`, its JSON in `raw`. Ollama decodes a WAV file itself and passes any other
container to the model whole. There is no default model: name an installed model that hears
audio, such as `gemma4:e4b`; a call without one, a blank `language` or `source_language` and a
missing or empty file throw `ArgumentError` before the request. A language Ollama does not
translate into or from is its 400, an [`AudioFailure`](@ref), and transient statuses are retried
as a transcription's are.

## Transcription

Transcription takes the service's default model when `model` is left `nothing`:
`gpt-transcribe` on OpenAI, `gemini-3.5-transcribe` on `GEMINIServiceEndpoint`. Its `languages` and `keywords` options
are vectors of strings sent as repeated multipart fields. A singular `language`
is translated to a one-element `languages` list for this model; older models keep
their singular field. Setting both forms raises `ArgumentError`.

OpenAI's further options ([create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create))
are typed fields of [`TranscriptionRequest`](@ref), each sent as OpenAI reads it:

| option | values | sent as |
|:--|:--|:--|
| `include` | `["logprobs"]` | `include[]`; the reply's `logprobs` stay in `raw` |
| `timestamp_granularities` | `"word"`, `"segment"`, with `response_format = "verbose_json"` | `timestamp_granularities[]` |
| `chunking_strategy` | `"auto"` or a [`TranscriptionVAD`](@ref) | one field: `auto`, or the server VAD settings as JSON |
| `known_speaker_names`, `known_speaker_references` | up to 4 pairs; a reference is a data URL or a path to a 2–10 s clip | `known_speaker_names[]`, `known_speaker_references[]` (data URLs) |
| `stream` | `true` | `stream`; the call streams (below) |

OpenAI reads `chunking_strategy` as one field: the VAD settings as JSON with a bogus type were
answered with HTTP 400, the bracketed fields a form gives other objects
(`chunking_strategy[type]`) were ignored with a 200 (17 November 2026). A request that asks for
what OpenAI would refuse or ignore throws `ArgumentError` before it is sent: an `include` item other
than `"logprobs"`, timestamps without `verbose_json` (with `json`, `gpt-transcribe` answered without
them), a stream on `whisper-1` (OpenAI: "Streaming is not supported for the whisper-1 model and will
be ignored"), names without a reference each, more than four speakers, a VAD threshold outside
`0 … 1`. Known speakers are for the diarization model: `gpt-transcribe` answered them with HTTP 400
"Known speaker references are only supported for diarization models", an [`AudioFailure`](@ref).

A reply's segments and words are typed: a `verbose_json` reply's segments (`id` an index) and words,
a `diarized_json` reply's segments, each with its `speaker`, become the result's
[`TranscriptSegment`](@ref)s and [`TranscriptWord`](@ref)s; everything else the reply holds stays in
`raw`. Transcription is not priced: [`token_usage`](@ref) and [`estimated_cost`](@ref) throw for a
[`TranscriptionSuccess`](@ref), and the usage the reply reports stays in `raw["usage"]`.

### Streaming

With `stream = true`, `transcribe` returns a `Task`, as a streamed `Chat` does, and hands the
transcript to `callback(chunk, close)` as OpenAI's events arrive: each `transcript.text.delta` as a
`String`, each `transcript.text.segment` of a streamed diarization as a [`TranscriptSegment`](@ref).
The task's result is the same [`TranscriptionSuccess`](@ref) as unstreamed, built from the
`transcript.text.done` event (its `text`, and in `raw` its `usage`, `logprobs` and `languages`), its
usage as unpriced as the unstreamed reply's.

```julia
task = transcribe("memo.wav"; stream=true, callback=(chunk, close) -> chunk isa String && print(chunk))
r = fetch(task)
r isa TranscriptionSuccess && println("\n", r.raw["usage"])
```

`close[] = true` stops the stream at once (an [`AudioCallError`](@ref) whose `cause` is a
[`UniLMCancelled`](@ref)); an exception in the callback ends the call with it as `cause`. A status
before the stream is an [`AudioFailure`](@ref) and a transient one is sent again while no callback
has run; an error inside the stream (`{"error": …}`, Ollama's form) is an `AudioFailure` with
`status = nothing`; a stream that ends before `transcript.text.done`, a byte gap longer than
`stream_idle_timeout`, a timeout and a cancel are an `AudioCallError`. Ollama streams the same events.

### Gemini

On `GEMINIServiceEndpoint`, `transcribe` sends the file inline to `gemini-3.5-transcribe` in one
`generateContent` request, its options in `generationConfig.audioTranscriptionConfig`
([audio transcription](https://ai.google.dev/gemini-api/docs/transcribe)):

| option | Gemini field |
|:--|:--|
| `languages` (or one `language`), BCP-47 codes such as `"en-US"` | `languageCodes` |
| `keywords`, up to 1,000 terms | `customVocabulary` |
| `diarization = true` | `diarization` |
| `timestamp_granularities = ["word"]` | `wordTimestamp` |
| `mode`, `"verbatim"` (default) or `"smart"` | `mode` |
| `source_text`, with `mode = "smart"` | `sourceText` |

```julia
r = transcribe("call.wav"; service=GEMINIServiceEndpoint, diarization=true, timestamp_granularities=["word"])
for s in r.segments
    println(s.speaker, " (", s.start_time, "–", s.end_time, " s): ", s.text)
end
```

Each part of the reply is a speaker turn: a [`TranscriptSegment`](@ref) with its `speaker`
(`"spk:0"`, `"spk:1"`, …), its start and end those of its first and last word; the words, with their
speaker, are the result's [`TranscriptWord`](@ref)s, and the `text` is the turns joined with spaces.
Google's Interactions API takes the same options as `transcription_config` and gave the same
transcript (17 November 2026); UniLM uses generateContent, which stores nothing and answers a turn
per part. Gemini rejects custom vocabulary beside diarization or word timestamps — it answered HTTP
400 "custom_vocabulary is incompatible with diarization." — and smart mode beside either, so those
throw `ArgumentError`, as do the options Gemini takes none of (`prompt`, `temperature`,
`response_format`, `include`, `chunking_strategy`, known speakers, `stream`, segment timestamps). A
reply whose candidate did not finish is an [`AudioFailure`](@ref) with `status` 200; transient
statuses are retried as speech's are. `diarization`, `mode` and `source_text` are Gemini's: an
OpenAI-wire service throws `ArgumentError` for them.

### Ollama, and OpenAI's deprecated models

Ollama has no default: name a model that hears audio, such as `gemma4:e4b`. Its
`/v1/audio/transcriptions` (Ollama 0.40.12 or later) reads one `language`, a `prompt`,
`temperature` (any finite number; its default 0) and a `response_format` of `json`, `text` or `verbose_json`
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)), and UniLM reads
the reply by its `Content-Type`: JSON, or the bare text for `"text"`. A reply labelled
otherwise is an [`AudioCallError`](@ref) naming the label: Ollama before 0.40.12 labelled
its JSON `application/x-ndjson`. `languages`, `keywords` and any other format (`srt`,
`vtt`, `diarized_json`) throw `ArgumentError` before the request, in Ollama's words, as do
OpenAI's further options but `stream`, which Ollama lists. The
live suite `test/integration_ollama.jl`, whose 109 tests include a transcription,
passed on Ollama 0.40.16 (16 November 2026, `gemma4:e4b-it-q4_K_M`). An endpoint declares transcription with the
capability `:transcription`, which `:audio` implies.

OpenAI deprecated `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
`gpt-4o-transcribe-diarize`, and `whisper-1` on August 26, 2026, with shutdown
scheduled for February 26, 2027. The `srt`, `vtt` and `verbose_json` formats
(subtitles and timestamps) are documented only for `whisper-1`, and `diarized_json`
(speaker labels) only for `gpt-4o-transcribe-diarize`.
[OpenAI transcription guide](https://developers.openai.com/api/docs/guides/transcription),
[deprecation schedule](https://developers.openai.com/api/docs/deprecations).

## [Custom voices](@id audio_voices)

One family of verbs makes, lists and removes the voices [`speak`](@ref) can use, on OpenAI and on
Gemini, each returning a [`Voice`](@ref) whose `id` is what `speak` takes as `voice` on that
service; an endpoint declares them with the capability `:voices`.

| verb | OpenAI (`/v1/audio/voices`) | Gemini (`/v1beta/voices`) |
|:--|:--|:--|
| [`create_voice`](@ref) | a prompt voice (`prompt`), or one from a recording (`audio_sample` and `consent`) | a designed voice (`prompt`), or a replicated one (`audio_sample` and `consent_audio`) |
| [`list_voices`](@ref) | none: `ArgumentError` | the project's voices, then the prebuilt catalog, filtered |
| [`get_voice`](@ref) | none: `ArgumentError` | a stored voice |
| [`delete_voice`](@ref) | yes | a stored voice |
| [`create_voice_consent`](@ref), [`list_voice_consents`](@ref), [`get_voice_consent`](@ref), [`update_voice_consent`](@ref), [`delete_voice_consent`](@ref) | the speaker's consent recordings | none: the consent recording goes with the replicated voice |

```julia
# Gemini: design a voice, read with it, delete it
v = create_voice(; service=GEMINIServiceEndpoint, name="Narrator", prompt="A calm, clear adult narrator.")
if v isa VoiceSuccess
    s = speak("The bridge reopens on Monday."; service=GEMINIServiceEndpoint, voice=v.response.id)
    delete_voice(v.response.id; service=GEMINIServiceEndpoint)
end

# The catalog: prebuilt voices in a language, a page at a time
page = list_voices(; service=GEMINIServiceEndpoint, type="prebuilt", language="en-GB", limit=20)
```

On Gemini a designed voice is stored under an id `voice_…` (200 per project, one year) and its
reply carries a WAV preview (`raw["sample_audio"]`) and its token usage, which UniLM does not price;
a replicated voice made with `store = false` is not stored and comes back as a `key`
(`voicekey_…`, seven days), which `speak` takes as its voice. The catalog's prebuilt voices list by
their ids (`"achernar"`, `"am-et-assistant-1"`, …), and `speak` took both kinds by id on 17 November
2026; `get_voice` reads stored voices only (a prebuilt id is a 404). A designed voice was read in a
dialogue beside a prebuilt one. Untested: no live run has replicated a voice (`audio_sample` with
`consent_audio`), made one with `store = false` or edited a `base_voice`; those requests are built from
Gemini's [voice replication](https://ai.google.dev/gemini-api/docs/voice-replication) and
[voice design](https://ai.google.dev/gemini-api/docs/voice-design) guides, and tested offline only,
against scripted replies.

On OpenAI a voice made from a `prompt` is supported "only in Live, not in Realtime or the speech
endpoint": the Realtime session `speak` runs refused one with "Prompt-based voices are not
supported by the Realtime API" (17 November 2026), so a prompt voice cannot be read by `speak`. A
voice from a recording needs the speaker's consent ([`create_voice_consent`](@ref), then `consent =
"cons_…"`), an access OpenAI limits "to eligible customers": an organization without it gets HTTP
404 "Your organization does not have access to this endpoint" from both, a [`VoiceFailure`](@ref).
Untested: the organization the live tests run under gets that 404, so no live run has made a voice from
a recording or seen a consent verb succeed; those success paths are built from OpenAI's reference and
tested offline only, against scripted replies. OpenAI's API lists and looks up no voices, so `list_voices` and `get_voice` on it throw
`ArgumentError` naming `GEMINIServiceEndpoint`. The Realtime API takes a custom voice as an object,
`{"id": "voice_…"}` (a plain string answered "Supported values are: 'alloy', …"), and so `speak`
on OpenAI sends a `voice_…` id
([custom voices](https://developers.openai.com/api/docs/guides/custom-voices)).

Each verb is one bounded attempt, never retried (a create that timed out may still have made the
voice), and takes `config` and `cancel` as the other platform verbs do; a status the service answers
is a [`VoiceFailure`](@ref), a reply that is not the documented object (a voice without its type, a
deletion record of another kind, a page without `has_more`) a [`VoiceCallError`](@ref).

```@docs
Voice
VoiceList
VoiceConsent
VoiceConsentList
VoiceSuccess
VoiceListSuccess
VoiceConsentSuccess
VoiceConsentListSuccess
VoiceDeleteSuccess
VoiceFailure
VoiceCallError
create_voice
list_voices
get_voice
delete_voice
create_voice_consent
list_voice_consents
get_voice_consent
update_voice_consent
delete_voice_consent
```
