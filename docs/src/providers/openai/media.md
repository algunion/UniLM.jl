# [Images, audio & realtime](@id openai_media_page)

```@setup openai
using UniLM
```

## [Image generation](@id openai_images)

[`generate_image`](@ref) sends an [`ImageGeneration`](@ref) to OpenAI's Images API,
`POST /v1/images/generations`, and [`edit_image`](@ref) an [`ImageEdit`](@ref) to
`POST /v1/images/edits`, as `multipart/form-data` with each image and the mask as a file. Both
take `gpt-image-2.5-sunburst` unless the request names another `model`; the request's `model`
field stays `""` until the request is serialized, so `JSON.json(ig)` shows the model that goes.
OpenAI's options are sent as given, for OpenAI to judge:

- **`size`**: `"1024x1024"`, `"1536x1024"`, `"1024x1536"` or `"auto"`, or a `WIDTHxHEIGHT` of
  the GPT Image 2 and 2.5 models, where "Width and height must be multiples of 16, the aspect
  ratio must be between 1:3 and 3:1, and neither edge may exceed 3840 pixels. The total pixel
  count must be between 655,360 and 8,294,400 (4K)"
  ([image generation](https://developers.openai.com/api/docs/guides/image-generation#size-and-quality-options)).
- **`quality`**: `"low"`, `"medium"`, `"high"` or `"auto"`, and `"xhigh"` and `"max"` on the 2.5
  models, which "default to `auto`"; "Use `quality: "low"` for quick drafts"
  ([image generation](https://developers.openai.com/api/docs/guides/image-generation#size-and-quality-options)).
- **`background`, `output_format`, `output_compression`**: `"transparent"`, `"opaque"` or
  `"auto"`, a transparent one with `"png"` or `"webp"`; the format `"png"`, the default,
  `"jpeg"` or `"webp"`; the compression from 0 to 100 of a JPEG or WebP, which "defaults to 100".
- **`n`**: how many images one request makes, "between 1 and 10", which [`image_data`](@ref)
  lists in order.
- **`moderation`**: `"auto"`, the default, or `"low"` "for less restrictive filtering": the
  filtering of the image models' prompts and images, a setting of the Images API and not the
  Moderations API ([Moderation](@ref openai_moderation)).
- **`safety_identifier`**: "A stable identifier for an end user in your organization", for which
  OpenAI asks you to "Hash a username or email address to avoid sending identifying information";
  it takes the place of the deprecated `user`, which the image requests do not have, and UniLM
  checks that it is 1 to 128 characters ([Limits and refusals](@ref openai_media_limits)).

The quoted words are from OpenAI's
[Images reference](https://developers.openai.com/api/reference/resources/images) unless the line
names the guide. An image request is not recorded, so its output here is quoted from a live run,
at the smallest size and the lowest quality:

```julia
result = generate_image("A lighthouse on a rocky coast at dusk, flat vector illustration";
                        size="1024x640", quality="low", output_format="jpeg")
result isa ImageSuccess || error("the image request failed: $result")
save_image(image_data(result)[1], "lighthouse.jpg")
# Generated on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-image-2.5-sunburst:
# "lighthouse.jpg"

u = token_usage(result)
(; u.prompt_tokens, u.completion_tokens)
# (prompt_tokens = 18, completion_tokens = 107)
```

![A lighthouse on a rocky coast at dusk, flat vector illustration](../../assets/openai_lighthouse.jpg)

OpenAI's image models "always return base64-encoded images"
([Images](https://developers.openai.com/api/reference/resources/images)), which [`image_data`](@ref) lists and
[`save_image`](@ref) decodes into a file, and the reply repeats the `size`, `quality`,
`output_format` and `background` it used, in `result.response.raw`. An edit sends the image as a
file, with the prompt:

```julia
edited = edit_image("lighthouse.jpg", "Make it a winter night with snow on the rocks";
                    size="1024x640", quality="low", output_format="jpeg")
edited isa ImageSuccess || error("the edit failed: $edited")
save_image(image_data(edited)[1], "lighthouse_winter.jpg")
eu = token_usage(edited)
(; eu.prompt_tokens, eu.completion_tokens)
# Edited on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-image-2.5-sunburst:
# (prompt_tokens = 666, completion_tokens = 107)
```

![The same lighthouse on a snowy winter night](../../assets/openai_lighthouse_winter.jpg)

`image` is a path or a vector of paths, "up to 16 images", and `mask` the path of an image whose
transparent area marks what to change: "The image to edit and mask must be of the same format and
size (less than 50MB in size)", and "The mask image must also contain an alpha channel"
([image generation](https://developers.openai.com/api/docs/guides/image-generation#mask-requirements)).
`input_fidelity` is sent as given; OpenAI's reference documents it for `gpt-image-1.5` and
`gpt-image-1-mini` and says "For `gpt-image-2`, omit this parameter". An image or mask path that is no
file is refused before anything is sent, as `upload_file` and `transcribe` refuse one
([Limits and refusals](@ref openai_media_limits)).

[`token_usage`](@ref) reads the reply's usage: the prompt's text tokens, and an edit's input image
too, as input — 640 image and 26 text tokens in the edit above, by the `input_tokens_details` of its
`response.usage` — and the image as output. UniLM does not price images:
[`estimated_cost`](@ref) throws `ArgumentError` for an `ImageSuccess` unless you pass `model=`, which
it prices by a `pricing` row of your own. OpenAI bills `gpt-image-2.5-sunburst` text input at \$5,
image input at \$8 and image output at \$30 per million tokens
([GPT-Image-2.5 Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst)): the
generation above came to about \$0.0033 and the edit to about \$0.0085.

A request is sent again after a transient status, as a `Chat` is
([When something fails](@ref openai_failures)), and a cancel of its `cancel` token, the ambient one
unless given, ends it with an `ImageCallError` whose `cause` is a [`UniLMCancelled`](@ref). A prompt or image the filters stop is an
[`ImageFailure`](@ref) whose `response` holds OpenAI's error, `type` `"image_generation_user_error"`
and `code` `"moderation_blocked"`, which OpenAI asks not to send again as it is: "Don't
automatically retry these errors without modifying the prompt or input images"
([image generation](https://developers.openai.com/api/docs/guides/image-generation#handling-blocked-requests-and-other-errors)).
`image_data` of a failure throws an [`LLMResultError`](@ref). The `gpt-image-1-mini`,
`gpt-image-1.5` and `chatgpt-image-latest` models shut down on 1 December 2026, with
`gpt-image-2.5-sunburst` or `gpt-image-2.5-flare` the replacements
([deprecations](https://developers.openai.com/api/docs/deprecations#2026-06-02-gpt-image-model-deprecations)).
A `Respond` makes images with the [`image_generation_tool`](@ref) ([Hosted tools](@ref openai_hosted_tools)),
and the verbs every provider shares are on [Image Generation](@ref images_guide); every field is in
the [Image Generation API](@ref images_api) reference.

## [Audio](@id openai_audio)

[`speak`](@ref) reads text aloud through a session of OpenAI's Realtime API, not the Speech API:
OpenAI's `tts-1`, `tts-1-hd` and `gpt-4o-mini-tts` "are deprecated and will shut down on
**January 6, 2027**", and their replacement, `gpt-realtime-2.1-mini`, "isn't a supported model for
`/v1/audio/speech`" ([text to speech](https://developers.openai.com/api/docs/guides/text-to-speech)).
UniLM opens `wss://api.openai.com/v1/realtime` on `gpt-realtime-2.1-mini` unless `model` names
another, sets the voice, 24 kHz 16-bit mono PCM output and instructions to read the text word for
word, sends the text as a user message, asks for a response and collects its audio; `instructions`
follow its own as delivery directions. Realtime models are conversational, and a session's
instructions "are not guaranteed to be followed by the model"
([Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events),
`instructions`).

- **`voice`**: `"marin"`, the default, or one of OpenAI's other built-in voices, `"alloy"`, `"ash"`,
  `"ballad"`, `"coral"`, `"echo"`, `"sage"`, `"shimmer"`, `"verse"` and `"cedar"`: "We recommend
  `marin` and `cedar` for best quality". A custom voice, which a session takes by its id, is
  refused ([Limits and refusals](@ref openai_media_limits)).
- **`speed`**: from 0.25 to 1.5 — "1.0 is the default speed" —, "a post-processing adjustment to
  the audio after it is generated".
- **`response_format`**: `"wav"`, the default, a WAV file with a 44-byte header, or `"pcm"`, the
  samples alone (`content_type` `"audio/pcm"`).

The quoted words are from the session's `audio.output` in OpenAI's
[Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events).

Speech is not recorded, so its output here is quoted from a live run:

```julia
s = speak("The Harbor Club ferry leaves pier 7 at nine fifteen.";
          instructions="calm and clear, like a station announcement")
s isa SpeechSuccess || error("speech failed: $s")
s.content_type, length(s.audio)
# Spoken on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-realtime-2.1-mini:
# ("audio/wav", 175244)

(length(s.audio) - 44) / (24_000 * 2)   # seconds of 16-bit mono at 24 kHz, after the WAV header
# 3.65

save_audio(s, "ferry.wav")
```

[`transcribe`](@ref) uploads an audio file to `POST /v1/audio/transcriptions`, on
`gpt-transcribe` unless `model` names another. "Files can be up to 25 MB. Supported input formats
are `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, and `webm`"
([file transcription](https://developers.openai.com/api/docs/guides/speech-to-text)). `prompt`,
`keywords` and `languages` give the model context — "Literal terms that may appear in the audio"
and "A list of expected input languages" — and `gpt-transcribe` uses "`languages` instead of the
singular `language` field" ([transcription](https://developers.openai.com/api/docs/guides/transcription#improve-transcription-quality)):
UniLM sends a singular `language` as a one-element `languages` to a `gpt-transcribe` model, and
as `language` to the others.

```julia
t = transcribe("ferry.wav"; keywords=["Harbor Club"], languages=["en"])
t isa TranscriptionSuccess || error("transcription failed: $t")
transcript_text(t)
# Transcribed on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-transcribe:
# "The Harbor Club ferry leaves Pier 7 at 9:15."

t.raw
# Dict{String, Any} with 3 entries:
#   "usage"     => Dict{String, Any}("seconds"=>4, "type"=>"duration")
#   "text"      => "The Harbor Club ferry leaves Pier 7 at 9:15."
#   "languages" => Any[Dict{String, Any}("code"=>"en")]
```

[`transcript_text`](@ref) reads the text, and `raw` keeps the JSON reply: the languages
`gpt-transcribe` detected — "When the model can't make a reliable language prediction, it returns
`"languages": []`" ([file transcription](https://developers.openai.com/api/docs/guides/speech-to-text)) —
and the usage, here a duration. A `response_format` of `"text"`, `"srt"` or `"vtt"` returns plain
text, which `raw` does not hold; subtitles, timestamps and speaker labels come from models OpenAI
deprecated, which "will shut down on **February 26, 2027**" with `/v1/audio/translations`
([transcription](https://developers.openai.com/api/docs/guides/transcription#existing-integrations-with-deprecated-models)).
The default model is `gpt-transcribe` on OpenAI, and on an endpoint of your own that declares
`:audio` or `:transcription`; an endpoint that declares its capabilities without either cannot
transcribe: `transcribe` throws `ArgumentError` before any request
([Limits and refusals](@ref openai_media_limits)), as it does for one that declares none
([Platform APIs](@ref openai_platform_shared)).

[`translate`](@ref) streams the speech of a WAV file through a Realtime translation session,
`wss://api.openai.com/v1/realtime/translations` on `gpt-realtime-translate`, into `language`
(English unless named), and returns the translated transcript as a
[`TranscriptionSuccess`](@ref); the translated speech the session also sends is not kept. OpenAI's
translation sessions take "base64-encoded 24 kHz PCM16 audio" over WebSockets
([Realtime translation](https://developers.openai.com/api/docs/guides/realtime-translation)), so
UniLM reads a 16-bit PCM WAV file of any rate and channel count, mixes it down to one channel,
resamples it by linear interpolation, with no anti-aliasing filter, and sends it in 200 ms frames;
any other file is refused ([Limits and refusals](@ref openai_media_limits)).

```julia
tr = translate("ferry.wav"; language="fr")
tr isa TranscriptionSuccess || error("translation failed: $tr")
transcript_text(tr)
# Translated on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-realtime-translate:
# "La navette du Harbor Club quitte le quai 7 à 9 h 15."
```

Speech already in the target language can come back as empty text, and a short clip's translation
varies from call to call, as the [`translate`](@ref) reference reports.

What `speak` and `translate` return when they fail:

- **`AudioFailure`** for what OpenAI reports: an HTTP error status, a refused WebSocket upgrade
  included (a bad key is a 401, `response` then HTTP.jl's report of the status); or, inside a
  session, an `error` event or a `response.done` that did not complete, with `status` `nothing` and
  the event's JSON as `response`, as for a language the session does not translate into.
- **`AudioCallError`** for a session that closed before its result, with the close's status and
  reason in `error` and the close as `cause`, a timeout, a cancel, and a reply without usable audio.
  Audio received before a failure is never returned.

Each makes one attempt. `total_deadline` bounds the whole call whatever the socket does,
`connect_timeout` the WebSocket's open, and `stream_idle_timeout` every send and every wait for the
next event. Both observe the ambient [`CancelToken`](@ref): a token already cancelled sends
nothing, and a cancel ends the call at once with an `AudioCallError` whose `cause` is a
[`UniLMCancelled`](@ref). `transcribe` is one HTTP request, sent again after a transient status, as
a `Chat` is, the whole file each time ([Timeouts & Retries](@ref timeouts_retries)); its results are
typed as a platform verb's are ([Platform APIs](@ref openai_platform_shared)).

UniLM prices none of them: [`token_usage`](@ref) and [`estimated_cost`](@ref) throw for a
`SpeechSuccess` and a `TranscriptionSuccess`. OpenAI bills `gpt-transcribe` at "\$0.0045 / minute",
`gpt-realtime-translate` at "\$0.034 / minute", and a speech session at the Realtime model's token
rates ([pricing](https://developers.openai.com/api/docs/pricing#transcription-and-speech)). A `Chat`
on an audio model answers in speech too ([Audio replies](@ref chat_audio_output)), and every
type is in the [Audio API](@ref audio_api) reference.

## [Realtime sessions](@id timeout_realtime)

[`realtime_connect`](@ref)`(handler; model="gpt-realtime-2.1")` opens a WebSocket to OpenAI's
Realtime API, `wss://api.openai.com/v1/realtime`, and runs `handler(session)` once it is open.
Inside the handler, [`realtime_send`](@ref) sends a client event, a `Dict` that
[`session_update`](@ref), [`input_audio_append`](@ref), [`response_create`](@ref) and
[`realtime_event`](@ref) build, and [`realtime_receive`](@ref) returns the next server event as a
`Dict`. The socket closes when the handler returns, and `realtime_connect` returns what it
returned. Audio travels as base64 PCM inside the events: UniLM covers the WebSocket transport, not
WebRTC or SIP, and not GPT-Live's sessions at `wss://api.openai.com/v1/live/sessions`
([WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets)). A session is not
recorded, so its output here is quoted from a live run:

```julia
events, reply = String[], IOBuffer()
realtime_connect(config=RequestConfig(connect_timeout=10.0, stream_idle_timeout=30.0)) do session
    realtime_send(session, session_update(Dict("type" => "realtime", "output_modalities" => ["text"],
                                               "instructions" => "Answer in one short sentence.")))
    realtime_send(session, realtime_event("conversation.item.create";
        item=Dict("type" => "message", "role" => "user",
                  "content" => [Dict("type" => "input_text", "text" => "Which is the largest moon of Saturn?")])))
    realtime_send(session, response_create())
    while true
        event = realtime_receive(session)
        push!(events, event["type"])
        event["type"] == "response.output_text.delta" && print(reply, event["delta"])
        event["type"] in ("response.done", "error") && break
    end
    session.model
end
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# "gpt-realtime-2.1"

String(take!(reply))
# "Saturn’s largest moon is Titan."

length(events), events[1], events[2], events[end]
# (21, "session.created", "session.updated", "response.done")
```

The events and the session's fields are OpenAI's
([Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events)):
`session.type` `"realtime"`, `output_modalities`, and the output audio under `audio.output`.
UniLM does not read a session's usage: [`estimated_cost`](@ref) has nothing to price, and OpenAI
bills the session at the model's Realtime rates
([pricing](https://developers.openai.com/api/docs/pricing)). A session is bounded in two phases,
and deliberately not in a third:

- **Opening** is bounded by `connect_timeout`. A peer that accepts the TCP connection but never
  finishes the upgrade throws `UniLMTimeout(:connect, …)`; an open that fails for another reason,
  such as an upgrade answered with 401, throws that error, even when it fails just before the
  bound; an upgrade that completes only after the timeout is closed without running the handler.
- **`realtime_receive`** is bounded by the `stream_idle_timeout` of the config the session took at
  connect time, and a breach throws `UniLMTimeout(:stream_idle, …)`. A parked socket read cannot be
  polled out of, so the bound ends the read and the connection, without a close handshake and
  without HTTP.jl's send lock: the breach surfaces just after the limit on `ws://`, and on `wss://`
  up to about 5 s later, the time the TLS close may wait to send its close notice, even while a
  `realtime_send` on another task is parked on a peer that stopped reading (that send throws too);
  the session is closed afterwards. On a WebSocket opened over your own stream
  (`HTTP.WebSockets.open(io)`), the stream stays yours: the breach ends the read, and a parked send
  waits until you close the stream. `Inf` disables the bound.
- **The session's lifetime** is not bounded: once the handler runs, how long it stays connected is
  yours to decide, since a Realtime session is meant to sit idle waiting for input.

Unlike the other verbs, `realtime_connect` throws rather than returning a typed result, makes a
single attempt, which `max_attempts` does not change, and refuses another `service` before any
I/O ([Limits and refusals](@ref openai_media_limits)). A `realtime_connect` session — its open and
`realtime_receive` — does not observe the [`CancelToken`](@ref); `speak` and `translate` run
Realtime sessions of their own, and those do observe it ([Audio](@ref openai_audio)).

[`mint_realtime_secret`](@ref) makes a client secret, `POST /v1/realtime/client_secrets`: "Client
secrets are short-lived tokens that can be passed to a client app, such as a web frontend or mobile
client, which grants access to the Realtime API without leaking your main API key"
([Create client secret](https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create)).
`session` is an optional session configuration; UniLM sends no `expires_after`, so a secret lasts
OpenAI's default, "600 seconds (10 minutes)":

```julia
secret = mint_realtime_secret()
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# RealtimeSecretSuccess(value="ek_6…[redacted]", raw=<3 keys>)
```

The verb makes one bounded attempt and returns a
[`RealtimeSecretSuccess`](@ref), whose `value` is the secret — shown redacted, so a log or the REPL
does not hand it out —, a [`RealtimeFailure`](@ref) with the status of an error and its `request_id`, or a
[`RealtimeCallError`](@ref), also for a 200 that carries no secret. Every type is in the
[Realtime API](@ref realtime_api) reference.

## [Limits and refusals](@id openai_media_limits)

OpenAI's limits on a request's media — an edit's 50 MB, a transcription's 25 MB — are OpenAI's to
check: UniLM checks neither the size nor the format of a file, except that `translate` reads its
WAV file. These are refused with an `ArgumentError` before anything is sent:

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| `aspect_ratio` or `resolution` on an image request | `the OpenAI Images API has no`: OpenAI sizes an image with one `WIDTHxHEIGHT`; those fields are Gemini's | `size="WIDTHxHEIGHT"` |
| a `safety_identifier` of no character or of more than 128 on an image request | `the OpenAI Images API takes a safety_identifier of 1 to 128 characters`: OpenAI reads an empty one as none, and its guide keeps it to 128 | a hash of your user's id: a SHA-256 hex digest is 64 characters |
| `edit_image` with an empty vector of images | `an image edit needs at least one input image` | a path, or a vector of paths |
| `edit_image` with an image or a `mask` path that is no file | `image not found:`, `mask not found:` | the path of the file |
| `speak` with empty or blank text | `speak needs text to read: input is empty` | the text to read |
| a `voice` that is not one of the ten built-in voices, a custom voice among them | `OpenAI Realtime voices are`: UniLM sends the built-in voices only | one of them |
| a `speak` `response_format` other than `"wav"` and `"pcm"` | `response_format must be "wav"`: the session's audio is 24 kHz PCM, which UniLM returns as a WAV file or as samples | `"wav"` or `"pcm"` |
| a `speed` outside 0.25 to 1.5 | `OpenAI Realtime speed must be between 0.25 and 1.5`: the session's range | a speed in the range |
| `transcribe` or `translate` with a path that is no file | `file not found:` | the path of the file |
| a transcription with `language` and `languages` both set | `Set language or languages, not both`: OpenAI asks for one of the two | one of them |
| a transcription keyword with `<`, `>` or a line break | `Transcription keywords cannot contain <, >, or line breaks`: OpenAI "rejects the entire request when it encounters one of these characters" ([file transcription](https://developers.openai.com/api/docs/guides/speech-to-text)) | the keyword without them |
| a transcription `response_format` other than `"json"`, `"text"`, `"srt"`, `"verbose_json"`, `"vtt"` and `"diarized_json"` | `response_format must be one of`: the formats of OpenAI's transcriptions | one of them |
| a transcription `temperature` outside 0 to 1 | `temperature must be between 0 and 1` | a value in the range |
| `transcribe` on an endpoint that declares no capabilities, or declares them without `:audio` or `:transcription`: of the endpoints UniLM ships, every one but `OPENAIServiceEndpoint`, `OllamaEndpoint` and `GenericOpenAIEndpoint` (`MistralEndpoint()` included), among them `AZUREServiceEndpoint`, `GEMINIOpenAIServiceEndpoint` and `DeepSeekEndpoint` | `Audio API is not supported by`: the endpoint has no transcription route | `OPENAIServiceEndpoint` |
| `translate` with a blank `language` | `translate needs a target language` | a language code, such as `"fr"` |
| `translate` with a file that is not 16-bit PCM WAV, or that holds no samples | `cannot translate`, `holds no audio samples`: a translation session takes 24 kHz PCM16, which UniLM makes from a 16-bit PCM WAV file only | a 16-bit PCM WAV file, as `ffmpeg -i in.m4a -c:a pcm_s16le out.wav` makes |
| `realtime_connect`, `translate` or `speak` on another endpoint that declares its capabilities (`speak` but on `GEMINIServiceEndpoint`, which speaks through Gemini TTS), and `mint_realtime_secret` on one that declares them without `:realtime` | `is not supported by`, `supports only OPENAIServiceEndpoint`, `has no speech route`: the Realtime WebSocket is served by `api.openai.com`, and another endpoint's key must not be sent there | `OPENAIServiceEndpoint` |
