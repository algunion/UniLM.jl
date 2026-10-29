# [Images, speech, embeddings & the compatible endpoint](@id gemini_media_page)

```@setup gemini
using UniLM
```

## [Image generation](@id gemini_images)

With `service=GEMINIServiceEndpoint`, [`generate_image`](@ref) and [`edit_image`](@ref) call
generateContent on an image model, `gemini-nano-banana-2.1` unless the request names another, with
image output (`responseModalities: ["IMAGE"]`). The request's `model` field stays `""` until the
request is sent, and the model goes in the URL. Size the image with two fields of Google's:

- `aspect_ratio`: `"1:1"`, `"1:4"`, `"4:1"`, `"1:8"`, `"8:1"`, `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`,
  `"4:5"`, `"5:4"`, `"9:16"`, `"16:9"` or `"21:9"`; unset, the model's own choice;
- `resolution`: `"512"`, `"1K"`, `"2K"` or `"4K"`, of which `gemini-nano-banana-2.1` takes `"1K"`,
  `"2K"` and `"4K"`, the "output resolutions (default `1K`)" of its model page
  ([Nano Banana 2.1](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1)).

They go in `generationConfig.imageConfig`, and `safety_identifier` in the request label
([Request options](@ref gemini_request_options)). A request returns one image, so `n` takes only 1,
and the image fields generateContent has no counterpart for — `size`, `quality`, `background`,
`output_format`, `output_compression`, `moderation`, and an edit's `mask` and `input_fidelity` — are
refused before the request ([Limits and refusals](@ref gemini_limits)). An image request is not recorded, so its
output here is quoted from a live run:

```julia
result = generate_image("A lighthouse on a rocky coast at dusk, oil painting";
                        service=GEMINIServiceEndpoint, aspect_ratio="16:9")
result isa ImageSuccess || error("the image request failed: $result")
save_image(image_data(result)[1], "lighthouse.jpg")
# Generated on 25 October 2026 with UniLM's development branch after 0.26.0 on gemini-nano-banana-2.1:
# "lighthouse.jpg"

parts = result.response.raw["candidates"][1]["content"]["parts"]
[p["inlineData"]["mimeType"] for p in parts if haskey(p, "inlineData")]
# 1-element Vector{String}:
#  "image/jpeg"

u = token_usage(result)
(; u.prompt_tokens, u.completion_tokens, u.reasoning_tokens)
# (prompt_tokens = 12, completion_tokens = 2224, reasoning_tokens = 793)
```

The reply's image parts are the result's images, in order, each as base64 in `b64_json`, which
[`image_data`](@ref) lists and [`save_image`](@ref) decodes into a file; their MIME type is in the
reply, `result.response.raw`. Text the model sends with an image is its `revised_prompt`, and the interim
images of the model's thinking are not results. A reply with no image — a filter stopped it, say — is
an [`ImageCallError`](@ref) naming the finish reason (`IMAGE_SAFETY`, `NO_IMAGE`, …), and a reply has
no creation time, so `result.response.created` is the time it arrived. "All generated images include a
SynthID watermark" ([image generation](https://ai.google.dev/gemini-api/docs/image-generation)).

An edit sends its input images inline, before the prompt, and decodes the same way:

```julia
edited = edit_image("lighthouse.jpg", "Make it a winter night with snow on the rocks";
                    service=GEMINIServiceEndpoint)
edited isa ImageSuccess || error("the edit failed: $edited")
length(image_data(edited)), edited.response.data[1].revised_prompt
# (1, nothing)

eu = token_usage(edited)
(; eu.prompt_tokens, eu.completion_tokens, eu.reasoning_tokens)
# (prompt_tokens = 1131, completion_tokens = 2325, reasoning_tokens = 858)
```

[`token_usage`](@ref) reads the reply's usage as a `Chat`'s ([First call](@ref gemini_first_call)),
its thinking counted as output; the image is output too — 1,120 of the generation's tokens above, by
the `candidatesTokensDetails` of its `raw` usage, as Google says "Output images at 1K (1024x1024px)
consume 1120 tokens" ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). UniLM does not price
images: [`estimated_cost`](@ref) throws `ArgumentError` for an image result unless you pass `model=`,
which it prices by a `pricing` row of your own, and estimates at `0.0`, with a warning, where no row
names it. The options every provider shares are on [Image Generation](@ref images_guide).

## [Audio](@id gemini_audio)

[`speak`](@ref) with `service=GEMINIServiceEndpoint` reads text aloud through Gemini TTS: one
generateContent request to `gemini-3.8-flash-tts`, unless `model` names another, in the voice `Kore`,
unless `voice` names another. Gemini 3.8 TTS takes "a prebuilt voice name, an Extended Voice Library
ID, a custom Voice design ID (`voice_...`), or a Voice replication ID (`voice_...`, or optional
stateless `voicekey_...`)" ([speech generation](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation)):
one of the 30 studio voices that page lists goes as a prebuilt voice, any other as the request's
`voice`. The Extended Voice Library's hundreds of voices are listed by `GET /v1beta/voices`, not by
UniLM, so a voice is sent as given, and Gemini answers a name it does not have.
UniLM asks for 24 kHz, 16-bit mono PCM and returns a [`SpeechSuccess`](@ref): a WAV file with
`response_format="wav"`, the default, which UniLM wraps in a 44-byte header, or the samples with
`"pcm"` (`content_type` `"audio/pcm"`). Speech is not recorded, so its output here is quoted from a
live run:

```julia
s = speak("The bridge reopens on Monday morning."; service=GEMINIServiceEndpoint,
          instructions="calm and clear, like a radio announcement")
s isa SpeechSuccess || error("speech failed: $s")
s.content_type, length(s.audio)
# Spoken on 25 October 2026 with UniLM's development branch after 0.26.0 on gemini-3.8-flash-tts:
# ("audio/wav", 142124)

(length(s.audio) - 44) / (24_000 * 2)   # seconds of 16-bit mono at 24 kHz, after the WAV header
# 2.96

save_audio(s, "bridge.wav")
```

"Gemini 3.8 TTS treats the `text` field strictly as a verbatim transcript", and the delivery goes in
the speech style, `speech_metadata.style`: "emotions, delivery style, prosody, pacing, and volume"
([speech generation](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation)).
`instructions` become that style, and the pace is part of it (`"speaking slowly"`): Gemini TTS has
no speed, so any `speed` is refused, as are a blank `voice`, a format other than `"wav"` and `"pcm"`
and an empty input ([Limits and refusals](@ref gemini_limits)).

A reply whose candidate did not finish with `STOP` — `MAX_TOKENS`, `SAFETY`, … — or that has no
candidate, a blocked prompt, is an [`AudioFailure`](@ref) with `status` 200 and the reply as
`response`; an HTTP error status is an `AudioFailure` too. A reply without audio, a timeout and a
cancel are an [`AudioCallError`](@ref). Speech is not priced: [`token_usage`](@ref) and
[`estimated_cost`](@ref) throw for a `SpeechSuccess`. `speak` refuses `GEMINIOpenAIServiceEndpoint`,
whose wire has no speech route.

[`transcribe`](@ref) refuses both endpoints: neither has the `transcription` capability. Google serves
transcription with a model of its own, `gemini-3.5-transcribe`
([transcribe](https://ai.google.dev/gemini-api/docs/transcribe)), which UniLM does not wrap. To get
the words of a clip, send it as an `AudioAttachment` and ask for them, as
[Attachments](@ref gemini_media) does. The verbs every provider shares are on [Audio API](@ref audio_api).

## [Embeddings](@id gemini_embeddings)

UniLM embeds text with Gemini through `GEMINIOpenAIServiceEndpoint`, on Google's OpenAI-compatible
`/v1beta/openai/embeddings`; `GEMINIServiceEndpoint` has no embeddings, and Google's native
embedding methods are not wrapped. The default model is `gemini-embedding-2`, which takes "Input
token limit 8,192" and an output dimension that is "Flexible, supports: 128 - 3072, Recommended: 768,
1536, 3072"
([Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2)); UniLM sends
`dimensions` unchecked.

```@example gemini
emb = Embeddings(["How long do refunds take?", "Is shipping free within the EU?"];
                 service=GEMINIOpenAIServiceEndpoint, task=:search_query, dimensions=768)
result = embeddingrequest!(emb)
length.(embedding_vectors(result)), result.usage
```

`gemini-embedding-2` reads a task from the text itself — "You cannot use the `task_type` field for
the `gemini-embedding-2` model. Instead, include the task as an instruction in your prompt"
([embeddings](https://ai.google.dev/gemini-api/docs/embeddings)) — so `task` writes Google's prompt in
front of each text: `"task: search result | query: "` for `:search_query`, `"title: … | text: "`
for `:document`, with an [`EmbeddingInput`](@ref)'s `title` (`none` without one), and the others as
[Embeddings](@ref embeddings_guide) lists them. A text without a task goes as it is. Google normalizes
every size on `gemini-embedding-2`: "While the default 3072-dimension embeddings are always normalized,
Gemini Embedding 2 also auto-normalizes truncated dimensions (e.g., 768, 1536)"; only on
`gemini-embedding-001` "you must manually normalize non-3072 dimensions"
([embeddings](https://ai.google.dev/gemini-api/docs/embeddings)).

The endpoint takes text: an `EmbeddingInput` with an image or a sound clip is refused before the
request ([Limits and refusals](@ref gemini_limits)). Its reply carries no `usage`, so `result.usage` is
`nothing` and [`estimated_cost`](@ref) gives `0.0`, with one warning per model, though Google lists
a "Text input price" for the model ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). The
reply leaves the index out of its first row, which UniLM reads as row 0. `gemini-embedding-001` is
served until its shutdown on "May 14, 2028", with `gemini-embedding-2` the replacement Google names
([deprecations](https://ai.google.dev/gemini-api/docs/deprecations)), and "The embedding spaces
between `gemini-embedding-001` and `gemini-embedding-2` are **incompatible**"
([embeddings](https://ai.google.dev/gemini-api/docs/embeddings)): re-embed your data to move.

## [The OpenAI-compatible endpoint](@id gemini_compatible)

`GEMINIOpenAIServiceEndpoint` speaks Google's OpenAI-compatible endpoint,
`https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`, with the key as
`Authorization: Bearer`. Google says its support "is still in beta while we extend feature support"
([OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)). A `Chat` reaches the same
models, with the default model of `GEMINIServiceEndpoint`:

```julia
chat = Chat(service=GEMINIOpenAIServiceEndpoint)
```

There UniLM is an OpenAI-compatible client, and of these pages only these rules hold:

- **Sampling and tiers.** `temperature` and `top_p` follow the rule of
  [Request options](@ref gemini_request_options), and `service_tier` takes `"standard"`, `"flex"` and
  `"priority"`, the tiers Google matches "in name and logic" there; another value is refused. The
  replies named no billed tier on 22 October 2026, so a result records the one its request asked for
  (a tier a reply names is kept): `"flex"` is priced at the Flex rates and `"priority"` at the Priority
  rates, an upper bound, since a downgraded Priority request bills at the standard rate and this wire
  did not say so.
- **Everything else is sent as set.** No other check runs: `reasoning_effort` goes as it is, Google
  mapping it to a thinking level, and `prompt_cache_options`, a request that ends with an assistant
  turn and the other refusals of [Limits and refusals](@ref gemini_limits) reach Google, which answered
  the first two with HTTP 400 on 22 October 2026, returned as an `LLMFailure`.
- **No thinking; signatures go back.** UniLM keeps no provider content from this wire:
  `reasoning_text` gives `nothing`. A function call's thought signature, which this wire gives in the
  call's `extra_content.google.thought_signature`, stays on its `ToolCall` (`thought_signature`) and
  goes back with the call, as Google requires ("Gemini 3 supports OpenAI compatibility for thought
  signatures in chat completion APIs"). A two-turn tool loop on `gemini-3.8-flash`, streamed and not,
  was answered on 29 October 2026; without the signature, its second request was answered HTTP 400.
- **Usage.** A stream asks for its usage, `stream_options = {"include_usage": true}`, when the `Chat`
  sets none, so a streamed turn is priced. The usage counted no audio apart on 23 October 2026, so a
  clip is priced at the text rate, a lower bound on the four models with an audio price of their own,
  for which such a request warns once per model.
- **No continuation, no other verbs.** A long reply is not continued ([Long generations](@ref gemini_long_decode)),
  though a `gemini-4-` model keeps its 20-minute default bounds; `respond`, `list_models`, `speak` and
  the image verbs refuse the endpoint, and `embeddingrequest!` is its own job
  ([Embeddings](@ref gemini_embeddings)).

Recorded on 25 October 2026 on `gemini-embedding-2`, through Google's OpenAI-compatible endpoint, with UniLM's
development branch after 0.26.0 and the Gemini API at `v1beta`.
