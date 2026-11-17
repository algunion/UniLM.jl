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
([Request options](@ref gemini_request_options)). Two more go in `generationConfig.responseFormat.image`,
Gemini's "Configuration for image output format":

- `output_format`: `"jpeg"`, its `mimeType` `IMAGE_JPEG`, the one format Gemini's API reference lists;
  another is refused before the request. Gemini answers JPEG unasked, and on 17 November 2026 it answered
  `IMAGE_PNG`, a value its reference does not list, with a JPEG too;
- `delivery`: `"inline"` or `"uri"`, "The delivery mode for the image output": the image in the reply, or
  a URI to fetch it from, which becomes the image's `url`. On 17 November 2026 `gemini-nano-banana-2.1`
  answered `"uri"` with the image inline, and the result carries it as base64 as it does any image.

`responseFormat.image` takes an aspect ratio and a size too, the controls that `aspect_ratio` and
`resolution` set, and those stay in `imageConfig`: one name for each control. On 17 November 2026 Gemini
took the two objects side by side.

A request returns one image, so `n` takes only 1, and the image fields generateContent has no counterpart
for — `size`, `quality`, `background`, `output_compression`, `moderation`, OpenAI's GPT-Image-2.6
controls (`complexity`, `frame_count`, `frame_duration_ms` and `svg`), `stream` and `partial_images`,
and an edit's `mask`, `markup` and `input_fidelity` — are refused before the request
([Limits and refusals](@ref gemini_limits)). Gemini has no partial images to stream: on 17 November 2026
its `streamGenerateContent` answered an image request with one event that held the whole image. An image
request is not recorded, so its output here is quoted from a live run:

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
[`image_data`](@ref) lists and [`save_image`](@ref) decodes into a file — or, for a part Gemini delivers
by URI (`fileData`), as its `url`; their MIME type is in the reply, `result.response.raw`. Text the model sends with an image is its `revised_prompt`, and the interim
images of the model's thinking are not results. A reply with no image — a filter stopped it, say — is
an [`ImageCallError`](@ref) naming the finish reason (`IMAGE_SAFETY`, `NO_IMAGE`, …), and a reply has
no creation time, so `result.response.created` is the time it arrived. "All generated images include a
SynthID watermark" ([image generation](https://ai.google.dev/gemini-api/docs/image-generation)).

An edit sends its input images inline, before the prompt, and decodes the same way; it reads them from
files, so an [`ImageRef`](@ref), OpenAI's image by reference, is refused:

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
consume 1120 tokens" ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). [`image_usage`](@ref)
gives the same counts with those apart, as on OpenAI: the input's text and image tokens, by modality
(`promptTokensDetails`), and the output's image tokens and the rest, its text and thinking, which Google
prices apart ("\$30.00 (images)", "\$7.50 (text and thinking)" per million output tokens for
`gemini-nano-banana-2.1`, 17 November 2026). UniLM does not price
images: [`estimated_cost`](@ref) throws `ArgumentError` for an image result unless you pass `model=`,
which it prices by a `pricing` row of your own, and estimates at `0.0`, with a warning, where no row
names it. The options every provider shares are on [Image Generation](@ref images_guide).

## [Audio](@id gemini_audio)

[`speak`](@ref) with `service=GEMINIServiceEndpoint` reads text aloud through Gemini TTS: a
generateContent request to `gemini-3.8-flash-tts`, unless `model` names another, in the voice `Kore`,
unless `voice` names another, sent again after a transient status or a transport failure, as a `Chat`
is ([Timeouts & Retries](@ref timeouts_retries)). Gemini 3.8 TTS takes "a prebuilt voice name, an
Extended Voice Library ID, a custom Voice design ID (`voice_...`), or a Voice replication ID
(`voice_...`, or optional stateless `voicekey_...`)"
([speech generation](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation)):
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

UniLM embeds text with Gemini on both endpoints. `GEMINIServiceEndpoint` sends Google's own embedding
methods ([embeddings reference](https://ai.google.dev/api/embeddings)): `models/{model}:embedContent` for
one input, `models/{model}:batchEmbedContents` for a vector of them, up to 100 — the API answered 101 with
HTTP 400 "at most 100 requests can be in one batch" on 17 November 2026, and UniLM refuses more before the
request. `GEMINIOpenAIServiceEndpoint` sends Google's OpenAI-compatible `/v1beta/openai/embeddings`. The
default model of both is `gemini-embedding-2`, which takes "Input token limit 8,192" and an output
dimension that is "Flexible, supports: 128 - 3072, Recommended: 768, 1536, 3072"
([Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2)); UniLM sends
`dimensions` unchecked (the native API answered 4000 with HTTP 400, naming the range it takes, 1 to 3072).

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

The same request on `GEMINIServiceEndpoint` goes to `batchEmbedContents`, one request per text, each
naming its model:

```julia
emb = Embeddings(["How long do refunds take?", "Is shipping free within the EU?"];
                 service=GEMINIServiceEndpoint, task=:search_query, dimensions=768)
result = embeddingrequest!(emb)
length.(embedding_vectors(result)), result.usage.prompt_tokens
```

On `GEMINIServiceEndpoint`, `gemini-embedding-2` reads its task from the text too: a `taskType` or a
`title` sent beside it left its vector unchanged on 17 November 2026, so UniLM sends neither and writes the
prompt. `gemini-embedding-001` takes the task as its `taskType` instead ("For `gemini-embedding-001`, you
can specify the `task_type` in the `embedContent` method"), with no prompt in the text:

| `task` | `taskType` |
| :--- | :--- |
| `:search_query` | `RETRIEVAL_QUERY` |
| `:document` | `RETRIEVAL_DOCUMENT`, with an `EmbeddingInput`'s `title` as `title` |
| `:question_answering` | `QUESTION_ANSWERING` |
| `:fact_checking` | `FACT_VERIFICATION` |
| `:code_retrieval` | `CODE_RETRIEVAL_QUERY` |
| `:classification` | `CLASSIFICATION` |
| `:clustering` | `CLUSTERING` |
| `:sentence_similarity` | `SEMANTIC_SIMILARITY` |
| `:none`, or none | no task type |

A title beside any other task is refused before the request: the API answered one with HTTP 400 "the model
does not support the title parameter unless the task_type is RETRIEVAL_DOCUMENT". The OpenAI-compatible
endpoint has no task field, so there `gemini-embedding-001` takes `task = :none` or none. UniLM sends
`taskType`, `title` and `outputDimensionality` at the request's top level, as the REST examples of Google's
embeddings guide do, though the API's discovery document marks them deprecated in favour of
`embedContentConfig`: on 17 November 2026 `gemini-embedding-001` ignored all three inside
`embedContentConfig` (its vector came back with 3072 dimensions, unchanged by the task) and honoured them at
the top, and `gemini-embedding-2` honoured `outputDimensionality` in both places.
`embedContentConfig`'s other options are not sent: `autoTruncate` would "silently truncate the input
content if it's longer than the maximum sequence length", and UniLM returns no vector of part of an input;
`documentOcr` and `audioTrackExtraction` apply to documents and video, which UniLM does not embed. An
`Embeddings`' `user` has no field there and is refused.

The native reply carries `usageMetadata` for `gemini-embedding-2`, whose `promptTokenCount` is the result's
`usage`, priced at the model's row as on the compatible endpoint; `gemini-embedding-001`'s replies carried
none on 17 November 2026, so its result is estimated at `0.0`, with one warning per model. A reply without
the vectors it owes is an [`EmbeddingCallError`](@ref), never zeros. Many requests at half price go in a
batch ([Batches of embeddings](@ref gemini_embedding_batches)).

Both endpoints take text: an `EmbeddingInput` with an image or a sound clip is refused before the
request ([Limits and refusals](@ref gemini_limits)), though `gemini-embedding-2` embeds images, audio,
video and PDFs, which UniLM does not send it. The compatible endpoint's reply carries no `usage`, so
`result.usage` is `nothing` and [`estimated_cost`](@ref) gives `0.0`, with one warning per model, though
Google lists a "Text input price" for the model ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). That
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
- **Thinking levels.** `reasoning_effort` is checked against the model's row of the
  [thinking table](@ref gemini_reasoning), as on generateContent, and an effort the row lacks is refused
  before the request is sent. Google's compatibility page maps `"minimal"` to `low` on Gemini 4, but the
  endpoint answered `"minimal"` on `gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-argon` and
  `gemini-3.8-flash` with HTTP 400 "Thinking level MINIMAL is not supported for this model", and took
  `"none"` on `gemini-3.8-flash` and thought anyway (16 November 2026). A `gemini-2.5-` model takes a
  thinking budget there, which `"none"` turns off, so its effort goes as set, as does an effort for an
  id no row names.
- **A context cache.** `cached_content` goes as `extra_body.google.cached_content`, the field Google's
  compatibility page names, under the rules of [Context caching](@ref gemini_caching): a system message,
  `tools` or a `tool_choice` beside it, and a Gemini 4 model, are refused before the request (this wire
  answered the first two with the same HTTP 400 as generateContent on 17 November 2026). A reply that
  read the cache reports it in `prompt_tokens_details.cached_tokens`, read as `cached_tokens`.
- **Everything else is sent as set.** No other check runs: `prompt_cache_options`, a request that ends
  with an assistant turn and the other refusals of [Limits and refusals](@ref gemini_limits) reach
  Google, which answered the first two with HTTP 400 on 22 October 2026, returned as an `LLMFailure`.
- **No thinking; signatures go back.** UniLM keeps no provider content from this wire:
  `reasoning_text` gives `nothing`. A function call's thought signature, which this wire gives in the
  call's `extra_content.google.thought_signature`, stays on its `ToolCall` (`thought_signature`) and
  goes back with the call, as Google requires ("Gemini 3 supports OpenAI compatibility for thought
  signatures in chat completion APIs"). A two-turn tool loop on `gemini-3.8-flash`, streamed and not,
  was answered on 29 October 2026; without the signature, its second request was answered HTTP 400.
- **Usage.** A stream asks for its usage, `stream_options = {"include_usage": true}`, when the `Chat`
  sets none, so a streamed turn is priced. The endpoint counts the thinking only in `total_tokens`
  (`gemini-4-hydrogen` at `"xhigh"`: 38 prompt, 5 completion and 124 total tokens, on 16 November 2026),
  so the tokens a total holds beyond the prompt and the completion join `completion_tokens` and
  `reasoning_tokens` and bill at the output rate, as generateContent's thoughts do; a reply that names
  its reasoning tokens keeps that count, and a total below the prompt and the completion is kept as
  reported, with a warning. The usage counted no audio apart on 23 October 2026, so a
  clip is priced at the text rate, a lower bound on the four models with an audio price of their own,
  for which such a request warns once per model.
- **No continuation, no other verbs.** A long reply is not continued ([Long generations](@ref gemini_long_decode)),
  though a `gemini-4-` model keeps its 20-minute default bounds; `respond`, `list_models`, `speak` and
  the image verbs refuse the endpoint, and `embeddingrequest!` is its own job
  ([Embeddings](@ref gemini_embeddings)).

Recorded on 25 October 2026 on `gemini-embedding-2`, through Google's OpenAI-compatible endpoint, with UniLM's
development branch after 0.26.0 and the Gemini API at `v1beta`.
