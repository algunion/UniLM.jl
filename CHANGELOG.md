# Changelog

## 0.28.0

### Breaking
- **A `Chat` with `prompt_cache_options` `mode="explicit"` is refused when it is built.** Explicit
  mode caches only the prefixes a request marks with `prompt_cache_breakpoint`, and a `Chat`'s
  message content, plain text, marks none: "If there are no explicit breakpoints, the request does
  not use prompt caching" (OpenAI's Chat Completions reference). Such a `Chat` was sent on every
  OpenAI-wire endpoint, and its prompt caching silently turned off. Its constructor now throws an
  `ArgumentError` naming why, whatever the endpoint; Anthropic's encoder, which refused it with its
  own message, no longer meets it. A `Respond`, whose `input_text(text; cache_breakpoint=true)`
  marks a breakpoint, keeps explicit mode. Upgrading: use `mode="implicit"` or leave `mode` unset,
  or send a `Respond` that marks its breakpoints.
- **`add_container_file` and `edit_image` refuse a path that is no file before sending, as `upload_file`,
  `transcribe` and `translate` do.** A missing local file threw an `ArgumentError` (`file not found:`)
  from those three before any request, but `add_container_file` returned a `ContainerCallError`, and
  `edit_image` an `ImageCallError` for a missing image or mask. Both now throw the `ArgumentError`
  (`file not found:`; `image not found:`, `mask not found:`) before any request, on Gemini's edits too.
  Upgrading: code that checked for that call error checks the path first, or catches the
  `ArgumentError`.
- **An endpoint that declares no capabilities is refused by name, by every platform verb, `transcribe`
  and the image verbs.** Such an endpoint, an endpoint type of your own with no `provider_capabilities`
  method, got a `MethodError` for that method from the files, uploads, vector stores, conversations,
  containers, batch, fine-tuning, moderation, realtime and audio verbs, `fim_complete` and
  `prefix_complete`, while `generate_image` and `edit_image` sent it on, unvalidated. Each now throws
  the `ArgumentError` that `list_models` and `ask` throw for it, "… is not supported by YourEndpoint:
  it declares no capabilities", before any request. `chatrequest!`, `embeddingrequest!` and `respond`
  still send it unvalidated. Upgrading: an endpoint type of your own that generates or edits images
  declares `:images` or `:image_edits` in `provider_capabilities`.
- **A file whose size is not yet known decodes: `FileObject.bytes` is `Union{Int,Nothing}`.** OpenAI's
  Files reference gives a file's `bytes` as null "In a completed file upload response, … when the file
  size is not yet available". Such a file object did not decode, so `complete_upload` returned no
  `UploadSuccess` for a completed upload, and `retrieve_file` and `upload_file` none for such a file.
  It now decodes with `bytes` `nothing`. Upgrading: code that reads a `FileObject`'s `bytes` as an
  `Int` handles `nothing`, a size the service has not given yet.
- **`:speech` names `speak`'s capability, and Ollama and generic endpoints declare `:agentic` and
  `:audio`.** No capability symbol said where `speak` goes, so `has_capability` was false for Gemini's
  native endpoint whatever the symbol, while Gemini TTS answered `speak`; and `OllamaEndpoint` and
  `GenericOpenAIEndpoint` (`MistralEndpoint()` among them) sent `respond` and `transcribe` without
  declaring `:agentic` and `:audio`, the symbols the capability table gives those verbs.
  `OPENAIServiceEndpoint` and `GEMINIServiceEndpoint` now declare `:speech`, which `speak` checks where
  it checked `:realtime`, and `OllamaEndpoint` and `GenericOpenAIEndpoint` declare `:agentic` and
  `:audio` too; speech and translation stay refused on both. Upgrading: code that compares a
  `provider_capabilities` set with a fixed set of its own adds them, or tests one capability with
  `has_capability`; an endpoint type of your own that `speak` reaches through a Realtime session
  declares `:speech` beside `:realtime`.
- **Gemini Interactions refuses an input that ends with an assistant message before sending.** From
  Gemini 3.6 Flash and 3.5 Flash-Lite on, Google answers an input whose last step is the model's with
  HTTP 400 ("Request contains an invalid argument.", gemini-3.8-flash, 26 October 2026), as
  generateContent answers a request that ends with a model turn, which a `Chat` already refuses before
  sending. `respond` on `GEMINIServiceEndpoint` sent such an input and returned the failure; it now
  throws an `ArgumentError` before sending, an input already in the Interactions shape included. The
  earlier models, which continue a prefilled answer, are unchanged. Upgrading: end the input with a
  user message, and shape the answer with `instructions` or `text`.

### Added
- **One item of a conversation can be read: `retrieve_conversation_item`.** `ConversationItemSuccess`
  was exported and documented as a Conversations result, yet no verb returned it: UniLM had no call for
  OpenAI's `GET /v1/conversations/{conversation_id}/items/{item_id}`, which gets "a single item from a
  conversation with the given IDs". `retrieve_conversation_item(conversation_id, item_id; include)`
  sends it, each `include` value as an `include[]` query parameter, and returns that
  `ConversationItemSuccess`.

### Fixed
- **A `Chat`'s log probabilities and moderation results reach its result.** A `Chat` with
  `logprobs=true` (and `top_logprobs`) or a `ModerationConfig` sent them and the request was billed,
  but `LLMSuccess` had no field for what came back, so the reply's log probabilities (the choice's
  `logprobs`) and its moderation results (the completion's `moderation`) were dropped. They are now
  the result's `logprobs` and `moderation`, each the reply's object as it came: `"content"` lists the
  output tokens, with `"refusal"` or DeepSeek's `"reasoning_content"` where the reply gives them, and
  `"input"` and `"output"` hold the moderation results; a streamed reply's chunks are joined, and a
  `prefix_complete` result keeps them too. On an `OllamaEndpoint`, `logprobs` and `top_logprobs` were
  refused; Ollama's `/api/chat` takes both, so they are now sent, and the reply's token list is
  `logprobs["content"]`. Native Gemini, Anthropic and Gemini Interactions still refuse them, UniLM
  mapping no such field there, and Ollama still refuses `moderation`. `LLMSuccess` gains the two
  fields last; built positionally, its five earlier fields still build one.
- **`temperature` and `top_p` can be set together.** The `Chat` and `Respond` constructors refused
  the pair on every provider, as "mutually exclusive", though no provider's page refuses it:
  OpenAI's, DeepSeek's and Mistral's references recommend altering one "but not both", and Azure's,
  Google's and Ollama's state no rule. Both are now kept, and sent where the endpoint takes sampling
  parameters. Anthropic still refuses each of them (its API version 2026-10-21 takes neither), and
  native Gemini and `GEMINIOpenAIServiceEndpoint` still take them only on the models released before
  Gemini 3.6 Flash.
- **A tool loop keeps every turn's result, so a `respond` loop's earlier turns can be priced.**
  `tool_loop` on a `Respond` sent a request per turn, each billed, and returned only the last
  response, so the usage of the turns before it was unreachable, while `tool_loop!` adds every turn of
  a `Chat` to its `cumulative_cost`. `ToolLoopResult` gains `responses`, every result the loop
  received, in order, `response` being its last: `sum(estimated_cost, loop.responses)` prices them
  all. The results are kept rather than one usage summed over the turns because a provider prices
  each request on its own, a long-context rate applying per request. Built positionally, the five
  earlier fields still build one, whose `responses` holds its `response`.
- **A `RealtimeFailure` keeps the request id the service sent.** `mint_realtime_secret` returned an
  error status as a `RealtimeFailure` holding only `response` and `status`, while every other platform
  `*Failure` — and the `RealtimeCallError` of the same verb — keeps the reply's `x-request-id`
  (`request-id`) header as `request_id`, the id a support request names. `RealtimeFailure` now has
  that field; built positionally, `response` and `status` still build one.
- **The System One seam's docstrings say which checks a protocol of your own gets.**
  `decode_system_one` promised a success "whose answers have passed the checks every backend's
  answers pass", but only the built-in decoders run them, through the internal
  `UniLM._check_answers`; a protocol of your own returned whatever its decoder built.
  `decode_system_one`, `encode_system_one` and `ask` now state that each built-in encoder and
  decoder calls the shared checks (`_check_request`, `_check_box_image`, `_check_answers`) and that
  a protocol of your own gets them only by calling them. No behaviour changes.
- **`url_citations` reads a Gemini interaction's citations.** Google puts a grounded answer's sources on
  its text as `annotations`, `url_citation` among them, and the message UniLM built from an interaction's
  `model_output` step kept the text alone, streamed or not, so `url_citations` found none. The answer's
  `output_text` part now keeps every annotation Google sent, and `url_citations` returns its
  `url_citation`s. Google counts `start_index` and `end_index` in bytes of the text, so an annotation of a
  part joined after other text is moved by that text's bytes; the steps an unstreamed interaction keeps in
  `response.raw` stay as Google sent them.
- **A Gemini `Chat` that sets `reasoning_effort` returns its thought summaries.** `reasoning_text` named
  Gemini's thought parts, but a native `Chat` never asked for them, so it gave `nothing`. Such a `Chat` now
  sends `includeThoughts: true` beside its thinking level, as Google's thinking page asks: the reply's
  summaries, parts marked `thought`, stay out of the answer's text and go back with the next request like
  the reply's other parts, and `reasoning_text` reads them, streamed or not. They add nothing to the bill:
  Google prices the thinking, "despite only the summary being output from the API". A `Chat` without
  `reasoning_effort` sends no thinking config, as before. A recording of such a request made before keys on
  the old bytes: record it again.
- **A Gemini call that read audio from the cache is priced.** On the four models whose pricing lists
  audio apart — `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-flash` and
  `gemini-2.5-flash-lite` — a call with audio that also hit the implicit cache was estimated at `0.0`,
  with a warning, since its usage did not say how many cached tokens were audio. `TokenUsage` gains
  `cached_audio_tokens`, the audio among the cached tokens, read from generateContent's
  `cacheTokensDetails` and an interaction's `cached_tokens_by_modality`, streamed or not (0 for every
  other provider, none of which reports it). Such a call now bills that audio at the model's
  cached-audio price — \$0.05, \$0.10, \$0.10 and \$0.03 per 1M tokens, at the billed tier's multiple —
  and its other cached tokens at the cached rate. A usage that reports more cached tokens than text
  tokens and cached audio together is still unpriced, with the warning.
- **The reasoning of a Mistral reply and of another generic server reaches `reasoning_text`, streamed or
  not.** An unstreamed `GenericOpenAIEndpoint` reply whose content is a list of chunks, as Mistral's
  reasoning models answer, kept its `text` chunks and dropped its `thinking` chunks, which a streamed reply
  keeps; and the `reasoning_content` field llama-server returns the reasoning in, or vLLM's `reasoning`,
  was dropped, streamed or not. `reasoning_text` gave `nothing` for each. The thinking chunks are now kept
  as `ProviderContent(:mistral, …)` and the reasoning field as `ProviderContent(:generic, …)`, whose text
  `reasoning_text` reads; neither goes back with the next request, and the replies of every other
  endpoint are read as before.

## 0.27.1

### Added
- **`gpt-6.1-terra` is priced and ruled.** OpenAI lists GPT-6.1 Terra, "a fast, efficient model for
  visual reasoning, precise computer use, and clear, natural writing" (its model page, 29 October
  2026). Its row in `DEFAULT_PRICING` holds the page's standard rates — \$1.00 input, \$0.05 cached
  input and \$5.00 output per 1M tokens —, with GPT-6's cache-write and long-context rates, Flex at
  half and Fast at twice; it serves no Ultrafast. The native endpoint checks its requests as the page
  and OpenAI's GPT-6 guide state: reasoning efforts `"none"` to `"max"` (Chat Completions without
  `"max"`), sampling controls and log probabilities only at `"none"`, and `prompt_cache_retention`
  refused as on the other GPT-6 models. Chat Completions function tools work at `"none"` only: the
  model page says "Tool calling requires the Responses API", but on 29 October 2026 the API took
  them at `"none"` and refused them at any other effort, as on GPT-6 Sol and Luna.

### Fixed
- **A tool loop on `GEMINIOpenAIServiceEndpoint` works, streamed or not.** Gemini 3 returns
  a function call's thought signature on the OpenAI-compatible wire, in the call's
  `extra_content.google.thought_signature`, and requires it back with the call. UniLM dropped
  it, so the request carrying the tool's result was answered HTTP 400 ("Function call is
  missing a thought_signature in functionCall parts"); streamed, the call itself was dropped,
  as this wire sends each call whole, without the `index` an OpenAI stream gives every call
  fragment. The call now keeps its signature on its `ToolCall` (`thought_signature`), streamed
  or not, and the endpoint sends it back with the call; no other endpoint sends it.

## 0.27.0

### Breaking
- **`DeepSeekEndpoint()` and `MistralEndpoint()` read their key when a request is sent.**
  Building one without `DEEPSEEK_API_KEY` or `MISTRAL_API_KEY` set no longer throws a
  `KeyError`: the variable is read each time a request is built, so a key exported after
  the endpoint was built is used, and a missing key fails the request with a call error
  naming the variable, as a missing `OPENAI_API_KEY` does. `DeepSeekEndpoint().api_key` is
  `nothing` (the field is now `Union{String,Nothing}`), and `MistralEndpoint()` is
  `GenericOpenAIEndpoint("https://api.mistral.ai", "", "MISTRAL_API_KEY")`. A key passed
  as `api_key` is sent as is, as before. `GenericOpenAIEndpoint` gains a third field,
  `key_env`: the variable read per request when `api_key` is empty (`""` reads none); the
  two-argument constructor is unchanged. Upgrading: code that takes the key from
  `DeepSeekEndpoint().api_key` (now `nothing`) or `MistralEndpoint().api_key` (now `""`), or relies on
  building the endpoint to check that the key is set, reads the variable itself.
- **Inside a `:replay` scope, streamed calls, the Responses lifecycle calls and FIM no
  longer reach the network.** At 0.26.0 a streamed `chatrequest!` or `respond` (and so a
  streamed `tool_loop!`), `get_response`, `delete_response`, `list_input_items`,
  `cancel_response`, `compact_response`, `count_input_tokens`, `fim_complete` and
  `prefix_complete` went to their service inside any `with_recorded_answers` scope. Now a
  scope answers them as it answers every other recorded call (see Added): in a `:replay`
  scope one without a recording throws `ReplayMissError`, and a polling loop recorded with
  `:record_missing` gets its first status back on every later poll. Upgrading: record what a
  `:replay` scope now needs (`mode = :record_missing`), and record a polling loop with
  `:record`, which keeps the last status.
- **`code_interpreter()` sends the container OpenAI requires.** OpenAI's code interpreter
  tool "requires a container object"; `CodeInterpreterTool` sent none unless one was
  given, and sent `file_ids` beside the tool. It now sends `{"type": "auto"}` unless
  given another container, sends `file_ids` inside that auto container, and takes a
  container id (`code_interpreter(container="cntr_…")`), which it refused before.
  `container` is a `Union{String,AbstractDict}` defaulting to `Dict("type" => "auto")`
  (was `Union{AbstractDict,Nothing}`, default `nothing`); `file_ids` with a container id,
  or beside a container `Dict` that names `file_ids`, is an `ArgumentError`. Upgrading:
  drop `container=nothing`; the auto container is the default.
- **A field-less built-in endpoint, called, is its type.** `OPENAIServiceEndpoint()`,
  `AZUREServiceEndpoint()`, `GEMINIServiceEndpoint()`, `GEMINIOpenAIServiceEndpoint()` and
  `TYPESAFEServiceEndpoint()` built an instance that `service` accepted but that none of
  the endpoint's methods, all defined on the type, served: it got no default model,
  skipped the type's encoding rules and failed at request time with a `MethodError` (on
  TypeSafe, `ask` refused it with an `ArgumentError` saying it declares no capabilities).
  Each now returns its type (`OPENAIServiceEndpoint() === OPENAIServiceEndpoint`), so it
  behaves as the type does. Upgrading: a value tested with `x isa T` is now tested with
  `x === T`.
- **GPT-5.6 takes sampling controls and log probabilities at reasoning effort `"none"` only.**
  The GPT-5.6 models (`gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` and their dated
  snapshots) had no such rule, so `temperature`, `top_p` and log probabilities went out at any
  effort; on 2026-10-25 the API answered `temperature` on `gpt-5.6-luna` at effort `"low"` with
  HTTP 400 on both APIs (Chat Completions: "Unsupported value: 'temperature' does not support 0.5
  with this model. Only the default (1) value is supported.") and took it at `"none"`, and OpenAI's
  GPT-5.4 and GPT-5.2 guides state the rule for the GPT-5 family: requests with `temperature`,
  `top_p` or `logprobs` "will raise an error for GPT-5.4 or GPT-5.2 with any other reasoning effort
  setting, or for older GPT-5 models". As on GPT-6 Sol and Luna, `temperature`, `top_p`,
  `logprobs=true`, `top_logprobs` and `include=["message.output_text.logprobs"]` at another effort
  — the default `"medium"` included — now throw `ArgumentError` before anything is sent. Upgrading: set `reasoning_effort="none"` on a `Chat` or
  `reasoning=Reasoning(effort="none")` on a `Respond`, or remove them.
- **`get_response` and `cancel_response` return a failed response as a `ResponseFailure`.** Both
  wrapped every response read back with HTTP 200 in a `ResponseSuccess`, also one whose `status`
  is `"failed"`, which `respond` returns as a `ResponseFailure`, so `issuccess` reported a failed
  background response as a success. Now each returns it as `respond` does: a `ResponseFailure`
  whose `status` is 200, whose `response` is the response's JSON, its `error` included, and whose
  `request_id` is the reply's; `response_status` still reads `"failed"`. Every other status is
  still a `ResponseSuccess`. Upgrading: code that read a failed response's fields from
  `result.response` parses `result.response` (a `String`) instead, or branches on `issuccess`.
- **`verify_webhook` refuses a signing secret that is not base64.** It promised an
  `ArgumentError` for such a secret, but `base64decode` skips characters outside the base64
  alphabet, so a secret holding one — a placeholder such as `whsec_***`, a stray character —
  decoded to another key and every delivery verified `false`. The secret after its `whsec_`
  prefix is now checked before it is decoded: `A-Z`, `a-z`, `0-9`, `+` and `/` only, then at most
  the `=` padding its last group lacks (a last group of two or three characters may come
  unpadded), as RFC 4648 reads base64: "Implementations MUST reject the encoded data if it
  contains characters outside the base alphabet". Whitespace around the secret, such as the line
  break a file ends it with, is ignored, as `base64decode` ignored it. Whitespace or another such
  character inside the secret, which `base64decode` skipped too, so that such a secret could
  still decode to the right key, now throws. Upgrading: remove anything but the base64 text from
  inside the secret, such as the line break of a wrapped paste.
- **Ollama's decision questions need `instructions`.** Ollama's API reference lists
  `instructions` among the required fields of a Choice, a Noul and a Score question, and `ask`
  on an `OllamaEndpoint` sent a question without them: most decision models answered it with
  HTTP 400 (on Ollama 0.40.6, `tev1:4b`, `nimble:9b`, `strands-decider:2b` and
  `kev:0.8b-q8_0`), and `clef-flash:9b`, which by Ollama's source reads the question's name in
  their place, answered it. Such a question, a `noul` of `yes`/`no` criteria alone included, is
  now an `ArgumentError` before anything is sent. Upgrading: give each question its
  instructions, the first argument of `choice`, `score` or `noul`.
- **Ollama's decision Choice takes two options or more.** Ollama's API reference gives a
  Choice's `criteria` `minProperties: 2`, and `ask` on an `OllamaEndpoint` sent a Choice of one
  option: on Ollama 0.40.6 the MLX decision models answered it with their own HTTP 400, and the
  models on the llama.cpp path (`kev`, `d1`, `d1-omni`) answered it. It is now an
  `ArgumentError` before anything is sent, as on OpenAI's Decisions API, so a natural-language
  verb whose call offers a single option is refused too. Upgrading: add an option, such as a
  fallback `"other"`.
- **`DeepSeekEndpoint` and `MistralEndpoint()` send a `Chat`'s `max_completion_tokens` as
  `max_tokens`.** DeepSeek's Chat Completions reference and Mistral's chat reference name the cap
  `max_tokens` and list no `max_completion_tokens`, yet a `Chat` that set it sent the cap under that
  name, which those references do not list. `DeepSeekEndpoint` and a `GenericOpenAIEndpoint` on
  Mistral's host (`MistralEndpoint()`) now send it as `max_tokens`, taken over a `max_tokens` also
  set, as the native encoders take it. Every other endpoint sends the names the `Chat` sets, as
  before: a generic endpoint elsewhere (Azure's v1 route, whose reasoning models take only
  `max_completion_tokens`, or vLLM, which deprecates `max_tokens`), `AZUREServiceEndpoint` and
  Gemini's OpenAI-compatible endpoint. A `Chat` that sets no `max_completion_tokens` sends the same
  bytes as before. A recording keys on the request's bytes, so one of a changed request made before
  does not replay: record it again. Upgrading: a `Chat` that sets both caps on these two endpoints
  sends the `max_completion_tokens` value as `max_tokens`; set one of them.
- **Gemini's language, escalation and Prohibited Use Policy stops are refusals.** A native Gemini
  candidate that ended with `finishReason` `LANGUAGE` (content "flagged for using an unsupported
  language"), `ESCALATION` ("filtered by an escalation rule") or `PUP_LIMITED_DISABLED` (the account
  "limited or disabled due to Prohibited Use Policy (PUP) violations") reported that value in lower
  case, `"language"`, `"escalation"` or `"pup_limited_disabled"`, and a turn without text had the
  content `""` and no `refusal_message`, so Google's `finishMessage` was lost. Each now gives
  `"content_filter"`, as Gemini's other filter stops do, streamed or not; a turn without text has
  no content, and its `refusal_message` is the `finishMessage`, else the stop's name and the enum's
  definition of it, such as "ESCALATION: Request was filtered by an escalation rule.". Upgrading: code that matched those three values tests
  `finish_reason == "content_filter"` and reads `refusal_message`; `text(result)` of such a turn is
  `nothing`, no longer `""`.
- **`list_input_items` refuses `GEMINIServiceEndpoint`.** Google's Interactions API gets, cancels
  and deletes an interaction, and its reference lists no method that lists an interaction's input
  items, yet `list_input_items(id; service=GEMINIServiceEndpoint)` sent
  `GET /v1beta/interactions/{id}/input_items`, a path the reference does not list, and returned what
  came back. It now throws `ArgumentError` before any request, as it does for an `OllamaEndpoint`.
  Upgrading: keep the `input` you sent with the interaction.
- **Gemini Interactions refuses `service_tier="deferred"` before sending.** A `Respond` on
  `GEMINIServiceEndpoint` sent `"deferred"` as given. The Interactions API reference no longer lists
  it — its tiers are `flex`, `standard` and `priority` — and on 2026-10-26 the API answered a request
  naming it with HTTP 400: "The value 'deferred' is not supported for 'service_tier' on the Gemini API,
  it is allowed on the Gemini Enterprise Agent Platform." It now throws `ArgumentError` before any
  request, as a `Chat` on either Gemini endpoint already did, whose message no longer sends such a
  request to `respond`. Upgrading: send `"flex"`, `"standard"` or `"priority"`, or leave
  `service_tier` unset.
- **An Ollama chat result names the model its reply gives.** An `/api/chat` reply names the "Model
  name used to generate this message", and each line of a stream the "Model name used for this
  stream event", but UniLM dropped it, so `LLMSuccess.model` was `nothing` on an `OllamaEndpoint`,
  while the other providers' results carry the model they report. It now holds the reply's name,
  streamed or not. A cloud model whose reply names it without its cloud tag keeps the requested
  name, as its `respond`, FIM and System One results do, so it is still priced as the cloud model
  it is; a local model's result still costs `0.0` unless you gave the model a price row. Upgrading:
  code that read `nothing` from an Ollama chat result's `model` reads the reply's model name, or a
  cloud model's requested name; its price is unchanged.
- **A DeepSeek `Chat` with a strict tool goes to DeepSeek's beta base URL.** DeepSeek's strict mode
  is a beta feature: "To use `strict` mode, you need to: Use `base_url="https://api.deepseek.com/beta"`".
  `DeepSeekEndpoint()` sent every `Chat` to `https://api.deepseek.com/v1/chat/completions`, so a
  tool's `strict` never reached the URL where it applies. A `Chat` whose tools include a strict one
  now goes to `https://api.deepseek.com/beta/v1/chat/completions`, streamed or not; one without
  goes where it went. There "The server will validate the JSON Schema of the Function provided by the
  user. If the schema does not conform to the specifications or contains JSON schema types that are
  not supported by the server, an error message will be returned", and "All properties of every
  `object` must be set as `required`, and the `additionalProperties` attribute of the `object` must be
  set to `false`": a strict tool whose schema breaks these rules, sent before to the non-beta URL,
  where strict mode does not apply, now gets that error (DeepSeek's page; neither observed). A
  recording keys on the request's path, so one of such a request made before does not replay:
  record it again. Upgrading: give a strict tool a schema DeepSeek's strict mode takes (every
  property `required`, `additionalProperties: false`, the types its tool-calls page lists), or leave
  `strict` unset.
- **Bridged MCP tool names fit the provider they go to.** `mcp_tools` and `mcp_tools_respond`
  kept an MCP tool's name up to 128 characters, but OpenAI's Chat Completions takes a function
  name of "a maximum length of 64", so a name of 65 to 128 characters went to OpenAI as it was.
  Both now take `service`, the endpoint the tools go to (`OPENAIServiceEndpoint` unless given),
  and keep each name within its provider's limit: 128 characters for `ANTHROPICServiceEndpoint`,
  whose Messages API takes `^[a-zA-Z0-9_-]{1,128}$`, and for `GEMINIServiceEndpoint` and
  `DeepSeekEndpoint`, whose generateContent and Chat Completions take "a maximum length of 128",
  and 64 for any other endpoint. A longer name is cut to the limit less 9 characters and followed
  by `_` and the first 8 hex digits of the SHA-256 of its MCP name, so names that agree up to the
  cut stay apart; a name over 128 characters was cut to 128 before. Upgrading: pass `service=` the
  endpoint the tools go to, to keep names of up to 128 characters on Anthropic, native Gemini and
  DeepSeek; a `tool_choice` that names a bridged tool takes the name the bridge now advertises.
- **A price row you give a local Ollama model is used.** `estimated_cost` returned `0.0` for a result
  of a local model on an `OllamaEndpoint` before it read any price row, so a row you gave the model,
  in `pricing=` or added to `DEFAULT_PRICING`, which `cumulative_cost` reads, was ignored without a
  word; so was one for a local model's System One answer. A row you give the model's exact name now
  prices its results as any model's row does. The rows UniLM ships price hosted models only: a local
  model that bears a hosted model's name, as Ollama suggests for tools that expect OpenAI's names
  (`ollama cp llama3.2 gpt-3.5-turbo`), still costs `0.0`, also when a copy of those rows such as
  `merge(DEFAULT_PRICING, …)` is passed; a row equal to a shipped one counts as the shipped one. A
  local model with no row of yours still costs `0.0`, with no missing-price warning, and `model=`
  still prices a result at the row of the model it names. Upgrading: to keep a local model's
  estimates at `0.0`, remove the row you gave its name.

### Added
- **`with_recorded_answers` records streamed calls, the Responses lifecycle calls and
  FIM.** Inside a scope a streamed `chatrequest!` or `respond` (and so a streamed
  `tool_loop!`) is replayed through the same parsers and callbacks as a live stream, or
  sent and, once its reply is whole and read with no stop, recorded as the lines of its
  body (`response.stream`); so are `get_response`, `delete_response`, `list_input_items`
  (keyed with its `limit`, `order` and `after`), `cancel_response`, `compact_response`,
  `count_input_tokens`, `fim_complete` and `prefix_complete`. A stream is recorded once it
  is whole, also when its connection fails after the final event, and never when it was
  cut before that event. A recording also keeps the response headers a replay gives back (`response.headers`: the request-id header and
  `x-gemini-service-tier`). A miss on a stream's first request is thrown by the verb; its
  lost recording, or a miss on a later request of a turn the provider continues, is
  thrown inside the call's task, so `fetch` (and `tool_loop!` on a streamed chat) throws a
  `TaskFailedException` around it. Every earlier recording replays unchanged; what this
  changes for code that ran these calls inside a scope is under Breaking.
- **`OllamaOptions` takes `draft_num_predict` and `typical_p`.** Ollama's runner option for
  speculative decoding — the most draft tokens predicted per step when a model has a draft
  model, `0` turning drafting off — and `typical_p`, the sampling option Ollama deprecates
  and still reads per request (it logs "deprecated option provided"), are sent in the
  request's `options` like the other options; a negative `draft_num_predict`, or a
  `typical_p` outside [0, 1], is an `ArgumentError`. `penalize_newline` and `numa`, which
  Ollama's API reference also lists, are still refused as unknown keywords: Ollama 0.40.11
  does not read them (it logs "invalid option provided" and goes on without them).
- **`transcribe` reaches a `GenericOpenAIEndpoint`.** The endpoint declares `:transcription`,
  as it declares the other OpenAI-compatible routes, so `transcribe` sends the request to its
  server's `/v1/audio/transcriptions`, where vLLM serves OpenAI's Transcriptions API; it
  refused the endpoint before. As for its chat, the endpoint has no default model: a call that
  names none is an `ArgumentError` before any request. `speak` and `translate` still refuse it.
- **`Chat` takes `prompt_cache_retention`.** OpenAI's Chat Completions reference still lists
  the field, deprecated ("Use `prompt_cache_options.ttl` instead"), and a `Chat` could not set
  it, while a `Respond` could. `OPENAIServiceEndpoint` sends it, and refuses it with an
  `ArgumentError` on the GPT-6 and GPT-5.6 models, whose cache lifetime `prompt_cache_options`
  sets, as on a `Respond`; Azure, `GEMINIOpenAIServiceEndpoint`, `DeepSeekEndpoint` and generic
  endpoints send it as given; native Gemini, Anthropic and Ollama refuse it, having no such
  field. Built positionally, a `Chat` takes it as the 38th argument.

### Changed
- **The language-model documentation is a chapter organised by provider.** The manual's *Guide*
  is now the *Language models* chapter: an overview of when to use a language model and which
  providers UniLM reaches, an entry for each provider, the concept pages every provider shares and
  the pages across providers. An entry holds what is its provider's own — setup, a first call, the
  rules and limits UniLM applies to it, its failures — and each concept page ends in a table that
  links the providers' entries. *Attachments*, *Results* and *Testing and recordings* are new
  concept pages. The API reference follows as *Reference*, the LLM reference for coding agents at
  its end.
- **OpenAI has an entry of its own.** Seven pages: *Start here* (setup, Chat Completions or
  Responses, a first call, conversations, streaming, embeddings, failures), *Models & request
  rules* (models, reasoning, prompt caching, cost, limits), *Chat Completions*, *Responses & hosted
  tools*, which holds the Responses API guide, *Retrieval & file search*, which holds the Retrieval
  & File Search guide, *Images, audio & realtime* and *Platform APIs* (files, conversations,
  containers, batches, fine-tuning, moderation, webhooks).
- **Anthropic has an entry of its own.** Three pages: *Start here* (setup, the endpoint's options, a
  first call, conversations, streaming), *Thinking, tools & output* (tools, structured output,
  reasoning, attachments, request options, limits) and *Caching, cost & limits* (prompt caching,
  cost, rate limits and failures, models, Anthropic's OpenAI-compatible endpoint).
- **Google Gemini has an entry of its own.** Four pages: *Start here* (setup, which of Google's APIs
  serves what, a first call, conversations, models, limits, failures), *The generateContent API*,
  which `chatrequest!` reaches, *The Interactions API*, which `respond` reaches, and *Images,
  speech, embeddings & the compatible endpoint*.
- **Ollama has an entry of its own.** Three pages: *Ollama* (setup, conversations, streaming, tools,
  structured output, reasoning, attachments, transcription, `respond` and FIM, cost, limits),
  *Embeddings* (EmbeddingGemma 2, images and sound, long inputs, local retrieval) and *Runtime &
  model management* (the endpoint's options, runners, models, failures). It holds the Local Models
  with Ollama guide.
- **vLLM, LM Studio and llama.cpp have an entry.** One page, on `GenericOpenAIEndpoint` against a
  server of your own: setup, a first call, streaming, tools, structured output, the embeddings,
  `respond`, FIM and transcription routes, limits and failures.
- **Azure OpenAI has an entry.** One page, on `AZUREServiceEndpoint` and a deployment of yours:
  setup, conversations, streaming, tools, structured output, reasoning, attachments, prompt caching,
  cost, models, limits and failures, and Azure's v1 API through a generic endpoint.
- **DeepSeek has an entry.** One page, on `DeepSeekEndpoint()`: setup, conversations, streaming,
  tools and DeepSeek's strict mode, structured output, reasoning, FIM and prefix completion, prompt
  caching, cost, models, limits and failures.
- **Mistral and other OpenAI-compatible services have an entry.** One page, on `MistralEndpoint()`
  and on `GenericOpenAIEndpoint` for a hosted service: setup, conversations, streaming, tools,
  structured output, attachments, embeddings, transcription, FIM, cost, models, limits and failures,
  with the facts that are Mistral's alone marked as Mistral's.
- **Jev (TypeSafe) has an entry of its own.** Three pages in the Decisions chapter: *Start here*
  (setup, a first decision, the models, the questions, limits), *Images, Box, long context,
  structured guidance* and *Confidence, errors, cost & testing*. It holds *Start Here: Decisions in
  Five Minutes*.
- **OpenAI's Decisions API has an entry of its own.** One page, on `ask` with
  `OPENAIServiceEndpoint`: setup and options, a first decision, the models and questions, declined
  questions, limits, confidence, cost, recordings and failures.
- **Local decision models have an entry of their own.** One page, on `ask` with `OllamaEndpoint()`,
  which Ollama's System One route answers: setup and options, a first decision, the models,
  questions and images, limits, confidence, recordings, throughput and failures.
- **Decisions is a chapter of its own.** The *Decisions (Jev)* section is now the *Decisions*
  chapter: an overview of when a decision model serves better than a language model and which
  providers answer `ask`, the three providers' entries, the concept pages every decision provider
  shares — *Route and Decide*, *Dispatch on Meaning*, *Many Items at Once*, *Decisions with LLMs*
  and *Test and Develop*, each ending in a table that links the providers' entries — and, across
  providers, *Choosing a decision provider*, which holds *Backends: Jev, OpenAI and Local Models*,
  and *Write your own decision provider*, a new page.
- **One API across providers says what every provider shares and where they differ.** The
  Multi-Backend guide, at its address, is now *One API across providers*: what every provider
  shares, how to switch providers, where they differ and how to keep up with their models, ending
  in a table that links each provider's entry. Beside it, *Capabilities by provider*, a new page,
  shows what each provider's endpoints serve, and *Write your own backend* is the Custom Backends
  guide.
- **Every page address and declared section id of the v0.26.0 manual still resolves.** The five
  pages that moved — the Responses API, Retrieval & File Search and Local Models with Ollama guides,
  *Start Here: Decisions in Five Minutes* and *Backends: Jev, OpenAI and Local Models* — each leave
  a page at their old address that forwards the reader to the page that replaced it, at the section
  the link named where the site still defines it; every other page keeps its address, and the API
  pages keep every docstring they held. Every section id the v0.26.0 pages declared still resolves;
  the six that were retired lead to their successors. The Documentation workflow writes the forwarding
  pages and fails when an address or a declared section id of v0.26.0 resolves to nothing.

### Fixed
- **A replayed Gemini reply keeps its service tier.** A recording now keeps the reply's
  `x-gemini-service-tier` header, which names the tier Gemini billed it at (Gemini 4 Argon
  names it nowhere else), and a replay gives it back: a replayed reply reports that tier
  and costs what the live one cost. Before, a `GEMINIServiceEndpoint` reply whose body
  named no tier replayed at the standard rates — a Flex reply at twice its live cost. A
  recording made before keeps replaying without the header: record it again to keep its
  tier.
- **A replay miss or lost write that reaches a tool dispatcher or an MCP tool handler
  wrapped fails loudly.** A dispatcher or handler that fetches a streamed call — or waits on
  any task, runs an `@sync` block or an `asyncmap` that hit one — gets the
  `ReplayMissError` or `RecordingWriteError` inside a `TaskFailedException`,
  `CompositeException` or `CapturedException`. The tool loops took that for a tool error
  and went on, and an MCP server answered the client with it as `isError` tool content, so
  the text, a local recording path with it, reached the model or the client. Now a loop
  throws it as it was thrown, and an MCP server logs it and answers the generic JSON-RPC
  `-32603`, as for a bare one.
- **A user's interrupt that reaches a tool dispatcher or an MCP handler wrapped propagates.**
  A dispatcher or handler that waits on a task gets an `InterruptException` inside a
  `TaskFailedException` (or a `CompositeException`, a `CapturedException`). The tool loops
  took it for a tool error and went on, an MCP server answered a tool handler's as tool
  content, and a resource or prompt handler's it logged and answered `-32603`. Now each
  throws it as it was thrown, as for a bare `InterruptException`. Any other error inside
  such a wrapper goes where it went: a tool loop still takes it for a tool error, an MCP
  server still answers a tool handler's as tool content, and a resource or prompt
  handler's it still logs and answers `-32603`.
- **A streamed turn interrupted during a later reply keeps the earlier replies' cost.** When
  an `InterruptException` ended a streamed turn the provider continued over several requests
  (a long Gemini generation), `cumulative_cost` left out the replies already received; they
  are now added before the interrupt propagates, as an unstreamed turn adds them.
- **`prefix_complete` on `DeepSeekEndpoint` keeps and sends back `reasoning_content`.** The
  prefix completion was read with the generic Chat Completions decoder, which drops it, so
  `reasoning_text` was `nothing` and a later request with `tools` on that chat sent the
  turn without it, which DeepSeek answers with HTTP 400; and its own request, built
  without the endpoint's encoder, sent no earlier turn's reasoning even when it carried
  `tools`. It now goes through the endpoint's encoder and decoder: the turn keeps its
  reasoning, as a `chatrequest!` turn does, and every request with `tools`, a prefix
  completion included, sends it back.
- **Anthropic's monthly spend cap is returned at once.** At the spend cap Anthropic answers
  HTTP 429 with the `rate_limit_error` type of a rate limit and
  `error.details.error_code` `enforced_spend_limit_reached`, and says retrying "fails until
  access resumes". UniLM read only `error.code` and `error.type`, so it sent the request
  again, up to `max_attempts` attempts in all, each after its backoff. That 429 is now
  final, as OpenAI's spend and quota codes are; a `rate_limit_error` without it is still
  retried.
- **A tool marked `strict=true` asks native Gemini for schema adherence.** A Gemini function
  declaration has no `strict` flag, so `GEMINIServiceEndpoint` sent a strict tool — on a
  `Chat`, and on `respond` (Gemini Interactions) — as a plain one, and the caller silently
  got no adherence. A strict tool now sets the request's function-calling mode: with
  `tool_choice` unset or `"auto"`, `VALIDATED` (`validated` on `respond`), in which Gemini
  "ensures function schema adherence" and the model may still answer in text. The mode
  covers every function the request declares. `"required"` and a named function are still
  sent as `ANY` (`any`), and `"none"` as `NONE`.
- **A recorded call's miss or lost write in a stream callback reaches the caller.** A
  `ReplayMissError` or `RecordingWriteError` that a recorded call made in a streamed
  `chatrequest!`'s `callback` or `on_tool_call`, or a streamed `respond`'s `callback`,
  threw was returned as the stream's `LLMCallError` (`ResponseCallError`) `cause`, a call
  error a fallback would take for a service failure. Now it propagates, as it does from a
  tool loop's dispatcher: `fetch` throws a `TaskFailedException` around it — around the
  wrapper too when the callback threw it inside one, such as the `TaskFailedException` of
  a fetched task. So does an `InterruptException` a callback throws inside such a wrapper,
  as a bare one already did. The chat's `cumulative_cost` still adds the usage the stream
  had reported, as for any other exception a callback throws.
- **Ollama takes every reasoning effort, `"max"` among them.** A `reasoning_effort` on an
  `OllamaEndpoint` `Chat`, or a `Reasoning` effort on its `respond`, other than `"none"`,
  `"low"`, `"medium"` and `"high"` threw an `ArgumentError`, though Ollama's API reference
  lists `"max"` among the levels of `think` and a model's thinking levels are its own. Every
  effort is now sent by name (`"none"` still as `think: false`) for Ollama to resolve against
  the model: one with thinking metadata applies a level it names and uses its default for any
  other; to one without, `/api/chat` answers a name other than `"low"`, `"medium"`, `"high"`
  and `"max"` with HTTP 400, and `/v1/responses` reads `"minimal"` as `"low"` and `"xhigh"`
  as `"max"`.
- **Ollama's cloud gets its API key.** Direct cloud inference at `https://ollama.com`, where
  `OLLAMA_HOST=ollama.com` points `OllamaEndpoint()`, "requires an API key" in the
  `Authorization: Bearer` header, and UniLM sent none, so every inference request went out
  without one. An endpoint whose server is `https://ollama.com` now sends the `OLLAMA_API_KEY`
  environment variable that way whenever it is set, read each time a request is built. Unset
  or blank, an inference request — chat, embeddings, `respond`, compaction, FIM, transcription,
  System One, a model's load or unload — is not sent, and the call error's `cause` is an
  `ArgumentError` naming the variable; `list_models` (which Ollama's cloud page runs without a
  key), `model_info`, `running_models`, `pull_model` and `delete_model` go without it. No
  other server gets the key. Such a call is priced by the name it requests, as a cloud model
  reached through the local server is — the cloud names its models without a cloud tag — and
  never as a free local call: a name with no price row logs the missing-price warning.
- **A `GenericOpenAIEndpoint` `respond` sends its `seed`.** A `Respond` with a `seed` was
  refused on every endpoint of the OpenAI Responses wire, a generic one included, though
  vLLM's Responses API forwards a seed to its engine and a generic endpoint sends every other
  field as given. A generic endpoint now sends it as `seed`; `OPENAIServiceEndpoint`, Azure
  and `GEMINIOpenAIServiceEndpoint` still refuse it, since OpenAI's Responses API has none.
- **`gpt-5-search-api` is priced.** The Chat Completions search model, which a `Chat` with
  `web_search_options` runs on, had no row in `DEFAULT_PRICING`, so `estimated_cost` and
  `cumulative_cost` read its calls as `0.0`, with the missing-price warning. Its row holds the
  rates OpenAI's pricing page lists on 2026-10-25: \$1.25 input, \$0.125 cached input and \$10.00
  output per 1M tokens. A fee for web search apart from the tokens is not included.
- **`jev-1.14.0` has its price row.** TypeSafe's Models page lists Jev 1.14 (`jev-1.14.0`)
  beside Jev 1.13 at \$42 per Btok of input, output free, and `DEFAULT_PRICING` had no row for
  it: a result naming it was priced through the `jev-latest` fallback, at that same rate, and
  the `estimated_cost` docstrings said TypeSafe lists a price for Jev 1.13 alone. The row is in
  the table, and the docstrings name both versions.
- **Five docstrings say what the code does.** `Chat`'s names `GEMINIOpenAIServiceEndpoint`
  among the endpoints whose stream asks for usage; `MCPTool`'s gives `connector_id` OpenAI's
  legacy label (deprecated for models released after September 1, 2026);
  `ReplayMissError`'s and `with_recorded_answers`' give OpenAI's Decisions API its keyed path,
  `/openai-decisions/v1/decisions` (they gave every System One backend `/<namespace>/v1/systemone`),
  and `with_recorded_answers`' lists `prompt_cache_options` among the options never keyed and
  names each backend's access for recording, not `TYPESAFE_API_KEY` for all; `@branch`'s says
  which model an omitted `model` means, and that Ollama has none. No behaviour changes.
- **An OpenAI audio reply goes back as its transcript once its audio expires.** A `Chat` on
  `OPENAIServiceEndpoint` sent an earlier audio reply back by its id whatever the reply's
  `expires_at`, the time from which OpenAI says the audio "will no longer be accessible on the
  server for use in multi-turn conversations", so a conversation continued after it sent the id
  of audio OpenAI no longer keeps. From a minute before `expires_at` on — a request built later
  could reach OpenAI after the expiry — the turn goes as its content alone, its transcript unless
  it holds other text, as every other endpoint gets it; before then, or with no expiry known, it
  still goes by its id. The request bytes of such a follow-up thus depend on the clock: one
  recorded with `with_recorded_answers` while the audio was live replays only until a minute before
  the expiry, then throws `ReplayMissError`; record it after the expiry.
- **A background `respond` on OpenAI may set `store=false`.** A `Respond` with
  `background=true` and `store=false` was refused for every endpoint, though OpenAI's
  background guide says "Background requests can use `store=false`, but response data is
  temporarily stored to support asynchronous execution and polling". It is now sent to OpenAI,
  and to a generic server, as set; `get_response` reads the response while OpenAI holds it.
  Gemini Interactions answers it with HTTP 400, so `respond` still refuses it there, now when
  the request is encoded rather than when the `Respond` is built; an `OllamaEndpoint`, whose
  Responses API keeps no response, refuses `background` and `store` as it did.
- **Between the attempts of a call, UniLM waits what `retry-after-ms` asks for.** The retry
  backoff read only `Retry-After`, so a reply that named its wait in `retry-after-ms` alone — the
  header Azure OpenAI's 429 carries, "The recommended wait time (in milliseconds) before
  retrying", and which the OpenAI, Anthropic and TypeSafe SDKs read first — got the jittered
  backoff without it, which can be shorter, while `ask_each` and a `SystemOneFailure`'s
  `retry_after` already read it. Every verb, streamed or not, now waits at least what
  `retry-after-ms` asks for, else `Retry-After`.
- **A retry wait no timer can hold is capped.** With `total_deadline = Inf`, a `Retry-After` or
  `retry-after-ms` asking for more than 1e9 seconds threw `InexactError` from the backoff's timer
  instead of waiting. The asked wait is now capped at 1e9 seconds, as `ask_each` caps its shared
  wait, and a cancel ends it as it ends any backoff; a finite `total_deadline` still returns the
  last response at once.
- **The TypeSafe docstrings use TypeSafe's words.** `TypeSafeProtocol`'s said TypeSafe marks
  images and Box questions experimental; TypeSafe calls them "Dreamer-ready", not yet "ready for
  production". `TYPESAFE_DEFAULT_MODEL`'s advised pinning `jev-1.13.0`, while `jev-latest` now
  names `jev-1.14.0`; it says to send the version a confidence threshold was tuned against, as
  TypeSafe's Models page does. No behaviour changes.
- **Four more docstrings say what the code does.** `DocumentAttachment`'s said its `filename` is
  the name the provider is given; only the OpenAI Chat Completions `file` part and the Responses
  `input_file` part carry it, and Anthropic, native Gemini and Gemini Interactions get a document
  without it. `FunctionSignature`'s called an omitted `strict` the API's non-strict default and
  said OpenAI rejects an invalid strict schema with a 400: OpenAI's Responses API "will attempt
  to normalize your schema into strict mode when possible" when `strict` is omitted, and names
  no status for the rejection. `respond`'s named OpenAI's Responses API as its one target, and
  the keyword `chatrequest!`'s listed `config` and `cancel` among the `Chat` fields. No behaviour
  changes.
- **OpenAI's Decisions API takes up to 400 questions a request.** `ask` on
  `OPENAIServiceEndpoint` refused a request of more than 200 questions, the public beta's
  limit, though the generally available API takes "1–400 questions" (OpenAI's reference), "up
  to 400 questions per request, up from 200 during the public beta" (its guide); a request of
  201 questions was answered on 2026-10-26. Up to 400 are now sent, and more is an
  `ArgumentError` before anything is sent, as more than 200 was.
- **A Decisions answer's cache reads and writes are priced at OpenAI's caching rates.**
  `estimated_cost` priced every input token of an answer from OpenAI's Decisions API at the
  model's input rate, cache reads and writes included, as the public beta's guide had it ("no
  cache-read, cache-write, or output-token charges"). The guide now says "Prompt cache hits
  receive discounted input pricing; cache writes are charged", at the standard `gpt-6-luna`
  caching rates: a cache read now costs the cached-input rate (\$0.01 per 1M tokens) and a
  cache write the write rate (\$0.125 per 1M, \$0.25 for a 12-hour lifetime), each twice for a
  request of more than 272K input tokens; output is still not charged.
- **OpenAI's Decisions API gets an image's detail level.** Every System One backend refused an
  `ImageAttachment` with a `detail` level, though the Decisions API's `input_image` part takes
  `detail` — `"low"`, `"high"`, `"auto"` or `"original"`, "using the selected model's image
  profile". `OPENAIServiceEndpoint` now sends it on the image's part of the state, and a
  recording keys it beside the image's digest. TypeSafe and Ollama still refuse it before
  anything is sent, and now say why: a TypeSafe image asset holds the image's bytes and format
  alone, and Ollama's `images` list each image's base64 bytes alone.
- **A Decisions field over 1,048,576 characters is refused before it is sent.** OpenAI's reference
  caps the model's name and each question's name, string instructions, choice value and
  description and level label at 1,048,576 characters, and `ask` on `OPENAIServiceEndpoint`
  checked only the 10,485,760 characters of a text input, so a longer field went out though the
  schema rules it out. Each is now an `ArgumentError` before anything is sent, naming the field
  and its length, counted in characters as the reference counts them.
- **OpenAI's Decisions API takes a tuple of strings and images as a state, as the other backends
  do.** `ask` on `OPENAIServiceEndpoint` sent a vector of strings and images as one user message
  of text and image parts, but refused the same items in a tuple, as an image "nested in JSON",
  while TypeSafe and Ollama take a tuple as they take the vector. A tuple is now sent as the
  vector is.
- **A tuple is guidance, as it is a state.** A tuple given as a question's instructions, an
  option's description, a Score level or a Noul criterion was an `ArgumentError` when the
  question was built, though a tuple state, and a tuple nested in guidance, are written as JSON
  arrays. `SystemOneEntry` now holds `Tuple`, so a tuple stands wherever an array does and every
  backend takes it as the vector of its items: TypeSafe writes it as an array, Ollama takes it
  as array instructions, and OpenAI's Decisions API refuses it where it refuses a vector, in
  guidance that must be a string.
- **Ollama's refusal of an image after a text says what Ollama does.** `ask` on an
  `OllamaEndpoint` refused an image after a text in the state with "Ollama shows a request's
  images before its text", of every model, and the `OllamaSystemOneProtocol` docstring said the
  same. Ollama takes a request's images in a list apart from its text, and by its source
  (v0.40.11) only Clef's encoder places them, before the state's text; the others that take
  images leave their place to the runner. The message now says so: "Ollama takes a request's
  images in a list apart from its text, so an image after a text cannot be sent where it is
  (Clef reads the images first)". What is refused is unchanged.
- **`OpenAIDecisionsProtocol`'s docstring calls OpenAI's Decisions API generally available.** It
  called the API a public beta; OpenAI's guide opens "The Decisions API is now generally
  available". No behaviour changes.
- **A streamed reply whose content arrives as a list of chunks loses no text.** Mistral streams
  a reasoning reply's thinking as a list holding a `ThinkChunk`, then "a single list containing
  both a closing `ThinkChunk` and the first `TextChunk`", then the answer as strings. The
  OpenAI-wire stream handler read a delta's `content` only when it was a string and dropped a
  list whole, so a streamed reasoning reply lost the answer's first chunk, in the callback and in
  the final message, which the same reply unstreamed kept. A list is now read chunk by chunk:
  its text chunks reach the callback and the message in order, and its thinking chunks are kept
  apart as the turn's reasoning (`ProviderContent(:mistral, …)`), which `reasoning_text` reads.
- **A streamed call on an endpoint type without a stream handler is refused before it is sent, alike
  on both stream seams.** On a `ServiceEndpoint` subtype with no `handle_sse_event!` method, a
  streamed `chatrequest!` was sent and every payload of the reply dropped as undecodable, so the
  call ended as an `LLMFailure` with status 200 that did not name the cause; on one with no
  `decode_agentic_stream` method, a streamed `respond` was sent and returned a `ResponseCallError`
  holding the `MethodError`. Both now throw the missing method's `MethodError` before anything is
  sent, as a missing `encode_request` does.
- **`prefix_complete` on a streaming `Chat` sends a request it can read.** It sent the `Chat`'s
  `"stream": true` (and DeepSeek's `stream_options`), so the server answered with a stream that the
  call, which reads one whole reply, could not decode: a billed request that returned an
  `LLMCallError`. The request now leaves `stream` and `stream_options` out and is read whole; the
  `Chat` keeps both for `chatrequest!`. `prefix_complete` takes no callback, so nothing a stream
  would deliver is lost. A recording keys on the request's bytes, so one of such a request made
  before does not replay: record it again.
- **`prefix_complete` adds its turn's cost to `cumulative_cost`.** It appended the continued turn
  to the `Chat` but left its cost out of the `Chat`'s running total, which every `chatrequest!`
  turn adds. A successful prefix completion now adds its result's `estimated_cost`, with or without
  `history`; a failure adds nothing, as it carries no billed usage.
- **`transcribe` retries a transient status, as the other inference verbs do.** It made one attempt,
  so a 429, a 503 or a dropped connection returned at once, while `chatrequest!`, `respond`,
  `embeddingrequest!` and `edit_image` on the same endpoint send the request again. It now retries
  the same statuses and transport failures under its `RequestConfig` (`max_attempts`,
  `total_deadline`, the wait a reply asks for), and each attempt sends the whole file again, as an
  image edit does; a 429 that no retry can serve is returned at once.
- **A DeepSeek `Chat` takes `reasoning_effort="ultra"`.** DeepSeek's thinking-mode page maps a
  requested `ultra` to `max`, and the `Chat` constructor refused it for every endpoint. A `Chat` on a
  `DeepSeekEndpoint` now takes `"ultra"` and sends it as set (DeepSeek answered such a request on
  2026-10-26); on every other endpoint it is refused when the `Chat` is built, as before.
- **`MistralEndpoint()` sends a `Chat`'s seed as `random_seed`.** Mistral's chat reference names the
  seed `random_seed`, "The seed to use for random sampling", and lists no `seed`, which the endpoint
  sent. A `Chat` on Mistral's host (`MistralEndpoint()`, or a `GenericOpenAIEndpoint` on
  `api.mistral.ai`) now sends its seed as `random_seed`; every other server's request is unchanged,
  byte for byte. A recording keys on the request's bytes, so one of such a request made before does
  not replay: record it again.
- **`MistralEndpoint()` sends an embedding size as `output_dimension`.** Mistral's embeddings
  reference names the vector size `output_dimension` and lists no `dimensions`, which the endpoint
  sent. An `Embeddings` request on Mistral's host now sends its `dimensions` as `output_dimension`;
  every other server's request is unchanged, byte for byte. A recording keys on the request's bytes,
  so one of such a request made before does not replay: record it again.
- **Azure takes an earlier audio reply back by its id, as OpenAI does.** Azure's audio quickstart adds
  "a history message referencing the previous turn's audio by ID", but `AZUREServiceEndpoint` sent an
  audio turn back as its transcript alone, so the model lost the audio it had spoken. It now sends
  `"audio": {"id": …}` beside the transcript, until a minute before the reply's `expires_at`, and the
  transcript alone from then on, as `OPENAIServiceEndpoint` does. A recording of such a follow-up made
  before still replays once the audio has expired, as a replayed reply's has; a turn whose reply named
  no expiry (a stream cut before its last update) now goes by id, so record that follow-up again.
- **`list_models` on an endpoint without a listing names that endpoint alone.** An endpoint with
  no model listing of its own reaches TypeSafe's, its fallback System One protocol's, so
  `list_models` on `GEMINIOpenAIServiceEndpoint`, `AZUREServiceEndpoint`, `DeepSeekEndpoint`, a
  `GenericOpenAIEndpoint` (`MistralEndpoint()` among them) or an endpoint type of your own that
  declares capabilities without `:models` threw "TypeSafe models listing is not supported by …",
  naming a provider the call had nothing to do with, and on an endpoint type of your own that
  declares no capabilities it threw a `MethodError`. Each now throws an `ArgumentError` that reads
  "Model listing is not supported by" the endpoint, then lists its capabilities, as before, or says
  that it declares none, as `ask` does.
- **`speak` on Gemini takes the Extended Voice Library and designed and replicated voices.** Gemini
  3.8 TTS takes "a prebuilt voice name, an Extended Voice Library ID, a custom Voice design ID
  (`voice_...`), or a Voice replication ID (`voice_...`, or optional stateless `voicekey_...`)", yet
  a `SpeechRequest` on `GEMINIServiceEndpoint` threw `ArgumentError` for any voice but the 30 studio
  voices. The library's hundreds of voices are listed only by Gemini's `GET /v1beta/voices`, so a
  voice is now sent as given, and Gemini answers a name it does not have: one of the 30 as
  `prebuiltVoiceConfig.voiceName`, as before, any other as `voiceConfig.voice`, the field Google's
  reference gives for a stored voice's id and a replication key. A blank voice still throws.
- **`verify_webhook` refuses an empty signing secret.** A secret with nothing after its
  `whsec_` prefix and the whitespace around it — `""`, as an unset variable read with a default
  gives, whitespace, or `whsec_` alone — was used as an empty HMAC key, with which anyone can
  sign: a delivery signed with it verified `true`, and every delivery OpenAI signed `false`. It
  now throws `ArgumentError` before any check, whatever the delivery, as the Standard Webhooks
  libraries refuse an empty secret ("webhook secret may not be empty").
- **`gpt-5.2-pro` is priced.** It had no row in `DEFAULT_PRICING`, so `estimated_cost` and
  `cumulative_cost` read its calls, and those of its snapshot `gpt-5.2-pro-2025-12-11`, as `0.0`,
  with the missing-price warning. Its row holds the rates OpenAI's pricing page lists on
  2026-10-26: \$21.00 input and \$168.00 output per 1M tokens, and no cached-input rate, so a
  cache read bills as input, as on the other pro models.
- **`score` refuses levels that are not a vector with an `ArgumentError`, as `choice` refuses its
  criteria.** `score("q", ("a", "b"))`, a tuple of levels, threw a `MethodError`, while `choice`
  with a tuple of options throws an `ArgumentError` that names the forms it takes. `score` now
  throws an `ArgumentError`, "Score `levels` must be a vector of levels, lowest first", for a
  tuple or any other container, and for a call that gives instructions alone.
- **`Respond`'s docstring states its constructor's checks of `max_output_tokens` and
  `top_logprobs`.** The constructor throws `ArgumentError` for a `max_output_tokens` below 1 and a
  `top_logprobs` outside 0 to 20, the range OpenAI's reference gives ("An integer between 0 and
  20"); the docstring stated neither and did not list `top_logprobs`. No behaviour changes.

## 0.26.0

### Breaking
- **Current default models.** `Chat` and `Respond` on `OPENAIServiceEndpoint` default to
  `gpt-6-sol` (was `gpt-5.6-sol`), and so do `compact_response` and `count_input_tokens`;
  image generation and edits default to `gpt-image-2.5-sunburst` (was `gpt-image-2`, at the
  same token prices); `realtime_connect` opens `gpt-realtime-2.1` (was `gpt-realtime-2`);
  and `Embeddings` on `GEMINIOpenAIServiceEndpoint` default to `gemini-embedding-2` (was
  `gemini-embedding-001`, which shuts down on 2028-05-14). Upgrading: pass `model=` to keep
  the model you used; the two Gemini embedding models' vectors are not comparable, so embed
  a corpus again before comparing it with `gemini-embedding-2` vectors.
- **`Respond` loses `prompt` and gains `seed`, and old positional calls shift.** OpenAI
  shuts down reusable prompts (`v1/prompts`) on 2026-11-30, so the `prompt` field is gone;
  `seed` (see Added) is appended as the 32nd field. `Respond` still has 32 fields, and its
  positional constructor takes exactly those 32 in declaration order, so a 30- or
  31-argument call is a `MethodError` — but a 32-argument positional call written for 0.25
  still builds and puts each value from the 24th argument on into the next field: the
  old `prompt` slot into `prompt_cache_key`, `prompt_cache_key`'s value into
  `prompt_cache_retention`, `prompt_cache_retention`'s into `safety_identifier`,
  `safety_identifier`'s into `conversation`, and so on, `moderation`'s into `seed`. Where a
  value cannot convert to its new field's type the call fails loudly; where it can — a
  `nothing`, or a string landing in another string field — it is silently misassigned.
  Upgrading: build every `Respond` with keywords, and move a reusable prompt's content into
  `instructions` and `input`.
- **Native OpenAI follows OpenAI's field deprecations.** On `OPENAIServiceEndpoint`, `user`
  throws `ArgumentError` on `Chat` and `Respond` (OpenAI replaces it with
  `safety_identifier` and `prompt_cache_key`), and a `Chat`'s `max_tokens` goes out as
  `max_completion_tokens`, its current name (setting both throws). Upgrading: set
  `safety_identifier` (a stable end-user id, at most 128 characters) or `prompt_cache_key`
  instead of `user`, and one token cap.
- **One OpenAI rule table, by exact model id.** The native endpoint checks `gpt-6-astra`,
  `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6` and `gpt-5.6-sol`/`-terra`/`-luna`, each
  by its exact id or a dated snapshot (`gpt-6-sol-2026-09-14`), before any request; no
  prefix carries one generation's rules to another, and any other model is left to the
  server. GPT-6 Astra and GPT-6.1 Sol take reasoning effort `"low"` to `"max"` on Responses
  and `"low"` to `"xhigh"` on Chat Completions, no sampling controls or log probabilities,
  and tools only through `Respond`; GPT-6 Sol, GPT-6 Luna and GPT-5.6 take `"none"` to
  `"max"` on Responses and `"none"` to `"xhigh"` on Chat Completions (which answered `"max"`
  with HTTP 400 on every one of these models on 2026-10-22, though its reference lists it),
  and Chat tools only at `"none"` — GPT-6 Sol and Luna also sampling controls and log
  probabilities only at `"none"`; none takes `"minimal"`, and the `gpt-5.6` alias takes
  GPT-5.6 Sol's rules. A refusal names the model, the efforts it takes and the next one up,
  or `Respond` for `"max"`. `service_tier` is checked per API: Chat Completions takes `"auto"`, `"default"`,
  `"flex"`, `"scale"`, `"priority"` and `"fast"`, Responses also `"ultrafast"` on GPT-6
  Astra and GPT-6.1 Sol. Every rule model refuses `prompt_cache_retention`, GPT-6.1 Sol now
  too, which 0.25's prefix rule did not reach (see Changed). Upgrading: use `"low"` for
  `"minimal"`; send `"max"` and the Astra and 6.1 Sol tools through `Respond`; set
  `reasoning_effort="none"` for Chat tools, and on GPT-6 Sol and Luna for sampling
  controls; set a cache lifetime with `prompt_cache_options` instead of
  `prompt_cache_retention`.
- **`output_text` returns the answer only.** A Responses message the model labels
  `phase: "commentary"` — an intermediate update, such as a preamble before a tool call —
  is left out, so a turn that stops at its commentary has no text yet; a message without a
  phase still counts. Upgrading: read the commentary from `r.response.output` where you
  need it.
- **Costs are priced at the model that served the call.** `estimated_cost` and
  `cumulative_cost` price a Chat result at the model the provider reports having served
  (`LLMSuccess.model`, see Added) before `self.model`, so a routed id or an alias bills at
  its target's rates; `estimated_cost` of an `ImageSuccess` throws `ArgumentError` unless
  `model=` is given, instead of returning `0.0`, since UniLM does not model image-token
  rates (a `model=` with no row is estimated at `0.0`, with the no-row warning).
  `LLMSuccess` has a fifth field, so its 4-argument positional constructor is gone.
  Upgrading: pass `model=` (with a `pricing=` row of your own for images) to price at
  another row, or add your row under the served name (`result.model`), which
  `cumulative_cost` prices too — an Azure deployment or a vLLM or LM Studio alias may report
  a name other than the one requested; build `LLMSuccess` by keyword.
- **Retired OpenAI surfaces are gone.** `LocalShellTool` (`local_shell`) and
  `ComputerUseTool` (`computer_use`) are removed — OpenAI no longer supports the legacy
  local shell tool, and `computer-use-preview` shut down on 2026-07-23; the `"evals"` file
  purpose throws `ArgumentError` naming the shutdown in `FileUpload`, `upload_file` and
  `create_upload` (OpenAI's Evals API is read-only from 2026-10-31 and shuts down on
  2026-11-30), and `WEBHOOK_EVENTS` no longer lists `eval.run.succeeded`,
  `eval.run.failed` and `eval.run.canceled`, which OpenAI sends until then (match
  `event.type` yourself to keep them); the `o4-mini` and `gpt-4.1-nano` price rows
  are gone (both shut down on 2026-10-23); and the unexported `UniLM.GPTImageContent`
  (text with image URLs as one content value) is removed. Upgrading: use `ShellTool`
  (`shell`) and `ComputerTool` (`computer_tool`); attach images as `ImageAttachment`s, or
  pass an image URL with `input_image` on the Responses API.
- **Image requests take `safety_identifier`, not `user`.** OpenAI deprecates the Images
  API's `user`: `ImageGeneration` has no `user` field, and `ImageGeneration` and `ImageEdit`
  take `safety_identifier` — 1 to 128 characters on OpenAI (an empty or longer one throws
  `ArgumentError` before the request), the request label `safety_identifier` on Gemini.
  Upgrading: rename `user=` to `safety_identifier=`.
- **OpenAI speech runs on a Realtime session.** OpenAI shuts down `tts-1`, `tts-1-hd` and
  `gpt-4o-mini-tts` on 2027-01-06, with no replacement on `/v1/audio/speech`, so `speak`
  on `OPENAIServiceEndpoint` opens a Realtime API session on `gpt-realtime-2.1-mini` and
  returns its 24 kHz 16-bit mono audio as a WAV file (`response_format = "wav"`, the
  default) or as raw PCM (`"pcm"`): `"mp3"`, `"opus"`, `"aac"` and `"flac"` throw, the
  default voice is `marin`, `fable`, `nova` and `onyx` are not Realtime voices and throw,
  and `speed` takes 0.25 to 1.5. It needs `:realtime`, so an endpoint that declares
  `:audio` alone no longer speaks. It still observes the ambient `CancelToken`. Upgrading:
  ask for `"wav"` or `"pcm"` and convert other formats yourself; pick a Realtime voice
  (`alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar`).
- **`translate` runs on a Realtime translation session.** OpenAI shuts
  `/v1/audio/translations` down on 2027-02-26, so `translate(path; language="en")` streams
  a 16-bit PCM WAV file (any rate and channel count, mixed down and resampled to 24 kHz
  mono) through a `v1/realtime/translations` session on `gpt-realtime-translate` into
  `language` and returns the translated text; any other file throws `ArgumentError` naming
  the conversion, the `translate(::TranscriptionRequest)` method is gone, and the call needs
  `:realtime`. Speech already in the target language can come back as empty text, and a
  short clip's translation varies from call to call. It
  observes the ambient `CancelToken`. Upgrading: call `translate(path; language)` with a
  WAV file.
- **`AudioFailure.status` may be `nothing`**: a failure a Realtime session reports as an
  event carries no HTTP status (its `response` is the event's JSON), an `error` event the
  provider sent just before closing the session included. A session that closes before its
  result is an `AudioCallError` whose `error` names the close's status and reason, with the
  `HTTP.WebSockets.WebSocketError` in `cause`. Upgrading: handle `status === nothing` where
  you read it.
- **Native Anthropic speaks API version `2026-10-21`, which removed sampling.** Requests
  send `anthropic-version: 2026-10-21`, and `temperature` or `top_p` on a Claude request
  throws `ArgumentError`: the API answers either with HTTP 400 on every model, even at its
  default value. Upgrading: remove `temperature` and `top_p` from Claude requests.
- **Claude rules match the exact model id.** A row applies to its id, or the id with a
  `-YYYYMMDD` snapshot suffix, only: `claude-sonnet-5-5` no longer inherits
  `claude-sonnet-5`'s rules (it has its own, see Added), and an id no row names gets only
  the rules every Claude model shares. The rows of the retired Claude Opus 4.1, Opus 4,
  Sonnet 4 and Claude 3 models are gone. Upgrading: nothing for current models; a request
  to a retired one goes to the API, which refuses it.
- **A system message after a Claude reply keeps the prompt append-only.** On Claude Fable
  5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5 and any Claude id without a rule row, a system
  message after an assistant turn that cannot stay in place goes right after the next user
  turn instead of joining the top-level `system` prompt: these models refuse an earlier
  turn's thinking blocks once that prompt changes. On every Claude model, a conversation in
  which no user turn follows such a system message — counting a refusal or an empty reply,
  which are left out of the request, as no turn — throws `ArgumentError` naming what to push;
  it was sent as a response prefill on the models that take one. Upgrading: push a user
  message, or the tool results, after a mid-conversation system message before sending.
- **`ANTHROPICServiceEndpoint` is a struct with options** (`betas`, `user_profile_id`,
  `workspace_id`, `inference_geo`; see Added). The bare type keeps working everywhere it
  worked, as the endpoint with no options. Upgrading: a method of your own defined on
  `::Type{ANTHROPICServiceEndpoint}` covers the bare type only; define it on
  `::ANTHROPICServiceEndpoint` too for endpoints built with options.
- **Claude requests that set a `reasoning_effort` level ask for summarized thinking**
  (`thinking.display = "summarized"`; see Added), so their request bodies change.
  Upgrading: a recording of such a request made by 0.25 does not replay
  (`ReplayMissError`); record it again (`mode = :record_missing`).
- **Gemini refuses the sampling it ignores.** Google ignores `temperature` and `top_p` on
  Gemini 3.6 Flash, 3.5 Flash-Lite and every later model, so native Chat, `respond`
  (Interactions) and `GEMINIOpenAIServiceEndpoint` throw `ArgumentError` for them on every
  model but the earlier ones that honour them (`gemini-2.5-*`, `gemini-3-flash-preview`,
  `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, Gemma 4 and the earlier image models),
  unknown ids and the `-latest` aliases included. Upgrading: drop `temperature` and `top_p`
  on current Gemini models.
- **Gemini rules match exact model ids.** Thinking levels come from rows by exact id: new
  rows for Gemini 4 Argon (`"low"`, `"medium"`, `"high"`), `gemini-robotics-er-2-preview`
  and the `-latest` aliases, which take their targets' levels (so `"minimal"` throws on
  `gemini-pro-latest` and `gemini-flash-latest`); the `gemini-3-pro-preview` row is gone,
  and an id no row names is left to the server. From Gemini 3.6 Flash and 3.5 Flash-Lite
  on, a native Chat that ends with an assistant turn (a prefilled answer) throws, since the
  API answers it with HTTP 400. Upgrading: use `"low"` where `"minimal"` is refused, and end
  a Gemini conversation with a user message.
- **`GEMINIOpenAIServiceEndpoint` takes only Google's service tiers.** Its `service_tier`
  must be `"standard"`, `"flex"` or `"priority"`, the values Google's compatibility page
  names; any other value throws `ArgumentError` before the request, where 0.25 sent it
  unchecked. Upgrading: use one of the three on that endpoint in place of an OpenAI tier
  such as `"auto"` or `"default"`.
- **Gemini Interactions joins an answer's consecutive steps with no separator.** The
  consecutive `model_output` steps of one answer — the hops of a long decode, thoughts
  between them or not — and the text parts inside a step are one part, joined as Google's
  SDK joins them, so `output_text` reads `"…42 lines long"` where it read
  `"…4\n2 lines long"`; a function call or result step, a hosted tool's step or a
  `user_input` step still starts a new message. Upgrading: none, unless you split
  `output_text` on its newlines.
- **Ollama's Responses API refuses every `input_file` part**: Ollama's OpenAI-compatibility
  page lists no file inputs for `/v1/responses`, so `respond` on an `OllamaEndpoint` throws
  `ArgumentError` before any request for an `input_file` part in any form —
  `input_file(document)` and `input_file(; url, id, file_data)` alike — which 0.25 sent
  unchecked. Upgrading: put the document's text in an `input_text` part, or send the file
  to `OPENAIServiceEndpoint`.

### Added
- **PDF documents and an image's detail level in user messages.**
  `DocumentAttachment("report.pdf")` (or bytes with `filename=`; recognised by its `%PDF-`
  header) attaches a PDF, and `ImageAttachment(…; detail=…)` sets the level of detail an
  image is read at (`"auto"`, `"low"`, `"high"`, `"original"`); an image without one is sent
  exactly as before. The OpenAI Chat Completions wire (OpenAI, Azure, the Gemini
  OpenAI-compatible endpoint, DeepSeek, generic endpoints) sends a document as a `file`
  part and the level as `image_url.detail`, and `input_file(document)` makes the document a
  Responses `input_file` part. `OllamaEndpoint`, System One requests and embeddings refuse
  both before any request.
- **Native Anthropic and Gemini send attachments.** Native Anthropic sends an
  `ImageAttachment` and a `DocumentAttachment` as base64 `image` and `document` blocks
  before the message's text, and refuses a sound clip and an image's detail level, which
  the Messages API has no field for. Native Gemini sends images, sound clips and PDF
  documents as `inlineData` parts ahead of the text, and an image's detail as its per-part
  media resolution (`"low"`, `"high"`; `"auto"` sends none; `"original"`, or any level on
  the Gemini 2.5 and Gemma 4 models, throws); `respond` on `GEMINIServiceEndpoint` takes the
  same attachments in an `InputMessage`'s content, sent as Interactions `image`, `audio`
  and `document` parts, as are `input_file(document)` and a data-URL `input_image`.
- **OpenAI Chat audio replies, kept and priced.** A `Chat` with
  `modalities=["text", "audio"]` and an `audio` voice and format (for example on
  `gpt-audio-1.5`) gets the reply's audio on the new `Message.audio_output`, an
  `AudioOutput` — its bytes, the format their signature shows, OpenAI's `id`, `expires_at`
  (`nothing` when the reply names none, or when a stream closed after the turn's finish,
  before the update that carries it) and the `transcript` — streamed or not; an
  `expires_at` that is not an integer fails a call that is not streamed, while a stream
  drops that update (counted in `sse_dropped`, with one warning) and keeps the turn. The
  reply's `content` is the transcript, which a stream also passes to the callback as text.
  The next request sends the turn back to OpenAI as its content and `"audio": {"id": …}`;
  every other endpoint receives the content alone. `TokenUsage` gains
  `prompt_audio_tokens` and `completion_audio_tokens` (OpenAI's `audio_tokens` details,
  streamed and not; `prompt_audio_tokens` also counts native Gemini's and Interactions'
  audio input), and `estimated_cost` and `cumulative_cost` bill `gpt-audio-1.5` audio at
  \$32 / \$64 per 1M tokens and its text at \$2.50 / \$10.
- **Usage reports cache writes, the billed tier, the region and the served model.**
  `TokenUsage` gains `cache_write_tokens`, `cache_write_extended_tokens` (the writes at the
  extended retention: OpenAI `"12h"`, Anthropic `"1h"`), `service_tier` and
  `inference_geo`, filled by OpenAI (Chat streamed and not, Responses, Decisions),
  Anthropic (cache writes and 1-hour writes, the tier — `"fast"` in fast mode — and the
  region, streamed too) and Gemini (the tier); the 5-count positional form still builds.
  `LLMSuccess.model` is the model the provider reports having served: the OpenAI wire's and
  DeepSeek's `model`, Anthropic's, and Gemini's `modelVersion`. For custom backends,
  `decode_response` may return a `model`, and a stream handler may set `state.model`.
- **OpenAI prompt caching for 12 hours, per breakpoint, and on Decisions.**
  `PromptCacheOptions` takes a `ttl` of `"5m"` or `"1h"` (Anthropic's) or `"30m"` or `"12h"`
  (OpenAI's), and each endpoint refuses the ones it does not take: OpenAI's Responses API
  and Decisions API keep a prefix 12 hours, Chat Completions 30 minutes only.
  `input_text(text; cache_breakpoint=true, cache_ttl="12h")` gives one breakpoint its own
  lifetime. `ask` on OpenAI's Decisions API takes `prompt_cache_options` (`mode`, `ttl`),
  left out of recordings as `prompt_cache_key` is; another ttl, `prewarm` or
  `comparison_response_id` throws before the request.
- **`InputMessage(…; phase)`** labels a replayed assistant message `"commentary"` or
  `"final_answer"`, as OpenAI asks callers to resend it; another value, or a phase on
  another role, throws, and Gemini Interactions and `OllamaEndpoint` refuse it.
- **OpenAI price rows** for `gpt-6.1-sol` (\$2 / \$0.10 cached / \$10), `chat-latest`
  (\$5 / \$0.50 / \$30), `gpt-5.5-pro` and `gpt-5.4-pro` (\$30 / \$180, no cached rate),
  `text-embedding-ada-002` (\$0.10) and `gpt-audio-1.5` (text \$2.50 / \$10); the OpenAI,
  Anthropic and Gemini rows were verified on 2026-10-22.
- **Claude Sonnet 5.5 and Haiku 5.5.** Sonnet 5.5 sends `reasoning_effort="none"` as
  `thinking = {type: "between_tools"}`, refuses a forced `tool_choice` and a prefill, and is
  priced at \$2 / \$0.10 cached / \$10; Haiku 5.5 sends `"none"` as thinking disabled, takes
  a forced `tool_choice`, refuses a prefill, and is priced at \$0.10 / \$0.01 / \$0.50, every
  rate 5× for a prompt over 100,000 tokens. `push!` accepts a system message later in a
  conversation (see Breaking for where native Anthropic places it).
- **Anthropic prompt caching, summarized thinking, fast mode and endpoint options.** On
  `ANTHROPICServiceEndpoint`, `prompt_cache_options` becomes a top-level `cache_control`
  (automatic caching) with `ttl` `"5m"` or `"1h"` and `mode` `"implicit"` or unset —
  `"explicit"`, `prewarm=true`, `comparison_response_id` and the OpenAI ttls throw; a
  `reasoning_effort` level turns adaptive thinking on with `display: "summarized"`, so
  `reasoning_text` holds the summarized thinking on the current models; and
  `service_tier="fast"` sends `speed: "fast"` on Claude Opus 5.5 (on Opus 5 and Opus 4.8
  only through an endpoint whose `betas` hold `"fast-mode-2026-02-01"`; refused on other
  models). `ANTHROPICServiceEndpoint(; betas, user_profile_id, workspace_id,
  inference_geo)` sends the betas comma-joined in `anthropic-beta`, the ids in
  `anthropic-user-profile-id` and `anthropic-workspace-id`, and `inference_geo` (`"global"`
  or `"us"`, refused on the Claude 4.5 models) in the body; a blank value, a control
  character or a comma in a beta name throws. `estimated_cost` bills Anthropic's cache
  writes at 1.25× (5-minute) or 2× (1-hour) input, every rate 2× in fast mode and 1.1× for
  US-only inference on Claude 4.6 and later (the 4.5 models take no `inference_geo` and
  bill the standard rates), and a reply at Anthropic's Priority Tier at the list rates
  (Priority Tier draws down a contracted capacity commitment).
- **Gemini image generation and editing (Nano Banana).** `generate_image` and `edit_image`
  with `service=GEMINIServiceEndpoint` call `generateContent` on `gemini-nano-banana-2.1`
  with image output (`GEMINIServiceEndpoint` declares `:images` and `:image_edits`); the new
  `aspect_ratio` and `resolution` fields size the image, and the other provider's fields
  throw before the request. The image parts come back as the result's images (JPEG), a
  reply with no image as an `ImageCallError` naming the finish reason, and `token_usage`
  reads `usageMetadata`; image cost is not modelled.
- **Gemini speech.** `speak(text; service=GEMINIServiceEndpoint)` reads text with Gemini 3.8
  Flash TTS (`gemini-3.8-flash-tts`, one of its 30 prebuilt voices, `instructions` as the
  speech style) and returns 24 kHz WAV or PCM, as on OpenAI; a candidate that did not
  finish is an `AudioFailure`. `GEMINIOpenAIServiceEndpoint` has no speech route: `speak`
  and `SpeechRequest` there throw `ArgumentError` naming `GEMINIServiceEndpoint`.
- **Gemini service tiers, in the request and in the cost.** `service_tier` `"standard"`,
  `"flex"` or `"priority"` is sent as `serviceTier` on native Chat, as `service_tier` on
  `GEMINIOpenAIServiceEndpoint` (see Breaking) and on `respond`, which also takes
  `"deferred"`; any other value is an `ArgumentError`. The tier that billed the call is
  `usage.service_tier`: the `x-gemini-service-tier` header first — Google names it as the
  sign of a downgrade, and Gemini 4 Argon names the tier nowhere else — else
  `usageMetadata.serviceTier`, or an interaction's `service_tier`. The compatible endpoint's
  replies name no tier, so a Chat result there records the tier its request asked for:
  `"flex"` priced at Flex, `"priority"` at Priority, an upper bound, since Google bills a
  downgraded Priority request at Standard and that endpoint does not report it.
  `estimated_cost` bills input and output at half the standard rates on Flex and 1.8 times
  on Priority, and a cache hit at the same multiple except where the pricing page lists
  another price (Flex's standard rate on 3.1 Pro Preview, 3 Flash Preview and the 2.5
  models; \$0.02 on Flex and \$0.05 on Priority for 3.5 Flash-Lite); `"deferred"`,
  unpriced, is estimated at the standard rates with a warning. A Flex request can queue for
  minutes: give it a `RequestConfig` with longer bounds.
- **Gemini `seed`, `store` and safety labels.** Native Chat sends `seed` as
  `generationConfig.seed` and `store` as the request's `store` (they were refused);
  `respond` sends `Respond.seed` as `generation_config.seed` and `safety_identifier` as the
  request label `safety_identifier`. A seed outside the int32 range is an `ArgumentError`
  on both. OpenAI's Responses API has no seed, so the OpenAI-wire Responses endpoints and
  `OllamaEndpoint` refuse `Respond.seed` before any request.
- **Long Gemini generations continue inside the call.** A Gemini 4 Argon reply that stops
  at its per-request cap with `CONTINUATION` and a `continuationToken` is continued by
  `chatrequest!` — the same request again with the latest token, in as many requests as
  the output budget (`max_completion_tokens`, else `max_tokens`) needs at Google's
  per-request cap (32,768 output tokens, 65,536 on a stream), plus one, and at least 8; a
  `CONTINUATION` reply that produced no output token ends the turn — and the result is one
  assistant turn whose usage is the requests' summed, priced at the last reply's tier; a
  streamed turn streams every reply into the same callback. A request that fails, or a
  stop or cancel before the turn's last reply, ends the call with nothing committed, and
  the replies already received are added to `cumulative_cost`.
  Proven against a local mock; a live CONTINUATION reply was not observed. Custom backends
  read the token from `StreamState.continuation_token`.
- **Gemini 4 calls wait longer by default.** `chatrequest!` and `respond` to a
  `gemini-4-…` model use 1200 s for `request_timeout`, `stream_idle_timeout` and
  `total_deadline` while no `RequestConfig` is set in any channel: Google ends a
  non-streamed Argon call after 10 minutes and a stream after 20, and advises a 20-minute
  read timeout.
- **Gemini 4 Argon and current Gemini rows.** Price rows for `gemini-4-argon` (\$2 / \$0.10
  cached / \$10, doubling on 2026-12-01), `gemini-3.1-pro-preview` and
  `gemini-3.1-pro-preview-customtools` (\$2 / \$0.20 / \$12), `gemini-2.5-pro` (\$1.25 /
  \$0.125 / \$10), `gemini-3-flash-preview` (\$0.50 / \$0.05 / \$3.00) and
  `gemini-robotics-er-2-preview` (\$1.00 / \$0.10 / \$5.00, doubling on 2027-01-01); a
  prompt over 200K tokens on 3.1 Pro Preview, its custom-tools endpoint and 2.5 Pro bills
  at the page's long-context rates, the whole request (twice the input and cache-hit
  rates, 1.5 times the output rate); `respond` on Argon takes a `max_output_tokens` above
  65,536 only with `background=true` (the API answers HTTP 400 otherwise). A Gemini result's `model` is the reply's
  `modelVersion`; an interaction's `errors` reach `response.error` (the first; all in
  `raw`); a filtered turn's `refusal_message` is Google's `finishMessage`; and
  `gemini-embedding-2` takes the same task prompts as EmbeddingGemma (`task`, optional).
- **Provider model listings.** `list_models` on `OPENAIServiceEndpoint`,
  `ANTHROPICServiceEndpoint` and `GEMINIServiceEndpoint` lists the models the provider
  serves the key, every page included, as `ProviderModel`s: id, display name, creation
  time, token limits, the provider's capabilities (Anthropic's thinking types and effort
  levels; Gemini's generation methods and thinking) and lifecycle dates (OpenAI's
  `shutdown_date`; Anthropic's `deprecated_at` and `retires_at`). The results are
  `ModelListSuccess` / `ModelListFailure` / `ModelListCallError`, and the three endpoints
  declare `:models`. An Anthropic listing sends the key, `anthropic-version` and the
  endpoint's `anthropic-workspace-id`, the headers Anthropic's List Models reference names
  — not the endpoint's user profile, nor its betas (Anthropic marks `anthropic-beta`
  deprecated on `GET /v1/models`).
- **`WEBHOOK_EVENTS` lists the current events**: `live.transport.incoming`,
  `live.call.incoming` (deprecated by OpenAI in its favour, still sent to existing
  subscriptions), `safety.warning_issued`, `safety.deactivation_issued`,
  `safety.alert.created` and `safety.org_alert.created`.

### Changed
- **OpenAI estimates include OpenAI's surcharges.** `estimated_cost` bills cache writes at
  1.25× input (`"30m"`) or 2.5× (`"12h"`) on GPT-6 and GPT-5.6, a prompt over 272K input
  tokens at 2× the input and cache rates and 1.5× the output rate, and the billed tier at
  the pricing page's multiple for the model (for example Flex 0.5×, Fast 2× on GPT-6 and
  GPT-5.6, Ultrafast 6× on Astra and 6.1 Sol); a tier with no listed price is estimated at
  the standard rates, with one warning per model and tier.
- **Gemini results are priced as served.** A routed id or a `-latest` alias bills at the
  rates of the model that served it (`gemini-3.5-flash`, routed to `gemini-3.6-flash`, at
  \$0.75 / \$3.75 per 1M where 0.25 billed \$1.50 / \$9.00), and a tool-use prompt counts
  as prompt tokens, at the input rate, on native Chat and Interactions alike — native Chat
  left it out, and Interactions billed it at the output rate.
- **Audio tokens are priced at an audio rate, or estimated at `0.0`.** `estimated_cost`
  bills counted audio tokens only at a rate stated for audio: OpenAI's audio-token rates
  for `gpt-audio-1.5`, and for Gemini audio input on native Chat (streamed and not) and
  Interactions the input price Google's pricing page states for audio, at the tier's and a
  long prompt's multiples — the model's own audio price where the page lists one apart (3.1
  Flash-Lite \$0.50 per 1M, 3 Flash Preview and 2.5 Flash \$1.00, 2.5 Flash-Lite \$0.30),
  else its input price, which covers every modality when it names none or names audio among
  them (a cache hit on such a model bills at its cached rate). Other counted audio — a
  model the page prices without audio or does not list, another endpoint (Azure, DeepSeek,
  generic, custom), Gemini audio output, audio in a call that also hit the cache on one of
  the four models with an audio price of their own — is estimated at `0.0`, with one
  warning per model and endpoint, like an unpriced model. `GEMINIOpenAIServiceEndpoint`
  counts no audio apart, so its audio input is priced at the text rate, a lower bound on
  the four models with an audio price of their own, for which a request carrying a sound
  clip warns once per model; send audio through `GEMINIServiceEndpoint` for an exact
  estimate.
- **An embeddings reply without usage says its estimate is `0.0`.** `estimated_cost` of an
  embeddings result whose reply reported no usage — `GEMINIOpenAIServiceEndpoint`'s never
  does — warns once per model, as for a model without a price row.
- **Spend and quota 429s are not retried.** A 429 whose `error.code` is
  `credit_balance_exhausted`, `organization_spend_limit_exceeded`,
  `project_spend_limit_exceeded` or `organization_usage_limit_exceeded`, or whose
  `error.type` (or, in older replies, `error.code`) is `insufficient_quota`, is returned at
  once, streamed or not, and ends a `poll_batch` or `poll_file_batch` poll: no retry
  restores spent credit or a reached limit. So is a Gemini Interactions 429 whose
  `error.code` is `quota_exceeded`, its daily quota; Gemini's rate limits
  (`rate_limit_exceeded`, generateContent's `RESOURCE_EXHAUSTED`) are still retried.
- **`prompt_cache_retention` and `prompt_cache_options` are no longer refused together.**
  The Responses reference lists the two fields independently and documents
  `prompt_cache_options` for gpt-5.6 and later. The GPT-6 and GPT-5.6 rule models refuse
  `prompt_cache_retention`, since `prompt_cache_options` sets their cache lifetime (GPT-6.1
  Sol's refusal is new: see Breaking); on older models UniLM sends both unchecked and leaves
  them to the server (the reference takes only `"24h"` retention on gpt-5.5 and
  gpt-5.5-pro).
- **Decisions usage reports cache writes and the tier.** `token_usage` of an OpenAI
  Decisions answer counts `cache_write_tokens` (the 12-hour ones in
  `cache_write_extended_tokens`) and keeps a reported `service_tier`; the price is
  unchanged — every input token at the input rate, twice it above 272K tokens.
- **UniLM depends on the `Dates` standard library.**

### Fixed
- **A request no longer goes out on a connection its server has closed.** HTTP.jl
  before 2.11.2 reused an idle keep-alive connection without checking it, so after a
  server closed one (on shutdown, or after its own idle timeout) the next request
  could be written on it and fail without reaching the server — a broken pipe, a
  reset or an unexpected end of stream. HTTP.jl does not retry a POST, so with the
  default three attempts the call spent one of them; with `max_attempts = 1` it
  failed. UniLM now requires HTTP.jl 2.11.2, which checks an idle connection before
  reusing it.
- **A tool call cut off inside an escape is dropped, not a failed turn.** UniLM requires
  JSON.jl 1.12, whose `isvalidjson` agrees with `parse` on an invalid escape (`\x`): a turn
  cut at the token limit inside such an escape kept a call whose arguments could not be
  parsed, and failed as a whole instead of losing only the partial call.
- **Claude turns go back as they came.** A thinking-only assistant turn (such as one cut
  at `max_tokens` before any text) is sent back instead of dropped, and an empty text block
  (a streamed block that got no delta) is never sent back — a turn holding nothing else is
  left out.
- **Multi-input Gemini embeddings work.** `GEMINIOpenAIServiceEndpoint` leaves the index out
  of its first reply row, and every request with more than one input failed with "row
  index nothing is not an integer"; that row is read as row 0.
- **Gemini Interactions takes `InputMessage`s.** `respond` sent an `InputMessage` as a
  `{role, content}` turn, which the Interactions API answers with HTTP 400 ("use
  step_list input format instead of turn_list"); a user message is now a `user_input`
  step and an assistant message a `model_output` step, their content Interactions parts. A
  system or developer message (use `instructions`), a message `phase`, a media part given
  by web URL or OpenAI file id, and a cache breakpoint are `ArgumentError`s before any
  request.
- **`cumulative_cost` counts what an unfinished stream was billed.** A streamed Chat turn
  that ends without committing — stopped, cancelled, timed out, cut off, or ended by an
  exception from its `callback` or `on_tool_call` — adds the usage its provider had
  reported by then, a lower bound of the bill: Anthropic reports a request's input when
  its stream starts, Gemini the usage so far on every chunk, and a reply whose final usage
  had arrived counts whole. A request stopped before any usage arrived adds nothing, though
  the provider bills it (OpenAI reports usage only after the finish reason), nor does an
  attempt that was retried or a stream the provider ended with an error event. 0.25 added
  nothing for such a turn.
- **Streamed `GEMINIOpenAIServiceEndpoint` turns are costed.** A stream with no
  `stream_options` asks for `include_usage`, as Google's compatibility page does, so its
  usage arrives and is priced; without it the turn reported no usage and accrued `\$0`.
- **`realtime_receive` keeps its idle bound while a send is parked.** Its guard closed the
  WebSocket, which waits for the send lock that a `realtime_send` parked on a peer that
  stopped reading holds, so the read could outlast `stream_idle_timeout`; it now ends the
  read and the connection without that lock, and the parked send throws too — except on a
  session over the caller's own stream (`HTTP.WebSockets.open(io)`), whose parked send waits
  until the caller closes that stream.
- **Attachment refusals say what to do.** An attachment built from a `data:` URL is
  refused as a URL, saying to pass the bytes it holds, where 0.25 tried to open it as a
  file (`SystemError`); and an `EmbeddingInput` given another kind of attachment (a
  document, or a clip as its image) throws `ArgumentError` naming the kinds it holds, where
  it was a `MethodError`.

### CI
- **The release gate runs the hosted live suites.** A `hosted-providers` job, on
  `release/**` pushes and manual dispatch, after the JET gate and the keyless test suite
  pass, runs every OpenAI, Anthropic and Gemini live suite (`test/live_hosted.jl`) against
  the real APIs with the repository's keys, and fails when a key or flag is missing, a
  suite is missing, or a suite ran no test. Among them, a drift check reads the providers'
  free model listings and fails when a listed current model of a family UniLM documents —
  on Gemini, every listed text model UniLM has thinking levels for — has no price row or
  rule row (a model whose provider publishes no price, such as `claude-mythos-preview`, is
  exempt by name); when a model UniLM names in its defaults, its live suites or the Julia
  code blocks of its docs is not listed (an Anthropic alias counts as its dated snapshot)
  or is retired, from its shutdown day on; or when a Claude rule disagrees with the listed
  thinking types and effort levels. It logs the listed ids no family and no known kind
  covers, so a new generation shows in the gate's log. Routine CI and the gate's other
  jobs stay keyless.
- **The local-models job runs Ollama 0.40.8**, pinned by the SHA-256 Ollama publishes for
  its Linux build.

## 0.25.0

### Breaking
- **An EmbeddingGemma request needs a `task`.** EmbeddingGemma — v1 and v2, any
  model whose name contains `embeddinggemma`, on any service — reads a short
  prompt that says what a text is for: Ollama does not add it, and the
  OpenAI-compatible request has no field for it.
  `Embeddings(…; model = "embeddinggemma-2:740m-q8_0")` without `task` is an
  `ArgumentError` before the request, naming the tasks;
  `task = :none` sends the text as it is. A task other than `:none` for a model
  whose prompts UniLM does not know is refused, a `title` goes only with
  `task = :document`, EmbeddingGemma's `dimensions` must be 768, 512, 256 or 128,
  and `dimensions` of 0 or less is refused for every model. A `Vector{String}`
  input is copied, so changing your vector afterwards no longer changes the
  request. Upgrading: pass `task = :search_query` with a query and
  `task = :document` with the documents (see Added), or `task = :none` to send
  what 0.24 sent; a renamed copy of EmbeddingGemma is not recognised — send it
  `task = :none` with the prompt written into the text.
- **An embeddings reply that holds no real vector is a malformed reply** for every
  provider: an empty or all-zero vector, a value that is not a finite number (a
  JSON `true` included), a row without a vector, or a row index that is not an
  integer is an `EmbeddingCallError` naming the row, and no row of the reply is
  stored. Ollama has answered all-zero vectors with HTTP 200 under load
  ([ollama#17878](https://github.com/ollama/ollama/issues/17878)); a zero vector
  compared and ranked like a real one. Upgrading: a mock that answers zero
  placeholder vectors must answer non-zero ones.
- **Ollama System One refuses locally what Ollama's route checks or documents; a
  model's own limits come back as its 400.** Before the request, `ask` on an
  `OllamaEndpoint` now refuses what Ollama's route refuses in its shared checks or
  documents as its maximum (Ollama 0.40.6): more than 64 questions; a Choice of more
  than 255 options (was 26; one option is now sent); a Score of fewer than 2 or more
  than 10 levels; a blank text state without images; Box questions. It also refuses
  what Ollama's API reference does not list, though some models read it:
  `instructions` that are not a string, an object or an array, and descriptions,
  levels and Noul criteria that are not strings — plus one refusal of UniLM's own:
  blank (whitespace-only) `instructions`, which Ollama would read as none.
  `instructions` are optional on the route. A model that takes less answers with its
  own HTTP 400, a `SystemOneFailure` carrying Ollama's message. Measured on Ollama
  0.40.6 (Apple M4 Max, 21 October 2026): tev1 takes at most 24 options and answers
  27 with `criteria must contain 2–26 candidates`; nimble and clef-flash take 26;
  kev, strands-decider, d1 and d1-omni answered 27; a Choice of one option was
  answered by `kev:0.8b-q8_0`, `kev:9b-q4_K_M`, `d1-omni:600m` and `d1:3b` (the
  llama.cpp path), while `tev1:0.8b`, `tev1:4b`, `nimble:9b` and `clef-flash:9b`
  answered 400 `criteria must contain 2–26 candidates` and `strands-decider:2b` 400
  `criteria must contain 2–255 candidates`; and asked a question without
  `instructions`, `tev1:4b`, `nimble:9b` and `strands-decider:2b` answered 400
  `instructions must be a nonempty string, object, or array`, `kev:0.8b-q8_0` 400
  `questions.<name>: "instructions" must be provided`, and `clef-flash:9b` answered
  the question: pass `instructions` with local models. Upgrading: a request the
  wrapper used to refuse — one option, 27 to 255 options, a question without
  `instructions` — now reaches the server; handle its 400 as a `SystemOneFailure`,
  or keep to 2–24 options and give every question `instructions`.
- **New fields.** `OllamaEndpoint.runner` (before `options`);
  `OllamaModel.runner::Union{Symbol,Nothing}` and `.format::Union{Symbol,Nothing}`
  (before `capabilities`); `OllamaModelInfo.runner` and `.format` (before `raw`);
  `OllamaRunningModel.runner` (before `raw`); `ResponseSuccess.service` and
  `FIMSuccess.service`, the endpoint that answered (last; `nothing` for a result
  built by hand); `Embeddings.task`, and `Embeddings.input` also holds
  `EmbeddingInput`s. A detail an Ollama listing reports as `""` — some MLX models
  leave `family`, `parameter_size` and `quantization` empty — reads as `nothing`.
  Upgrading: positional constructors take the new fields; build `ResponseSuccess`
  and `FIMSuccess` by keyword; compare a runner or a format with a `Symbol`
  (`m.runner === :mlx`, `m.format === :gguf`).
- **`TranscriptionRequest.model` defaults to `nothing`,** resolved per service
  when the request is sent: `gpt-transcribe` on OpenAI, while Ollama has no
  default and a request to it names a model. Upgrading: a default request's
  `model` field is `nothing`, not `"gpt-transcribe"`.
- **`:transcription` joins `provider_capabilities`** of `OPENAIServiceEndpoint`
  and of an `OllamaEndpoint` (see Added), so code that compares those sets with a
  fixed set of its own no longer matches them. Upgrading: test one capability with
  `has_capability(s, cap)`, or add `:transcription` to the expected set.
- **`get_response`, `cancel_response`, `delete_response` and `list_input_items`
  refuse an `OllamaEndpoint`** with an `ArgumentError` before any request, naming
  what to do instead: Ollama's Responses API stores no response, so each failed
  at the server. `compact_response` stays allowed on Ollama with a `model` named:
  Ollama has no default model, so a call there that names none is now the
  `ArgumentError` a `Chat` gives, before any request (it sent OpenAI's default
  model, which Ollama answered with a 404).
- **`load_model` asks `/api/show` first:** two requests, both bounded by one
  `total_deadline`. A chat model loads with an empty chat request as before; an
  embedding or decision model, which Ollama's chat route refuses, loads with an
  empty embeddings request. Upgrading: a mock Ollama server must answer
  `/api/show` for `load_model`.
- **Refusal messages changed.** An image in an Ollama System One request is sent
  now, and `transcribe` on an `OllamaEndpoint` goes through (see Added), so their
  old refusals are gone; the new refusals — an image out of place, blank
  `instructions`, what Ollama's transcription cannot carry, an embeddings item
  holding both an image and a sound clip for Ollama, a `compact_response` to
  Ollama without a model — name the rule and the fix. Upgrading: match an
  `ArgumentError` by its type, or a fragment with `occursin`, not a whole message.
- **Recordings: media sent to Ollama are keyed and written by digest.** The
  images and sound clips of an embeddings request to an `OllamaEndpoint` and the
  images of an Ollama System One request read `data:<mime>;sha256,<hex>` (the
  SHA-256 of their base64 text) in a recording's key and file, never their bytes.
  An EmbeddingGemma 2 request carries its window options (see Added), which are
  part of its key. Upgrading: a recording of an EmbeddingGemma 2 request made by
  0.24 does not replay (`ReplayMissError`), and one of an EmbeddingGemma request
  without a task cannot be sent at all: record them again with a task
  (`mode = :record_missing`).

### Added
- **Task prompts for EmbeddingGemma.** `Embeddings(…; task = …)` writes the model
  card's prompt in front of each text: `:search_query`, `:question_answering`,
  `:fact_checking` and `:code_retrieval` for queries (`"task: search result |
  query: "` and the like), `:document` for the corpus (`"title: <title, else
  none> | text: "`), `:classification`, `:clustering`, `:sentence_similarity`, and
  `:none` for the text as it is. `EmbeddingInput(text; title)` gives a document
  its title; an empty or whitespace-only title is none, sent as the model card's
  `title: none`. On SciFact (BEIR: 300 claims, 5,183 abstracts; Ollama 0.40.6,
  Apple M4 Max, 21 October 2026) the prompts raised nDCG@10 on every build
  measured:
  `embeddinggemma-2:740m-mxfp8` 0.8515 against 0.8245 without them (+0.027, 95%
  interval +0.009 to +0.046), `embeddinggemma-2:270m-q8_0-text` 0.8536 against
  0.8207 (+0.033, +0.014 to +0.053), `embeddinggemma` 0.7921 against 0.7727
  (+0.019, +0.003 to +0.037).
- **Images and sound in embeddings, through Ollama.** An `ImageAttachment` or
  `AudioAttachment` is an item of its own, and `EmbeddingInput(text, image)` — or
  `EmbeddingInput(; text, title, image, audio)` — embeds a text with its picture
  or its sound clip as one item, for the models that see or hear. Measured on
  Ollama 0.40.6 (Apple M4 Max, 21 October 2026) with `embeddinggemma-2:740m-q8_0`
  and `:740m-mxfp8`; the 440m (text and images) and 570m (text and audio) builds
  are described from their tags, untested. Texts, images and sounds share one
  space: on `740m-q8_0` a red circle scored 0.754 against "a red circle" and 0.633
  against "a blue square", a spoken "banana" 0.792 against "banana" and 0.626
  against "apple"; "a red circle" sent with its picture as one item gave a vector
  of its own, cosine 0.870 with the text alone and 0.903 with the picture alone
  (both 740m builds: 0.845–0.958 with each part, pictures and clips alike). An
  item holds an image or a clip, not both: Ollama passes an item's media to the
  model in a random order, and ten requests of one item holding both gave
  `embeddinggemma-2:740m-q8_0` two different vectors, so such an item is refused
  before any request — send the image and the audio as separate items. Every
  other service embeds text only and refuses media before the request.
- **EmbeddingGemma 2 gets its trained 8,192-token window.** Ollama reports a
  262,144-token context for its llama.cpp builds (the MLX build, `740m-mxfp8`,
  loads at 8,192), and at its default batch the llama.cpp builds measured refused
  long inputs with HTTP 413 (`740m-q8_0` anything over about 2,037 tokens). An
  embeddings request — and `load_model` — for an EmbeddingGemma 2 model now
  carries `num_ctx` and `num_batch` of 8192 unless the endpoint sets either: on
  Ollama 0.40.6 every build measured (`740m-q8_0`, `740m-mxfp8`,
  `270m-q8_0-text`) embedded 5,991 tokens whole and refused 8,331 with HTTP 400.
- **Pictures for local decision models.** `ask` on an `OllamaEndpoint` sends the
  images of the state to a vision decision model — measured on Ollama 0.40.6:
  `clef-flash:9b`, `d1:3b` and `d1-omni:600m`; other decision models whose
  `/api/show` capabilities include vision take the same request, untested. The
  state is an `ImageAttachment`, or a vector of images followed by texts
  (`[photo, "Is the item damaged?"]`, a blank text allowed); the images go in
  Ollama's `images`, before the text, shared by every question. Refused before
  the request: an image after a text, inside a JSON object or array, beside values
  that are not texts, or inside a question. A model without vision answers with
  its own 400, in its own words: `image inputs are not supported by this decision
  model` from `tev1:0.8b`, `registry.ollama.ai/library/kev:0.8b-q8_0 does not
  support vision` from `kev:0.8b-q8_0`.
- **`OllamaEndpoint(runner = :mlx | :llamacpp | :ggml)`** pins the runner that
  serves a model. Ollama matches it against the runner the model's manifest names,
  the one `list_models` reports as `OllamaModel.runner`: a model published for one
  runner — every model measured on Ollama 0.40.6 — answers as if unpinned when
  that runner is pinned, and any other is the server's error, never a fallback:
  HTTP 400 `no compatible manifest found for runners: <runner>` (404 from
  `model_info`, and so from `load_model`), even while the model is loaded; a model
  published for several runners is served by its variant for the one pinned (by
  Ollama's source; untested). Pin the runner the listing names, not the one
  `running_models` reports: a GGUF model listed as `:ggml` is served by llama.cpp
  and is the 400 pinned to `:llamacpp`. The pin goes with chat, embeddings, System
  One, `model_info`, `load_model`, `unload_model` and `pull_model` requests, so a
  pinned endpoint fails for each model without a variant for its runner, and it is
  part of a recording's key. `OllamaModel`, `OllamaModelInfo` and
  `OllamaRunningModel` report a model's `runner` (a `Symbol`) and the first two its
  `format` (`:gguf`, `:safetensors`).
- **`delete_model(name)`** removes an installed model from the server's disk
  (`DELETE /api/delete`). `load_model` loads embedding and decision models too,
  each with what its own requests carry: an embedding model the options of its
  embeddings requests, a decision model no options, as System One requests carry
  none — so an endpoint's `num_ctx`, or any other option, does not apply to a
  decision model, neither when it is loaded nor when it is asked. Measured on
  Ollama 0.40.6 (Apple M4 Max, 21 October 2026), `kev:0.8b-q8_0` through
  `OllamaEndpoint(num_ctx=4096)`: the first ask after `load_model` took
  0.036–0.049 s, an ask without it, which loads the model, 0.81–1.36 s.
- **Local transcription:** `transcribe(path; service = OllamaEndpoint(), model =
  "gemma4:e4b")` goes to Ollama's `/v1/audio/transcriptions` (Ollama 0.40.6:
  "The secret word is banana" for the manual's recording). There is no default
  model; `language` and `prompt` are sent, and the transcript is asked for as JSON
  only, the one form in which Ollama reports a failure that arrives after its
  reply has begun (an `AudioCallError`; by Ollama 0.40.6's source, not
  reproduced): `response_format = "text"`, which would read such a failure as an
  empty transcript, is refused before the request — leave it unset and read the
  text with `transcript_text` — and so are `temperature`, `languages`, `keywords`
  and other formats. Ollama labels these replies `application/x-ndjson`, so they
  are read as JSON whatever the label. A new capability, `:transcription`, is
  declared by OpenAI and Ollama, and `has_capability(s, :transcription)` is true
  for any endpoint that declares `:audio`, so an endpoint of your own keeps
  transcribing; `speak` and `translate` still need `:audio`.
- The manual: *Local Models with Ollama* covers EmbeddingGemma 2 (task prompts,
  smaller vectors, images and sound, long inputs), transcription, runners and model
  deletion; *Embeddings* adds task prompts, a local retrieval recipe and media;
  *Backends* adds pictures, what each local model takes, and nine local decision
  models measured on the 129 labelled tickets — accuracy from 0.690
  (`kev:0.8b-q8_0`) to 0.938 (`tev1:4b`, expected calibration error 0.038, 0.24 s
  a ticket), against 0.969 for Jev and 0.946 for OpenAI's Decisions API — with
  UniLM's answers equal to the server's reply to the same bytes on all nine.

### Fixed
- **`num_ctx` and `truncate` are described per runner.** Measured on Ollama 0.40.6:
  on MLX (Apple Silicon) a model loads with `num_ctx`, keeps that window when a
  later request asks for more, and is not bounded by it — a code word at the start
  of a 10,842-token prompt was read on a 4,096-token load; on llama.cpp another
  `num_ctx` loads the model again, and a prompt that does not fit is an HTTP 400
  with `truncate = false`. `truncate = true` on chat can cut inside a single
  over-long message: 5,407 tokens became 2,050, the first 4 and the tail kept. A
  model with vision loads with at least 2,048 tokens whatever `num_ctx` asks
  (`embeddinggemma-2:740m-q8_0` asked for 1,024 loaded with 2,048 on an Apple M4
  Max on 22 October 2026, and on Ollama 0.40.7 on Linux).
  The docstrings and the guide said, for every runner, that another window reloads
  the model and that a prompt that does not fit is refused.
- **Ollama cloud models are priced by the requested name**, not as free local
  calls: a model whose tag is `cloud` or ends in `-cloud` (`gpt-oss:120b-cloud`)
  runs on Ollama's servers and bills its account. Its chat, embeddings and System
  One results were priced at 0.0 as local, and its `respond` and FIM results by
  the reply's name, which drops the tag (`gpt-oss:120b`), so a price row for the
  cloud model never applied. A System One answer, or a `respond` (streamed or
  not) or FIM result, whose reply drops the tag now keeps the requested name, and
  the server's stays in its `raw`; a name missing from the price table logs the
  missing-price warning. That the reply drops the tag is inferred from Ollama
  0.40.6's source (its cloud proxy sends the name without the tag and passes the
  reply through), not observed. A model is recognised as cloud by its tag alone,
  surrounding whitespace trimmed: a cloud model copied under a name without it
  (`ollama cp`) is priced as local, at 0.0, with no warning.
- **Local `respond` and FIM results cost `0.0`** without the missing-price warning,
  as chat and embeddings did: the results now record the endpoint that answered.

## 0.24.0

### Breaking
- **Provider limits are checked when a request is sent, not when a question is
  built.** `choice`, `score`, `ChoiceQuestion` and `ScoreQuestion` take any number
  of options or levels from one up, so a Choice of 256 options builds, and `ask`
  refuses it on TypeSafe with an `ArgumentError` before the request. Each backend
  has its own bounds: TypeSafe at most 255 options and 10 levels, OpenAI's
  Decisions API 2–255 and 2–10, Ollama 2–26 and 2–10. A keyed table of
  `nl_dispatch` or `nl_classify` likewise holds any number of entries, and the
  backend refuses more options than one question may offer. Upgrading: code that
  relied on the constructor to refuse an oversized question gets the
  `ArgumentError` from `ask`, `ask_each` (naming the item) or the
  natural-language verb instead; `meanings` and `meaning_gaps` accept a table of
  any size.
- **`SystemOneRequest.model` is a `Union{Nothing,String}`, resolved when the
  request is sent, and a request has a fourth field, `options`.** A request built
  without a model holds `nothing` and is sent with the model of the enclosing
  `with_system_one` scope, else the backend's default read at that moment
  (TypeSafe: `TYPESAFE_DEFAULT_MODEL`, else `jev-latest`). Before, the default
  was read when the request was built, so a `TYPESAFE_DEFAULT_MODEL` set in
  between was ignored. `JSON.json(request)` writes `"model"` only when the request
  names one. `options::NamedTuple` holds the provider options (see Added).
  Upgrading: read the model that answered from `SystemOneResponse.model`, not
  from the request; code that read `request.model` as a `String` must handle
  `nothing`. `SystemOneRequest(state, questions, model)` and the keyword
  constructor keep working.
- **The default `service` of `ask`, `list_models`, `nl_dispatch`, `nl_classify`
  and `@branch` is `nothing`:** the backend of the enclosing `with_system_one`
  scope, else `TYPESAFEServiceEndpoint` as before. Upgrading: nothing changes
  outside a `with_system_one` scope; inside one, pass
  `service = TYPESAFEServiceEndpoint` to keep a call on TypeSafe.
- **A reply whose answers do not fit the request is a `SystemOneCallError`**,
  carrying the request id, instead of a `SystemOneSuccess` built from it: a
  question left unanswered or one answered that was not asked; an answer of
  another kind than its question; a Choice of an option that was not offered or
  is not the most probable; probabilities missing, extra, outside `0 … 1` or not
  summing to 1 within the backend's rounding (`0.005` per outcome on TypeSafe and
  OpenAI, the allowance of values rounded to two decimals; `0.00005` on Ollama);
  Score levels other than `0` to `n − 1`; a Score whose `score` is not the
  expected level its probabilities give (`Σ i·pᵢ`), within that rounding, or whose
  `legend` does not repeat, unchanged, each level asked as a string (OpenAI's
  echoed labels likewise); a score, confidence, Noul probability or box edge out
  of range; a reply that names no model, or a blank one. `nl_dispatch`,
  `nl_classify` and `@branch` raise `SystemOneError` for such a reply, where they
  raised a `KeyError` or an `ArgumentError` after the request was billed.
  Upgrading: handle these as failed calls; every answer recorded in this
  repository passes the checks.
- **Calls to TypeSafe and to OpenAI's Decisions API are bounded by 60 s per
  attempt and 150 s in all** while no `RequestConfig` is set — no `config`, no
  `with_request_config` scope, no `set_default_config!` — instead of the field
  defaults' 600 s and 900 s: TypeSafe answers a typical request in about 0.3 s,
  and the slowest call measured took 5.3 s. Upgrading: pass a `config`, open a
  scope or set a process default to choose other bounds; any of them replaces the
  backend's default as a whole.
- **Two terse TypeSafe failures carry a hint after their message.** A context
  overflow (`error_type == "max_tokens_exceeded"`), whose `message` was that
  type alone, now goes on to name the limits and the fix; the bare
  `"Invalid request."` a Box question draws from a Jev before 1.14 goes on to
  name the models that answer one (`jev-1.14.0`, `jev-preview`). Upgrading:
  code that compares a failure's `message` with `==` must test its
  `error_type`, or match the message with `startswith` or `occursin`.
- **`DecisionDeclinedError.answer` is a `Union{ChoiceAnswer,RefusalAnswer}`**: it
  carries the `RefusalAnswer` when the backend declined to answer the question,
  as OpenAI's Decisions API may. Upgrading: check `e.answer isa ChoiceAnswer`
  before reading `e.answer.choice` or `e.answer.confidence`.
- **`nl_dispatch` refuses a default state that holds what is not data.** When it
  builds the state from its ordinary arguments, a function or closure, a type, a
  module, a `Ref` or `Ptr`, a `Task`, a `Channel`, a lock, a condition, an event,
  an `IO`, an `Expr` or a value JSON.jl cannot write, such as an
  `AudioAttachment`, anywhere in an argument — in its dicts, arrays, sets, tuples
  and fields — or a container nested more than 64 levels deep is an
  `ArgumentError` before the request, naming the argument and the path to the
  value. Before, JSON.jl wrote a function by its name and a task, lock or stream
  field by field, and the model read that. An `ImageAttachment` is data: the
  backend's wrapper sends it as it sends any image in a state (TypeSafe as an
  asset), or refuses it before the request. Every other value is still sent as
  JSON.jl writes it, every field of a struct included — a `String` credential a
  struct holds as well. Upgrading: pass `state = …` with only what the model
  should read.
- **`min_confidence` must be a finite number in `0 … 1`** for `nl_dispatch`,
  `nl_classify` and `@branch`; anything else is an `ArgumentError` before the
  request. Before, `NaN` declined nothing, so a `_` line or a `fallback` could
  never run, and a value above 1 declined every answer. Upgrading: pass a
  threshold in `0 … 1`.
- **A `Meaning` type prints as Julia prints any type**, `Meaning{Symbol("…")}`
  (for a sentence of several words; one word prints as `Meaning{:x}`), instead
  of `nl"…"`; an instance still prints as `nl"…"()`. The method that printed the
  type extended `show(::IO, ::Type)`, and invalidated compiled code in every
  session that loaded UniLM (see Performance). A `MethodError` and a
  `methods(f)` listing show the new form; the messages UniLM writes still quote
  the sentences. Upgrading: compare `Meaning` types with `===`, never their
  printed form; `meanings(f)` lists the sentences as strings.
- **A plain `Dict` of Choice options or of questions is sent in sorted key
  order**, not in its hash order, which changed between Julia versions (1.13
  changed the string hash); the order of the options is part of what the model
  reads. Any other dictionary, a `NamedTuple` and a vector of pairs keep their
  order, and a `Dict` state was already written in sorted order by JSON.jl.
  Upgrading: such a request's bytes change, and with them the key of its
  recording: one that 0.23.0 recorded in hash order no longer replays, unless that
  order happened to be sorted (`with_recorded_answers` throws `ReplayMissError`),
  so record it again (`mode = :record_missing`); pass a `NamedTuple` or a vector of
  pairs to fix the order you mean. The recordings of every other request keep
  their keys: the test suite rebuilds each of the 90 System One recordings this
  repository committed before this release from typed requests, to the same key.
- **Every base64 `data:` URL a request carries is recorded as its digest, not
  its bytes** — an image, a PDF, audio, any media. `with_recorded_answers` keys,
  writes and reports each one — an attachment on the OpenAI chat wire
  (`image_url`), an image sent to OpenAI's Decisions API, a Responses
  `input_image`, a PDF in a Responses `input_file` — as
  `data:<mime>;sha256,<hex>`, the SHA-256 of its base64 text: a recording names
  the media it answered without holding them, and the request sent is
  unchanged. Media sent as bare base64 is written as sent: Ollama's chat
  `images` and an audio clip's `input_audio`. Upgrading: a recording that 0.23.0
  made of a request with such a URL was keyed on its base64 text and no longer
  replays (`with_recorded_answers` throws `ReplayMissError`), so record it again
  (`mode = :record_missing`); a recording without one keeps its key.
- **An attachment where no request carries media is refused.** An
  `ImageAttachment` or `AudioAttachment` written as JSON anywhere but a message's
  `attachments` or a System One request's state, instructions or criteria —
  inside a struct of your own, say — is an `ArgumentError` (`JSON.lower` throws),
  where JSON.jl wrote its bytes as an array of numbers that no service reads as
  media. Upgrading: put the attachment where a request carries media, or write
  its fields yourself (`base64encode(a.data)`, `a.mime`).
- **A question type of your own is no longer sent.** A subtype of
  `SystemOneQuestion` that is not one of UniLM's is an `ArgumentError` before the
  request on every backend: TypeSafe's wire carries Choice, Score, Noul and Box
  questions and a `RawQuestion` verbatim. Upgrading: build such a question as a
  `RawQuestion(fields)`, whose answer is accepted whatever its kind, or a Box
  question with `box(instructions)` — which needs an image in the request.

### Added
- **One System One surface over three backends.** `ask`, `ask_each`,
  `list_models`, `nl_classify`, `nl_dispatch` and `@branch` run on:
  - TypeSafe's Jev, the default (`TYPESAFEServiceEndpoint`), with what its API
    offers in October 2026: images, Box questions and long context (below);
  - OpenAI's Decisions API, a public beta since October 6, 2026
    (`service = OPENAIServiceEndpoint`, `OPENAI_API_KEY`, model `gpt-6-luna`): a
    `noul` is sent as a `predicate`, a `choice` as a `choice` of 2–255 options, a
    `score` as a `score` of 2–10 levels; images go in the state (an
    `ImageAttachment`, or a `Vector` of strings and images, at most 128); the
    provider options are `prompt_cache_key` and `safety_identifier`, sent and left
    out of a recording's key; a question the API declines comes back as a
    `RefusalAnswer`, which the natural-language verbs take as a decline (the
    fallback, or `DecisionDeclinedError`). What its wire has
    no form for — a question without string `instructions`, a Noul's `yes`/`no`
    criteria, a description or level that is not a string, a Box question, an
    image outside the state — is an `ArgumentError` before the request.
    `estimated_cost` bills input tokens only, \$0.10 per million, cached ones at
    the same rate and twice that rate above 272K input tokens; the request id is
    the `x-request-id` header;
  - local decision models through Ollama 0.35 or later
    (`service = OllamaEndpoint()`, with `model = "tev1:0.8b"` or another decision
    model: there is no default): 1–64 questions, `instructions` on every question,
    2–26 options and 2–10 levels described by strings, text only, no Box
    questions, and one option, `keep_alive`, sent and left out of a recording's
    key. A model may cap lower and says so in its 400 (`tev1:0.8b`: 24 options),
    and Ollama answers 413 to a request whose text and schema exceed 64 KiB, a
    limit it counts itself. Answers cost `0.0` and carry no request id.
    Measured on 2026-10-20 in two runs (Apple M4 Max, Ollama 0.40.6, one request
    at a time): about 17 ms per warm request on `kev:0.8b-q8_0` and 25–140 ms on
    `tev1:0.8b`; for a first call that loads the model, about 0.9 s on `kev`
    and 2.6–2.8 s on `tev1`.

  Each backend's wrapper refuses before anything is sent what its wire cannot
  carry, naming the backend, the rule and the fix. `confidence` is one measure
  everywhere — TypeSafe's published formulas, which TypeSafe and OpenAI report and
  the Ollama wrapper computes from the probabilities (Ollama's generic decision
  models report `1 − H(p)/ln n`, kept in `raw`). Calibration belongs to the model,
  so tune `min_confidence` per backend: on 129 labelled support tickets with 7
  options (2026-10-20), Jev (`jev-1.13.0`) reached accuracy 0.969 with an
  expected calibration error of 0.016, and OpenAI (`gpt-6-luna`) 0.946 with 0.080,
  under-confident (its accuracy at least its mean top probability in every bin
  above 0.5); Jev answered in 0.25 s at the median and 0.33 s at the 95th
  percentile, OpenAI in 0.48 s and 2.44 s.
- `with_system_one(f, service; model = nothing, options...)` sets the backend of
  every System One call inside `f` that names none — tasks spawned there
  included — and, for those calls, the model and the provider options when a call
  names none of its own
  (`with_system_one(TYPESAFEServiceEndpoint; long_context = true) do … end`),
  checked by the backend's wrapper as each request is sent; scopes nest, and the
  innermost applies.
- `ask_each(states, questions...; …)` and `ask_each(requests; …)` send one request
  per item, concurrently, and return one typed result per item in input order:
  what `ask` returns for that item alone (nothing is packed in this release).
  Every request is built and checked before the first is sent, so an item that
  cannot be sent is an `ArgumentError` naming its position, and nothing is sent;
  the worker that sends an item builds its body again, so a batch holds no more
  request bodies than it has requests in flight. At most `ntasks` requests are in
  flight — any `Integer` of at least 1 (a `Bool` is refused), by default 16 on
  TypeSafe and OpenAI, 1 on Ollama, 8 on a protocol of your own; a 429 or 529 to
  any of them holds every attempt not yet started until the wait the service
  asked for has passed. An item's `total_deadline` runs from when a worker takes
  it, so a hold keeps it waiting no longer than its own budget: an item whose
  budget runs out while held is not sent, and its result is a
  `SystemOneCallError` whose `cause` is a `UniLMTimeout` of phase `:deadline`.
  `cancel` turns every unanswered item into a `SystemOneCallError` whose `cause`
  is a `UniLMCancelled`; `on_result(i, result)` — any callable that takes an
  index and each kind of result, checked before anything is sent — runs as each
  result lands, one call at a time; a `ReplayMissError`, a `RecordingWriteError`,
  an interrupt or an exception from `on_result` stops the batch and is rethrown
  once every worker has stopped. 50 support tickets took 3.28 s on TypeSafe,
  against 13.21 s for one `ask` after another, over one HTTP/2 connection
  (2026-10-20).
- Images on TypeSafe: an `ImageAttachment` anywhere in the state, the instructions
  or the criteria — directly, or inside a `Dict`, a vector, a tuple or a
  `NamedTuple` — is sent as a TypeSafe asset: each distinct image once, as a part
  of a `multipart/form-data` request, every place it stood holding its reference
  `<image:…>` (the first 32 hex digits of the SHA-256 of its bytes), with the
  `TypeSafe-Danger-Zone-Images: true` header TypeSafe requires. TypeSafe marks
  images experimental. Refused before the request: an image that is not a still
  PNG or JPEG, is over 16 MiB, has a side outside 280–1568 pixels or a header that
  cannot be read, or a JPEG that does not end with its end-of-image marker (cut
  short); a body over 32 MiB; more than 64 levels of nesting. Each use of
  an image is billed; on 2026-10-19 the input tokens of a W×H image fit
  2·⌈W/28⌉·⌈H/28⌉, plus about 4 per use. A Score level that is an image comes back
  in the answer's `legend` as that `ImageAttachment`. A recording of a request
  with images is keyed on its JSON part and the SHA-256 of each image, and holds
  no image.
- Box questions: `box(instructions)` builds a `BoxQuestion` — where, in an image of
  the state, is what the instructions name — answered by a
  `BoxAnswer(box::BoxCoordinates, confidence, raw)` whose `top`, `bottom`, `left`
  and `right` edges are fractions of the image's height and width. On TypeSafe a
  Box question needs Jev 1.14 or later (`model = "jev-1.14.0"` or
  `"jev-preview"`; `jev-latest` answered as 1.13 on 2026-10-20, and its bare
  400 `"Invalid request."` comes back with a hint naming those models) and an
  image in the request: one without is refused before it is sent, since TypeSafe
  answers it anyway — HTTP 200 and a box at a low confidence, 0.05 to 0.10 in
  probes on 2026-10-19 and 2026-10-20. A box turned inside out is a
  `SystemOneCallError`. OpenAI and Ollama refuse Box questions.
- Long context on TypeSafe: `ask(…; long_context = true)` sends
  `TypeSafe-Danger-Zone-Long-Context: true`, which raises the context limits from
  32k tokens per question and 64k per request to 262k, at the same price per
  token; TypeSafe documents it as slower and less accurate. A
  `max_tokens_exceeded` failure's `message` names the option and the limits.
- Provider options: every keyword of `ask` and `ask_each` that is not their own
  becomes the request's `options`, which the backend's wrapper checks — TypeSafe
  `long_context`, OpenAI `prompt_cache_key` and `safety_identifier`, Ollama
  `keep_alive` — and any other is an `ArgumentError` listing the accepted ones.
  `nl_dispatch`, `nl_classify` and `@branch` take them as
  `options = (long_context = true,)`, and a `with_system_one` scope's keywords set
  them for the calls inside that name no backend and no options. TypeSafe's
  `long_context` is part of a recording's key only when true, so a request
  without it, or with `long_context = false`, keeps its key; OpenAI's and
  Ollama's options change how a request is served, never what it is answered,
  and are neither keyed nor written.
- `RawQuestion(fields)`: a question of a kind this version has no type for, sent
  verbatim by each of the three backends (OpenAI adds its name). Its answer is
  accepted whatever its kind; one this version does not know is an
  `UnknownAnswer`.
- `RefusalAnswer`: a question the backend declined to answer, accepted only from a
  backend that documents refusals (OpenAI's Decisions API).
- `SystemOneResponse.service`: the backend that answered, as the call resolved it;
  the five-argument constructor fills `nothing`.
- The System One protocol seam, public and not exported:
  `UniLM.system_one_protocol(service)`, a Holy trait, returns the endpoint's
  `UniLM.SystemOneProtocol` (`TypeSafeProtocol`, `OpenAIDecisionsProtocol` or
  `OllamaSystemOneProtocol`), whose `UniLM.encode_system_one` builds the
  `UniLM.SystemOneWire` it sends — and its `record_body`, the body a recording is
  keyed on and writes — and whose `UniLM.decode_system_one` reads the
  reply. An endpoint that declares `:system_one` without a protocol speaks
  TypeSafe's wire, so a TypeSafe-compatible gateway needs no protocol code. A
  protocol other than TypeSafe's files its recordings under a namespace of its
  own (`/openai-decisions/v1/decisions`, `/ollama/v1/systemone`), so backends
  never replay each other's answers, while every TypeSafe recording keeps its
  key.
- Every System One exchange goes through one connection pool per process, which
  keeps up to 256 idle connections per host for 30 s and is created by the first
  live request. A retry of a TypeSafe request carries `X-TypeSafe-Retry-Count`
  (`1` on the second attempt), as TypeSafe's official SDKs send it.
- A `@branch` option name may be a `Symbol`, sent as its `String`.

### Changed
- `OPENAIServiceEndpoint` and `OllamaEndpoint` declare `:system_one`.
- An endpoint that declares no capabilities is refused by `ask`, `ask_each`,
  `with_system_one` and the natural-language verbs with an `ArgumentError` saying
  to declare `:system_one`, instead of a `MethodError`.
- A recording holds `key_extra`, what its key covers beyond the body (TypeSafe's
  `long_context`, image digests), when there is any. A `ReplayMissError` for a
  backend other than TypeSafe shows its namespaced path and says to record it with
  access to that backend — its API key set, or its local server running.
- The manual's decision pages form a **Decisions (Jev)** lane at the same
  addresses: *Start Here: Decisions in Five Minutes* (formerly *Start Here: Jev in
  Five Minutes*), a new *Backends: Jev, OpenAI and Local Models*, *Decisions with
  LLMs* (formerly *Jev with LLMs*), *Route and Decide*, now with images, Box
  questions and long context, *Dispatch on Meaning*, *Many Items at Once*, now
  led by `ask_each`, and *Test and Develop*. The API page *System One API (Jev,
  OpenAI Decisions, Ollama)* documents what the backends share and each one's
  setup, rules, errors and cost, and the result-type hierarchy lists the Ollama
  model-management results (68 result types in 17 families).

### Fixed
- A Score answer's levels are read strictly: `"-1"`, `"+1"`, `" 2"` and a level
  beyond the rubric were accepted, and `"01"` beside `"1"` read as one level,
  dropping a probability. Each is now a `SystemOneCallError`.
- A token count in a reply's `usage` that is not a non-negative whole number reads
  as 0 with a warning: `1e300` made the whole call a `SystemOneCallError`,
  discarding answers that were paid for, a negative count gave a negative cost,
  and `2.5` read as 2.
- Input that cannot be sent is an `ArgumentError` before the request:
  `ask(42, …)`, `ask(nothing, …)` and `nl_classify(42, …)` threw a `MethodError`;
  a `NaN` or `Inf` in the content came back as a `SystemOneCallError`, as though
  the service had failed; a question or option name of blanks, such as `" "`, was
  sent (both are now refused when the request or the question is built).
- A request whose questions changed after it was built is checked again when it
  is sent: 302 options and 14 levels pushed into built questions reached TypeSafe,
  and a repeated question name replaced the first question on the wire.
- The plans `nl_dispatch` caches per function and argument types are capped at
  1,024, as keyed plans already were: they grew without bound, one per distinct
  type of the ordinary arguments.
- A `@branch` option name that is neither a `String` nor a `Symbol` is an
  `ArgumentError` naming its line, instead of a `MethodError`.

### Performance
- A session's first System One call no longer waits for the compiler. A
  precompile workload, run once while the package image is built, compiles every
  System One verb on each backend, replaying answers it writes and then removes,
  so nothing leaves the machine. Measured in fresh processes, each first call
  timed once with `@timed @eval`, interleaved with 0.23.0 (Julia 1.13.1, HTTP.jl
  2.11, JSON.jl 1.11, an Apple M4 Max under background load; medians), against a
  local server that answers at once: the first `ask` took 0.054 s instead of
  2.33 s; the first `nl_dispatch` 0.048 s instead of 3.18 s with sentences in its
  signatures, and 0.118 s instead of 3.61 s with a table; the first `nl_classify`
  0.074 s instead of 2.64 s; the first `@branch` 0.028 s instead of 2.51 s. The
  first live answer from TypeSafe arrived in 0.48 s instead of 2.79 s. Replayed,
  the first `ask` on OpenAI's Decisions API or on Ollama took about 0.09 s, and
  the first `ask_each` about 0.04 s. `list_models` is left out of the workload, to
  keep loading short, and compiles at its first call (about 0.5 s).
- The workload is paid for when the package loads: `using UniLM` took 0.16–0.17 s
  instead of 0.13 s, about 25% more (+24.6%, +25.8% and +23.9% in three
  interleaved runs); the package image grew from 7.6 MB to about 23 MB, and
  precompiling UniLM from about 2 s to about 8 s.
- Loading UniLM's own definitions invalidated 1,466 compiled method instances in
  the session — among them the code that prints any type — nearly all through the
  `show` method of `Meaning` types; they now invalidate 13 (see Breaking).
- A function's first natural-language call compiles in about 23–35 ms instead of
  80–90 ms: `nl_dispatch` and `nl_classify` are compiled once, not once per
  function, argument type or keyword type.
- A table of new keys no longer compiles code for each key, nor for each pair of
  keys in its duplicate check: on 0.23.0, validating a table of 50 new keys
  took 0.50–0.53 s. The first `nl_dispatch` with a table of 255 new `Symbol`
  keys takes 0.12 s instead of 6.3–6.8 s, and the first `nl_classify` 0.014 s
  instead of 2.25–2.45 s (each timed up to the request, two runs on each
  version).
- Plans are read without a lock, and a table is validated once per contents
  instead of on every call.
- Checking an answer is linear in its options: a replayed Choice of 255 options
  takes about 187 µs per call.
- 64 workers calling a local HTTP/1.1 server in a closed loop for 60 s failed no
  call, over 66 connections (53,729 calls/s); HTTP.jl's default client, which
  keeps two idle connections per host, opened over 16,000 connections, ran out of
  ephemeral ports and failed 324 calls. A burst of 1,000 requests to a server that
  answers in 300 ms took 0.44–0.51 s (`ask_each`, three runs).

## 0.23.0

### Breaking
- `OllamaEndpoint` is its own type (`OllamaEndpoint <: OpenAIWireEndpoint`), no longer a
  `GenericOpenAIEndpoint`: code that dispatched on `GenericOpenAIEndpoint` for Ollama, or read
  its `api_key` field, must use `OllamaEndpoint`. Its chat requests go to Ollama's native
  `/api/chat` and its embeddings to `/api/embed` instead of the OpenAI-compatible routes, and
  its default server is `http://127.0.0.1:11434` (or `OLLAMA_HOST`) instead of
  `http://localhost:11434`. `respond` and `fim_complete` keep the OpenAI-compatible routes.
  `GenericOpenAIEndpoint("http://127.0.0.1:11434", "")` still reaches Ollama's
  OpenAI-compatible chat route, as `OllamaEndpoint()` did before.

### Added
- Ollama, with Gemma 4 as the reference model (`gemma4:e4b`, measured on Ollama 0.34.4):
  - The native API carries what the OpenAI-compatible route ignores: the context window
    and the other runtime options (`OllamaEndpoint(num_ctx=32_768, top_k=64, …)`, typed as
    `OllamaOptions`), `keep_alive` (seconds, `Inf` to stay loaded) and the `think` and
    `truncate` fields.
  - Thinking is captured (streamed or not) and sent back with its turn, which Gemma 4's
    tool loops rely on; `reasoning_text(result)` reads it, and also DeepSeek's
    `reasoning_content`, Anthropic thinking blocks and Gemini thought parts.
  - A `response_format` turns thinking off: Ollama constrains a reply only after thinking
    ends, and with thinking at its default `gemma4:e4b` answered plain text to 10 of 10
    schema requests (10 of 10 valid with it off). A format together with a thinking
    effort is refused; `respond` follows the same rule for a `json_schema` text format.
  - An input that does not fit the context window is an HTTP 400 ("exceeds the available
    context size") instead of an answer to a silently shortened prompt, for chat and for
    embeddings (`truncate=false` by default; `truncate=true` restores Ollama's behaviour).
  - Fields the native API or Ollama's stateless Responses API would ignore
    (`tool_choice` other than `"auto"`, strict tools, `logit_bias`, `previous_response_id`,
    `store`, …) are refused before any request; a `Respond` tool loop is refused in favour
    of `tool_loop!` on a `Chat`.
  - `OLLAMA_HOST` is honoured with the rules of Ollama's own clients; local turns and
    embeddings cost `0.0` without a missing-price warning; an unreachable server's error
    says how to start one.
  - Model management: `list_models(service=OllamaEndpoint())`, `model_info`, `pull_model`
    (streamed progress), `running_models`, `load_model` and `unload_model`, with typed
    results (`OllamaModel`, `OllamaModelInfo`, `OllamaRunningModel`, `OllamaPullProgress`;
    `OllamaSuccess`, `OllamaFailure`, `OllamaCallError`).
- Images and audio in a user `Message`: `Message(Val(:user), "…", ImageAttachment("shape.png"),
  AudioAttachment("clip.wav"))`, recognised by their leading bytes (PNG, JPEG, GIF, WebP;
  WAV, MP3). OpenAI-compatible endpoints send them as `image_url` / `input_audio` content
  parts, `OllamaEndpoint` in the message's `images`; the native Anthropic and Gemini
  encoders refuse them for now. `prefix_complete` sends each message in its full wire form.
- A new guide, *Local Models with Ollama*, and an *Ollama* API page.

### Fixed
- The `SystemOneRequest` docstring shows its example question, which names a state key
  in backticks, as text; the manual rendered it as a formula.

### Changed
- The manual's LLM examples that ask open questions now ask for a short plain-text
  answer, so their recorded replies read cleanly in an output block instead of showing
  raw Markdown (`**bold**`, `##` headings, code fences), and long output lines wrap
  instead of scrolling sideways.
- The manual is a new static site at the same addresses:
  https://algunion.github.io/UniLM.jl/dev/, `/stable/`, and `/vX.Y.Z/` for each release from
  now on, with every page at its URL and every section and docstring at its anchor.
  Documenter still runs every example, expands the docstrings, resolves the
  cross-references and checks that every export is documented; the site is built from its
  output and published with Documenter's versioned deployment. As before, `/stable/` points
  at the newest release, the root redirects to `/stable/`, and each release is added to the
  version list (`versions.js`) that the version menu of earlier releases' pages reads; those
  pages stay as they were, their menu included. The new site has no version menu: its
  header shows which version it documents. Documenter's `objects.inv` inventory, which
  other manuals resolve cross-references against, is not published for `dev`, for the
  releases from now on or for a redeployed release; earlier releases keep theirs.
- A manual run of the Documentation workflow on `main` with `redeploy_version` (such as
  `v0.22.0`) publishes the current manual as that release's docs, in its folder, without a
  new release. It is refused before anything is built unless it runs on `main`, the
  release's tag exists and `src/` and `Project.toml` are unchanged since the tag, so a
  release's docs never describe unreleased code.

## 0.22.0

### Added
- Keyed natural-language dispatch: with `nl_dispatch(f, args...; texts = table)` the methods
  dispatch on a short key — `route(::Val{:refund}, t)`, a singleton instance or type you own,
  an enum value — and a table maps each key to the sentence the model reads. The request
  carries only the sentences, byte for byte the one the same sentences written as `nl"..."`
  send; the chosen sentence maps back to its key locally (a `Symbol` key is passed as
  `Val(key)`, any other key as written), so no key reaches the model. A table is a
  `NamedTuple` of sentences or a vector of `key => sentence` pairs, and fills the one position
  whose declared type its keys have; several keyed arguments take a tuple of tables. Options
  follow the types of the ordinary arguments as they do for meanings, in table order. Before
  any request, a `Dict` (it has no order), an empty table, more than 255 entries, a blank
  sentence, a sentence given twice, two keys whose dispatched values are `isequal` (`:a` and
  `Val(:a)`), `nothing` or `missing` as a key, a key no method takes at its position whatever
  the other arguments are, a method that also pins a `Meaning`, and a gap in the method table
  are each an `ArgumentError`. `decide` may return an offered key as the table writes it.
  `meanings(f; texts)`, `meanings(f, argtypes; texts)` and `meaning_gaps(f, argtypes; texts)`
  preview a keyed call. Dispatch on `nl"..."` meanings is unchanged.
- `nl_classify(state, texts; ...)` returns the chosen key as the table writes it (`:refund`
  for a `NamedTuple` table) without dispatching: one Choice question named `"classify"` over
  every sentence, with the decision policies of `nl_dispatch`. A declined answer calls
  `fallback(state)`, or throws `LowConfidenceError` or `DecisionDeclinedError`.
- An `on_response` keyword on `nl_dispatch`, `nl_classify` and `@branch`: called once with the
  `SystemOneSuccess` before the decision policy runs, so an audit gets the response's
  `request_id`, `model` and `raw`, which `decide` cannot see. Its return value is ignored, and
  an exception from it propagates with nothing else run; it is not called for a failed call or
  when nothing was sent. A value that cannot be called on a `SystemOneSuccess` is an
  `ArgumentError` before the request.
- `with_recorded_answers` records and replays the language-model verbs as well: inside
  a scope, every non-streaming `chatrequest!` (and so its keyword form and
  `tool_loop!`), `respond` (and so its convenience forms and `tool_loop`) and
  `embeddingrequest!` exchanges through the recordings exactly as `ask` does — the
  same three modes, the same directory, the same key: the SHA-256 of the request line
  and the exact body, where the line holds the URL's path alone, never its host or its
  query (which can carry a credential). The credentials a body can carry are replaced
  by `"<redacted>"` in the key, the recording and a `ReplayMissError`, so none is
  written and a replay needs none: in the body's top-level `tools` array, an MCP tool's
  (`"type": "mcp"`, `MCPTool`) `authorization` and `headers`, a Gemini Interactions
  `"mcp_server"`'s `headers`, and the `api_key` of a Gemini `"retrieval"` tool's
  `exa_ai_search_config` and `parallel_ai_search_config`. The request sent still
  carries them. A replayed reply decodes like the live one:
  the message, finish reason and usage, the reply appended to the `Chat` and its cost
  accumulated, the `ResponseObject`, the embedding vectors. A streamed call, images,
  audio, files, MCP, the Responses lifecycle operations and every other verb reach
  the network as before, inside a scope or not. A recording now also keeps the name
  of the header its request id came in (`request_id_header`: `x-typesafe-request-id`,
  `x-request-id` or `request-id`), restores that one header on replay, and keeps no
  other; a recording without the name replays its id as `x-typesafe-request-id`, as
  before. A `ReplayMissError` for an LLM request names the request's `model` and says
  to record it with the provider's API key.

### Changed
- Inside a `with_recorded_answers` scope, `chatrequest!`, `respond` and
  `embeddingrequest!` — and so `tool_loop!` and `tool_loop` — throw `ReplayMissError`
  for a request the scope cannot replay, and a paid reply's recording that cannot be
  written throws `RecordingWriteError`, instead of returning an `LLMCallError`,
  `ResponseCallError` or `EmbeddingCallError` that a fallback would take for a service
  failure. Before, these verbs ignored the scope and called the provider. Upgrading:
  code that runs LLM calls inside a `:replay` scope must record them first (the same
  code once with the provider's key and `mode = :record_missing`) or make them
  outside the scope.
- A recording that cannot be written after a paid answer throws the new exported
  `RecordingWriteError` — its `file`, and the I/O error as its `cause` — out of `ask`,
  `list_models` (and so `nl_dispatch`, `nl_classify` and `@branch`), `chatrequest!`,
  `respond` and `embeddingrequest!`, where 0.21.0 threw the bare I/O error; its
  `showerror` names the file and the cause and says that the answer was billed. A tool
  loop no longer turns one raised by a dispatcher's request into a tool error the model
  reads while the loop goes on: it propagates, as `ReplayMissError` does. An MCP server
  answers one raised by a tool handler with the generic JSON-RPC `-32603` and logs it,
  instead of relaying the error, with its local path, to the client as `isError` tool
  content. Upgrading: catch `RecordingWriteError` where code caught `Base.IOError` or
  `SystemError` from these verbs.
- `nl_dispatch` and `nl_classify` refuse a `fallback` that cannot take what it would
  be called with — the call's arguments (`fallback(args...)`), or the state
  (`fallback(state)`) — with an `ArgumentError` before the request. Before, it failed
  with a `MethodError` only after a declined answer had been billed.
- On the Home, Getting Started, Chat Completions, Responses API, Embeddings, Tool
  Calling, Agentic Workflows, Structured Output, Cost Tracking and Multi-Backend pages,
  the examples that call `chatrequest!`, `respond` or `embeddingrequest!` without
  streaming render recorded real output instead of a failed request. What the
  recordings do not cover still runs, and fails, without a key: the two streamed
  examples of the Streaming guide and the `count_input_tokens` and `compact_response`
  examples of the Responses API guide render their failure, and its stored-response
  example shows the stored id but nothing from `get_response`, `list_input_items` or
  `delete_response`. The native Anthropic example of the Multi-Backend guide is shown
  without running. The documentation build chooses its mode by flag, not by the keys
  it finds: replay by default (with every provider key hidden, so it never calls a
  service), `UNILM_DOCS_RECORD=1` to record the missing answers, `UNILM_DOCS_LIVE=1`
  to call every service live — a check that the examples still run, which never
  deploys, because the prose quotes the recorded outputs.
- The Jev documentation is rebuilt around jobs, each page opening with a table of
  which pathway to take and what it gives: Start Here: Jev in Five Minutes (new), Jev
  with LLMs (new: route before an LLM call, check a draft, check claims against the
  source, pick the model, screen what the model reads, approve and shortlist tool
  calls), Route and Decide (formerly Typed Judgments with Jev), Dispatch on Meaning
  (keyed and `nl"..."` dispatch side by side), Many Items at Once and Test and
  Develop. Semantic Programs (`guide/semantic_programs/`) is gone: its sections moved to Route and Decide, Dispatch
  on Meaning and Jev with LLMs. Every example runs on recorded real answers,
  measurements sit in collapsible Evidence boxes, the typed-extraction example sends
  each option's sentence as its name (re-measured), natural-language dispatch has its
  own API page, and the LLM guides point to the Jev check for what they build.

### Fixed
- Cloudflare's origin errors 520–524 are retried like 502 and 504, within
  `RequestConfig.max_attempts` and honouring `Retry-After`. A Cloudflare-fronted API
  that answered one of them used to fail the call on the first attempt.
- An MCP server tool handler that throws `ReplayMissError` — a request with no
  recording inside `with_recorded_answers` — no longer answers the client with
  `isError` tool content that a model would try to correct from: the dispatcher logs
  it and answers the generic JSON-RPC `-32603`.

## 0.21.0

### Breaking
- **Options follow the argument types.** `nl_dispatch` offers only the meanings of
  methods that accept the types of its ordinary arguments: with `step(::Waiting,
  ::nl"gives an order number", msg)` defined, that meaning is no longer offered when
  the first argument is a `Confirming()`. Before, the model could pick a meaning with
  no method for the arguments and the call ended in a `MethodError` after a billed
  request. Arguments no natural-language method accepts are an `ArgumentError` before
  any request. `meanings(f)` still lists every method's options.
  Upgrading: for a function whose natural-language methods restrict the types of its
  ordinary arguments, the option list sent to the model is shorter, so its
  probabilities and `confidence` — and what a tuned `min_confidence` means — change;
  re-check thresholds tuned against the full list.
- **Gaps are refused before billing.** A gap in the method table — a combination of
  offered meanings with no method, or with an ambiguous one — is an `ArgumentError`
  raised before any request, instead of Julia's `MethodError` after a billed one. For
  the combinations with no method the error suggests defining them, or one method wild
  in every slot (`f(::Meaning, ::Meaning, x)`), which is not an option and covers
  every combination that has no method. Such a backstop is not more specific than the
  methods of an ambiguous combination and cannot settle it, so for those the error
  names the colliding methods and the intersection to define instead. A partial
  wildcard (`f(::nl"a", ::Meaning, x)`) is still refused, and the error now says that
  it is one and that it is not supported.
- **`@branch` refuses an unreachable `_` line.** `@branch` accepts a `_` fallback line
  with `decide` as well as with `min_confidence`. A `_` line whose policy cannot
  decline when the branch runs — `decide = nothing`, or a `min_confidence` of 0 or
  less — is an `ArgumentError` before the request; before, the request was sent and
  the line could never run.

### Added
- `with_recorded_answers(f, dir; mode)` records System One answers once and replays
  them offline. The service does not answer a repeated request identically, so a test
  or a docs build that calls it live cannot be reproduced. Inside the scope, every
  `ask` — and so `nl_dispatch` and `@branch` — and `list_models` exchanges through
  `<dir>/<key>.json`, keyed by the SHA-256 of the request line and the exact request
  body (a request with its options or state keys reordered is a different request).
  `:replay` (the default) needs no API key and no network. A request with no
  recording, or with a recording file that cannot be read, throws the new
  `ReplayMissError` out of `ask` instead of returning a `SystemOneCallError`; its
  `reason` says which. `:record` sends every request and writes each HTTP 200,
  replacing any earlier file; `:record_missing` replays what is recorded and records
  the rest, and throws on an unreadable file rather than overwrite it. Both create
  `dir` and prove it writable before `f` runs, so a directory that cannot take a
  recording is an `ArgumentError` before anything is billed; a recording that still
  cannot be written once an answer has arrived throws its I/O error out of `ask`
  rather than returning a `SystemOneCallError` a fallback would absorb. The scope
  covers tasks started inside it, scopes nest, and a cancelled token ends a call
  before any replay. The documentation's System One examples replay committed
  recordings when the build has no `TYPESAFE_API_KEY`; `UNILM_DOCS_RECORD=1` with the
  key records the missing ones.
- A `decide` keyword on `nl_dispatch` and `@branch`: a decision policy called with
  each question's `ChoiceAnswer` that returns any offered meaning — not only the
  argmax — or `nothing` to decline. `nl_dispatch` takes one callable for every slot or
  a vector with one per slot; every slot is decided before anything runs. A decline
  runs `fallback` (the `_` line of `@branch`) or throws the new `DecisionDeclinedError`,
  which carries the question name and the full answer. Any other return is an
  `ArgumentError`, and nothing runs. `nl_dispatch` refuses `decide` together with a
  nonzero `min_confidence` before any request; `@branch` refuses `decide` together
  with any `min_confidence` while the macro expands.
- `meanings(f, argtypes)`: the options `nl_dispatch` sends when its ordinary arguments
  have the types in `argtypes`, e.g. `meanings(step, Tuple{Waiting,String})`.
- `meaning_gaps(f, argtypes)`: the combinations of offered meanings that no method
  covers (missing or ambiguous), found without a request.
- Three guides: *Semantic Programs with Jev* (loss-matrix decisions, catch-all
  options, tail probabilities, taxonomies as type hierarchies, typed extraction,
  conversation state machines, tool guards and shortlists), *Semantic Algorithms with
  Jev* (keyed batches, rankings, search and bisection, joins, streaming with an early
  stop) and *Developing and Testing with Jev* (recorded answers, coverage checks,
  semantic assertions). Every System One example in the manual runs during the
  build, from recorded live answers when no key is set.

### Changed
- The state `nl_dispatch` builds from its ordinary arguments lists them in argument
  order (a `JSON.Object`) instead of a `Dict`'s hash order, so the same arguments send
  the same request bytes on every Julia version.
- `nl_dispatch` works out its options, argument names and gap check once per type of
  `f`, types of the ordinary arguments and state of the method table, instead of on
  every call; a method defined later is offered from the next call on.
- `tool_loop!` and `tool_loop` let a `ReplayMissError` raised by a dispatcher propagate,
  as they do an `InterruptException`, instead of sending it to the model as a tool error
  while the loop goes on.

### Fixed
- A System One error whose object `detail` carries an `error_type` and no `message`
  (the service's 400 `{"detail":{"error_type":"max_tokens_exceeded"}}`) reported the
  raw JSON body as `SystemOneFailure.message`; the message is now the `error_type`,
  used only when no message string appears anywhere in the body (`detail.message`,
  `error`, a top-level `message`).
- `nl_dispatch` on a function whose methods take keywords: Julia records an unnamed
  positional argument of such a method under the empty name, so an unnamed slot got a
  blank question name (an `ArgumentError` before the request) and an unnamed ordinary
  argument the state key `""`. They are now `meaning_<index>` and `arg<index>`, as
  without keywords.

## 0.20.1

### Fixed
- An MCP session over Streamable HTTP failed the first call after an idle gap longer
  than the server's keep-alive timeout (5 s for uvicorn and Node.js servers by
  default). HTTP.jl handed the connection the server had closed to the next request
  without checking it and never replays a POST, so the call failed with a transport
  error, or with the 408 an HTTP.jl server writes on a connection it times out. Each
  `HTTPTransport` now keeps its own connection pool and redials a connection that has
  been idle for more than a second; `mcp_disconnect!` releases the pooled sockets.

## 0.20.0

### Breaking
- **Requirements:** Julia 1.13 or later, HTTP.jl 2.7.1 or later within the 2.x major,
  and JSON.jl 1.8.1 or later. HTTP.jl 1.x is no longer supported, and with it go the
  1.x-only caveats (the `connect_timeout = Inf` restriction and the process-global
  connection-pool cap). HTTP.jl 2.7.1 fixes an HTTP/2 flow-control leak in which every
  response body closed unread shrank the connection's receive window until later
  requests on that connection hung.
- **The Videos API is removed.** OpenAI shut the Videos API and the `sora-2` models
  down on September 24, 2026. `VideoObject`, `VideoList`, `VideoSuccess`,
  `VideoListSuccess`, `VideoContentSuccess`, `VideoFailure`, `VideoCallError`,
  `create_video`, `retrieve_video`, `list_videos`, `video_content` and the `:video`
  capability are gone.
- **Changed defaults.** Native Anthropic defaults to `claude-opus-5-5` (was
  `claude-opus-4-8`) with a default `max_tokens` of 16000 (was 4096): `max_tokens` caps
  thinking plus text, and current Claude models think by default, so 4096 ended turns
  at `"length"` before an answer. DeepSeek defaults to `deepseek-flash` for chat and
  FIM (was `deepseek-chat`, which DeepSeek discontinued on July 24, 2026).
- **Local validation throws before any network I/O**, streaming or not.
  `chatrequest!` and `respond` now let an encoder's `ArgumentError` propagate — an
  option the provider or model cannot express (the GPT-6 and GPT-5.6 rules, the
  Anthropic and Gemini field mappings, the Interactions field set) — where 0.19.0
  returned it inside an `LLMCallError` / `ResponseCallError`. With `history=true`,
  `chatrequest!` throws `InvalidConversationError` for a conversation that ends with an
  assistant message, instead of billing a call whose reply could not be appended.
  `chatrequest!` throws `ArgumentError` for a `callback` or `on_tool_call`, and `respond`
  for a `callback`, passed without `stream=true`: they run only on a stream, and were
  silently ignored. `tool_loop!` passes them through, so it raises the same error.
  `chatrequest!(; kwargs...)` takes either `messages` or both `systemprompt` and
  `userprompt` and throws `ArgumentError` otherwise (it returned a fabricated
  `LLMFailure(status=499)`); it copies `messages` instead of emptying the caller's
  vector.
- **Stricter construction.** `Chat(n=…)` must be 1: a result carries one choice, and
  extra choices were silently dropped (or, streamed, merged into one garbled text).
  `Chat` and `Reasoning` reject a reasoning effort outside `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, `max`; `Chat` rejects `top_logprobs` outside [0, 20] and
  `logit_bias` values outside [-100, 100] (the field now takes any
  `AbstractDict{String,<:Real}`); `Message` rejects roles other than system, user,
  assistant and tool; `Embeddings` rejects an `encoding_format` other than `"float"`;
  `Respond` rejects `conversation` together with `previous_response_id`, and
  `background=true` with `store=false`; `FIMCompletion(stream=true)` throws, since
  `fim_complete` has no streaming path. Audio and upload requests are checked against
  the documented values: `TranscriptionRequest.response_format` (json, text, srt,
  verbose_json, vtt, diarized_json) and `temperature` (0–1), `SpeechRequest.speed`
  (0.25–4), `translate` (no `languages`, `keywords` or `diarized_json`) and
  `create_upload`'s `purpose`. `RequestConfig` and the MCP per-call `timeout` reject a
  finite value above 1e9 s (use `Inf` to disable a bound).
- **Stopping a stream is a typed cancellation.** Setting the callback's
  `close[] = true` — now honoured from any task, and at once — ends the call with
  `LLMCallError` / `ResponseCallError` (`status = nothing`, `cause =
  UniLMCancelled(:callback, …)`), where it returned `LLMFailure` / `ResponseFailure`
  with `status == 200` and the raw partial wire. A stop that arrives after the
  provider's completion marker keeps the turn: it is committed, the final-message
  callback runs, and usage may be missing (on the OpenAI wire the marker is the chunk
  carrying `finish_reason`, which precedes the usage chunk and `[DONE]`). An exception
  thrown by a streaming `callback` or `on_tool_call` now ends the call with that
  exception in `cause` — never retried, nothing appended; `on_tool_call` exceptions used
  to be logged and ignored.
- **Result accessors throw on failures.** `output_text` (which returned the error text
  as if it were model output), `embedding_vectors` (a `MethodError`), `image_data` (an
  empty list) and `fim_text` (`""`) throw `LLMResultError` on a non-success result, as
  `text` already did. `LLMResultError.result` now holds any `LLMRequestResponse`.
- **`fork` deep-copies every field except `service`.** It copied only `messages`, so a
  fork's `push!(fork.tools, t)`, a `metadata` edit or a `stop` change also changed its
  parent and siblings.
- **Tool loops.**
  - `tool_loop!` runs tool calls only on a turn whose `finish_reason` is
    `"tool_calls"`. A turn with calls that finished otherwise (`"length"`,
    `"content_filter"`, a provider-specific value) may hold partial calls: none runs, the
    loop returns `completed=false` with an `llm_error` naming the reason, and the
    unanswered assistant turn is removed from the chat so it stays sendable. A call cut
    off before its arguments formed a JSON object is dropped from such a turn by the
    decoder; a turn left without calls ends the loop as a text turn with that reason.
  - `tool_loop!` completes on a text turn only when it finished with `"stop"` or no
    reason, or with `"tool_calls"` but no calls (which used to resend the conversation);
    a content-filtered or otherwise cut-off text turn returns `completed=false` with an
    `llm_error` naming the reason, where it counted as completed.
  - `tool_loop!` on a `Chat` with `history=false` throws `ArgumentError`, and both
    loops throw `ArgumentError` for `max_turns < 1`.
  - On `max_turns` exhaustion, `response` is the last response the model sent, with
    `completed=false` and `llm_error = "max turns (N) exhausted"`; it was a fabricated
    call error.
  - A dispatcher result that is not a `String` is sent to the model JSON-encoded, not as
    its Julia `string` (`Dict("temp" => 21.5)` arrives as `{"temp":21.5}`, `nothing` as
    `null`).
  - The Responses loop chains a `Respond` that sets `conversation` through the
    conversation alone (the API rejects it together with `previous_response_id`);
    answers call arguments that are not a JSON object with an
    `"Error: invalid arguments: …"` output instead of throwing; and stops with
    `completed=false`, running none of the turn's calls, when a turn requests a
    client-side action it cannot execute (`custom_tool_call`, `apply_patch_call`,
    `local_shell_call`, `computer_call`, a `shell_call` the platform did not answer with
    a `shell_call_output` in the same output, `mcp_approval_request`) — such a turn used
    to end the loop as completed. The `llm_error` of a turn that did not complete names
    the `incomplete_details` reason with its status
    (`"Response did not complete (status=incomplete, reason=max_output_tokens)"`).
- **Platform verbs.**
  - `upload_file` makes a single attempt; `max_attempts` no longer applies. A POST that
    timed out, or drew a gateway 5xx after the backend stored the file, was retried and
    stored the file twice. A `FileFailure` / `FileCallError` therefore does not prove
    that no file was created.
  - `delete_file`, `delete_vector_store`, `delete_container` and
    `delete_conversation` report success only when the reply carries
    `"deleted": true`; anything else is the family's `*CallError`.
    `delete_conversation_item` returns a `ConversationSuccess` holding the updated
    conversation the service sends back.
  - `poll_batch` and `poll_file_batch` bound the poll by wall-clock time (`timeout`),
    capping each GET and pause at the time left; they poll through retryable statuses
    and per-attempt timeouts, require `interval` and `timeout` > 0 (`ArgumentError`),
    and on timeout return the family's call error with `cause =
    UniLMTimeout(:deadline, …)` and the last object seen in the new `last_observed`
    field.
  - `moderate` returns a `ModerationCallError` (on which `is_flagged` throws) for a
    `200` that does not carry exactly one result row per input: such a reply decoded as
    a success that was not flagged, so unmoderated content passed a moderation gate.
  - `add_container_file` returns a `ContainerSuccess` whose `response` is a new
    `ContainerFileObject`; `ContainerSuccess.response` is
    `Union{ContainerObject,ContainerFileObject}`.
  - `ChoiceQuestion`'s constructor validates descriptions the way `choice` does, and a
    System One answer without a string `type` fails to decode.
- **Result types gained fields**, so their positional constructors changed (keyword
  construction is unchanged): the `*Failure` types of the Files, Vector Stores,
  Conversations, Moderations, Audio, Batch, Fine-tuning, Containers and Uploads APIs,
  `ImageFailure`, `ImageCallError`, `SystemOneCallError` and `RealtimeCallError` carry
  `request_id`; the `*CallError` types of those APIs, `ImageCallError`,
  `SystemOneCallError` and `RealtimeCallError` carry `cause`; `BatchCallError` and `VectorStoreCallError` carry
  `last_observed`; `ImageObject` carries `url`. `MCPServer` dropped its internal
  `_initialized` field and gained a registry `lock`.
- **`DEFAULT_PRICING` is a lock-guarded `AbstractDict{String,PriceRow}`**, not a
  `Dict`: every Chat success reads it, streaming ones on their own task, while callers
  may add rows, and a `Dict` is not safe to read during a write. It keeps
  `DEFAULT_PRICING[model] = row`, `get`, `haskey`, `delete!`, `pop!`, `empty!`, `keys`,
  `length` and snapshot iteration; `copy`, `merge` and `filter` return a plain `Dict`,
  and `DEFAULT_PRICING isa Dict` is false. `estimated_cost(…; pricing)` accepts any
  `AbstractDict{String,PriceRow}`.
- **Native Anthropic fails closed.** A `Chat` field the Messages API cannot express —
  `seed`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`,
  `logit_bias`, `verbosity`, `store`, `prompt_cache_key`, `stream_options`,
  `prediction`, `modalities`, `audio`, `web_search_options` — throws `ArgumentError`
  naming the fields instead of being dropped, `n` must be 1, a `json_object`
  `response_format` throws (there is no schema-less JSON mode), an unknown
  `tool_choice` string throws (it became `auto`), and `metadata` may carry only
  `user_id`. Requests Claude answers with HTTP 400 are refused locally: `temperature`
  outside [0, 1]; any `top_p`, or a temperature other than 1.0, on Opus 4.7 and later;
  a forced `tool_choice` on Opus 5.5, Fable 5.1 and Mythos 5.1; a
  trailing assistant turn (prefill) from the 4.6 generation on; and a
  `reasoning_effort` the model does not take. Because `parallel_tool_calls` defaults to
  `false` whenever `tools` is set, every Anthropic tool request now sends
  `tool_choice.disable_parallel_tool_use = true` (at most one tool call per turn); set
  `parallel_tool_calls = true` for the previous wire.
- **MCP.**
  - A call on a closed or not-connected session throws the new
    `MCPSessionClosedError` (cause `:disconnected`, `:timeout` or `:crash`) instead of
    an `ErrorException`, on both transports; an HTTP session no longer keeps working
    after `mcp_disconnect!`.
  - The inferred-schema `register_tool!(server, name, description, handler)` binds the
    `arguments` object to the handler's positional parameters by name, as `@mcp_tool`
    does, and rejects a handler that takes one `Dict` or varargs with `ArgumentError`
    (register those with an explicit schema). Both bindings validate instead of
    coercing: a missing required argument or a value of the wrong JSON type is answered
    with an `isError: true` tool result naming the argument.
  - An exception from a resource or prompt handler is answered with a generic JSON-RPC
    `-32603` `"Internal error"` and logged on the server, instead of relaying its text
    (file paths, argument values) to the peer.
  - `mcp_tools` / `mcp_tools_respond` advertise provider-safe tool names: any character
    outside `[A-Za-z0-9_-]` becomes `_` and names are cut to 128 characters (the call
    still goes out under the MCP name); two tools that map to one name raise
    `ArgumentError`.
- **Realtime.** `realtime_connect` throws `ArgumentError` before any I/O for a service
  other than `OPENAIServiceEndpoint` — it always dialled OpenAI, so another endpoint's
  credentials would have been sent to `api.openai.com` — and `mint_realtime_secret`
  returns `RealtimeCallError` for a `200` without a non-empty secret.

### Added
- **Cooperative cancellation:** `CancelToken`, `cancel!`, `iscancelled`, `with_cancel`
  and the `UniLMCancelled` exception. A `cancel` keyword (default: the ambient
  `with_cancel` token) on `chatrequest!`, `respond`, `embeddingrequest!`,
  `generate_image`, `edit_image`, `fim_complete`, `prefix_complete`, `ask`,
  `list_models`, `poll_batch`, `poll_file_batch`, `tool_loop!` / `tool_loop`,
  `nl_dispatch` and `@branch`; every other HTTP verb observes the ambient token, which
  propagates into spawned tasks. A cancel — before connecting, during the header wait,
  mid-stream, during a retry backoff or a poll pause — ends the call with its call-error
  result (`status = nothing`, `cause = UniLMCancelled(:token, …)`), never retried and
  never committed — unless the provider's completion marker already arrived on a
  stream: then the turn is committed, the final-message callback runs, and usage may be
  missing. A pre-cancelled token sends nothing. Tokens are level-triggered and
  `cancel!` runs every registered hook, even when one of them raises an interrupt. A
  TCP connect or TLS handshake in progress is not interrupted (it completes or reaches
  `connect_timeout`), and the Realtime WebSocket and MCP stdio exchanges do not observe
  the token; a cancelled MCP call over HTTP throws `UniLMCancelled` and sends the
  server a best-effort `notifications/cancelled`.
- `tool_concurrency` on `tool_loop!` and `tool_loop`: up to that many of one turn's tool
  calls run at once on spawned tasks, with their results sent back in call order.
- **MCP:** the `:queue` phase of `MCPTimeoutError` (a call's `timeout` also bounds its
  wait for the session); a `stderr` keyword on `mcp_connect(::Cmd)` and
  `StdioTransport` (an `IO` or a file path, appended to); an exit hook that tears down
  stdio servers still running when Julia exits, and a watcher that kills a server's
  whole process group when its leader exits (so killing an `npx` wrapper also stops the
  server it launched); a best-effort `notifications/cancelled` for an HTTP request that
  timed out; `client_version` defaults to the package version; custom `MCPTransport`
  subtypes can connect (they bound their own IO).
- **Anthropic:** structured outputs (a JSON Schema `response_format` becomes
  `output_config.format`), `FunctionSignature.strict` as the tool's `strict`,
  `reasoning_effort` mapped to `output_config.effort` plus adaptive thinking per model
  family (`"none"` disables thinking where it can be disabled), `service_tier`
  (`"auto"`, `"standard_only"`), and `safety_identifier` / `user` sent as
  `metadata.user_id`.
- **DeepSeek:** `reasoning_content` is kept on the assistant `Message` as
  `ProviderContent(:deepseek, …)` and sent back on requests that carry tools, as
  thinking-mode tool use requires; context-cache hits (`prompt_cache_hit_tokens`) fill
  `TokenUsage.cached_tokens` and are priced at the cached rate.
- Price rows for the current Claude models (Fable 5.1, Mythos 5.1, Fable 5, Mythos 5,
  Opus 5.5, Opus 5, Sonnet 5 at \$2/\$10, and the 4.x family) and for `deepseek-flash`,
  `deepseek-v4-pro` and the legacy Flash names (DeepSeek peak rates; off-peak bills
  half). Anthropic's dated `-YYYYMMDD` ids resolve to their alias row, and an unpriced
  model id logs one warning. `token_usage` reads the usage an `ImageSuccess` carries,
  and FIM results report usage and cost.
- OpenAI and DeepSeek streams request `stream_options.include_usage` when it is unset,
  so a streamed turn carries usage and accrues cost like a non-streamed one.
- `ImageObject.url`: `image_data` returns each image's base64 data or, when it was
  delivered by URL, its URL. `limit` and `after` on `list_fine_tuning_events` and
  `list_fine_tuning_checkpoints`.
- The provider extension API — `get_url`, `auth_header`, `default_model`,
  `encode_request`, `decode_response`, `handle_sse_event!`, `StreamState`,
  `encode_agentic`, `decode_agentic`, `decode_agentic_stream`, `AgenticStreamState` —
  is declared `public` (not exported) and documented in the new Extension API reference.
- Documentation: a "Concurrency, Tasks and Cancellation" guide (what a concurrent call
  may share, fan-out patterns, streaming into a `Channel`, cancellation, the thread
  layout, HTTP, tool loops and MCP under load).

### Changed
- `Retry-After` is a floor under the jittered backoff instead of a replacement for it:
  clients that received the same header slept exactly the header's wait and retried in
  lockstep. The spread above the floor is at most the exponential backoff and at most
  half of the budget left after the floor, so the next attempt keeps at least half of
  what remains; a header whose own wait does not fit `total_deadline` still returns the
  last real response at once.
- Streams use HTTP/1.1, one connection per stream; non-streaming requests keep protocol
  negotiation (HTTP/2 over TLS).
- `stream_idle_timeout` measures wire idleness only: time spent inside a streaming
  callback or `on_tool_call` is not counted, and the clock restarts when it returns.
- An in-band stream error on the OpenAI wire that carries a numeric `code` from 400 to
  599 is an `LLMFailure` with that `status` (other in-band errors stay an
  `LLMCallError`), so a retryable one (`408`, `429`, `500`, `502`, `503`, `504`, `529`)
  is retried with a new request while no callback has run yet, like its HTTP-status twin.
- `update!` on a `Chat` with `history=false` logs at debug level instead of warning on
  every successful call: leaving the chat unchanged is what `history=false` means.
- Chat decoding keeps the provider's finish reason on a tool-call turn (`"tool_calls"`
  stands in only for `"stop"` or none), and a stream without a finish reason reports
  `nothing`. Partial calls are dropped: on a turn that keeps another reason, a call
  whose arguments were cut off before they formed a JSON object is removed and the turn
  kept with its text, reason and usage; under `"tool_calls"` such arguments are an
  `LLMCallError`, since those calls would run. Native Gemini maps `STOP` to `"stop"`,
  `MAX_TOKENS` to `"length"` and the
  content filters, the image ones included (`IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`,
  `IMAGE_RECITATION`), to `"content_filter"`; any other
  `finishReason` is reported as its lowercased wire value (`"malformed_function_call"`)
  instead of `"stop"`, and a candidate without one as `nothing`. `reasoning_summaries` also reads
  Gemini Interactions `thought` steps.
- A native Anthropic system message after the start of a conversation stays in place as
  a mid-conversation system message on models that accept one, where its placement is
  valid; anywhere else it is hoisted into the top-level `system` prompt as before.
- Requests send a refusal as `{"content": null, "refusal": …}` and never send the
  response-only `finish_reason`. `Respond(tools=…)` converts a Chat `Tool` to the
  equivalent `FunctionTool`, and `Chat(tools=Tool[])` stores `nothing`.
- MCP: stdio lines that are not JSON are skipped with a warning; `MCPError.data` holds
  any JSON value; the disconnect `DELETE` carries `MCP-Protocol-Version`; the server negotiates
  the protocol revision the client requests when it supports it, answers `400` over HTTP
  to a request whose `MCP-Protocol-Version` names an unsupported revision, runs
  `serve(:http)` handlers concurrently on the default thread pool (handlers must be
  thread-safe; protocol requests stay inline), runs stdio handlers one at a time with
  the process `stdout` pointed at `stderr`, and accepts a JSON-RPC response sent to it
  without answering (HTTP `202`).
- Capability errors name the endpoint (they printed `DataType` for marker types); a
  custom endpoint without a `default_model` method gets the documented `ArgumentError`
  instead of a `MethodError`; the OpenAI-wire endpoints (OpenAI, Azure, the Gemini
  compat shim, DeepSeek, generic) declare `:streaming` and `:json_output`.
- `nl_dispatch` and `meanings` offer options in definition order (REPL input sorted
  `REPL[10]` before `REPL[2]`), and a method defined after the call began is callable.
- CI runs a single latest-stable Julia leg; CompatHelper has the write permissions it
  needs to open its pull requests.

### Fixed
- Every non-streaming call paid a ~100 ms latency floor: the request watchdog polled
  for completion every 0.1 s. Completion now wakes the caller directly, and a timeout
  lands within a few milliseconds of its bound.
- A busy main task stalled calls running on other threads: closing a watchdog timer
  waits for the event loop, which shares thread 1 with the main task. Timers are now
  closed off the caller's path.
- Over HTTP/2, concurrent streams to one host shared a connection and its flow-control
  windows, so a stream whose consumer applied backpressure starved its siblings, and a
  stream closed early could leave the connection stalled. Streams now each own an
  HTTP/1.1 connection.
- A slow stream consumer lost a healthy, fully delivered stream to `:stream_idle`,
  because the time spent in its callback counted as a byte gap.
- A cancelled HTTP request (`HTTP.CanceledError`) was classified as a retryable
  transport failure.
- Streaming parsed each SSE line and accumulated each tool call's arguments in
  quadratic time: a 4 MiB line read in 16 KiB chunks allocated about 1 GB. Native
  Anthropic stream blocks (16,000 deltas allocated 3.9 GB) and Gemini Interactions
  steps accumulated the same way. All three are linear now.
- An `InterruptException` raised while parsing a streamed payload, inside
  `on_tool_call`, while rendering an error, while parsing a TypeSafe error body, while
  encoding a Gemini tool result, or inside an MCP server handler was swallowed or
  converted; it propagates.
- Chat decoding dropped text sent alongside tool calls, failed on array-shaped message
  content and on empty tool-call arguments (`""` is `{}`), relabelled a tool-call turn
  cut at `"length"` or filtered as `"tool_calls"` (so a tool loop could run partial
  calls), failed a turn cut off inside a call's arguments with an `LLMCallError` (a
  JSON parse error, its billed usage never accrued), invented `"stop"` for a stream
  that sent no finish reason, and did not treat an in-band OpenAI-wire `{"error": …}`
  payload as terminal. `request_id` also falls back to the `request-id` header, which
  Anthropic sends.
- Before its first byte, a stream was bounded by `stream_idle_timeout`, not by
  `min(request_timeout, remaining total_deadline)`: closing the stream does nothing in
  HTTP.jl 2.x while the request is uploading or the response headers are awaited, so a
  peer that never answered held each attempt until the read-idle timer fired (120 s by
  default, however small `request_timeout` was). The first-byte deadline now also
  cancels the attempt's request context and ends it typed `UniLMTimeout(:request, …)`.
- A Chat stream waited for the end of its HTTP body after `[DONE]`: a server holding
  the body open delayed the final-message callback and the result as long as it held
  it, and a body that never ended cost `stream_idle_timeout`. The turn is final at the
  sentinel; the body's end is awaited for at most 0.5 s (it returns the connection to
  the pool), after which the connection is closed instead of reused.
- An `InterruptException` from a `tool_loop!` dispatch left the assistant turn — and
  any tool results already appended — in the chat, so the next request carried tool
  calls without results. The interrupted turn is now removed first, with one tool at a
  time or `tool_concurrency > 1`.
- `ResponseCallError.cause` is set for every exception from `respond` and the lifecycle
  operations, not only for timeouts.
- A Chat `Tool` wrapped in a `CallableTool` — the form a Chat tool loop takes — went
  out from `Respond` in the Chat shape (the function nested under `function`), which
  the Responses API rejects. It now converts to the equivalent `FunctionTool` like a
  bare `Tool`, keeping its callable, alone or in a mixed tool list.
- FIM: a `200` whose body was not JSON escaped `fim_complete` as an `ArgumentError`, a
  body without choices decoded as an empty completion, call errors lost their `cause`,
  Mistral FIM was sent to `/v1/completions` (now `/v1/fim/completions`, with its
  `message.content` choices read), and `prefix_complete` replaced the prefix in history
  with the bare continuation.
- Native Anthropic advertised `:json_output` yet dropped `response_format`,
  `reasoning_effort` and `strict`; it now maps them. `model_context_window_exceeded`
  maps to `"length"`; a refusal carries no partial content, tool calls or captured
  blocks, and its `refusal_message` is the explanation Anthropic sends; a `200` without
  a content array or a `stop_reason` is an `LLMCallError`, not an empty success;
  `prompt_tokens` counts cache writes and `reasoning_tokens` comes from
  `thinking_tokens`; an assistant turn with neither text nor tool calls, which the API
  rejects, is left out of the next request.
- DeepSeek multi-turn tool loops failed with HTTP 400, because the earlier turns'
  `reasoning_content` was not sent back; the old default `deepseek-chat` had no price
  row, so its cost read `\$0`.
- Native Gemini: function calls under `MAX_TOKENS` or a safety filter read as
  `"tool_calls"`; a model turn with neither text nor function calls (a refusal, a turn
  spent on thinking) broke the follow-up request; a streamed refusal repeated its text
  once per trailing chunk. Gemini Interactions: a body without an `id` or `status` —
  non-streamed or as a streamed terminal — decoded as an empty success; streamed thought
  steps lacked their `summary`; streamed text parts were merged into one; `null`
  function arguments read as `"null"` (now `"{}"`).
- MCP client: a looping caller could starve a queued one (the session lock was not
  fair), a woken waiter that was interrupted stranded the callers behind it, a waiter's
  wait was unbounded, and an exception while a caller waited (a huge `timeout`, an
  interrupt) could wedge the session so that even `mcp_disconnect!` hung; the lock is
  now a FIFO hand-off. A stdio reply arriving after the request watchdog fired was
  returned as a success, and the timeout surfaced up to ~7 s after its bound; blank
  stdout lines read as EOF (a spurious `MCPCrashError`); a busy stdio server outlived
  its client process; Streamable HTTP treated SSE priming events as frames and ignored
  frames after the response; a concurrent `list_changed` could be lost to a refresh;
  and `mcp_disconnect!`'s `DELETE` (like a cancellation notice) was skipped inside a
  cancelled `with_cancel` scope.
- MCP server: registering a tool, resource or prompt while `serve` dispatched could
  corrupt the registry and crash the process; `serve(:http)` ran handlers on HTTP.jl's
  interactive pool, where a busy handler stalled the event loop; schema inference threw
  on `where` signatures and some unions; the by-name binding failed on `Symbol`,
  `Vector` and `Dict` parameters it advertised; a handler that printed corrupted the
  stdio protocol stream. The client reported version `0.8.0` in `clientInfo`.
- Realtime: an upgrade that completed just after `realtime_connect` had thrown
  `UniLMTimeout(:connect)` still ran the handler, and an open that failed just before
  the bound was reported as that timeout instead of its own error.
- `save_file_content` and `save_audio` truncated the destination before writing, so a
  failed write left a corrupt file. They now write a temporary file and rename it into
  place, keeping the destination's permission bits and writing through a symlink
  (dangling included); a directory destination throws `ArgumentError`.
- Platform: `image_data` returned `[]` for URL-delivered images; the poll helpers
  counted iterations instead of time, ended on the first transient failure and threw
  `InexactError` for `interval = 0`; `add_container_file` decoded the container-file
  object as a container; the fine-tuning event and checkpoint lists could not page.
- Azure read the deployment variable of `gpt-5.2` only; `AZURE_OPENAI_DEPLOY_NAME_<MODEL>`
  is read for every model, and a model without a deployment fails naming the variable
  instead of with a `KeyError`.
- The `Embeddings` missing-model error named `DataType` instead of the service.
- Documentation matches the code: examples that threw or no longer ran (the
  `chatrequest!(chat) do … end` form, which does not exist; examples that printed
  `output_text` of a failed result; the inferred-schema `register_tool!` example) are
  corrected, as are the claims on timeouts, retries, cancellation, cost accounting, MCP
  and the provider mappings.

## 0.19.0

### Breaking
- `ImageGeneration` no longer has an `input_fidelity` field: it is an image-edit
  parameter that `/images/generations` does not accept, so `generate_image` never
  sent anything the provider could act on. Set it on `ImageEdit` instead. Passing
  `input_fidelity=` to `ImageGeneration` or `generate_image` now throws
  `MethodError`, and the positional `ImageGeneration` constructor takes 11
  arguments instead of 12.

### Added
- GPT-6 Sol and Luna restrictions on the native OpenAI endpoint fail before a
  request is sent: Chat tool calling requires `reasoning_effort="none"` (as for
  GPT-5.6); unless reasoning effort is `"none"` (an omitted effort counts as the
  provider default, `"medium"`), `temperature`, `top_p`, `top_logprobs`, Chat
  `logprobs=true` and a Responses `include` of `"message.output_text.logprobs"`
  are refused; `prompt_cache_retention` is refused for every `gpt-6*` model.
  `encode_request` / `encode_agentic` throw `ArgumentError`; `chatrequest!` /
  `respond` return it as `LLMCallError` / `ResponseCallError`.
- `ModerationConfig` and a `moderation` field on `Chat` and `Respond` for OpenAI
  moderated completions: a required moderation `model` (for example
  `"omni-moderation-latest"`) plus an optional `"score"` / `"block"` policy mode
  for the input and output sides. A `respond` result carries the outcome in
  `r.response.raw["moderation"]`.
- `PromptCacheOptions` (now defined alongside `Chat`) gains two Responses-only
  options: `prewarm` prepares the prompt cache without generating output, and
  `comparison_response_id` requests prompt-cache diagnostics, returned in
  `r.response.raw["prompt_cache_diagnostics"]`. `Chat.prompt_cache_options`
  carries `mode` and `ttl` to Chat Completions for gpt-5.6 and later (older
  models answer 400: `gpt-5.4-mini` did on 2026-09-22); setting a Responses-only
  option on a native OpenAI `Chat` throws `ArgumentError`, and `mode="explicit"`
  on a `Chat` means no prompt caching because Chat messages cannot carry cache
  breakpoints.
- `input_text(text; cache_breakpoint=true)` marks an explicit prompt-cache
  breakpoint, and `configuration_update(; effort)` builds the GPT-6
  `configuration_update` input item that changes reasoning effort mid-conversation
  without rewriting the cached prefix.
- `FunctionTool` gains `async`, `allowed_callers` (validated: `"direct"` /
  `"programmatic"`), `defer_loading` and `output_schema`, and `function_tool`
  (dict and keyword forms) carries them. `ImageGenerationTool` gains `model`,
  `action`, `moderation`, `partial_images`, `input_fidelity` and
  `input_image_mask`; `action`, `moderation`, `partial_images` and
  `input_fidelity` are validated at construction. `output_schema` and
  `input_image_mask` accept any `AbstractDict`. Existing positional constructors
  keep working.
- Native Gemini structured output: `Chat(service=GEMINIServiceEndpoint,
  response_format=…)` sends JSON-object and JSON-Schema formats as
  `generationConfig.responseFormat`, and `respond(…; service=GEMINIServiceEndpoint,
  text=json_schema_format(…))` (or `json_object_format()`) sends the Interactions
  `response_format`. `GEMINIServiceEndpoint` now declares `:json_output`. Other
  shapes, including a set text `verbosity`, raise `ArgumentError`; the schema
  `name`, `description` and `strict` have no Gemini counterpart and are not sent.
- Native Gemini `Chat` sends `safety_identifier` as the request label
  `safety_identifier`, checked locally against Google's label rule (at most 63
  characters of lowercase letters, digits, `_` and `-`), so a 64-character hex
  digest throws `ArgumentError` before any request.
- Price rows for `gpt-6-sol`, `gpt-6-luna`, `gpt-5.4-nano`, `gemini-3.6-flash`,
  `gemini-3.5-flash-lite` and `gemini-embedding-2` (text input).

### Changed
- Gemini thinking levels are validated per model family against Google's
  thinking table, matching the longest family name: for example
  `gemini-3-pro-preview` accepts only `low`/`high` and
  `gemini-3.1-flash-lite-image` only `minimal`/`high`. Models outside the table
  keep the previous check.
- Gemini Interactions rejects a `FunctionTool` whose `async`, `allowed_callers`,
  `defer_loading` or `output_schema` is set, and native Anthropic rejects a `Chat`
  whose `moderation` or `prompt_cache_options` is set, instead of dropping the
  field silently.
- GPT-6 Astra accepts `logprobs=false` (only `logprobs=true` and `top_logprobs`
  are refused), matching the observed provider behaviour.
- The System One guide, API page and cost-tracking guide now execute their
  single-shot `ask` and `list_models` examples during the documentation build,
  like the OpenAI examples; a live documentation build renders real answers.
- Documentation: the `Reasoning` effort set is `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, `max`; Responses `service_tier` lists `scale` and
  `ultrafast`; reusable prompts (`v1/prompts`) shut down on November 30, 2026;
  the Videos API and `sora-2` models shut down on September 24, 2026 (the
  wrappers stay and return provider errors from that date); image `quality`
  accepts `xhigh` and `max` on the 2.5 models and `gpt-image-2` and later accept
  arbitrary `WIDTHxHEIGHT` sizes; the model table adds GPT-6 Sol/Luna and
  `gpt-5.4-mini` and drops `o3`; the Agentic Workflows examples use
  `gemini-3.8-flash` (`gemini-3.1-flash-lite` shuts down May 7, 2027);
  `gemini-embedding-001` shuts down May 14, 2028 and `gemini-embedding-2` works
  through `GEMINIOpenAIServiceEndpoint`; the README notes that the Agents API and
  GPT-Live sessions are not wrapped.

### Fixed
- `estimated_cost` prices a versioned Jev id that has no row of its own (for
  example `jev-1.14.0`, including on `SystemOneSuccess` results) at the
  `jev-latest` rate instead of `0.0`; TypeSafe lists one price (Jev 1.13), and a
  later version is assumed to keep it until the pricing page says otherwise.
- Five documentation examples rendered misleading output (an empty block for the
  FIM failure accessors, a leaked Azure deployment entry across pages, a validity
  comment that contradicted its output, a struct dump after `push!`, and a
  compaction example on the flagship model); each now renders what its prose says.

## 0.18.0

### Added
- TypeSafe System One (Jev) client on `POST /v1/systemone`: `ask` evaluates every
  question against one ingestion of the state in a single request, with the three
  question primitives `choice` / `score` / `noul` and the matching typed answers
  `ChoiceAnswer` / `ScoreAnswer` / `NoulAnswer` (an unrecognised answer type is
  preserved as `UnknownAnswer` rather than dropped). Results are
  `SystemOneSuccess` / `SystemOneFailure` / `SystemOneCallError`; the accessors
  `answers`, `answer`, `getindex`, `haskey` and `keys` throw `SystemOneError` on a
  failed call instead of reading as an empty set of answers. `list_models` lists
  the aliases the account may name, and `token_usage` / `estimated_cost` price a
  call against the versioned model that answered — only input tokens are billed.
  The endpoint type is `TYPESAFEServiceEndpoint` (`TYPESAFE_API_KEY`,
  `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`), which declares only
  `:system_one` and `:models`, so the chat, agentic and embedding verbs reject it
  up front.
- `SystemOneFailure.retry_after`: the wait a rate-limited or overloaded response
  asked for, in seconds, read off the final response from `retry-after-ms` (which
  wins, and keeps sub-second precision) or `Retry-After` (delta-seconds or an
  HTTP-date), and `nothing` when neither arrived or neither parsed. The retry seam
  already waits out `Retry-After` between attempts; the field is what a caller
  needs to schedule its own next attempt once `max_attempts` is spent, and
  `showerror` on a `SystemOneError` reports it.
- Natural-language control flow on top of that client. `@branch` is a `switch`
  whose cases are written in plain language: the option names become the criteria
  of one Choice question, only the selected body is evaluated, and a `_` line
  catches an answer below `min_confidence`. `nl"..."` lifts a description into the
  type domain as a `Meaning`, so a meaning is writable in an ordinary method
  signature; `nl_dispatch` resolves every `Meaning` slot of a generic function in a
  single request and calls the method Julia's own dispatch selects, leaving the
  remaining arguments to dispatch on their types. `meanings` previews the options
  exactly as they will be sent, and `LowConfidenceError` is raised when an answer
  falls below the threshold and no fallback was given.
- A "System One (TypeSafe Jev)" documentation section: a guide to typed judgments
  with Jev, a guide to multiple dispatch on natural language, and an API reference
  page.

### Fixed
- Auth-header masking indexed the result of its own re-match directly. `match`
  is typed `Union{Nothing,RegexMatch}` and a capture group `Union{Nothing,SubString}`,
  which Julia 1.13 surfaces as a type instability in the one renderer every
  `*CallError` passes through. Both are now guarded: an unmatched hit is handed
  back unchanged instead of being indexed.

## 0.17.0

### Breaking
- Default models are now `gpt-5.6-sol` for OpenAI chat and Responses,
  `gemini-3.8-flash` for Gemini, and `gpt-transcribe` for transcription.
  Set `model` explicitly to retain a previous model where the provider still
  offers it.
- Native Gemini 3.8 rejects unsupported sampling controls locally. Native Gemini Chat
  now rejects unmapped options instead of silently ignoring them. Remove
  `temperature` and `top_p` for Gemini 3.8, and use `reasoning_effort` to control
  thinking. Remove other unsupported options reported by validation.
- GPT-6 Astra requests validate unsupported reasoning and sampling options;
  tool callers receive guidance to use the Responses API.
- GPT-5.6 Chat tools require explicit `reasoning_effort="none"`; reasoning with
  tools uses `Respond` and `respond` instead.
- For GPT-5.6 and later, replace legacy `prompt_cache_retention` with typed
  `PromptCacheOptions`, passed through `Respond(prompt_cache_options=...)`.

### Changed
- Documentation builds and releases run examples offline by default. Live
  examples require the manual Documentation workflow's `live_examples` input.

### Added
- Provider availability notices for OpenAI video retirement and fine-tuning restrictions.
- OpenAI persisted reasoning (`Reasoning.context`), execution mode
  (`Reasoning.mode`), and validated `PromptCacheOptions` for GPT-5.6 and later.
- Gemini thinking control through `Chat.reasoning_effort` and `Respond.reasoning`,
  including Interactions thought summaries with `Reasoning(summary="auto")`.
- Typed transcription `languages` and `keywords`, with native multipart encoding
  and compatibility for a singular language hint.
- Current GPT-6, GPT-5.6, and Gemini 3.8 price estimates and inexpensive live
  coverage using GPT-5.6 Luna and Gemini 3.8 Flash.

### Fixed
- Tool loops preserve truncated or incomplete responses as unfinished results
  and do not execute partial function calls.
- Webhook verification rejects NaN and negative replay tolerances; only an
  explicit positive infinity disables the replay window.
- Gemini function schemas use `parametersJsonSchema`, allowing standard JSON
  Schema constraints that the restricted `parameters` field rejects.
- Streamed Gemini chat preserves all provider parts and thought signatures;
  thought summaries are excluded from answer text and in-band errors terminate.
- Interactions streaming preserves initial text in `step.start` and the order
  of text and tool steps, using per-step buffers for text accumulation.
- Deprecated `Reasoning.generate_summary` serializes as `summary`; conflicting
  aliases raise `ArgumentError`. Chat token limits reject nonpositive values.
- Cost estimates recognize dated model snapshots instead of silently reporting
  zero when only the corresponding base model is priced.
- Azure deployment registry: a registered value that did not carry the
  `/openai/deployments/` prefix every registration writes was silently encoded
  whole and re-prefixed, producing a request path nobody registered. The reader
  now throws an `ArgumentError` naming the malformed entry.

## 0.16.0

### Fixed
- Gemini Interactions streaming: an interaction whose terminal status is
  `"failed"` reported terminal completion, so the driver built a
  `ResponseSuccess` and dropped the wire's error and metadata. It now takes the
  typed-failure limb, and the response is overlaid on a copy of the raw
  interaction, so a streamed result keeps the same status/error/metadata/raw
  surface the non-streamed decode preserves.
- Gemini Interactions: a step with `content: null` — spec-legal for a
  `model_output` that produced no parts — threw a `MethodError` while decoding.
  It was user-visible on the non-streamed path and silently swallowed as a
  dropped SSE line on the streamed one. Null or absent content now decodes as no
  parts.
- Gemini Interactions streaming: a stream whose text deltas arrived in the same
  read as `interaction.completed` delivered **zero** deltas to the callback,
  because the terminal rebuild consumed the driver's text buffer. The rebuild
  now reads the buffer back without consuming it.
- MCP over HTTP: the transport's SSE reader required `data: ` with the space
  (rejecting spec-legal `data:foo`) and emitted one frame per data line,
  corrupting any event whose payload spans several `data:` lines. The body now
  goes through the shared SSE machine, cutting events at blank lines and joining
  each event's data lines with a newline.
- MCP over HTTP: notifications omitted the dual `Accept` header the request path
  sends and ignored the response status, so a server's rejection of
  `notifications/initialized` vanished and left a session the client believed was
  initialized. Both paths now share one header set and one status policy; a
  non-2xx notification fails loud, naming the status and body, which means a
  rejected `notifications/initialized` now fails `mcp_connect`.
- MCP over HTTP: connect-phase exchanges (handshake discovery, status
  `:initializing`) were bounded by `mcp_request_timeout` instead of
  `mcp_connect_timeout`, aborting handshakes the connect budget still covered. A
  breach there is now classified `:connect` and names that override.
- MCP over HTTP: `_mcp_notify!` reached the transport without the session lock;
  it now takes it re-entrantly, so notifications are serialized with exchanges.
- Multipart uploads: the request seam owns retries and passes `retry=false`,
  which also disables HTTP.jl's body rewind, so a single form handed to the retry
  loop was left consumed by the first attempt and the second put a zero-length
  multipart body on the wire — a transient 429/503 became a hard protocol
  failure. Non-replayable bodies now travel as a factory and are rebuilt per
  attempt, re-reading from source rather than buffering a copy. `edit_image` also
  validates its file paths before the loop, so a missing file is reported once, up
  front.
- `Retry-After` in its HTTP-date form is now honored. RFC 7231 allows
  delta-seconds **or** an HTTP-date and servers behind CDNs send both; only the
  seconds form was parsed, so the date form was silently ignored and the client
  under-waited a 429 storm by whatever the header actually asked for. The parsed
  date goes through the same clamping and deadline arithmetic as the seconds
  form: a past date yields zero, and a malformed value still yields the default
  backoff — a server sending garbage must never make a client throw.
- Decoders no longer fabricate text for an empty turn. The OpenAI, Anthropic and
  Gemini decoders each substituted `"No response from the model."` whenever a turn
  produced no text — content the provider never sent, which then landed in the
  returned `Message` and in the next request's history. Reasoning models hit this
  routinely, spending the whole completion budget on thought tokens. An empty turn
  is now reported as the empty turn it is. A response with no candidates or
  choices at all remains a different thing — there is no assistant turn to report
  — and still fails loud into the verb's typed error result.
- OpenAI-wire tool calls are now decoded whenever they are present, whatever the
  finish reason. A tool-only reply finishing with anything other than
  `"tool_calls"` — which some providers do — previously fell into the empty-turn
  fallback, which fabricated prose *and* dropped the calls.
- MCP server: survives spec-legal parameter shapes (`params: null` and positional
  `params` no longer crash a request). Errors below the tool handler now answer
  JSON-RPC `-32603` with a generic message, with the exception and backtrace
  logged locally rather than shipped to the peer, since an exception string can
  carry file paths and argument values a remote client has no business reading.
  Frames and request bodies are capped at 16 MiB on both transports and an
  oversized one is answered rather than parsed — parsing an attacker-sized payload
  allocates a multiple of it, which is an out-of-memory kill rather than a
  protocol error. Tool-handler exceptions are unaffected: they still reach the
  client as tool results (`isError`), which is what lets a model correct itself.
- Embeddings: a response that misses or duplicates a row index is now an
  `EmbeddingCallError` instead of a zero vector. The buffers are pre-zeroed, so an
  uncovered slot stayed a valid-looking all-zero embedding that still compares,
  normalizes and indexes — a silent corruption. A response must now cover every
  input exactly once.
- `save_image` decodes the payload **before** opening the destination. `open(…,
  "w")` truncates, so decoding inside the block let a malformed payload destroy
  whatever already lived at the path and leave a 0-byte stub. A failed save now
  costs nothing.
- The Azure model-to-deployment registry is now lock-guarded. It is read while
  building every Azure request URL and written by `add_azure_deploy_name!`,
  potentially from another task; a `Dict` write that rehashes reallocates the
  arrays a reader is walking, which yields a wrong hit or a bounds error rather
  than merely a stale answer.
- The keyword form of `set_default_config!` is now atomic. It is a
  read-modify-write on the process default, so two concurrent calls read the same
  snapshot and the second write installed a state predating the first, silently
  dropping its field. The merge now runs in a compare-and-swap loop.
- API keys and request bodies are redacted before they reach a result value. A
  mid-exchange transport failure on HTTP.jl 1.x arrives as an error whose
  rendering is a full request dump — every header and the whole body — and the
  library masks only `Authorization`, `Proxy-Authorization` and `Cookie` there, so
  a provider authenticating with its own header (Anthropic `x-api-key`, Gemini
  `x-goog-api-key`, Azure `api-key`) had its key in cleartext inside the
  user-visible `.error` of the returned result, and from there in logs and bug
  reports. Every error-construction site now renders through one helper that
  prefers the root cause's own text and then masks auth-shaped header values as
  defense in depth. The same redaction covers debug/warn logs, the displayed form
  of MCP transports and sessions, and a minted Realtime client secret. Call-error
  results additionally define a `show` that names `cause` by TYPE instead of
  letting Julia's default recurse into it and reprint the dump; `.cause` itself is
  unchanged and still reachable.
- The auth-header mask is bound to header position. The pattern had no leading
  word boundary, so any longer word merely *ending* in a header name was matched
  and everything after it erased — a message containing `reauthorization: <text>`
  lost the text. Fail-safe, but destructive of diagnostics that never carried a
  credential. The real names (`authorization`, `proxy-authorization`, `api-key`,
  `x-api-key`, `x-goog-api-key`) are still masked in both the wire form and the
  Julia pair form.
- The MCP HTTP server answers a bodyless `POST` with the same `-32700` JSON-RPC
  parse error a malformed body gets, instead of crashing the handler into a 500.
  On HTTP.jl 2.x a payload-less request carries a typed empty body that supports
  no `length`; body size is now read through a shape-dispatched helper on both
  majors.
- Caller-supplied values interpolated into request URLs are percent-encoded. Ids,
  model names, pagination cursors and list filters went into path segments and
  query values raw across the platform APIs (files, batches, uploads, containers,
  conversations, vector stores, fine-tuning, videos), the Responses lifecycle
  operations (response ids, pagination cursors and filters), the Azure deployment
  path, the Gemini model-in-URL and the Realtime model query. A value carrying `/`, `?`, `#`, `&`, `=` or a space
  silently re-shaped the request target — extra path segments, a spurious query
  string or fragment, a smuggled parameter — or, on HTTP.jl 2.x, was rejected
  client-side as a synthetic 400 that never reached the server. Only the
  interpolated value is encoded; the templates' own separators and Gemini's
  `:generateContent` verb colon stay literal. Ordinary identifiers draw from the
  unreserved set, so the common case is byte-identical on the wire.
- `edit_image` follows the declared-endpoint capability rule the four primary
  verbs use. It validated unconditionally, so a user-defined endpoint — one that
  declares no capabilities, because the package does not ship it — raised a raw
  `MethodError` instead of dispatching. An endpoint that declares nothing now
  passes through to the wire; one that declares its capabilities and omits image
  edits still gets the typed `ArgumentError`.
- Agentic streaming no longer costs O(n²) in delivered bytes. Emitting a delta
  meant `take!`-ing the whole accumulated text plus a mirror of what had already
  been sent, slicing the tail and re-printing both — four copies of the full text
  per read. Deltas are now drained from a pending buffer the decoders append to,
  matching the chat driver. Measured on the emission loop at 200/400/800/1600
  reads: 0.98/3.4/12.7/49.4 MB allocated before, 33/63/129/261 KB after, with
  byte-identical output. Callers see the identical delta sequence.

### Added
- `sse_dropped::Int` on `LLMSuccess`, `LLMFailure`, `ResponseSuccess` and
  `ResponseFailure`. A `data:` payload the parser cannot read is dropped rather
  than re-queued — a re-queued partial line is the stream-poisoning mechanism the
  drop policy exists to prevent — but the only trace was a process-global counter
  no caller could attribute to its own request, so a turn assembled from a
  truncated wire was indistinguishable from a clean one. Every drop is now counted
  on the stream that saw it and the count rides the result. It defaults to `0`, so
  a non-streamed call reports `0` rather than nothing. Both drivers state a
  non-zero count once per attempt at finalize, naming the model and the surface —
  a warning, since an undecodable provider payload is anomalous by construction,
  not the per-line debug message. A retried attempt starts from fresh state and
  reports only its own. On a truncated stream (HTTP 200, no terminal event) the
  `LLMFailure` now carries the count that explains why no message could be built.

### Changed
These entries change behavior or public surface that callers may depend on. Most
turn a silently-succeeding path into a typed exception; the rest change a result
type or drop a field. They are collected here pending a version decision.

- `Respond` fields the Gemini Interactions wire does not map now throw
  `ArgumentError` at encode time instead of being silently dropped. Structured
  output via `text`, `reasoning`, `metadata` and others simply vanished between
  the caller's request and the wire. The unmapped set is derived from
  `fieldnames(Respond)`, so a newly added field is unsupported until deliberately
  mapped, and the error names the offending fields.
- `is_flagged` throws `ArgumentError` on a failed moderation call. A failed call
  has no verdict, and returning `false` made "not flagged" indistinguishable from
  "never checked".
- `verify_webhook` throws `ArgumentError` for a signing secret that is not valid
  base64, and rejects non-finite timestamps, instead of failing verification as
  though the signature were wrong. A malformed secret is a configuration error,
  not a failed verification.
- `token_usage` and `estimated_cost` throw `ArgumentError` for result types
  outside the token-billed APIs (audio, batch, files, moderations, vector stores,
  video, …), where they previously raised `MethodError`. Those calls report no
  token usage at all, and a `0.0` would be indistinguishable from a genuinely free
  call.
- `mcp_disconnect!` now takes the session lock, so a disconnect racing a call in
  flight waits for that exchange to finish instead of tearing the transport down
  under its reader — the same concurrency-1 semantics every other call obeys,
  bounded transitively by the exchange's own `mcp_request_timeout`.
- **`HTTPTransport` no longer has a `lock` field.** Nothing ever took it.
  Exchanges, notifications and the disconnect are all ordered by the session lock,
  which is the only ordering an HTTP transport needs, so the field was a public
  promise the type never kept. Code reading `transport.lock` must drop it.
- `realtime_connect` and `realtime_receive` are bounded by `RequestConfig`, and
  `RealtimeSession` gained a `config` field carrying the budget resolved at
  connect time. The WebSocket was the one surface outside the bound-everything
  guarantee: a peer that accepted TCP and never completed the upgrade, or a
  connected peer that went silent, blocked the caller forever. `connect_timeout`
  now bounds **only** the open phase and `stream_idle_timeout` bounds
  `realtime_receive`; a live session's lifetime stays deliberately unbounded,
  since it is the caller's to decide. The two-argument `RealtimeSession`
  constructor is retained and inherits the ambient config.
- A non-streamed generation whose terminal status is `"failed"` is now a
  `ResponseFailure` on both agentic wires, carrying the wire body verbatim so
  error and metadata are preserved. The streamed limb already did this, so
  `issuccess` depended on how the call was made rather than on what happened.
  Only `"failed"` flips: `cancelled`, `expired`, `in_progress`, `queued` and
  `requires_action` are legitimate terminals of the background and tool-action
  flows and remain successes.
- A **streamed** generation whose terminal event is `response.incomplete` is now a
  `ResponseSuccess`, not a `ResponseFailure`. The streamed driver routed that
  terminal to the typed-failure limb while the non-streamed path decoded the very
  same object as a success, so whether a truncated generation had failed depended
  on `stream` rather than on what happened. `response.incomplete` carries a real
  terminal response object with usable partial output, and the truncation is
  reported in `status` and `incomplete_details`; the streamed limb now finalizes
  it exactly like `response.completed` — same result type, same status and
  details, usage recorded, raw capture complete, deltas already delivered. Only
  `"failed"` is a failure, on both paths. The typed-failure limb still catches an
  `incomplete` terminal carrying no response object at all, which has no result to
  hand back. Code that tested `result isa ResponseFailure` to detect a truncated
  stream must read `response_status`/`incomplete_details` instead.
- The four primary verbs — `chatrequest!`, `embeddingrequest!`, `respond` and
  `generate_image` — now validate provider capabilities before dispatch, giving a
  typed `ArgumentError` instead of a provider 404. Validation applies only to
  endpoints that **declare** their capabilities (`respond` accepts either
  `:responses` or `:agentic`, since the two agentic wires name the same surface
  differently). A custom endpoint that declares nothing passes through
  unvalidated: refusing to dispatch a backend the package knows nothing about
  would be a false negative.
- `call_tool` and `get_prompt` accept any `AbstractDict` for `arguments`, so the
  natural `Dict("k" => "v")` form works without conversion.

## 0.15.1

### Fixed
- Streaming: the byte-gap idle watchdog read the clock before loading the
  last-byte stamp, so a chunk arriving between the two reads made the unsigned
  gap wrap to an astronomically large value and killed a healthy,
  actively-streaming connection with a spurious `:stream_idle` timeout. The gap
  is now computed stamp-first and saturating. The probability per check was
  tiny, but it scaled with concurrent streams, so sustained high-concurrency
  streaming workloads would eventually hit it.
- Streaming: a completed turn now survives connection-teardown noise on both
  streaming surfaces. Previously, once the terminal event had been delivered, a
  transport error raised while the socket wound down (a reset from the peer, a
  framing fault on the read that would have carried the end-of-stream sentinel)
  could either discard the finished generation (surfacing a call error after
  the user's callback already received the result) or — when no callback was
  registered — re-send the entire request, billing a second generation for one
  call. The recorded outcome now stands: finalization and the terminal callback
  are structurally at-most-once, teardown noise after completion neither
  discards nor re-POSTs, and the chat surface also recognizes the provider's
  recorded finish reason as completion when the connection dies on the very
  read that would have carried the closing sentinel. Failures that are not
  teardown-shaped (a throwing user callback, a decoding defect) surface
  exactly as before.
- Streaming: a watchdog kill that lands while the driver is inside a user
  callback truncates the socket read into a clean end-of-stream, so the read
  loop ended without an exception and the kill was reported as a status-200
  failure carrying partial bytes (or, with a recorded finish reason, as a
  silently truncated success). Both drivers now consult the guards on the
  no-exception exit path and surface the same typed `:stream_idle`/`:request`
  timeout the throwing path produces.
- Streaming: when the request-phase bound fires at the same instant the guarded
  call completes, the watchdog closes the socket without anything unwinding;
  the next read then raised a bare `IOError` that classified as a retryable
  transport failure — one extra billed wire attempt and a lost phase. The
  fired bound is now detected from guard state and surfaces as the typed
  `:request` timeout.
- Responses/Interactions streaming: a terminal event arriving as the very last
  bytes of the stream without a trailing newline was never dispatched (the
  chat surface already handled this), so a completed response surfaced as a
  status-200 failure. The agentic read loop now feeds the terminating newline
  at end-of-stream.
- Streaming on HTTP 2.x with `stream_idle_timeout=Inf`: nothing bounded the
  response-header wait (the request-phase watchdog's close is inert before
  response headers exist on that major), so a mute peer stalled the stream
  task indefinitely. The header wait is now natively capped at the effective
  request bound whenever the idle bound is disabled, and breaches surface as
  typed `:request` timeouts.
- `SystemError` now classifies as a connection-level transport error: a peer
  reset can surface as a raw `SystemError("read", ECONNRESET)` rather than an
  `IOError`, and previously was neither retried nor mapped to a typed cause.
- MCP: `mcp_request_timeout` now bounds a call's own exchange, measured from
  the moment it acquires the session lock — not from the moment it asked. A
  caller queued behind a concurrent exchange on the same session previously
  burned its bound while waiting and, on breach, group-killed the server in
  the middle of the holder's healthy exchange, which then surfaced to the
  holder as a fabricated server-crash error.
- MCP: the liveness check, auto-respawn, and the exchange now run under one
  session-lock acquisition. Two concurrent calls on a session closed by a
  timeout or crash previously could both observe it closed and both respawn,
  spawning two server processes — one of them orphaned beyond the kill
  ladder — and interleaving their handshakes on shared session state.
- MCP: once a server process is spawned, any connect or respawn failure now
  tears it down before the error propagates. Previously only timeout- and
  crash-shaped failures did (a stdout banner that breaks JSON-RPC framing, or
  an `initialize` error reply, leaked a live child process holding the
  session's pipes), and a failed respawn attempt relabelled every recorded
  close cause as a crash; the recorded cause is now preserved unless the
  attempt itself diagnosed a new one.
- Streaming: when the request-phase bound (first-byte deadline) closes a mute
  connection, the typed `UniLMTimeout` is now recorded before it unwinds
  through HTTP.jl and is restored as the surfaced cause. Previously — observed
  on the HTTP 1.x major under CI load — the library's teardown of the closed
  socket could raise its own transport error (for example a broken-pipe
  `IOError` from writing the terminating chunk), which replaced the in-flight
  typed exception, so the result carried a raw transport error as its cause
  instead of the promised `UniLMTimeout`. Transport errors that arrive with no
  bound fired (for example a refused connection) classify exactly as before.
  Chat streaming and the Responses streaming surface share the fix.

### Added
- `test/load_probe.jl`: standalone, opt-in concurrency/load harness (not part
  of `Pkg.test`) driving hundreds of concurrent mixed streaming and
  non-streaming calls against local mock providers, with cross-talk markers,
  descriptor/task-leak plateaus, retry-storm and teardown-noise batches.

## 0.15.0

### Breaking
- `push!(chat, msg)` and `pop!(chat)` now throw `InvalidConversationError` on an
  invalid mutation (a conversation not started by a system message, a system
  message after the start, consecutive same-role non-tool messages, or popping an
  empty conversation) instead of emitting a warning and returning the chat
  unchanged. The error message names the violated rule and the offending role and
  never includes conversation content or endpoint configuration. Valid mutations
  are unchanged.

### Breaking (provider extension)
- The OpenAI-wire request/response encoding and SSE handling are now an explicit
  opt-in via the new exported `OpenAIWireEndpoint <: ServiceEndpoint` supertype. A
  `ServiceEndpoint` subtype that does not implement the wire seam
  (`encode_request` / `decode_response` / `handle_sse_event!`, the agentic
  encode/decode/stream defaults, and `_agentic_url` routing) now fails with a
  `MethodError` at call time instead of
  silently emitting OpenAI-shaped requests at a foreign API. Migration: subtype
  `OpenAIWireEndpoint` instead of `ServiceEndpoint` for OpenAI-compatible
  backends; backends with a genuinely different wire keep subtyping
  `ServiceEndpoint` and implement the seam (see the new Custom Backends guide).
  All built-in endpoints are unaffected.

### Added
- `OpenAIWireEndpoint` abstract supertype (exported) and a "Custom Backends"
  guide documenting the provider-extension contract.
- A "Versioning & Stability" policy page (what 0.x means here, breaking-change
  batching, alias guarantees, 1.0 intent) plus a "When to choose UniLM"
  positioning section in the README.
- Result-consumption helpers: `issuccess` / `isfailure` for any request result,
  `text(::LLMSuccess)` returning the reply content (`nothing` for tool-calls-only
  turns), and `LLMResultError` — thrown by `text` on a `LLMFailure`/`LLMCallError`;
  its `showerror` reveals only the status and a trimmed response excerpt.
- Cost accounting: pricing entry for `gemini-3.7-flash` (introductory rate through
  2026-12-31; the output rate includes thinking tokens). The live Gemini suites now
  pin this model.
- "Release gate" workflow: on `release/**` branches and manual dispatch, runs a JET
  whole-package type-stability analysis and then the full test suite. JET is
  expensive, so it gates releases instead of running in routine push/PR CI; the
  workflow sets no provider keys and makes no billed calls. The analysis is its own
  step in its own process, ahead of the suite and without coverage instrumentation,
  so it reports on the package exactly as `using UniLM` leaves it — not on a session
  the test files have extended with mock endpoints and test-only method overloads,
  which widen dispatch and change what inference concludes about package code. The
  step fails on any package-code finding. The same analysis stays available inside
  the test suite for local use behind `UNILM_RUN_JET=true`, where it likewise runs
  ahead of the behavioural tests.

### Changed
- Type-strengthening pass driven by the new JET whole-package release gate: over-wide
  internal seam signatures were narrowed and URL-construction inference made concrete,
  bringing JET to zero findings. One user-visible effect: calling a platform API with a
  native (non-OpenAI-wire) endpoint now raises a descriptive `ArgumentError` instead of
  a bare `MethodError`.
- Streaming keep-alive test margins widened for loaded CI runners (test-only).
- Tool-surface types renamed to provider-neutral canonical names: `GPTTool` →
  `Tool`, `GPTToolCall` → `ToolCall`, `GPTFunctionSignature` →
  `FunctionSignature`, `GPTFunctionCallResult` → `FunctionCallResult`.
  Non-breaking: the former `GPT*` names remain exported as `const` aliases —
  construction, dispatch, `isa`, field access, and the parametric
  `GPTFunctionCallResult{T}` all keep working unchanged. The `GPT*` aliases are
  scheduled for removal at 1.0 per the new stability policy; docs teach the
  neutral names as canonical.
- `GenericOpenAIEndpoint` and `DeepSeekEndpoint` now redact `api_key` when shown: a
  short prefix plus a `…[redacted]` marker (never the full key or its length),
  inherited when an endpoint prints nested inside a `Chat` or a result value.
- Aqua's method-ambiguity check is enabled in the test suite (`ambiguities=true`)
  after measuring zero ambiguities in the package.

### Fixed
- Streaming: a byte-gap idle breach is now always classified as
  `UniLMTimeout(:stream_idle)` — non-retryable — on the HTTP.jl 2.x major.
  Previously the classification depended on when the breach fired: HTTP 2.x also
  applies its native `read_idle_timeout` (armed at `stream_idle_timeout`) to the
  response-header wait, so a breach landing before the first byte was reported as
  a retryable `:request`-phase timeout with `request_timeout` as its limit, and a
  mute stream could be silently retried. Whether real runs hit that window is
  timing-sensitive (newer HTTP 2.6.x releases changed server task scheduling,
  flipping it under load); the classifier now decides from which native timers
  the streaming seam arms, so the phase is deterministic. Chat streaming and the
  Responses streaming surface share the fix; HTTP 1.x behavior (idle-guard
  enforcement only) is unchanged.

### CI and docs
- The test environment admits JET 0.12 alongside 0.11 (`JET = "0.11, 0.12"`), keeping
  the whole-package release gate on the current JET release; 0.12 is the first series
  to support Julia 1.13. The gate still measures zero findings.
- The CI matrix, documentation build, and release gate track the latest stable Julia
  release instead of a pinned 1.12; the supported floor stays `julia = "1.12"`.
  CompatHelper now also sweeps the `test/` and `docs/` environments, so their bounds
  cannot go stale unnoticed.
- Pull-request documentation builds run without provider API keys — examples render
  offline — eliminating live-API spend on PRs; push, tag, and manual builds still
  render live outputs. The routine CI test workflow no longer receives provider
  keys at all.
- Live integration suites are explicitly opt-in: set `UNILM_LIVE=1` in addition to
  the provider key (the live MCP suite keeps its own `UNILM_LIVE_MCP=1` gate).
- Executed chat and agentic documentation examples that make live calls are pinned
  to each provider's cheapest model (`gpt-5.4-mini`, `gemini-3.1-flash-lite`,
  `claude-haiku-4-5`); image-generation examples are static code blocks that reuse
  the committed sample image, so docs builds never bill for image generation.
  Token-counting, response-compaction, embeddings, and FIM examples keep their API
  defaults.

## 0.14.0

### Breaking
- Abrupt stdio MCP server death (killed, crashed, or broken pipe) now surfaces the
  new typed `MCPCrashError` and closes the session, instead of leaking a raw
  `Base.IOError`/`ErrorException` while the session stayed `:ready`. Code that
  caught raw transport errors around MCP calls should catch `MCPCrashError`.
- `mcp_connect` itself now throws `MCPCrashError` (instead of a raw transport
  error) when the server dies during the connect handshake.

### Added
- `MCPCrashError` (exported): message with recovery guidance plus best-effort
  `exitcode`/`termsignal` diagnostics.
- `auto_respawn=true` now also respawns after a server crash (previously hangs
  only); the crashing call always throws, the next call heals the session.

## 0.13.0

### Added
- `RequestConfig` and four resolution channels bound and tune every request.
  Fields (all `Float64` seconds unless noted; `Inf` disables a bound; the
  constructor rejects `NaN` and non-positive values): `connect_timeout` (10.0),
  `request_timeout` (600.0, whole non-streaming exchange), `stream_idle_timeout`
  (120.0, byte-gap between raw stream chunks), `total_deadline` (900.0, across
  all retry attempts; for streams, until the first byte), `max_attempts` (`Int`,
  3), `mcp_connect_timeout` (120.0), `mcp_request_timeout` (120.0). Resolve a
  config, in precedence order, via the per-call `config=` keyword on every
  request verb, a `with_request_config(f; kwargs...)` dynamic scope (propagates
  into `Threads.@spawn`), a process default set by `set_default_config!(cfg)` /
  `set_default_config!(; kwargs...)`, or the built-in defaults. `current_config()`
  returns the config in force. `RequestConfig(base; kwargs...)` copies with
  overrides.
- Typed timeout errors: `UniLMTimeout` (`phase` ∈ `:connect`/`:request`/
  `:stream_idle`/`:deadline`, with `elapsed`/`limit` seconds) and
  `MCPTimeoutError` (`phase` ∈ `:connect`/`:request`, whose message names the
  applicable per-connect/per-call override). Value-returning surfaces carry the
  timeout on a new `cause::Union{Nothing,Exception}` field of their call-error
  result (`LLMCallError`, `EmbeddingCallError`, `ResponseCallError`, …) with
  `status = nothing`; no fabricated HTTP status is invented for a timeout.
- `mcp_connect(...; auto_respawn=false)` and a call-time `timeout` keyword on
  `call_tool` / `list_tools!`. The request-phase timeout resolves at call time
  (explicit `timeout` > ambient `with_request_config` scope > the config captured
  at `mcp_connect`), so tools bridged into a tool loop — which pass no keywords —
  still honor an ambient scope.
- `tool_loop(input::String; tools, …)`: a no-dispatcher convenience method that
  wraps the prompt and `tools` in a `Respond` and runs the Responses-API tool
  loop, so `tool_loop("…"; tools=mcp_tools_respond(session))` works as a
  documented one-liner. `max_turns` drives the loop and `config` bounds each
  underlying request; every other keyword is forwarded to the `Respond`
  constructor (an unknown one raises).

### Changed
- **Breaking:** the `retries` keyword is removed from every request verb
  (`chatrequest!`, `embeddingrequest!`, `respond`, `fim_complete`,
  `prefix_complete`, `generate_image`, `edit_image`, `upload_file`, and the
  `tool_loop!` / `tool_loop` verbs). Retry behavior is now governed by
  `RequestConfig.max_attempts` (default `3`). Migrate by intent: code that passed
  `retries=30` — the real disable switch, since `retries=N` meant "N attempts
  already spent" and `30` hit the ceiling — now passes
  `config=RequestConfig(max_attempts=1)`; code that passed `retries=0` or nothing
  (the old default seed, which allowed up to thirty attempts) needs no change and
  takes the new default (`max_attempts=3`). There is no compatibility alias: the
  old `retries` counted toward a 30-attempt ceiling (`retries=30` disabled
  retries, `retries=0` allowed thirty) — the inverse of an attempt count — so any
  silent shim would invert real call sites. A removed keyword raises `MethodError`.
- **Breaking:** requests now carry bounded default timeouts, so an operation that
  previously blocked forever on a silent or stalled peer instead fails with a
  typed error once its bound elapses. Defaults (all overridable per call, per
  scope, or process-wide via `RequestConfig`; set any field to `Inf` to disable
  it): connect `10s`, whole non-streaming request `600s`, stream byte-gap idle
  `120s`, total across retries `900s`, `max_attempts` `3`, MCP connect `120s`,
  MCP request `120s`. Non-streaming timeouts land inside the existing error
  result (`status = nothing`, timeout on `cause`); streaming timeouts surface when
  the returned `Task` is `fetch`ed.
- **Breaking:** an MCP **stdio** request that exceeds `mcp_request_timeout` now
  closes the session (`status = :closed`) and throws `MCPTimeoutError`. Stdio
  framing has no response-id demultiplexing, so a late reply could be misdelivered
  to the next caller as a fabricated result — tearing the session down is the only
  safe response. A subsequent call raises unless the session was opened with
  `mcp_connect(...; auto_respawn=true)`, which respawns the server (fresh
  handshake, logged with `@warn`) and retries the call once; in-memory server
  state is lost on respawn, so respawn is opt-in. MCP **HTTP** request timeouts
  are not session-fatal (request/response correlation is per-POST).
- **Breaking:** `call_tool(session, name, args)` now returns an `MCPToolResult`
  instead of a bare `String`, and no longer throws when the tool reports an
  execution error (`isError: true`). The struct carries `content::String` (the
  rendered text, unchanged from before), `structured::Union{Nothing,Dict{String,Any}}`
  (the server's `structuredContent` verbatim), `is_error::Bool`, and
  `parts::Vector{Any}` (the raw content array) — so a tool-execution error is now
  distinguishable from a JSON-RPC protocol error (still thrown as `MCPError`). The
  `mcp_tools` / `mcp_tools_respond` tool-loop bridges are unchanged for callers:
  they still hand the model a string (the faithful `content`), now surfacing
  `structuredContent` when the content is otherwise empty and raising the tool's
  own error content on `isError`.
- `Chat(; tools=…)` now accepts a `Vector{<:CallableTool}` and stores the
  unwrapped inner `GPTTool`s, so MCP tools bridge in directly —
  `Chat(tools=mcp_tools(session))`, with no manual `map(t -> t.tool, tools)`.
  The `tools` field type is unchanged (`Union{Vector{GPTTool},Nothing}`);
  conversion happens only at construction.

### Fixed
- A user interrupt (`InterruptException`, e.g. Ctrl-C) raised during a chat
  request or inside a streaming task now propagates instead of being laundered
  into a call-error result and retried. Every request catch layer rethrows an
  interrupt first; a streamed interrupt surfaces as a `TaskFailedException` when
  the returned `Task` is `fetch`ed. (Same class as the tool-loop interrupt fix
  below.)
- Azure OpenAI deployment names configured through `AZURE_OPENAI_DEPLOY_NAME_*`
  environment variables are now read when the request URL is built, instead of
  being captured once when the package loads. A deployment name exported after
  `using UniLM` is honored; `add_azure_deploy_name!` registrations still take
  precedence.
- The tool-calling loop no longer swallows a user interrupt: an
  `InterruptException` (e.g. Ctrl-C) raised inside a tool function now
  propagates and aborts the loop instead of being recorded as a tool-call
  failure and retried. All other exceptions still become tool-error outcomes.
- Tool-error messages now reach the model faithfully instead of wrapped in
  exception-constructor noise. A tool that raises `error("kaboom")` is reported
  to the model as `Error: kaboom` (previously `Error: ErrorException("kaboom")`);
  other exceptions render through `showerror` (e.g. `KeyError: key "x" not found`)
  rather than their `string(e)` constructor form. The single loop-level `Error: `
  prefix is unchanged, so a tool whose own message already begins with `Error:` is
  still not rewritten.
- MCP client now validates and honors the negotiated protocol version and
  recovers expired HTTP sessions (Streamable HTTP, MCP spec 2025-11-25). After
  `initialize`, a server `protocolVersion` outside the client's supported set
  (`2025-11-25`, `2025-06-18`, `2025-03-26`) closes the transport and raises an
  error naming both the requested and the returned version, instead of being
  accepted unchecked; a supported older version is stored and used. Every HTTP
  request after `initialize` now sends the negotiated `Mcp-Protocol-Version`
  header (previously a fixed constant), while the `initialize` request itself
  advertises the client's latest supported revision. A `404` on a request that
  carries a live session id triggers a single re-initialize (fresh session id)
  followed by one replay of the request before giving up, rather than failing
  outright. `401`/`403` responses raise an error that names the status and
  directs credentials to the `headers` kwarg of `mcp_connect`.

## 0.12.0

### Added
- `ProviderContent` and `Message.provider_content`: provider-native assistant
  content (Anthropic thinking/redacted_thinking blocks, Gemini parts with
  text-part `thoughtSignature`s) is captured verbatim at decode time — for
  both non-streaming and streaming Anthropic responses, and non-streaming
  Gemini responses — and echoed verbatim when the same provider encodes the
  conversation again. Never serialized on the OpenAI wire.

### Changed
- **Breaking:** `serve(server; transport=:http)` now blocks until the server
  is closed (matching stdio serving and `HTTP.serve`). Pass `block=false` for
  the previous behavior: it returns the running server handle, which you
  `close` yourself.
- The agentic streaming decode seam now threads one `AgenticStreamState`
  (text buffer, line carry, sticky event name, per-step assembly registry)
  instead of three loose buffer arguments. The seam is unexported; provider
  packages overriding `decode_agentic_stream` must adopt the new signature.
- MCP HTTP server transport now validates the `Origin` header (a Streamable
  HTTP requirement; DNS-rebinding defense): requests without an `Origin` and
  requests from localhost origins are accepted, anything else gets 403 unless
  listed in the new `allowed_origins` kwarg of `serve`.

### Fixed
- Anthropic tool calling on thinking models (e.g. `claude-sonnet-5`): assistant
  turns rebuilt from text+tool_calls dropped the thinking blocks the API
  requires back verbatim, so multi-turn tool use failed with HTTP 400.
- Gemini Interactions `thought` steps are now surfaced verbatim in
  `ResponseObject.output` instead of being collapsed into an empty
  `reasoning` stub (their `signature` was previously lost; `reasoning_items`
  no longer returns stub entries for them; filter `output` for type ==
  `"thought"`).
- Gemini Interactions streaming with tools: function-call steps
  (`step.start` + `arguments_delta` + `step.stop`) are now assembled and
  surfaced in the terminal response's `output`, so streamed
  `respond(...; tools=…)` returns a usable `requires_action` result instead
  of failing on a 200 stream. Streamed thought steps keep their signature.
- Gemini chat: parallel tool calls without wire `id`s now receive unique
  synthetic positional ids (reserved prefix `unilm_call_`), fixing tool-result
  correlation that previously collapsed to the last call; synthetic ids are
  omitted on re-encode.
- `fork(chat)` now copies every `Chat` field by construction (previously 15
  config fields — `reasoning_effort`, `max_completion_tokens`,
  `stream_options`, `verbosity`, `store`, `metadata`, `service_tier`,
  `logprobs`, `top_logprobs`, `prediction`, `modalities`, `audio`,
  `web_search_options`, `prompt_cache_key`, `safety_identifier` — were
  silently dropped, so forked chats behaved differently). Forks no longer
  normalize `parallel_tool_calls`; the copy is verbatim.
- MCP server: a syntactically valid JSON frame that is not a JSON object
  (array, string, number — e.g. a legacy JSON-RPC batch) now gets a `-32600`
  Invalid Request response; the stdio serve loop continues with the next frame
  (previously a `MethodError` killed it) and the HTTP transport returns 400
  instead of 500.
- MCP client: a request now reads frames until the response with its own id
  arrives — interleaved server notifications are skipped instead of being
  returned as the (empty) result and desyncing every subsequent call, and
  server-initiated `ping` requests are answered inline. After
  `notifications/tools/list_changed` the session marks its cached tool list
  stale (`session.tools_stale`; refresh with `list_tools!`). The whole
  exchange, including request-id allocation, runs under a session lock
  (previously only writes were locked and the id counter was racy). Custom
  `MCPTransport` subtypes must now also implement `_transport_read!`.

## 0.11.3

### Fixed
- Streaming (all providers, one shared SSE machine in `src/sse.jl`): `on_tool_call` now fires exactly once per completed streamed tool call (was: never); `stream_options.include_usage` no longer loses usage or turns a successful stream into `LLMFailure` (chat EOS is `data: [DONE]` only — `finish_reason` never ends the stream, and empty-`choices` chunks are tolerated, unbreaking Azure preambles and `: keep-alive` proxies); Anthropic mid-stream `error` events now produce `LLMFailure`(529 for `overloaded_error`)/`LLMCallError` instead of a truncated `LLMSuccess`; streamed messages keep assistant text alongside tool calls, and zero-argument streamed tool calls parse as `Dict{String,Any}()` instead of throwing. Failed SSE lines are logged and dropped, never re-queued; partial lines carry over verbatim (no whitespace loss at chunk boundaries).
- Gemini chat streaming reads to EOF (no sentinel exists), so trailing `usageMetadata` chunks are consumed instead of being cut off at `finishReason`.
- Streaming callbacks now receive each text delta as parsed (verbatim forwarding replaces O(n²) buffer re-diffing that could split multibyte characters).

### Changed
- Behavior change: an Anthropic stream truncated after `message_delta` but before `message_stop` now completes as `LLMSuccess` (the driver's EOF + recorded-finish_reason rule) where it previously produced `LLMFailure`.
- SSE parsing tolerates `data:` lines with or without the single optional space
  after the colon (`data:{…}` and `data: {…}` are equivalent), per the SSE spec.

### Removed
- Internal (unexported, documented) streaming seam `decode_stream_chunk` and `_parse_chunk`, replaced by `handle_sse_event!(service, event, payload, state) -> :continue | :done | :error` in `src/sse.jl` (no known external overriders).

## 0.11.2

### Added
- Exposed `x-request-id` header on all non-success/failure result structs (`LLMFailure`, `LLMCallError`, `FIMFailure`, `FIMCallError`, `ResponseFailure`, `ResponseCallError`).
- Implemented streaming residual flushing in the `:completed` terminal branch of `_respond_stream` to ensure text deltas received in the final stream chunk are not dropped.

## 0.11.1

Documentation-only release: an exhaustive review pass over the docs. No API or behavior changes.

### Changed
- Onboarding: the registered install (`Pkg.add("UniLM")`) now leads the README, docs home, and Getting Started (the GitHub-URL install is kept for tracking unreleased changes); the Julia 1.12+ prerequisite is stated up front.
- Positioning: reframed around first-class **native** backends (OpenAI, Anthropic, Gemini) plus the OpenAI-compatible tier, rather than "via the OpenAI-compatible API standard"; the Chat-Completions-vs-Responses comparison is now scoped to OpenAI.
- LLM reference (`llm.md`): the Complete Exports List is regenerated from `names(UniLM)` (previously missing ~40 symbols and the platform-API families); provider-compatibility table, capability sets, `tool_choice`/`DEFAULT_PRICING` types, and documented default models corrected.
- New guides: **Cost Tracking** (incl. the silent-`$0` behavior for unpriced models) and **Retrieval & File Search** (end-to-end Files → Vector Store → `file_search`).

### Fixed
- Corrected stale documentation: image default `gpt-image-2`, chat default `gpt-5.5`, the Gemini embeddings example (`GEMINIOpenAIServiceEndpoint`), retry-status lists, and the module / `provider_capabilities` docstrings. Added a root `llms.txt` for LLM/agent consumers.

## 0.11.0

### Breaking changes

- **`GEMINIServiceEndpoint` now targets Google's native `generateContent` API** (auth header `x-goog-api-key`; model in the URL), not the OpenAI-compatibility endpoint. The OpenAI-compatible Gemini shim is renamed **`GEMINIOpenAIServiceEndpoint`**. Migrate any code that used `GEMINIServiceEndpoint` for the OpenAI-compatible path — including `Embeddings(...; service=GEMINIServiceEndpoint)`, which the native endpoint does not support — to `GEMINIOpenAIServiceEndpoint`.

### Added

- Native Google Gemini chat (`GEMINIServiceEndpoint`): `generateContent` messages, tools (with Gemini-3 `thoughtSignature` echo for multi-turn tool calls), SSE streaming, and usage/cost accounting. Default model `gemini-3.5-flash`.
- Native Anthropic chat (`ANTHROPICServiceEndpoint`): Messages API messages, tools, streaming, usage/cost.
- Optional `GPTToolCall.thought_signature` field (set by the Gemini decoder; ignored by other providers).
- Unified agentic verb across providers: `respond`/`Respond` now also targets Google's **Gemini Interactions** API via `service=GEMINIServiceEndpoint` (in addition to OpenAI Responses), sharing inputs, `tool_loop`/`tool_loop!`, lifecycle (`get_response`/`cancel_response`), and usage/cost accounting.
- Gemini hosted-tool constructors `gemini_google_search`, `gemini_code_execution`, `gemini_url_context` for use in `respond(...; tools=[...])`.
- Cross-provider `estimated_cost`/`token_usage` for Gemini Interactions results (usage normalized to the shared shape; `gemini-3.5-flash` priced in `DEFAULT_PRICING`).

## 0.10.3

Chat-path `strict` structured outputs are now expressible. Additive and non-breaking:
when `strict` is not set, request bodies are identical to 0.10.2 (the field is omitted,
which is the API default — non-strict).

### Added
- `GPTFunctionSignature` gains `strict::Union{Bool,Nothing} = nothing` (strict function
  calling on the Chat Completions tool path). `true`/`false` are emitted inside the
  `"function"` object per the Chat Completions wire shape; `nothing` omits the field.
  `GPTTool(::AbstractDict)` reads `"strict"` back symmetrically (bare and wrapped
  formats), so dict-rendered tool definitions carrying `"strict": true` are now
  transmitted instead of silently dropped.
- `JsonSchemaAPI` gains the documented `strict` field for chat `response_format`
  structured outputs, and `json_schema(name, description, schema; strict=...)` threads
  it. `JsonSchemaAPI` now declares `JSON.omit_null` (all its previous fields were
  required, so existing serialized output is unchanged).

### Behavior note (deliberate bug fix)
- Tool definitions ingested as dicts (`GPTTool(::AbstractDict)` / `to_tool(::AbstractDict)`)
  that already carry a `"strict"` key now transmit it — previously the key was silently
  dropped. If a stored `"strict": true` definition has a strict-invalid schema, the API
  will now reject it with a 400; that rejection reflects what the definition always
  declared. Non-Bool `"strict"` values raise a descriptive `ArgumentError` instead of a
  raw `MethodError`.
- The pre-0.10.3 3-argument positional constructors `GPTFunctionSignature(name,
  description, parameters)` and `JsonSchemaAPI(name, description, schema)` are preserved
  via explicit methods (`@kwdef` field defaults do not extend positional constructors).

UniLM does not validate schemas against strict-mode rules (transport, not policy); the
API rejects strict-invalid schemas with a 400. Live transmission is witnessed by a
key-gated integration test: the same tool is accepted without `strict` and rejected
(400, `invalid_function_parameters`) with `strict=true` on a strict-invalid schema.

## 0.10.2

Documentation and CI maintenance only — **no functional changes to the library** (`src/`
is unchanged from 0.10.1). Released so the versioned/`stable` documentation reflects 0.10.x.

### Documentation
- MCP guide macro examples are now **executed at build time** (realistic, self-verifying),
  which prevents silent doc rot; added usage-contract notes for `@mcp_tool`/`@mcp_resource`.

### CI / tooling
- Updated GitHub Actions to current majors (`actions/checkout` v7, `julia-actions/setup-julia`
  v3, `julia-actions/cache` v3, `codecov/codecov-action` v7) and fixed CI-vs-Documentation drift.
- `Documentation` workflow gains `workflow_dispatch` for manual versioned-docs rebuilds.
- `CompatHelper` workflow now installs Julia (a `setup-julia` step was missing).

## 0.10.1

Patch release: correctness fixes for the MCP server macros, plus a large test-coverage
hardening pass. No breaking changes.

### Fixed
- `@mcp_resource` template form now binds the matched path parameters into the handler's
  declared arguments. The documented example
  `@mcp_resource server "file://{path}" function(path::String) read(path, String) end`
  previously raised `UndefVarError` at read time because the params were never unpacked;
  each declared argument is now bound from the matched URI `{param}` of the same name.
- `@mcp_prompt` now supports the documented anonymous `function(arg) … end` form. The first
  declared argument was previously dropped (omitted from the prompt's argument schema and
  left unbound in the handler); a shared, signature-shape-robust argument extractor now
  handles both the named and anonymous forms.
- `@mcp_tool` registers under the function name, so it now raises a clear error when given
  an anonymous function instead of silently registering a tool named after the first
  argument node. Use the documented named form `@mcp_tool server function name(args…) … end`.

### Internal
- Test suite substantially expanded — the MCP client/server operation layers, streaming,
  retry recursion, and error paths are now exercised end-to-end via deterministic in-process
  mocks. Project coverage ~99.6%. This work is tests-only; no behavior change.

## 0.10.0

"OpenAI first-class" release: correctness fixes, a fully-modeled Responses API, cache-aware
cost accounting, and broad new endpoint coverage.

> Model ids and `DEFAULT_PRICING` figures were verified against the live OpenAI `/v1/models` and
> official model pricing pages on 2026-06-21 (18/18 model ids present; the `gpt-5.2` rate was
> corrected to 1.75/0.175/14.0 per 1M tokens in that pass). Prices drift — re-verify over time.

### Breaking changes
- **`embeddingrequest!`** now returns `EmbeddingSuccess` / `EmbeddingFailure` / `EmbeddingCallError`
  (previously a `(dict, emb)` tuple on success and **threw** on failure). `emb.embeddings` is still
  filled in place; use `embedding_vectors(result)` to read the vectors.
- **`extract_message`** preserves partial assistant `content` for any `finish_reason` (e.g.
  `"length"`) instead of replacing it with `"No response from the model."` (that fallback is kept
  only for genuinely-empty responses).
- **`WebSearchTool`** defaults to the GA `type = "web_search"` (was `"web_search_preview"`). Pass
  `WebSearchTool(type="web_search_preview")` to restore the previous wire output.
- **`Respond.tool_choice`** widened to `Union{String,AbstractDict,Nothing}` (source-compatible;
  existing `String`/`nothing` callers are unaffected).
- **Default models** bumped: OpenAI chat/responses → `gpt-5.5` (was `gpt-5.2`), image → `gpt-image-2`
  (was `gpt-image-1.5`). These cost more per token — pin an explicit `model=` to control spend.

### Fixed
- Chat now sends **`max_completion_tokens`**; `max_tokens` is deprecated and rejected by reasoning models.
- Refusals are captured in both non-streaming and streaming paths, regardless of `finish_reason`.
- Responses streaming recognizes terminal `response.failed` / `response.incomplete` / `error` events and
  surfaces structured failures (previously lost as `ResponseFailure(status=200, raw)`); unknown/future
  events degrade gracefully.
- Embeddings `dimensions` / `encoding_format` supported; `update!` is resize-tolerant for non-1536-dim
  models (e.g. 3072-dim `text-embedding-3-large`).
- `_is_retryable` adds 408 / 502 / 504 / 529.

### Added — Responses API
- `tool_choice` builders: `tool_choice_function` / `_hosted` / `_mcp` / `_custom` / `_allowed`.
- `text.verbosity`; `WebSearchTool` `filters`; `MCPTool` `connector_id` / `authorization` /
  `server_description` / `tunnel_id` + `mcp_approval_response`.
- Typed output accessors: `reasoning_summaries`, `reasoning_items`, `refusals`, `url_citations`,
  `web_search_results`, `file_search_results`, `image_generation_results`, `code_interpreter_outputs`,
  `mcp_call_outputs`, `mcp_approval_requests`, `response_status`, `incomplete_details`, `usage_details`.
- New tool types: `LocalShellTool`, `ShellTool`, `ApplyPatchTool`, `CustomTool` (incl. grammar format),
  and GA `ComputerTool`. Input parts: `input_image(file_id=…)`, `input_file(file_data=…, filename=…)`.

### Added — Chat Completions
- `reasoning_effort`, `stream_options`, `verbosity`, `store`, `metadata`, `service_tier`, `logprobs` /
  `top_logprobs`, `prediction`, `modalities`, `audio`, `web_search_options`, `prompt_cache_key`,
  `safety_identifier`.

### Added — Accounting
- `TokenUsage` gains `cached_tokens` / `reasoning_tokens`. `estimated_cost` bills cached input at the
  discounted rate (no longer overcharges cache-heavy workloads); pricing table gains a `cached_input`
  column and refreshed/extended rows (incl. embeddings).

### Added — new endpoints
Files, Vector Stores (+ `poll_file_batch`), Conversations, Audio (TTS + transcription/translation),
Batch (+ `poll_batch`), Moderations, Image edits, Fine-tuning, Webhooks (HMAC-SHA256 verification),
Containers, Uploads (resumable), Videos, and Realtime (WebSocket transport + ephemeral client-secret
minting; WebRTC/SIP out of scope). Each is gated by a provider capability; non-OpenAI providers reject them.

### Dependencies
- Added `SHA` (stdlib) for webhook signature verification (no TLS dependency).
