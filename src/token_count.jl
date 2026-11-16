# ============================================================================
# Token counting: the input tokens a request would bill, counted by its provider without
# running it, for the request object the generation verb takes.
#   OpenAI     POST /v1/responses/input_tokens          a Respond (Responses API)
#              https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count
#   Anthropic  POST /v1/messages/count_tokens           a Chat (Messages API)
#              https://platform.claude.com/docs/en/api/messages/count_tokens
#   Gemini     POST /v1beta/models/{model}:countTokens  a Chat (generateContent)
#              https://ai.google.dev/api/tokens
# Each body is the one the generation verb's encoder builds, without the fields the count does
# not take. Anthropic's token counting "is free to use"
# (https://platform.claude.com/docs/en/build-with-claude/token-counting); OpenAI and Google
# publish no price for theirs, and no count reports usage.
# ============================================================================

"""
    TokenCountSuccess

Successful [`count_tokens`](@ref) result:

- `input_tokens::Int`: the input tokens the request would bill — OpenAI's and Anthropic's
  `input_tokens`, Gemini's `totalTokens`;
- `raw::Dict{String,Any}`: the reply as the provider sent it, with what one provider adds:
  Gemini's per-modality `promptTokensDetails`.
"""
struct TokenCountSuccess <: LLMRequestResponse
    input_tokens::Int
    raw::Dict{String,Any}
end
"[`count_tokens`](@ref) error result: HTTP `status`, the raw `response` body, and the `request_id` the provider sent (`x-request-id`/`request-id` header), if any."
@kwdef struct TokenCountFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"[`count_tokens`](@ref) call that produced no count (a missing key, no connection, a timeout, a cancel, or a reply that is not the documented count); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel."
@kwdef struct TokenCountCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── The three wires ─────────────────────────────────────────────────────────

# How a provider counts a request: its route, its headers, the body it takes and the key of its total.
abstract type _TokenCounting end
struct _ResponsesCount <: _TokenCounting end
struct _AnthropicCount <: _TokenCounting end
struct _GeminiCount <: _TokenCounting end

_token_counting(::Type{OPENAIServiceEndpoint}, ::Respond) = _ResponsesCount()
_token_counting(::_AnthropicEndpoint, ::Chat) = _AnthropicCount()
_token_counting(::Type{GEMINIServiceEndpoint}, ::Chat) = _GeminiCount()
_token_counting(service, request) = throw(ArgumentError(
    "count_tokens of a $(nameof(typeof(request))) is not supported by $(_service_name(service)): " * _what_counts(service)))
_what_counts(::Type{OPENAIServiceEndpoint}) =
    "OpenAI counts the input of a Responses API request (POST /v1/responses/input_tokens); count a Respond"
_what_counts(::Type{GEMINIServiceEndpoint}) =
    "Gemini counts a generateContent request (models.countTokens), and its Interactions API counts none; count a Chat"
_what_counts(::_AnthropicEndpoint) = "Anthropic counts a Messages API request (POST /v1/messages/count_tokens); count a Chat"
_what_counts(_) = "UniLM counts a Respond on OPENAIServiceEndpoint, and a Chat on ANTHROPICServiceEndpoint and GEMINIServiceEndpoint"

_count_url(::_ResponsesCount, service, ::Respond)::String = _api_base_url(service) * RESPONSES_PATH * "/input_tokens"
_count_url(::_AnthropicCount, _, ::Chat)::String = ANTHROPIC_BASE_URL * ANTHROPIC_MESSAGES_PATH * "/count_tokens"
_count_url(::_GeminiCount, _, chat::Chat)::String = GEMINI_NATIVE_BASE * "/models/" * _uripart(chat.model) * ":countTokens"

# The headers of the generation request. Anthropic's SDK (v2.9.0) counts on the GA route with no
# `anthropic-beta` header, and on the beta route, the one an endpoint with betas takes, with the
# caller's betas and "token-counting-2024-11-01".
const _ANTHROPIC_COUNT_BETA = "token-counting-2024-11-01"
_count_headers(::_TokenCounting, service) = auth_header(service)
function _count_headers(::_AnthropicCount, service::_AnthropicEndpoint)::Vector{Pair{String,String}}
    beta(v::String) = _ANTHROPIC_COUNT_BETA in split(v, ',') ? v : v * "," * _ANTHROPIC_COUNT_BETA
    [k == "anthropic-beta" ? k => beta(v) : k => v for (k, v) in auth_header(service)]
end

# The fields of the generation body that the count does not take, each one that bounds or
# delivers the output, names the end user, or picks the tier, the region or the stored copy: none
# changes the input. The APIs answer each with HTTP 400 ("Extra inputs are not permitted",
# "Unknown parameter"), so the count leaves them out.
const _ANTHROPIC_NOT_COUNTED = ("max_tokens", "stop_sequences", "stream", "metadata", "service_tier", "inference_geo")
const _RESPONSES_NOT_COUNTED = ("temperature", "top_p", "max_output_tokens", "top_logprobs", "stream",
    "stream_options", "background", "store", "include", "max_tool_calls", "metadata", "user", "safety_identifier",
    "service_tier", "prompt_cache_key", "prompt_cache_retention", "prompt_cache_options", "moderation")
_without(body::String, keys::Tuple) =
    JSON.json(filter(((k, _),) -> !(k in keys), JSON.parse(body; dicttype=Dict{String,Any})))

_count_body(::_AnthropicCount, service, chat::Chat)::String = _without(encode_request(service, chat), _ANTHROPIC_NOT_COUNTED)
# countTokens takes the whole generateContent request, which names its model.
function _count_body(::_GeminiCount, service, chat::Chat)::String
    request = JSON.parse(encode_request(service, chat); dicttype=Dict{String,Any})
    request["model"] = "models/" * chat.model
    JSON.json(Dict("generateContentRequest" => request))
end
# Compaction and the image policy change the input the model reads, and the count takes neither
# (HTTP 400 "Unknown parameter"): counted without them, the total would not be the request's.
function _count_body(::_ResponsesCount, service, r::Respond)::String
    for (field, what) in ((:context_management, "context_management (server-side compaction)"),
                          (:image_context_compression, "image_context_compression (image.context_compression)"))
        isnothing(getfield(r, field)) || throw(ArgumentError(
            "count_tokens: OpenAI's input token count takes no $what, which changes the input the model reads; " *
            "unset $field to count the input without it"))
    end
    _without(encode_agentic(service, r), _RESPONSES_NOT_COUNTED)
end

_count_key(::Union{_ResponsesCount,_AnthropicCount})::String = "input_tokens"
_count_key(::_GeminiCount)::String = "totalTokens"

function _token_count(wire::_TokenCounting, resp::HTTP.Response)::TokenCountSuccess
    d = JSON.parse(resp.body; dicttype=Dict{String,Any})
    d isa Dict{String,Any} || throw(ArgumentError("the token count is not a JSON object"))
    key = _count_key(wire)
    n = get(d, key, nothing)
    n isa Int && n >= 0 || throw(ArgumentError("the token count has no $(repr(key)) count of tokens (got $(repr(n)))"))
    TokenCountSuccess(n, d)
end

# ─── The verb ────────────────────────────────────────────────────────────────

"""
    count_tokens(request::Union{Chat,Respond}; config=nothing, cancel=nothing)

How many input tokens `request` would bill, counted by its provider without generating:

- a [`Chat`](@ref) on [`ANTHROPICServiceEndpoint`](@ref) (`POST /v1/messages/count_tokens`): the
  body [`chatrequest!`](@ref) sends, without `max_tokens`, `stop_sequences`, `stream`,
  `metadata` (the user id), `service_tier` and the endpoint's `inference_geo`, which the count does
  not take and which do not change the input. An endpoint with `betas` counts on the beta route:
  its betas and `token-counting-2024-11-01` in the `anthropic-beta` header, as Anthropic's SDK sends
  them; one without sends no beta header.
- a `Chat` on [`GEMINIServiceEndpoint`](@ref) (`models.countTokens`): the whole `generateContent`
  body, system instruction, tools and generation config included, as its `generateContentRequest`.
- a [`Respond`](@ref) on [`OPENAIServiceEndpoint`](@ref) (`POST /v1/responses/input_tokens`): the
  body [`respond`](@ref) sends, without its output, delivery, storage, tier, caching, end-user and
  moderation fields, which the count does not take. A `Respond` with `context_management` or
  `image_context_compression` throws `ArgumentError`: each changes the input the model reads, and
  the count takes neither. [`count_input_tokens`](@ref) counts on the same route by keyword, and
  sends `personality`, which a `Respond` does not hold.

The encoder's own refusals hold, as in the generation verb. Any other pairing throws
`ArgumentError` before any request, naming what the provider counts: OpenAI counts no `Chat`, and
Gemini's Interactions API (a `Respond` on Gemini) has no count; so does an endpoint without the
`:token_count` capability.

Returns a [`TokenCountSuccess`](@ref) (`input_tokens`, and `raw`, the reply), a
[`TokenCountFailure`](@ref) for an error status, or a [`TokenCountCallError`](@ref), which is also
what a missing key gives (it names the variable) and what a reply without its count gives.
`config` (a [`RequestConfig`](@ref)) bounds the call and its retries of the retryable statuses;
`cancel` (default: the ambient [`with_cancel`](@ref) token) ends it with a
[`UniLMCancelled`](@ref) in `cause`. A count is not recorded by [`with_recorded_answers`](@ref):
inside a scope it goes to the provider, as outside one.

```julia
chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5")
push!(chat, Message(Val(:user), "Hello"))
r = count_tokens(chat)
issuccess(r) && r.input_tokens   # 11
```
"""
function count_tokens(request::Union{Chat,Respond}; config::Union{Nothing,RequestConfig}=nothing,
                      cancel::Union{Nothing,CancelToken}=nothing)
    service = request.service
    validate_capability(service, :token_count, "Token counting")
    wire = _token_counting(service, request)
    url, body = _count_url(wire, service, request), _count_body(wire, service, request)
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()
    try
        resp = _http_with_retries(cfg, t0, "POST", url, _count_headers(wire, service), body; cancel=tok)
        resp.status == 200 ? _token_count(wire, resp) : _failure(TokenCountFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(TokenCountCallError, e)
    end
end
