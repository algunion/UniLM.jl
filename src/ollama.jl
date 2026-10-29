# ============================================================================
# Ollama (https://docs.ollama.com/api). Chat and embeddings speak the native API
# (`/api/chat`, `/api/embed`): it is the only one that takes the runtime `options`
# (context window, sampling knobs), `keep_alive`, `think` and `truncate`; the
# OpenAI-compatible routes accept and ignore them, and truncate over-long input
# without a word. `respond`, compaction, FIM and transcription keep the OpenAI-compatible
# routes (`/v1/responses`, `/v1/responses/compact`, `/v1/completions`,
# `/v1/audio/transcriptions`), which carry none of the endpoint's `keep_alive`, `options`
# or `runner`.
# ============================================================================

get_url(s::OllamaEndpoint, ::Chat)::String = s.base_url * OLLAMA_CHAT_PATH
get_url(s::OllamaEndpoint, ::Embeddings)::String = s.base_url * OLLAMA_EMBED_PATH
get_url(s::OllamaEndpoint, ::FIMCompletion)::String = s.base_url * COMPLETIONS_PATH
_resolve_base_url(s::OllamaEndpoint) = s.base_url

# A local server takes no credentials. Ollama's cloud takes its key, read from OLLAMA_API_KEY (the
# variable https://docs.ollama.com/api/authentication exports) each time a request is built and sent
# to that host alone, "in the `Authorization: Bearer` header", whenever it is set. The same page says
# "Direct cloud inference at https://ollama.com/api and https://ollama.com/v1 requires an API key",
# while the cloud's own page lists its models with no key, "curl https://ollama.com/api/tags"
# (https://docs.ollama.com/cloud): a `required` request — inference — is refused without the key
# before it is sent, any other goes without it.
_ollama_cloud(s::OllamaEndpoint)::Bool = occursin(r"^https://ollama\.com(?::\d+)?(?:/|$)"i, s.base_url)
function _ollama_auth(s::OllamaEndpoint; required::Bool)::Vector{Pair{String,String}}
    json = "Content-Type" => "application/json"
    _ollama_cloud(s) || return [json]
    key = strip(get(ENV, OLLAMA_API_KEY, ""))
    isempty(key) || return ["Authorization" => "Bearer $key", json]
    required && throw(ArgumentError("$OLLAMA_API_KEY is not set: Ollama's cloud ($(s.base_url)) takes an API key " *
        "as a Bearer token for inference (https://ollama.com/settings/keys)"))
    [json]
end
# Every request the shared verbs send to Ollama — chat, embeddings, respond, compaction, FIM,
# transcription, System One — is inference; the model verbs choose for themselves (ollama_models.jl).
auth_header(s::OllamaEndpoint)::Vector{Pair{String,String}} = _ollama_auth(s; required=true)

provider_capabilities(::OllamaEndpoint) =
    Set([:chat, :embeddings, :fim, :tools, :streaming, :json_output, :responses, :agentic, :models, :system_one, :audio,
         :transcription])

_model_hint(::OllamaEndpoint) =
    " (an installed model, e.g. model=\"gemma4:e4b\"; list_models(service=OllamaEndpoint()) lists them)"
# Ollama has no default model: a compaction that names none is refused before any request, as a Chat is.
_compaction_model(s::OllamaEndpoint)::String = _resolve_model(s, "")

# A model on the caller's own machine bills no one (`estimated_cost` prices it only by a row the caller
# gave it, `_callers_row`). A cloud model reached through the local server
# (a tag of `cloud`, or one ending in `-cloud`, such as `clef:cloud` or `gpt-oss:120b-cloud`) runs on
# Ollama's servers and bills its account: it is priced by the name requested, like any other model. As
# in Ollama's own name parser, surrounding whitespace is trimmed, the tag follows the last ':' only when
# no '/' comes after it (in `localhost:5000/library/clef-cloud` that ':' starts a registry port, and the
# tag is the default `latest`), and Ollama compares names case-insensitively (`GEMMA4:E4B` is `gemma4:e4b`).
function _ollama_cloud_model(name::AbstractString)::Bool
    model = strip(name)
    i = findlast(':', model)
    (isnothing(i) || something(findlast('/', model), 0) > i) && return false
    tag = lowercase(model[nextind(model, i):end])
    tag == "cloud" || endswith(tag, "-cloud")
end
# A request made to Ollama's cloud itself bills its account as well, whatever the model's name: the cloud
# names its models without a cloud tag, "such as `gemma4:31b`" (https://docs.ollama.com/cloud).
_is_local(s::OllamaEndpoint, model::AbstractString) = !_ollama_cloud(s) && !_ollama_cloud_model(model)

# A cloud model may answer without its tag: Ollama's cloud proxy sends the cloud the name stripped of its
# cloud tag (`gpt-oss:120b` for `gpt-oss:120b-cloud`) and passes the reply through. A result keeps the
# requested name, so it is priced as the cloud model it is; the server's name stays in its `raw`. A
# request that named no model has no name to keep.
_ollama_billed_model(requested::Union{AbstractString,Nothing}, replied::AbstractString)::String =
    !isnothing(requested) && _ollama_cloud_model(requested) && !_ollama_cloud_model(replied) ?
        String(requested) : String(replied)
_billed_response(::OllamaEndpoint, requested::AbstractString, x::ResponseObject) =
    (m = _ollama_billed_model(requested, x.model); m == x.model ? x :
     ResponseObject(x.id, x.status, m, x.output, x.usage, x.error, x.metadata, x.raw))
_billed_response(::OllamaEndpoint, requested::AbstractString, x::FIMResponse) =
    (m = _ollama_billed_model(requested, x.model); m == x.model ? x : FIMResponse(x.choices, x.usage, m, x.raw))
# A chat reply names its model too, and a Chat result records the name that bills.
_billed_response(::OllamaEndpoint, requested::AbstractString, replied::AbstractString) =
    _ollama_billed_model(requested, replied)

# ─── Chat request (neutral Chat → /api/chat body) ───────────────────────────

# Fail closed: a Chat field /api/chat has no counterpart for is refused, never
# dropped (the request would not be the one the caller asked for).
const _OLLAMA_CHAT_MAPPED_FIELDS = (:service, :model, :messages, :history, :tools, :tool_choice,
    :parallel_tool_calls, :temperature, :top_p, :n, :stream, :stop, :max_tokens,
    :max_completion_tokens, :presence_penalty, :response_format, :frequency_penalty, :seed,
    :reasoning_effort, :logprobs, :top_logprobs, :_cumulative_cost)
const _OLLAMA_CHAT_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Chat), _OLLAMA_CHAT_MAPPED_FIELDS))

# reasoning_effort → `think`: "none" turns thinking off; any other effort goes by name,
# since thinking levels are the model's own (https://docs.ollama.com/capabilities/thinking):
# a model with thinking metadata applies a level it names and uses its default for any other
# name, and one without takes "low", "medium", "high" and "max" — the levels of Ollama's API
# reference — and answers another name with HTTP 400 on /api/chat (Ollama 0.40.11,
# api.ValidateLegacyThinking), while /v1/responses reads "minimal" as "low" and "xhigh" as
# "max" (https://docs.ollama.com/api/openai-compatibility). Unset, the
# model's default applies (Gemma 4 thinks) — except with a response_format: Ollama
# constrains a reply to the format only after a thinking block ends, so a thinking
# model that answers without thinking is not constrained at all (gemma4:e4b answered
# plain text to 10 of 10 schema requests with thinking at its default, and valid JSON
# to 10 of 10 with it off, on Ollama 0.34.4). A format therefore turns thinking off, and a format
# with a thinking effort is refused rather than sent to fail at random.
function _ollama_think(effort::Union{String,Nothing}, constrained::Bool)
    isnothing(effort) && return constrained ? false : nothing
    effort == "none" && return false
    constrained && throw(ArgumentError(
        "Ollama applies a response_format only after the model's thinking ends, so a reply that " *
        "skips thinking is unconstrained; use reasoning_effort=\"none\" (or leave it unset) with a " *
        "response_format, or drop the response_format to let the model think"))
    effort
end

# keep_alive on the wire: seconds, or -1 for "stay loaded".
_ollama_keep_alive(ka::Float64) = isinf(ka) ? -1 : (isinteger(ka) ? Int(ka) : ka)

# The runner the endpoint pins, on a native request body.
_ollama_runner!(body::AbstractDict, s::OllamaEndpoint) = (isnothing(s.runner) || (body["runner"] = String(s.runner)); body)

function _ollama_validate(chat::Chat)::Nothing
    fields = Symbol[f for f in _OLLAMA_CHAT_UNMAPPED_FIELDS if !isnothing(getfield(chat, f))]
    isempty(fields) || throw(ArgumentError(
        "Ollama /api/chat does not support field(s) $(join(fields, ", "))"))
    tc = chat.tool_choice
    isnothing(tc) || tc == "auto" || throw(ArgumentError(
        "Ollama has no tool_choice: the model decides whether to call a tool (\"auto\"); " *
        "remove tool_choice=$(repr(tc)), or send no tools to rule calls out"))
    for t in something(chat.tools, Tool[])
        t.func.strict === true && throw(ArgumentError(
            "Ollama does not enforce strict tool schemas (tool $(repr(t.func.name))); leave strict unset"))
    end
    nothing
end

# The Chat sampling fields that are Ollama options, merged with the endpoint's own
# options (disjoint by construction: OllamaOptions has none of these names).
function _ollama_request_options(chat::Chat, endpoint::OllamaEndpoint)::Dict{String,Any}
    opts = _ollama_options(endpoint.options)
    num_predict = something(chat.max_completion_tokens, chat.max_tokens, Some(nothing))
    for (k, v) in ("temperature" => chat.temperature, "top_p" => chat.top_p, "seed" => chat.seed,
                   "num_predict" => num_predict, "presence_penalty" => chat.presence_penalty,
                   "frequency_penalty" => chat.frequency_penalty,
                   "stop" => (chat.stop isa String ? [chat.stop] : chat.stop))
        isnothing(v) || (opts[k] = v)
    end
    opts
end

# response_format → `format`: a JSON schema is sent as is; json_object becomes the
# schema of "any JSON object", which Ollama enforces while generating.
_ollama_format(::Nothing) = nothing
function _ollama_format(rf::ResponseFormat)
    js = rf.json_schema
    rf.type == "text" && isnothing(js) && return nothing
    rf.type == "json_object" && return Dict("type" => "object")
    rf.type == "json_schema" || throw(ArgumentError(
        "Ollama supports response_format text, json_object or json_schema (got $(repr(rf.type)))"))
    schema = js isa JsonSchemaAPI ? js.schema :
             js isa AbstractDict ? get(js, "schema", get(js, :schema, nothing)) : nothing
    schema isa AbstractDict || throw(ArgumentError("Ollama json_schema response_format needs a schema object"))
    schema
end

# The thinking an Ollama reply carried, captured on its assistant turn.
function _ollama_thinking(m::Message)::Union{String,Nothing}
    pc = m.provider_content
    m.role == RoleAssistant && pc isa ProviderContent && pc.provider === :ollama || return nothing
    parts = String[b["thinking"] for b in pc.blocks if b isa AbstractDict && get(b, "thinking", nothing) isa AbstractString]
    isempty(parts) ? nothing : join(parts)
end

# A call's id goes back with it, and a result names the call it answers by id and by
# function name (the Gemma 4 template matches ids first); a synthetic id stays local.
function _ollama_wire_call(tc::ToolCall)::Dict{String,Any}
    d = Dict{String,Any}("function" => Dict{String,Any}("name" => tc.func.name, "arguments" => tc.func.arguments))
    _is_synthetic_call_id(tc.id) || (d["id"] = tc.id)
    d
end

# A message's `images` entry: the raw base64 of an image or a sound clip
# (https://docs.ollama.com/api/chat). The list has no field for an image's detail level and
# none for a document, so either is refused rather than dropped.
_ollama_media(a::AudioAttachment)::String = base64encode(a.data)
function _ollama_media(a::ImageAttachment)::String
    isnothing(a.detail) || throw(ArgumentError(
        "Ollama's /api/chat takes an image's bytes alone and no detail level: build the ImageAttachment " *
        "without `detail` (got $(repr(a.detail)))"))
    base64encode(a.data)
end
_ollama_media(a::DocumentAttachment) = throw(ArgumentError(
    "Ollama's /api/chat takes images and sound clips, not documents ($(a)): send the document through an " *
    "OpenAI-compatible endpoint, or its text in the message"))

function _ollama_messages(msgs::Vector{Message})::Vector{Dict{String,Any}}
    names = Dict{String,String}()          # tool-call id → function name, for the results
    out = Dict{String,Any}[]
    for (i, m) in pairs(msgs)
        isnothing(m.name) || throw(ArgumentError("Ollama messages have no name field (message $i sets one)"))
        d = Dict{String,Any}("role" => m.role, "content" => something(m.content, m.refusal_message, ""))
        isnothing(m.attachments) || (d["images"] = String[_ollama_media(a) for a in m.attachments])
        if m.role == RoleAssistant
            thinking = _ollama_thinking(m)
            isnothing(thinking) || (d["thinking"] = thinking)
            if !isnothing(m.tool_calls)
                d["tool_calls"] = [_ollama_wire_call(tc) for tc in m.tool_calls]
                foreach(tc -> names[tc.id] = tc.func.name, m.tool_calls)
            end
        elseif m.role == RoleTool
            id = m.tool_call_id::String          # a tool message always names its call
            name = get(names, id, nothing)
            isnothing(name) && throw(ArgumentError(
                "tool result $(repr(id)) (message $i) answers no earlier tool call; " *
                "Ollama identifies a tool result by the called function's name"))
            d["tool_name"] = name
            _is_synthetic_call_id(id) || (d["tool_call_id"] = id)
        end
        push!(out, d)
    end
    out
end

function encode_request(service::OllamaEndpoint, chat::Chat)::String
    _ollama_validate(chat)
    body = Dict{String,Any}("model" => chat.model, "messages" => _ollama_messages(chat.messages),
                            "stream" => chat.stream === true)   # the API streams unless told not to
    isnothing(chat.tools) || (body["tools"] = chat.tools)
    format = _ollama_format(chat.response_format)
    isnothing(format) || (body["format"] = format)
    think = _ollama_think(chat.reasoning_effort, !isnothing(format))
    isnothing(think) || (body["think"] = think)
    opts = _ollama_request_options(chat, service)
    isempty(opts) || (body["options"] = opts)
    isnothing(service.keep_alive) || (body["keep_alive"] = _ollama_keep_alive(service.keep_alive))
    body["truncate"] = service.truncate
    isnothing(service.shift) || (body["shift"] = service.shift)
    # ChatRequest's `logprobs`, "Whether to return log probabilities of the output tokens", and
    # `top_logprobs` (https://docs.ollama.com/api/chat).
    isnothing(chat.logprobs) || (body["logprobs"] = chat.logprobs)
    isnothing(chat.top_logprobs) || (body["top_logprobs"] = chat.top_logprobs)
    _ollama_runner!(body, service)
    JSON.json(body)
end

# ─── Chat reply (/api/chat JSON → neutral Message) ──────────────────────────

# Ollama names its tool calls (`call_` + 8 characters); a call without an id gets a
# synthetic `unilm_call_<k>` (the prefix Gemini's decoder uses too), which is never
# sent back — its result is matched by function name.
function _ollama_tool_calls(raw, first_index::Int=0)::Vector{ToolCall}
    calls = ToolCall[]
    raw isa AbstractVector || return calls
    for (k, c) in enumerate(raw)
        f = c isa AbstractDict ? get(c, "function", nothing) : nothing
        f isa AbstractDict || continue
        args = get(f, "arguments", Dict{String,Any}())
        args isa AbstractString && (args = _parse_tool_arguments(args))
        args isa AbstractDict || throw(ArgumentError("Ollama tool-call arguments must be a JSON object"))
        id = get(c, "id", nothing)
        id = id isa AbstractString && !isempty(id) ? String(id) : "unilm_call_$(first_index + k)"
        push!(calls, ToolCall(id=id, func=GPTFunction(String(f["name"]), Dict{String,Any}(args))))
    end
    calls
end

# The token counts of a final reply: the prompt tokens evaluated for this request
# and the generated ones (thinking included).
function _ollama_usage(d::AbstractDict)::Union{TokenUsage,Nothing}
    p, c = get(d, "prompt_eval_count", nothing), get(d, "eval_count", nothing)
    p isa Integer || c isa Integer || return nothing
    p = p isa Integer ? Int(p) : 0
    c = c isa Integer ? Int(c) : 0
    cached = get(d, "prompt_eval_cached_count", 0)
    TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c,
               cached_tokens=cached isa Integer ? Int(cached) : 0)
end

# done_reason → finish reason: "stop" and "length" as OpenAI names them; a turn
# with tool calls reads "tool_calls" (see `_tool_finish_reason`).
_ollama_finish(reason, has_calls::Bool) =
    has_calls ? _tool_finish_reason(reason isa AbstractString ? String(reason) : nothing) :
    reason isa AbstractString ? String(reason) : nothing

# The reply names the "Model name used to generate this message", and its `logprobs` holds the
# "Log probability information for the generated tokens" (https://docs.ollama.com/api/chat): the
# result's `logprobs["content"]`, as an OpenAI-wire reply lists its output tokens.
_ollama_logprobs(tokens::AbstractVector)::Dict{String,Any} = Dict{String,Any}("content" => tokens)
_ollama_logprobs(_)::Nothing = nothing

function decode_response(::OllamaEndpoint, resp::HTTP.Response)
    d = JSON.parse(resp.body; dicttype=Dict{String,Any})
    d isa AbstractDict || error("Ollama reply is not a JSON object")
    m = get(d, "message", nothing)
    m isa AbstractDict || error("Ollama reply carries no message")
    text = get(m, "content", "")
    text = text isa AbstractString ? String(text) : ""
    calls = _ollama_tool_calls(get(m, "tool_calls", nothing))
    thinking = get(m, "thinking", nothing)
    pc = thinking isa AbstractString && !isempty(thinking) ?
         ProviderContent(:ollama, Any[Dict{String,Any}("thinking" => String(thinking))]) : nothing
    finish = _ollama_finish(get(d, "done_reason", nothing), !isempty(calls))
    msg = isempty(calls) ?
        Message(role=RoleAssistant, content=text, finish_reason=finish, provider_content=pc) :
        Message(role=RoleAssistant, content=(isempty(text) ? nothing : text), tool_calls=calls,
                finish_reason=finish, provider_content=pc)
    (; message=msg, usage=_ollama_usage(d), model=_wire_string(d, "model"),
       logprobs=_ollama_logprobs(get(d, "logprobs", nothing)))
end

# ─── Chat stream (newline-delimited JSON) ───────────────────────────────────
# Each line is one JSON object: a delta of `message.content` / `message.thinking`,
# whole tool calls, and finally `done: true` with the reason and token counts.
# A line `{"error": …}` is a failure reported on the stream.

_stream_frames(::OllamaEndpoint, carry::IOBuffer, ::Ref{String}, chunk::String) =
    Tuple{String,SubString{String}}[("", line) for line in _sse_complete_lines!(carry, chunk)]

function _ollama_thinking_buffer!(state::StreamState)::IOBuffer
    blk = get!(state.raw_pending, 0) do
        state.raw_provider = :ollama
        Dict{String,Any}("thinking" => IOBuffer())
    end
    blk["thinking"]::IOBuffer
end

function _ollama_finalize_thinking!(state::StreamState)::Nothing
    blk = pop!(state.raw_pending, 0, nothing)
    isnothing(blk) ||
        push!(state.raw_blocks, Dict{String,Any}("thinking" => takestring!(blk["thinking"])))
    nothing
end

function handle_sse_event!(service::OllamaEndpoint, ::AbstractString, payload::AbstractString,
                           state::StreamState)::Symbol
    d = JSON.parse(payload; dicttype=Dict{String,Any})
    d isa AbstractDict || return :continue
    err = get(d, "error", nothing)
    if !isnothing(err)
        state.error = Dict{String,Any}("error" => err)
        return :error
    end
    # Each line names the "Model name used for this stream event", and carries its tokens' log
    # probabilities, which the unstreamed reply joins (Ollama v0.40.11, server/routes.go).
    model = _wire_string(d, "model")
    isnothing(model) || (state.model = model)
    _logprobs_delta!(state, _ollama_logprobs(get(d, "logprobs", nothing)))
    m = get(d, "message", nothing)
    if m isa AbstractDict
        c = get(m, "content", nothing)
        if c isa AbstractString && !isempty(c)
            print(state.content, c)
            print(state.pending_delta, c)
        end
        t = get(m, "thinking", nothing)
        t isa AbstractString && !isempty(t) && print(_ollama_thinking_buffer!(state), t)
        for tc in _ollama_tool_calls(get(m, "tool_calls", nothing), length(state.tool_calls))
            idx = length(state.tool_calls)
            state.tool_calls[idx] = Dict{String,Any}("id" => tc.id, "type" => "function", "complete" => true,
                "function" => Dict{String,Any}("name" => tc.func.name, "arguments" => JSON.json(tc.func.arguments)))
        end
    end
    get(d, "done", false) === true || return :continue
    _ollama_finalize_thinking!(state)
    state.finish_reason = _ollama_finish(get(d, "done_reason", nothing), !isempty(state.tool_calls))
    state.usage = _ollama_usage(d)
    :done
end

# A connection that could not be made: most often, no server is running locally.
_failure_hint(s::OllamaEndpoint, e) = _unwrap_exception(e) isa HTTP.ConnectError ?
    " — no Ollama server answered at $(s.base_url): start it with `ollama serve` (or the Ollama " *
    "app), or point OLLAMA_HOST or base_url at the server" : ""

# ─── respond (Ollama's OpenAI-compatible /v1/responses) ─────────────────────
# Ollama serves the Responses API statelessly and reads only the fields below
# (https://docs.ollama.com/api/openai-compatibility); anything else it ignores
# without an error — a previous_response_id would silently lose the conversation.
# Those fields are refused here instead, each with its reason: a field that relies on a
# stored response because Ollama keeps none, any other because it reads none. A seed is
# refused on its own: that page lists none for /v1/responses, while Ollama's chat API
# reads one, so its fix is a Chat.
const _OLLAMA_RESPOND_MAPPED_FIELDS = (:service, :model, :input, :instructions, :tools, :tool_choice,
    :temperature, :top_p, :max_output_tokens, :stream, :text, :reasoning)
const _OLLAMA_RESPOND_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Respond), _OLLAMA_RESPOND_MAPPED_FIELDS, (:seed,)))
# Of those, the fields that ask the server to keep a response, or to read one it kept.
const _OLLAMA_RESPOND_STATE_FIELDS = (:store, :previous_response_id, :background, :conversation)

_stateless_responses(::OllamaEndpoint) = true

# An input image named by file id, which Ollama skips (it reads data URLs only).
_file_image(x) = false
_file_image(v::AbstractVector) = any(_file_image, v)
_file_image(m::InputMessage) = _file_image(m.content)
_file_image(d::AbstractDict) =
    (get(d, "type", get(d, :type, nothing)) == "input_image" && (haskey(d, "file_id") || haskey(d, :file_id))) ||
    _file_image(collect(values(d)))

# An input file (a document, by data, URL or file id): Ollama's OpenAI-compatibility page lists
# no file inputs for /v1/responses (https://docs.ollama.com/api/openai-compatibility), so the
# file might never reach the model.
_input_file(x) = false
_input_file(v::AbstractVector) = any(_input_file, v)
_input_file(m::InputMessage) = _input_file(m.content)
_input_file(d::AbstractDict) = get(d, "type", get(d, :type, nothing)) == "input_file" || _input_file(collect(values(d)))

# A replayed assistant message's phase: only OpenAI's Responses API reads it.
_input_phase(x) = false
_input_phase(v::AbstractVector) = any(_input_phase, v)
_input_phase(m::InputMessage) = !isnothing(m.phase)

# Keys of a function tool that Ollama's Responses tools do not have (OpenAI-only).
const _OLLAMA_RESPOND_TOOL_KEYS = ("async", "allowed_callers", "defer_loading", "output_schema")

function encode_agentic(service::OllamaEndpoint, r::Respond)::String
    fields = Symbol[f for f in _OLLAMA_RESPOND_UNMAPPED_FIELDS if !isnothing(getfield(r, f))]
    state = filter(in(_OLLAMA_RESPOND_STATE_FIELDS), fields)
    isempty(state) || throw(ArgumentError(
        "Ollama's Responses API ignores $(join(state, ", ")): it keeps no state between calls" *
        (isempty(intersect(state, (:previous_response_id, :conversation))) ? "" :
         "; send the whole conversation in `input`, or use a Chat")))
    isempty(fields) || throw(ArgumentError(length(fields) == 1 ?
        "Ollama's Responses API reads no $(only(fields)) (its OpenAI-compatibility page does not list it for " *
        "/v1/responses), so it would be dropped: unset it" :
        "Ollama's Responses API reads none of $(join(fields, ", ")) (its OpenAI-compatibility page does not list " *
        "them for /v1/responses), so they would be dropped: unset them"))
    isnothing(r.seed) || throw(ArgumentError(
        "Ollama's Responses API reads no seed — its OpenAI-compatibility page lists none for /v1/responses — " *
        "so the seed would be dropped: unset seed, or send the request as a Chat, whose seed Ollama reads"))
    isnothing(r.tool_choice) || r.tool_choice == "auto" || throw(ArgumentError(
        "Ollama has no tool_choice: the model decides whether to call a tool (\"auto\")"))
    rs = r.reasoning
    isnothing(rs) || all(isnothing, (rs.generate_summary, rs.summary, rs.context, rs.mode)) ||
        throw(ArgumentError("Ollama reads only reasoning.effort"))
    t = r.text
    constrained = false
    if !isnothing(t)
        isnothing(t.verbosity) || throw(ArgumentError("Ollama ignores text.verbosity"))
        t.format.type in ("text", "json_schema") || throw(ArgumentError(
            "Ollama's Responses API constrains output only to a json_schema text.format (got $(repr(t.format.type)))"))
        # Without a schema Ollama sends no format at all.
        constrained = t.format.type == "json_schema"
        constrained && !(t.format.schema isa AbstractDict) && throw(ArgumentError(
            "a json_schema text.format needs its schema: without one Ollama constrains nothing"))
    end
    # The Chat rule, for the same reason: a format turns thinking off, and a format with a
    # thinking effort is refused (Ollama applies the format only after thinking ends).
    effort = isnothing(rs) ? nothing : rs.effort
    think = _ollama_think(effort, constrained)
    _file_image(r.input) && throw(ArgumentError(
        "Ollama reads input images from data URLs only; an input_image with a file_id would be skipped"))
    _input_file(r.input) && throw(ArgumentError(
        "Ollama's Responses API documents no input_file parts, so the file might never reach the model: put " *
        "the document's text in an input_text part, or send the file to OPENAIServiceEndpoint"))
    _input_phase(r.input) && throw(ArgumentError(
        "Ollama's Responses API has no message phase: unset the InputMessage phase, or send the " *
        "request to OPENAIServiceEndpoint"))
    body = JSON.parse(invoke(encode_agentic, Tuple{OpenAIWireEndpointSpec,Respond}, service, r))
    think === false && isnothing(effort) && (body["reasoning"] = Dict("effort" => "none"))
    for tool in something(get(body, "tools", nothing), Any[])
        type = get(tool, "type", nothing)
        type == "function" || throw(ArgumentError(
            "Ollama's Responses API runs function tools here (got a $(repr(type)) tool)"))
        get(tool, "strict", nothing) === true && throw(ArgumentError(
            "Ollama does not enforce strict tool schemas (tool $(repr(get(tool, "name", "")))); leave strict unset"))
        extra = filter(k -> haskey(tool, k), _OLLAMA_RESPOND_TOOL_KEYS)
        isempty(extra) || throw(ArgumentError(
            "Ollama's Responses tools have no $(join(extra, ", ")) (tool $(repr(get(tool, "name", ""))))"))
    end
    JSON.json(body)
end

# ─── Embeddings (/api/embed) ────────────────────────────────────────────────

# One /api/embed input item: a (prompted) text as a string; with media, an object whose image and
# audio are bare base64 — or, for a recording's key, their digests. Ollama reads an item's fields in Go map
# order, which is random, and passes its media to the model in that order, so an item holding both an image
# and a sound clip embeds differently from one request to the next (Ollama 0.40.6: two vectors in ten
# requests to embeddinggemma-2:740m-q8_0, largest |Δ| 0.018); it is refused before any request.
_ollama_embed_item(emb, x::String, ::Bool) = _prompted(emb.task, x, nothing)
function _ollama_embed_item(emb, x::EmbeddingInput, digest::Bool)
    isnothing(x.image) || isnothing(x.audio) || throw(ArgumentError(
        "Ollama passes an item's image and sound clip to the model in a random order, so an item holding both " *
        "embeds differently from one request to the next: send the image and the audio as separate items"))
    text = isnothing(x.text) ? nothing : _prompted(emb.task, x.text, x.title)
    _item_media(x) || return text
    d = Dict{String,Any}()
    isnothing(text) || (d["text"] = text)
    isnothing(x.image) || (d["image"] = digest ? _media_digest(x.image) : base64encode(x.image.data))
    isnothing(x.audio) || (d["audio"] = digest ? _media_digest(x.audio) : base64encode(x.audio.data))
    d
end
_ollama_embed_input(emb, digest::Bool) = emb.input isa AbstractVector ?
    Any[_ollama_embed_item(emb, x, digest) for x in emb.input] : _ollama_embed_item(emb, emb.input, digest)

# EmbeddingGemma 2 was trained on 8,192 tokens, but at Ollama's default batch the llama.cpp builds measured
# (740m-q8_0, 270m-q8_0-text) refuse long inputs with HTTP 413 — 740m-q8_0 any longer than about 2,037
# tokens: the longest input they take follows `num_batch`, not the 262,144-token context the server
# reports for them. Unless the endpoint sets `num_ctx` or `num_batch`, both are sent as the trained
# window; with it, the builds measured (740m-q8_0, 740m-mxfp8, 270m-q8_0-text) embedded inputs of up to
# 6,251 tokens and refused one of 8,331 (Ollama 0.40.6).
const _EMBEDDINGGEMMA2_WINDOW = 8192
# `embeddinggemma-2`, `EmbeddingGemma2`, but not a v1-derived name whose number only starts with 2
# (`embeddinggemma-256d`, `embeddinggemma-2048ctx`).
_embeddinggemma2(model::AbstractString) = occursin(r"embeddinggemma-?2(?!\d)"i, model)

# The options an /api/embed request for `model` carries: the endpoint's, plus EmbeddingGemma 2's
# trained window when the endpoint sets neither `num_ctx` nor `num_batch`.
function _ollama_embed_options(s::OllamaEndpoint, model::AbstractString)::Dict{String,Any}
    opts = _ollama_options(s.options)
    if _embeddinggemma2(model) && !haskey(opts, "num_ctx") && !haskey(opts, "num_batch")
        opts["num_ctx"] = _EMBEDDINGGEMMA2_WINDOW
        opts["num_batch"] = _EMBEDDINGGEMMA2_WINDOW
    end
    opts
end

function _ollama_embed_body(s::OllamaEndpoint, emb::Embeddings, digest::Bool)::String
    isnothing(emb.user) || throw(ArgumentError("Ollama embeddings take no user field"))
    body = Dict{String,Any}("model" => emb.model, "input" => _ollama_embed_input(emb, digest), "truncate" => s.truncate)
    isnothing(emb.dimensions) || (body["dimensions"] = emb.dimensions)
    isnothing(s.keep_alive) || (body["keep_alive"] = _ollama_keep_alive(s.keep_alive))
    opts = _ollama_embed_options(s, emb.model)
    isempty(opts) || (body["options"] = opts)
    _ollama_runner!(body, s)
    JSON.json(body)
end
_encode_embeddings(s::OllamaEndpoint, emb::Embeddings)::String = _ollama_embed_body(s, emb, false)
# A recording keys and stores media by digest; a text-only body is keyed as sent.
_record_embeddings_body(s::OllamaEndpoint, emb::Embeddings, body::String)::String =
    _has_media(emb) ? _ollama_embed_body(s, emb, true) : body

# `{"embeddings": [[…], …], "prompt_eval_count": n}` → the OpenAI-shaped rows, in input order.
function _decode_embeddings(::OllamaEndpoint, data::Dict{String,Any})
    vecs = get(data, "embeddings", nothing)
    vecs isa AbstractVector || throw(ArgumentError("Ollama embeddings reply carries no \"embeddings\" array"))
    rows = [Dict{String,Any}("embedding" => v, "index" => i - 1) for (i, v) in enumerate(vecs)]
    n = get(data, "prompt_eval_count", nothing)
    rows, n isa Integer ? TokenUsage(prompt_tokens=Int(n), total_tokens=Int(n)) : nothing
end
