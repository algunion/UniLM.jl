# ============================================================================
# Ollama (https://docs.ollama.com/api). Chat and embeddings speak the native API
# (`/api/chat`, `/api/embed`): it is the only one that takes the runtime `options`
# (context window, sampling knobs), `keep_alive`, `think` and `truncate`; the
# OpenAI-compatible routes accept and ignore them, and truncate over-long input
# without a word. `respond` and FIM keep the OpenAI-compatible routes
# (`/v1/responses`, `/v1/completions`).
# ============================================================================

get_url(s::OllamaEndpoint, ::Chat)::String = s.base_url * OLLAMA_CHAT_PATH
get_url(s::OllamaEndpoint, ::Embeddings)::String = s.base_url * OLLAMA_EMBED_PATH
get_url(s::OllamaEndpoint, ::FIMCompletion)::String = s.base_url * COMPLETIONS_PATH
_resolve_base_url(s::OllamaEndpoint) = s.base_url

# A local server takes no credentials.
auth_header(::OllamaEndpoint)::Vector{Pair{String,String}} = ["Content-Type" => "application/json"]

provider_capabilities(::OllamaEndpoint) =
    Set([:chat, :embeddings, :fim, :tools, :streaming, :json_output, :responses, :models])

_model_hint(::OllamaEndpoint) =
    " (an installed model, e.g. model=\"gemma4:e4b\"; list_models(service=OllamaEndpoint()) lists them)"

# A model on the caller's own machine is free: its turns cost 0.0, with no
# missing-price warning.
_is_local(::OllamaEndpoint) = true

# ─── Chat request (neutral Chat → /api/chat body) ───────────────────────────

# Fail closed: a Chat field /api/chat has no counterpart for is refused, never
# dropped (the request would not be the one the caller asked for).
const _OLLAMA_CHAT_MAPPED_FIELDS = (:service, :model, :messages, :history, :tools, :tool_choice,
    :parallel_tool_calls, :temperature, :top_p, :n, :stream, :stop, :max_tokens,
    :max_completion_tokens, :presence_penalty, :response_format, :frequency_penalty, :seed,
    :reasoning_effort, :_cumulative_cost)
const _OLLAMA_CHAT_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Chat), _OLLAMA_CHAT_MAPPED_FIELDS))

# reasoning_effort → `think`: "none" turns thinking off; the effort levels pass
# through (models with a single thinking mode treat any level as on). Unset, the
# model's default applies (Gemma 4 thinks) — except with a response_format: Ollama
# constrains a reply to the format only after a thinking block ends, so a thinking
# model that answers without thinking is not constrained at all (gemma4:e4b answered
# plain text to 10 of 10 schema requests with thinking on, and valid JSON to 10 of 10
# with it off, on Ollama 0.34.4). A format therefore turns thinking off, and a format
# with a thinking effort is refused rather than sent to fail at random.
const _OLLAMA_EFFORTS = ("none", "low", "medium", "high")
function _ollama_think(effort::Union{String,Nothing}, constrained::Bool)
    isnothing(effort) && return constrained ? false : nothing
    effort in _OLLAMA_EFFORTS || throw(ArgumentError(
        "Ollama takes reasoning_effort $(join(repr.(_OLLAMA_EFFORTS), ", ")) (got $(repr(effort)))"))
    effort == "none" && return false
    constrained && throw(ArgumentError(
        "Ollama applies a response_format only after the model's thinking ends, so a reply that " *
        "skips thinking is unconstrained; use reasoning_effort=\"none\" (or leave it unset) with a " *
        "response_format, or drop the response_format to let the model think"))
    effort
end

# keep_alive on the wire: seconds, or -1 for "stay loaded".
_ollama_keep_alive(ka::Float64) = isinf(ka) ? -1 : (isinteger(ka) ? Int(ka) : ka)

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

function _ollama_messages(msgs::Vector{Message})::Vector{Dict{String,Any}}
    names = Dict{String,String}()          # tool-call id → function name, for the results
    out = Dict{String,Any}[]
    for (i, m) in pairs(msgs)
        isnothing(m.name) || throw(ArgumentError("Ollama messages have no name field (message $i sets one)"))
        d = Dict{String,Any}("role" => m.role, "content" => something(m.content, m.refusal_message, ""))
        isnothing(m.attachments) || (d["images"] = [base64encode(a.data) for a in m.attachments])
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
    (; message=msg, usage=_ollama_usage(d))
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
# Those fields are refused here instead.
const _OLLAMA_RESPOND_MAPPED_FIELDS = (:service, :model, :input, :instructions, :tools, :tool_choice,
    :temperature, :top_p, :max_output_tokens, :stream, :text, :reasoning)
const _OLLAMA_RESPOND_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Respond), _OLLAMA_RESPOND_MAPPED_FIELDS))

_stateless_responses(::OllamaEndpoint) = true

# An input image named by file id, which Ollama skips (it reads data URLs only).
_file_image(x) = false
_file_image(v::AbstractVector) = any(_file_image, v)
_file_image(m::InputMessage) = _file_image(m.content)
_file_image(d::AbstractDict) =
    (get(d, "type", get(d, :type, nothing)) == "input_image" && (haskey(d, "file_id") || haskey(d, :file_id))) ||
    _file_image(collect(values(d)))

# Keys of a function tool that Ollama's Responses tools do not have (OpenAI-only).
const _OLLAMA_RESPOND_TOOL_KEYS = ("async", "allowed_callers", "defer_loading", "output_schema")

function encode_agentic(service::OllamaEndpoint, r::Respond)::String
    fields = Symbol[f for f in _OLLAMA_RESPOND_UNMAPPED_FIELDS if !isnothing(getfield(r, f))]
    isempty(fields) || throw(ArgumentError(
        "Ollama's Responses API ignores $(join(fields, ", ")): it keeps no state between calls" *
        (isempty(intersect(fields, (:previous_response_id, :conversation))) ? "" :
         "; send the whole conversation in `input`, or use a Chat")))
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

function _encode_embeddings(s::OllamaEndpoint, emb::Embeddings)::String
    isnothing(emb.user) || throw(ArgumentError("Ollama embeddings take no user field"))
    body = Dict{String,Any}("model" => emb.model, "input" => emb.input, "truncate" => s.truncate)
    isnothing(emb.dimensions) || (body["dimensions"] = emb.dimensions)
    isnothing(s.keep_alive) || (body["keep_alive"] = _ollama_keep_alive(s.keep_alive))
    opts = _ollama_options(s.options)
    isempty(opts) || (body["options"] = opts)
    JSON.json(body)
end

# `{"embeddings": [[…], …], "prompt_eval_count": n}` → the OpenAI-shaped rows, in input order.
function _decode_embeddings(::OllamaEndpoint, data::Dict{String,Any})
    vecs = get(data, "embeddings", nothing)
    vecs isa AbstractVector || throw(ArgumentError("Ollama embeddings reply carries no \"embeddings\" array"))
    rows = [Dict{String,Any}("embedding" => v, "index" => i - 1) for (i, v) in enumerate(vecs)]
    n = get(data, "prompt_eval_count", nothing)
    rows, n isa Integer ? TokenUsage(prompt_tokens=Int(n), total_tokens=Int(n)) : nothing
end
