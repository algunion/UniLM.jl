# ============================================================================
# Google Gemini native generateContent API
# Plugs into the wire-translation seam (encode_request / decode_response /
# handle_sse_event! from sse.jl) so all chat orchestration is shared.
# https://ai.google.dev/gemini-api/docs/latest-model
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────
# Model is in the URL (like Azure); streaming is the URL METHOD, not a body flag.

function get_url(::Type{GEMINIServiceEndpoint}, chat::Chat)::String
    # The model names a path segment; the `:<verb>` suffix and `?alt=sse` are the
    # template's own structure, so only the model is encoded.
    m = _uripart(chat.model)
    if chat.stream === true
        "$(GEMINI_NATIVE_BASE)/models/$m:streamGenerateContent?alt=sse"
    else
        "$(GEMINI_NATIVE_BASE)/models/$m:generateContent"
    end
end

_resolve_base_url(::Type{GEMINIServiceEndpoint}) =
    throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))

auth_header(::Type{GEMINIServiceEndpoint})::Vector{Pair{String,String}} = [
    "x-goog-api-key" => ENV[GEMINI_API_KEY],
    "Content-Type"   => "application/json",
]

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::Type{GEMINIServiceEndpoint}) = Set([:chat, :tools, :streaming, :agentic, :json_output])

default_model(::Type{GEMINIServiceEndpoint}) = "gemini-3.8-flash"

function _gemini_validate_sampling(model::String, temperature, top_p)
    if _model_family(model, "gemini-3.8-flash") &&
       (!isnothing(temperature) || !isnothing(top_p))
        throw(ArgumentError("$model does not support temperature or top_p; use reasoning effort instead"))
    end
    nothing
end

# Thinking levels per model family, transcribed from the "Controlling thinking" table
# (https://ai.google.dev/gemini-api/docs/thinking, 2026-09-22).
const _GEMINI_THINKING_LEVELS = (
    ("gemini-3.8-flash",            ("low", "medium", "high")),
    ("gemini-3.7-flash",            ("low", "medium", "high")),
    ("gemini-3.6-flash",            ("minimal", "low", "medium", "high")),
    ("gemini-3.5-flash-lite",       ("minimal", "low", "medium", "high")),
    ("gemini-3.1-pro-preview",      ("low", "medium", "high")),
    ("gemini-3.1-flash-lite-image", ("minimal", "high")),
    ("gemini-3-flash-preview",      ("minimal", "low", "medium", "high")),
    ("gemini-3-pro-preview",        ("low", "high")),
    ("gemini-3.5-flash",            ("minimal", "low", "medium", "high")),
    ("gemini-2.5-pro",              ("low", "medium", "high")),
    ("gemini-2.5-flash",            ("low", "medium", "high")),
    ("gemini-2.5-flash-lite",       ("low", "medium", "high")),
)

# A model resolves to its LONGEST matching family (gemini-3.5-flash-lite is also in
# the gemini-3.5-flash family); `nothing` for an unlisted family.
function _gemini_thinking_levels(model::AbstractString, table=_GEMINI_THINKING_LEVELS)
    rows = filter(row -> _model_family(model, first(row)), table)
    isempty(rows) ? nothing : last(argmax(row -> length(first(row)), rows))
end

function _gemini_thinking_level(model::String, effort::String; native::Bool=false)::String
    effort in ("minimal", "low", "medium", "high") || throw(ArgumentError(
        "Gemini thinking effort must be minimal, low, medium, or high (got $(repr(effort)))"))
    levels = _gemini_thinking_levels(model)
    isnothing(levels) || effort in levels || throw(ArgumentError(
        "$model supports $(join(levels, ", ", length(levels) == 2 ? " or " : ", or ")) thinking effort"))
    native && startswith(model, "gemini-2.5-") && throw(ArgumentError(
        "Gemini 2.5 generateContent uses thinkingBudget; use Respond with Reasoning for thinking levels"))
    effort
end

# Fail closed when a neutral option has no native translation. The parallel-tool
# default predates the native backend; Gemini determines parallelism itself.
const _GEMINI_CHAT_MAPPED_FIELDS = (:service, :model, :messages, :history, :tools,
    :tool_choice, :parallel_tool_calls, :temperature, :top_p, :n, :stream, :stop,
    :max_tokens, :max_completion_tokens, :response_format, :reasoning_effort,
    :safety_identifier, :_cumulative_cost)
const _GEMINI_CHAT_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Chat), _GEMINI_CHAT_MAPPED_FIELDS))

function _gemini_validate_chat(chat::Chat)
    _gemini_validate_sampling(chat.model, chat.temperature, chat.top_p)
    isnothing(chat.n) || chat.n == 1 || throw(ArgumentError("Native Gemini Chat returns one candidate; n must be 1"))
    fields = Symbol[f for f in _GEMINI_CHAT_UNMAPPED_FIELDS if !isnothing(getfield(chat, f))]
    isempty(fields) || throw(ArgumentError(
        "Native Gemini Chat does not support field(s) $(join(fields, ", ")); use a supported option or GEMINIOpenAIServiceEndpoint"))
    nothing
end

# Request label values (GenerateContentRequest.labels in the Gemini API discovery
# document): at most 63 characters (Unicode code points), only lowercase letters,
# numeric characters, underscores, and dashes; international characters are allowed.
const _GEMINI_LABEL_VALUE = r"\A[\p{Ll}\p{Lo}\p{N}_-]{0,63}\z"

function _gemini_safety_label(id::String)::String
    occursin(_GEMINI_LABEL_VALUE, id) || throw(ArgumentError(
        "Native Gemini sends safety_identifier as a request label; label values allow at most " *
        "63 characters, only lowercase letters (international characters allowed), numeric " *
        "characters, underscores, and dashes (got $(length(id)) characters). A 64-character " *
        "hex hash such as a SHA-256 digest is one character too long for a Gemini label."))
    id
end

# ─── Request encoding (neutral Chat → Gemini generateContent body) ───────────

function encode_request(::Type{GEMINIServiceEndpoint}, chat::Chat)
    _gemini_validate_chat(chat)
    body = Dict{Symbol,Any}()
    sysinstr, contents = _gemini_contents(chat.messages)
    isnothing(sysinstr) || (body[:systemInstruction] = Dict(:parts => [Dict(:text => sysinstr)]))
    body[:contents] = contents
    isnothing(chat.tools) ||
        (body[:tools] = [Dict(:functionDeclarations => [_gemini_tool(t) for t in chat.tools])])
    isnothing(chat.tool_choice) || (body[:toolConfig] = _gemini_tool_config(chat.tool_choice))
    gen = Dict{Symbol,Any}()
    # Gemini does NOT require maxOutputTokens; omit when unset (a low cap truncates
    # Gemini-3 thinking before any answer). No default_max_tokens override.
    # NB: plain `something(a, b, nothing)` THROWS when all are nothing — use a ternary
    # so "both unset" yields nothing (→ omitted), not an ArgumentError.
    mot = !isnothing(chat.max_completion_tokens) ? chat.max_completion_tokens : chat.max_tokens
    isnothing(mot)              || (gen[:maxOutputTokens] = mot)
    isnothing(chat.temperature) || (gen[:temperature] = chat.temperature)
    isnothing(chat.top_p)       || (gen[:topP] = chat.top_p)
    isnothing(chat.stop)        || (gen[:stopSequences] = chat.stop isa String ? [chat.stop] : chat.stop)
    if (effort = chat.reasoning_effort) !== nothing
        gen[:thinkingConfig] = Dict(:thinkingLevel => uppercase(_gemini_thinking_level(chat.model, effort; native=true)))
    end
    (fmt = _gemini_response_format(chat.response_format)) === nothing || (gen[:responseFormat] = fmt)
    isempty(gen) || (body[:generationConfig] = gen)
    # Request labels carry an aggregator's end-user id under Google's documented key.
    isnothing(chat.safety_identifier) ||
        (body[:labels] = Dict(:safety_identifier => _gemini_safety_label(chat.safety_identifier)))
    # NB: `stream` is expressed in the URL method (get_url), never in the body.
    JSON.json(body)
end

# Neutral response_format → generationConfig.responseFormat = {text: TextResponseFormat}.
# TextResponseFormat.mimeType is an enum on this wire: the API answers 400 to
# "application/json" and accepts APPLICATION_JSON (observed 2026-09-22). The OpenAI
# schema name, description, and strict flag have no generateContent counterpart.
_gemini_response_format(::Nothing) = nothing
function _gemini_response_format(rf::ResponseFormat)
    js = rf.json_schema
    rf.type == "text" && isnothing(js) && return nothing
    rf.type == "json_object" && isnothing(js) && return Dict(:text => Dict(:mimeType => "APPLICATION_JSON"))
    rf.type == "json_schema" || throw(ArgumentError(
        "Native Gemini Chat supports response_format text, json_object, or json_schema " *
        "(got $(repr(rf.type))$(isnothing(js) ? "" : " with a json_schema"))"))
    schema = js isa JsonSchemaAPI ? js.schema :
             js isa AbstractDict ? get(js, "schema", get(js, :schema, nothing)) : nothing
    schema isa AbstractDict || throw(ArgumentError(
        "Native Gemini Chat json_schema response_format needs a schema object"))
    Dict(:text => Dict(:mimeType => "APPLICATION_JSON", :schema => schema))
end

# Split neutral messages into (systemInstruction::Union{String,Nothing}, contents).
# - system → concatenated top-level systemInstruction text
# - user   → {role:"user", parts:[{text}]}
# - assistant → {role:"model", parts:[{text}?, {functionCall,thoughtSignature?}...]}
# - consecutive tool results → ONE {role:"user"} of functionResponse parts
# - a tool result with no preceding model functionCall of that id → loud ArgumentError.
function _gemini_contents(messages)
    sysinstr = nothing
    out = Vector{Dict{Symbol,Any}}()
    tool_names = Dict{String,String}()        # id → function name (functionResponse correlation)
    pending = Vector{Dict{Symbol,Any}}()      # buffered functionResponse parts
    function flush!()
        isempty(pending) && return
        push!(out, Dict{Symbol,Any}(:role => "user", :parts => copy(pending)))
        empty!(pending)
    end
    for m in messages
        if m.role == RoleSystem
            sysinstr = isnothing(sysinstr) ? m.content :
                       string(sysinstr, "\n\n", something(m.content, ""))
        elseif m.role == RoleTool
            tcid = something(m.tool_call_id, "")
            haskey(tool_names, tcid) || throw(ArgumentError(
                "functionResponse references unknown tool_call id $(repr(tcid)); no preceding model functionCall emitted it"))
            fr = Dict{Symbol,Any}(:name => tool_names[tcid], :response => _gemini_tool_response(m.content))
            _is_synthetic_call_id(tcid) || (fr[:id] = tcid)
            push!(pending, Dict{Symbol,Any}(:functionResponse => fr))
        elseif m.role == RoleAssistant
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "model", :parts => _gemini_model_parts(m, tool_names)))
        else  # RoleUser
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "user",
                :parts => [Dict{Symbol,Any}(:text => something(m.content, ""))]))
        end
    end
    flush!()
    (sysinstr, out)
end

# Assistant turn → Gemini model parts: echo captured provider-native parts
# verbatim when this provider produced them (text-part thoughtSignatures
# intact); otherwise rebuild optional text + functionCall parts. Either way,
# record id→name into `tool_names` so later functionResponse parts correlate.
function _gemini_model_parts(m::Message, tool_names)
    if !isnothing(m.tool_calls)
        for tc in m.tool_calls
            tool_names[tc.id] = tc.func.name
        end
    end
    pc = m.provider_content
    pc isa ProviderContent && pc.provider === :gemini && !isempty(pc.blocks) &&
        return pc.blocks
    parts = Vector{Dict{Symbol,Any}}()
    (isnothing(m.content) || isempty(m.content)) ||
        push!(parts, Dict{Symbol,Any}(:text => m.content))
    isnothing(m.tool_calls) && return parts
    for tc in m.tool_calls
        fcall = Dict{Symbol,Any}(:name => tc.func.name, :args => tc.func.arguments)
        _is_synthetic_call_id(tc.id) || (fcall[:id] = tc.id)
        part = Dict{Symbol,Any}(:functionCall => fcall)
        isnothing(tc.thought_signature) || (part[:thoughtSignature] = tc.thought_signature)
        push!(parts, part)
    end
    parts
end

function _gemini_tool(t::Tool)
    f = t.func
    d = Dict{Symbol,Any}(:name => f.name)
    isnothing(f.description) || (d[:description] = f.description)
    # Neutral tools carry JSON Schema, not Google's restricted OpenAPI Schema.
    isnothing(f.parameters)  || (d[:parametersJsonSchema] = f.parameters)
    d
end

_gemini_tool_config(tc::String) = Dict(:functionCallingConfig => Dict(:mode =>
    tc == "auto"     ? "AUTO" :
    tc == "none"     ? "NONE" :
    tc == "required" ? "ANY"  : throw(ArgumentError("Unknown Gemini tool_choice $(repr(tc))"))))
_gemini_tool_config(tc::GPTToolChoice) = Dict(:functionCallingConfig =>
    Dict(:mode => "ANY", :allowedFunctionNames => [string(tc.func)]))

# Gemini requires functionResponse.response to be a JSON OBJECT. Pass through a
# JSON-object string; otherwise wrap the raw string as {"result": ...}.
function _gemini_tool_response(content)
    s = something(content, "")
    try
        v = JSON.parse(s; dicttype=Dict{String,Any})
        v isa AbstractDict ? v : Dict{String,Any}("result" => s)
    catch
        Dict{String,Any}("result" => s)
    end
end

# ─── Response decoding (Gemini generateContent → neutral Message) ────────────

# The Gemini API's FunctionCall.id is OPTIONAL. Id-less calls get a unique
# synthetic positional id so tool results correlate within the turn; the
# prefix is reserved and such ids are OMITTED on re-encode (Gemini correlates
# positionally when ids are absent — echoing a fabricated id would be wrong).
_is_synthetic_call_id(id::AbstractString) = startswith(id, "unilm_call_")

# Gemini finishReason → neutral finish_reason. OPEN ENUM: Google adds values
# unannounced, so unknown → STOP (never throw). Tool-call detection is by
# functionCall presence in the decoder, NOT by this reason (Gemini says STOP).
function _gemini_finish_reason(fr)
    fr == "STOP"       ? STOP :
    fr == "MAX_TOKENS" ? "length" :
    fr in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY") ? CONTENT_FILTER :
    isnothing(fr)      ? STOP :
    STOP
end

# Gemini usageMetadata → neutral TokenUsage. Assumptions (verify vs live docs):
# promptTokenCount INCLUDES cachedContentTokenCount (cached is a subset), so we
# do NOT add; candidatesTokenCount EXCLUDES thoughtsTokenCount, and thoughts bill
# at the output rate → completion = candidates + thoughts.
function _gemini_usage(u)::Union{TokenUsage,Nothing}
    u isa AbstractDict || return nothing
    _i(x) = x isa Integer ? Int(x) : 0
    prompt   = _i(get(u, "promptTokenCount", 0))
    cand     = _i(get(u, "candidatesTokenCount", 0))
    thoughts = _i(get(u, "thoughtsTokenCount", 0))
    cached   = _i(get(u, "cachedContentTokenCount", 0))
    total    = _i(get(u, "totalTokenCount", prompt + cand + thoughts))
    TokenUsage(prompt_tokens = prompt, completion_tokens = cand + thoughts,
        total_tokens = total, cached_tokens = cached, reasoning_tokens = thoughts)
end

function decode_response(::Type{GEMINIServiceEndpoint}, resp::HTTP.Response)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    cands = get(data, "candidates", [])
    # No candidate is not "the model said nothing" — there is no assistant turn to
    # report at all (a prompt-level block is the usual cause). Fail loud with the
    # provider's own diagnostics; the verb turns this into its typed error result.
    isempty(cands) && error("Gemini response contained no candidates (promptFeedback: ",
                            JSON.json(get(data, "promptFeedback", Dict{String,Any}())), ")")
    cand = cands[1]
    fr_raw = get(cand, "finishReason", nothing)
    parts = get(get(cand, "content", Dict{String,Any}()), "parts", Any[])
    # Verbatim capture for round-trip: Gemini-3 attaches thoughtSignature to
    # text parts too; the neutral IR keeps it only on functionCall parts, so
    # echoing these parts verbatim is what preserves multi-turn thinking.
    pc = parts isa AbstractVector && !isempty(parts) ?
         ProviderContent(:gemini, parts) : nothing
    text = IOBuffer()
    tool_calls = ToolCall[]
    for p in (parts isa AbstractVector ? parts : Any[])
        if haskey(p, "text") && get(p, "thought", false) !== true
            print(text, p["text"])
        elseif haskey(p, "functionCall")
            fc = p["functionCall"]
            args = get(fc, "args", Dict{String,Any}())
            args isa AbstractDict || (args = Dict{String,Any}())
            raw_id = get(fc, "id", "")
            id = isempty(raw_id) ? "unilm_call_$(length(tool_calls) + 1)" : raw_id
            push!(tool_calls, ToolCall(id=id,
                func=GPTFunction(get(fc, "name", ""), args),
                thought_signature=get(p, "thoughtSignature", nothing)))
        end
    end
    finish = isempty(tool_calls) ? _gemini_finish_reason(fr_raw) : TOOL_CALLS
    usage = _gemini_usage(get(data, "usageMetadata", nothing))
    txt = String(take!(text))
    msg = if !isempty(tool_calls)
        Message(role=RoleAssistant, content=(isempty(txt) ? nothing : txt),
                tool_calls=tool_calls, finish_reason=finish, provider_content=pc)
    elseif finish == CONTENT_FILTER && isempty(txt)
        Message(role=RoleAssistant, refusal_message="Model response blocked by safety filter.",
                finish_reason=finish, provider_content=pc)
    else
        # A well-formed candidate that yields no text is a real, legitimate turn:
        # thinking models routinely spend the whole completion budget on thought
        # tokens (finishReason MAX_TOKENS, zero visible parts). Report the empty
        # turn the provider actually sent — substituting prose here would inject
        # content nobody generated into the reply and into the next request.
        Message(role=RoleAssistant, content=txt, finish_reason=finish, provider_content=pc)
    end
    (; message=msg, usage)
end

# ─── Streaming event handler for the shared SSE machine (src/sse.jl) —
# replaces decode_stream_chunk (removed in 0.11.3).
# Gemini streaming has NO sentinel EOS: this handler NEVER returns :done — the
# driver reads to EOF, so trailing usageMetadata-only chunks are consumed.
# finishReason only records state.finish_reason (EOS-on-finishReason was the
# old, wrong behavior: it skipped trailing usage). Each functionCall part
# arrives whole, so its entry is marked "complete" => true (immediate
# on_tool_call firing in the driver).
function handle_sse_event!(::Type{GEMINIServiceEndpoint}, event::AbstractString,
                           payload::AbstractString, state::StreamState)::Symbol
    ev = JSON.parse(payload; dicttype=Dict{String,Any})
    ev isa AbstractDict || return :continue
    if (err = get(ev, "error", nothing)) isa AbstractDict
        state.error = Dict{String,Any}(err)
        return :error
    end
    feedback = get(ev, "promptFeedback", nothing)
    if feedback isa AbstractDict && haskey(feedback, "blockReason")
        state.error = Dict{String,Any}("message" => "Gemini prompt blocked", "promptFeedback" => feedback)
        return :error
    end
    cands = get(ev, "candidates", nothing)
    if cands isa AbstractVector
        for cand in cands
            cand isa AbstractDict || continue
            for p in get(get(cand, "content", Dict{String,Any}()), "parts", Any[])
                p isa AbstractDict || continue
                # Preserve every part, including a signature-only final chunk.
                # Combining text parts can invalidate their thought signatures.
                state.raw_provider = :gemini
                push!(state.raw_blocks, p)
                if haskey(p, "text") && get(p, "thought", false) !== true
                    txt = p["text"]
                    if txt isa AbstractString
                        print(state.content, txt)
                        print(state.pending_delta, txt)
                    end
                elseif haskey(p, "functionCall")
                    fc = p["functionCall"]
                    fc isa AbstractDict || continue
                    idx = length(state.tool_calls)
                    raw_id = get(fc, "id", "")
                    state.tool_calls[idx] = Dict{String,Any}(
                        "id" => (isempty(raw_id) ? "unilm_call_$(idx + 1)" : raw_id),
                        "type" => "function",
                        "function" => Dict{String,Any}(
                            "name" => get(fc, "name", ""),
                            "arguments" => JSON.json(get(fc, "args", Dict{String,Any}()))),
                        "thought_signature" => get(p, "thoughtSignature", nothing),
                        "complete" => true)
                end
            end
            fr = get(cand, "finishReason", nothing)
            isnothing(fr) ||
                (state.finish_reason = isempty(state.tool_calls) ? _gemini_finish_reason(fr) : TOOL_CALLS)
            if state.finish_reason == CONTENT_FILTER && position(state.content) == 0
                print(state.refusal, "Model response blocked by safety filter.")
            end
        end
    end
    u = get(ev, "usageMetadata", nothing)
    u isa AbstractDict && (state.usage = _gemini_usage(u))
    :continue
end
