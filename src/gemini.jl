# ============================================================================
# Google Gemini native generateContent API
# Plugs into the wire-translation seam (encode_request / decode_response /
# decode_stream_chunk from requests.jl) so all chat orchestration is shared.
# Wire shape verified against ai.google.dev live docs on 2026-07-07.
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────
# Model is in the URL (like Azure); streaming is the URL METHOD, not a body flag.

function get_url(::Type{GEMINIServiceEndpoint}, chat::Chat)
    if chat.stream === true
        "$(GEMINI_NATIVE_BASE)/models/$(chat.model):streamGenerateContent?alt=sse"
    else
        "$(GEMINI_NATIVE_BASE)/models/$(chat.model):generateContent"
    end
end

_api_base_url(::Type{GEMINIServiceEndpoint}) =
    throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))

auth_header(::Type{GEMINIServiceEndpoint}) = [
    "x-goog-api-key" => ENV[GEMINI_API_KEY],
    "Content-Type"   => "application/json",
]

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::Type{GEMINIServiceEndpoint}) = Set([:chat, :tools, :streaming])

default_model(::Type{GEMINIServiceEndpoint}) = "gemini-3.5-flash"

# ─── Request encoding (neutral Chat → Gemini generateContent body) ───────────

function encode_request(::Type{GEMINIServiceEndpoint}, chat::Chat)
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
    isempty(gen) || (body[:generationConfig] = gen)
    # NB: `stream` is expressed in the URL method (get_url), never in the body.
    JSON.json(body)
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
            push!(pending, Dict{Symbol,Any}(:functionResponse => Dict{Symbol,Any}(
                :id => tcid, :name => tool_names[tcid], :response => _gemini_tool_response(m.content))))
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

# Assistant turn → Gemini model parts: optional text + functionCall parts (id, args,
# thoughtSignature echoed). Records id→name into `tool_names` for later functionResponse.
function _gemini_model_parts(m::Message, tool_names)
    parts = Vector{Dict{Symbol,Any}}()
    (isnothing(m.content) || isempty(m.content)) ||
        push!(parts, Dict{Symbol,Any}(:text => m.content))
    isnothing(m.tool_calls) && return parts
    for tc in m.tool_calls
        tool_names[tc.id] = tc.func.name
        part = Dict{Symbol,Any}(:functionCall => Dict{Symbol,Any}(
            :id => tc.id, :name => tc.func.name, :args => tc.func.arguments))
        isnothing(tc.thought_signature) || (part[:thoughtSignature] = tc.thought_signature)
        push!(parts, part)
    end
    parts
end

function _gemini_tool(t::GPTTool)
    f = t.func
    d = Dict{Symbol,Any}(:name => f.name)
    isnothing(f.description) || (d[:description] = f.description)
    isnothing(f.parameters)  || (d[:parameters] = f.parameters)
    d
end

_gemini_tool_config(tc::String) = Dict(:functionCallingConfig => Dict(:mode =>
    tc == "auto"     ? "AUTO" :
    tc == "none"     ? "NONE" :
    tc == "required" ? "ANY"  : "AUTO"))
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
