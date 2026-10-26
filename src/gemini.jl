# ============================================================================
# Google Gemini native generateContent API
# Plugs into the wire-translation seam (encode_request / decode_response /
# handle_sse_event! from sse.jl) so all chat orchestration is shared.
# https://ai.google.dev/api/generate-content
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

provider_capabilities(::Type{GEMINIServiceEndpoint}) =
    Set([:chat, :tools, :streaming, :agentic, :json_output, :images, :image_edits, :models])

default_model(::Type{GEMINIServiceEndpoint}) = "gemini-3.8-flash"

# The models released before Gemini 3.6 Flash (2026-07-21) and still served, by exact id: the text
# and image models of the 2026-10-22 model listing. From Gemini 3.6 Flash and 3.5 Flash-Lite on, for
# those models and all future releases, "the API ignores" temperature, top_p and top_k, and a
# request ending with a model turn is an HTTP 400
# (https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6#api-changes-and-parameter-updates;
# gemini-4-argon and gemini-3.8-flash answered sampled requests with HTTP 200 on 2026-10-22), so
# both rules refuse every other id. An alias is not one of these models: it moves to newer ones.
const _GEMINI_PRE_3_6_MODELS = (
    "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-flash-lite",                                   # shuts down 2027-05-07
    "gemini-3-flash-preview",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemma-4-31b-it", "gemma-4-26b-a4b-it",
    "gemini-3.1-flash-lite-image", "gemini-3-pro-image",
    "gemini-3.1-flash-image",                                  # deprecated, no shutdown date announced
    "gemini-2.5-flash-image")                                  # shuts down 2027-03-15

function _gemini_validate_sampling(model::String, temperature, top_p)
    set = _set_fields("temperature" => !isnothing(temperature), "top_p" => !isnothing(top_p))
    isempty(set) || model in _GEMINI_PRE_3_6_MODELS || throw(ArgumentError(
        "Gemini ignores $(join(set, " and ")) on Gemini 3.6 Flash, 3.5 Flash-Lite and every later model " *
        "(https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6), so UniLM sends sampling parameters " *
        "only to $(join(_GEMINI_PRE_3_6_MODELS, ", ")): unset $(join(set, " and ")) for $model"))
    nothing
end

# From Gemini 3.6 Flash and 3.5 Flash-Lite on, a request whose last non-empty turn is the model's
# is an HTTP 400 "Requests ending with a model turn are not supported."
# (https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6#api-changes-and-parameter-updates;
# observed on gemini-3.8-flash, 2026-10-22). The earlier models continue a prefilled turn.
function _gemini_validate_last_turn(model::String, contents::Vector{Dict{Symbol,Any}})::Nothing
    isempty(contents) || last(contents)[:role] != "model" || model in _GEMINI_PRE_3_6_MODELS ||
        throw(ArgumentError("$model answers HTTP 400 to a request that ends with a model turn: Gemini " *
            "3.6 Flash, 3.5 Flash-Lite and every later model do not continue a prefilled answer. End the " *
            "conversation with a user message; to shape the answer, use a system message or response_format"))
    nothing
end

# Gemini caches repeated prompt prefixes implicitly, with no option to set
# (https://ai.google.dev/gemini-api/docs/caching). Its explicit caching — cached contents, which
# generateContent offers on Gemini 3 and 2.5 models — is an API of its own that UniLM does not
# wrap, so a cache option has nowhere to go.
_gemini_refuse_cache_options(::Nothing) = nothing
_gemini_refuse_cache_options(::PromptCacheOptions) = throw(ArgumentError(
    "Gemini takes no prompt_cache_options: it caches repeated prompt prefixes implicitly, with no option " *
    "to set, and UniLM does not wrap its explicit caching (cached contents) " *
    "(https://ai.google.dev/gemini-api/docs/caching); unset prompt_cache_options"))

# The service tiers a request names (https://ai.google.dev/gemini-api/docs/optimization):
# generateContent takes standard, flex and priority (GenerateContentRequest.serviceTier, whose
# "unspecified" is standard), and so does the Interactions API, whose reference lists flex,
# standard and priority (https://ai.google.dev/api/interactions-api). It no longer lists deferred,
# which the API answered with HTTP 400 on 2026-10-26: "The value 'deferred' is not supported for
# 'service_tier' on the Gemini API, it is allowed on the Gemini Enterprise Agent Platform."
# A reply reports the tier that billed it.
const _GEMINI_SERVICE_TIERS = ("standard", "flex", "priority")

# GenerationConfig.seed is an int32 (its format in the API's discovery document,
# https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta), so a seed outside
# that range has no Gemini value.
function _gemini_seed(seed::Integer)::Int64
    typemin(Int32) <= seed <= typemax(Int32) || throw(ArgumentError(
        "Gemini takes a seed from $(typemin(Int32)) to $(typemax(Int32)), an int32 (got $seed)"))
    Int64(seed)
end

_gemini_service_tier(::Nothing, ::Tuple, ::String)::Nothing = nothing
function _gemini_service_tier(tier::String, tiers::Tuple, api::String)::String
    tier in tiers || throw(ArgumentError("$api takes service_tier $(join(repr.(tiers), ", ", " or ")) " *
        "(got $(repr(tier)))" * (tier == "deferred" ?
        ": the Gemini API answers \"deferred\" with HTTP 400, a tier \"allowed on the Gemini Enterprise Agent " *
        "Platform\"" : "")))
    tier
end

# An image's detail level as Gemini's per-part media resolution — a generateContent part's
# `mediaResolution` (https://ai.google.dev/gemini-api/docs/generate-content/media-resolution), an
# Interactions content item's `resolution` (https://ai.google.dev/gemini-api/docs/media-resolution):
# "low" and "high" are its low and high levels (low took a 1,089-token image to 256 tokens on
# gemini-3.8-flash, observed 2026-10-22); "auto", the model's default, sets none; "original", an
# image at its own dimensions on OpenAI, has no Gemini level.
function _gemini_detail_level(detail)::Union{String,Nothing}
    (isnothing(detail) || detail == "auto") && return nothing
    detail == "original" && throw(ArgumentError(
        "Gemini has no media resolution for an image at its own dimensions (detail=\"original\", an " *
        "OpenAI level): use detail=\"high\", \"low\" or \"auto\""))
    detail in ("low", "high") || throw(ArgumentError(
        "an image's detail level on Gemini is \"auto\", \"low\" or \"high\" (got $(repr(detail)))"))
    String(detail)
end

# Which models take one: "Setting media_resolution on individual Part objects is supported on
# Gemini 4 and Gemini 3 models." (generateContent page) and "Setting the resolution on individual
# content items is supported on Gemini 4 and Gemini 3 models." (Interactions page), both pages
# last updated 2026-10-13. That the earlier models still served — the Gemini 2.5 family and
# Gemma 4 — take no per-part level is inferred from the pages naming only Gemini 4 and Gemini 3
# (not probed on them): they refuse a level, by exact id, and any other id is left to the server.
const _GEMINI_NO_PART_RESOLUTION = ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                                    "gemini-2.5-flash-image", "gemma-4-31b-it", "gemma-4-26b-a4b-it")

function _gemini_part_resolution(detail, model::String)::Union{String,Nothing}
    level = _gemini_detail_level(detail)
    isnothing(level) || !(model in _GEMINI_NO_PART_RESOLUTION) || throw(ArgumentError(
        "$model takes no per-image media resolution: Google documents one per part for Gemini 4 and " *
        "Gemini 3 models only; unset the image's detail"))
    level
end

# An attachment as a generateContent part (Part.inlineData, a Blob,
# https://ai.google.dev/api/caching#Blob): its bytes inline, base64, under the MIME type its file
# signature gave (images image/png, image/jpeg, image/gif or image/webp; clips audio/wav or
# audio/mp3; documents application/pdf). The image-edit request builds its input images with it.
_gemini_mime(a::Union{ImageAttachment,DocumentAttachment})::String = a.mime
_gemini_mime(a::AudioAttachment)::String = "audio/" * a.format
_gemini_inline_data(a::Attachment)::Dict{Symbol,Any} =
    Dict{Symbol,Any}(:inlineData => Dict{Symbol,Any}(:mimeType => _gemini_mime(a), :data => base64encode(a.data)))

_gemini_media_part(a::Attachment)::Dict{Symbol,Any} = _gemini_inline_data(a)
function _gemini_media_part(a::ImageAttachment)::Dict{Symbol,Any}
    part = _gemini_inline_data(a)
    level = _gemini_detail_level(a.detail)
    isnothing(level) || (part[:mediaResolution] = Dict{Symbol,Any}(:level => "MEDIA_RESOLUTION_" * uppercase(level)))
    part
end

# A user turn's parts: its media, then its text, as Google advises for an image with a prompt
# ("place the text prompt after the image part"). A message without media is its text alone.
function _gemini_user_parts(m::Message)::Vector{Dict{Symbol,Any}}
    text = Dict{Symbol,Any}(:text => something(m.content, ""))
    attachments = m.attachments
    isnothing(attachments) && return Dict{Symbol,Any}[text]
    parts = Dict{Symbol,Any}[_gemini_media_part(a) for a in attachments]
    isnothing(m.content) || isempty(m.content) || push!(parts, text)
    parts
end

# Thinking levels per model, by exact id, transcribed from the "Controlling thinking" table
# (https://ai.google.dev/gemini-api/docs/thinking, 2026-10-22). Google's ids carry no dated
# snapshots, and distinct models share prefixes (gemini-3.8-flash-tts, gemini-3.8-live), so a
# row never extends to a longer id: an unlisted id gets the generic check and the server's
# answer. The -latest aliases carry the levels of the models that served them on 2026-10-22;
# gemini-3.7-flash and gemini-3.5-flash, deprecated on 2026-10-08, those of the models Google
# routes them to.
const _GEMINI_THINKING_LEVELS = Dict{String,Vector{String}}(
    "gemini-4-argon"               => ["low", "medium", "high"],
    "gemini-3.8-flash"             => ["low", "medium", "high"],
    "gemini-3.6-flash"             => ["minimal", "low", "medium", "high"],
    "gemini-3.5-flash-lite"        => ["minimal", "low", "medium", "high"],
    "gemini-3.1-pro-preview"       => ["low", "medium", "high"],
    "gemini-3.1-flash-lite-image"  => ["minimal", "high"],
    "gemini-3-flash-preview"       => ["minimal", "low", "medium", "high"],
    "gemini-2.5-pro"               => ["low", "medium", "high"],
    "gemini-2.5-flash"             => ["low", "medium", "high"],
    "gemini-2.5-flash-lite"        => ["low", "medium", "high"],
    "gemini-robotics-er-2-preview" => ["minimal", "low", "medium", "high"],
    "gemini-pro-latest"            => ["low", "medium", "high"],               # served by gemini-3.1-pro-preview
    "gemini-flash-latest"          => ["low", "medium", "high"],               # served by gemini-3.8-flash
    "gemini-flash-lite-latest"     => ["minimal", "low", "medium", "high"],    # served by gemini-3.5-flash-lite
    "gemini-3.7-flash"             => ["low", "medium", "high"],               # routed to gemini-3.8-flash
    "gemini-3.5-flash"             => ["minimal", "low", "medium", "high"])    # routed to gemini-3.6-flash

_gemini_thinking_levels(model::AbstractString)::Union{Vector{String},Nothing} =
    get(_GEMINI_THINKING_LEVELS, model, nothing)

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
    :safety_identifier, :seed, :store, :service_tier, :_cumulative_cost)
const _GEMINI_CHAT_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Chat), _GEMINI_CHAT_MAPPED_FIELDS))

function _gemini_validate_chat(chat::Chat)
    _gemini_validate_sampling(chat.model, chat.temperature, chat.top_p)
    isnothing(chat.n) || chat.n == 1 || throw(ArgumentError("Native Gemini Chat returns one candidate; n must be 1"))
    _gemini_refuse_cache_options(chat.prompt_cache_options)
    fields = Symbol[f for f in _GEMINI_CHAT_UNMAPPED_FIELDS if !isnothing(getfield(chat, f))]
    isempty(fields) || throw(ArgumentError(
        "Native Gemini Chat does not support field(s) $(join(fields, ", ")); use a supported option or GEMINIOpenAIServiceEndpoint"))
    for m in chat.messages, a in something(m.attachments, Attachment[])
        _gemini_check_part(a, chat.model)
    end
    nothing
end

# An image's detail level is one the model takes (`_gemini_part_resolution`); other media carry none.
_gemini_check_part(::Attachment, ::String)::Nothing = nothing
_gemini_check_part(a::ImageAttachment, model::String)::Nothing = (_gemini_part_resolution(a.detail, model); nothing)

# Request label values (GenerateContentRequest.labels in the Gemini API discovery
# document): at most 63 characters (Unicode code points), only lowercase letters,
# numeric characters, underscores, and dashes; international characters are allowed.
const _GEMINI_LABEL_VALUE = r"\A[\p{Ll}\p{Lo}\p{N}_-]{0,63}\z"

function _gemini_safety_label(id::String, api::String="Native Gemini")::String
    occursin(_GEMINI_LABEL_VALUE, id) || throw(ArgumentError(
        "$api sends safety_identifier as a request label; label values allow at most " *
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
    _gemini_validate_last_turn(chat.model, contents)
    isnothing(sysinstr) || (body[:systemInstruction] = Dict(:parts => [Dict(:text => sysinstr)]))
    body[:contents] = contents
    isnothing(chat.tools) ||
        (body[:tools] = [Dict(:functionDeclarations => [_gemini_tool(t) for t in chat.tools])])
    (tc = _gemini_tool_config(chat)) === nothing || (body[:toolConfig] = tc)
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
    isnothing(chat.seed)        || (gen[:seed] = _gemini_seed(chat.seed))
    if (effort = chat.reasoning_effort) !== nothing
        gen[:thinkingConfig] = Dict(:thinkingLevel => uppercase(_gemini_thinking_level(chat.model, effort; native=true)))
    end
    (fmt = _gemini_response_format(chat.response_format)) === nothing || (gen[:responseFormat] = fmt)
    isempty(gen) || (body[:generationConfig] = gen)
    # Request labels carry an aggregator's end-user id under Google's documented key.
    isnothing(chat.safety_identifier) ||
        (body[:labels] = Dict(:safety_identifier => _gemini_safety_label(chat.safety_identifier)))
    # `store` sets whether Google logs the request, over the project's logging setting.
    isnothing(chat.store) || (body[:store] = chat.store)
    tier = _gemini_service_tier(chat.service_tier, _GEMINI_SERVICE_TIERS, "Gemini generateContent")
    isnothing(tier) || (body[:serviceTier] = tier)
    # NB: `stream` is expressed in the URL method (get_url), never in the body.
    JSON.json(body)
end

# GEMINIOpenAIServiceEndpoint serves the same models on the OpenAI wire, so the same models take
# the sampling parameters: Google's page on Gemini 3.6 says the models ignore them, without
# naming an endpoint, and that they are ignored on this wire too is inferred from that page,
# not measured on it. It takes the service tiers Google's compatibility page names —
# `service_tier="priority"` or `"flex"`, and `"standard"`, which `service_tier` "defaults to"
# (https://ai.google.dev/gemini-api/docs/openai#flex-and-priority-inference) — and no other
# OpenAI tier. Its other refusals are the server's: it answers HTTP 400 to
# `prompt_cache_options` ("Unknown name") and to a request ending with an assistant turn
# ("Requests ending with a model turn are not supported."), observed on gemini-3.8-flash on
# 2026-10-22.
function encode_request(::Type{GEMINIOpenAIServiceEndpoint}, chat::Chat)::String
    _gemini_validate_sampling(chat.model, chat.temperature, chat.top_p)
    _gemini_service_tier(chat.service_tier, _GEMINI_SERVICE_TIERS, "Gemini's OpenAI-compatible endpoint")
    _compat_audio_estimate(chat)
    invoke(encode_request, Tuple{OpenAIWireEndpointSpec,Chat}, GEMINIOpenAIServiceEndpoint, chat)
end

# Gemini 3 gives a function call's thought signature on this wire in the call's
# `extra_content.google.thought_signature` and validates it when the call comes back: "Required and
# Validated" (https://ai.google.dev/gemini-api/docs/thought-signatures, "Signatures for OpenAI
# compatibility"). A follow-up whose call lacked it was answered HTTP 400 "Function call is missing
# a thought_signature in functionCall parts" (gemini-3.8-flash, 26 October 2026). The OpenAI-wire
# decoders keep it on the call's `ToolCall`, and this endpoint sends each call back with the
# signature it came with; of parallel calls, only the first has one.
function _google_signature(call::AbstractDict)::Union{String,Nothing}
    extra = get(call, "extra_content", nothing)
    google = extra isa AbstractDict ? get(extra, "google", nothing) : nothing
    sig = google isa AbstractDict ? get(google, "thought_signature", nothing) : nothing
    sig isa AbstractString && !isempty(sig) ? String(sig) : nothing
end

_wire_messages(::Type{GEMINIOpenAIServiceEndpoint}, messages::Conversation) =
    any(_signed_calls, messages) ? Any[_signed_calls(m) ? _signed_turn(m) : m for m in messages] : messages

_signed_calls(m::Message)::Bool = !isnothing(m.tool_calls) && any(tc -> !isnothing(tc.thought_signature), m.tool_calls)

function _signed_turn(m::Message)
    d = JSON.lower(m)
    d[:tool_calls] = Any[isnothing(tc.thought_signature) ? tc :
                         merge(JSON.lower(tc), Dict(:extra_content =>
                             Dict(:google => Dict(:thought_signature => tc.thought_signature))))
                         for tc in something(m.tool_calls)]   # a turn with signed calls (`_signed_calls`)
    d
end

# The compatible endpoint answers a sound clip but reports no audio token count: a WAV sent to
# gemini-3.1-flash-lite came back with 54 prompt tokens and no token details (observed 2026-10-23),
# where the native endpoint counts 38 of them as audio. So the clip is priced at the model's text
# rate, a lower bound on a model whose table states an audio rate of its own (each is higher than
# its text rate): a request that sends one to such a model says so, once per model.
function _compat_audio_estimate(chat::Chat)::Nothing
    _gemini_own_audio_rate(chat.model) && any(_holds_clip, chat.messages) || return nothing
    @warn "Gemini's OpenAI-compatible endpoint reports no audio token count, so this model's audio input is " *
          "estimated at its text rate, a lower bound of its audio rate; send audio through GEMINIServiceEndpoint " *
          "for an exact estimate" model = chat.model _id = Symbol("unilm_compat_audio_", chat.model) maxlog = 1
    nothing
end
_holds_clip(m::Message)::Bool = _holds_clip(m.attachments)
_holds_clip(::Nothing)::Bool = false
_holds_clip(attachments::Vector{Attachment})::Bool = any(_is_clip, attachments)
_is_clip(::AudioAttachment) = true
_is_clip(::Attachment) = false

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
# - user   → {role:"user", parts:[{inlineData}..., {text}]} (`_gemini_user_parts`)
# - assistant → {role:"model", parts:[{text}?, {functionCall,thoughtSignature?}...]};
#   a turn with no part to send (a refusal, a turn spent on thinking) is left out,
#   because an empty-parts model turn breaks the follow-up request
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
            parts = _gemini_model_parts(m, tool_names)
            isempty(parts) && continue
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "model", :parts => parts))
        else  # RoleUser
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "user", :parts => _gemini_user_parts(m)))
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

# A tool marked strict asks for arguments that conform to its schema. A declaration has no such
# flag; the request's function-calling mode has: VALIDATED "ensures function schema adherence"
# while the model may still answer in text, as under AUTO, and ANY, which requires a call, "ensures
# function schema adherence" too
# (https://ai.google.dev/gemini-api/docs/generate-content/function-calling). So a strict tool
# makes an unset or "auto" tool_choice VALIDATED (gemini-3.8-flash answered a strict tool under it
# with a function call on 2026-10-25); "required", a named function and "none" keep their modes.
function _gemini_tool_config(chat::Chat)
    tc = chat.tool_choice
    any(t -> t.func.strict === true, something(chat.tools, Tool[])) && (isnothing(tc) || tc == "auto") &&
        return Dict(:functionCallingConfig => Dict(:mode => "VALIDATED"))
    isnothing(tc) ? nothing : _gemini_tool_config(tc)
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
    catch e
        e isa InterruptException && rethrow()
        Dict{String,Any}("result" => s)
    end
end

# ─── Response decoding (Gemini generateContent → neutral Message) ────────────

# The Gemini API's FunctionCall.id is OPTIONAL. Id-less calls get a unique
# synthetic positional id so tool results correlate within the turn; the
# prefix is reserved and such ids are OMITTED on re-encode (Gemini correlates
# positionally when ids are absent — echoing a fabricated id would be wrong).
_is_synthetic_call_id(id::AbstractString) = startswith(id, "unilm_call_")

# Gemini finishReason (the FinishReason enum, https://ai.google.dev/api/generate-content)
# → neutral finish_reason; the decoder and the stream handler share this one rule.
# STOP is the only normal completion, and a STOP turn that carries function calls is a
# tool-call turn (Gemini reports STOP for those). MAX_TOKENS → "length"; the stops the enum
# describes as a filter or a policy refusal → "content_filter": the content filters, the image
# ones included (safety, prohibited content, recitation), content "flagged for using an
# unsupported language", a request "filtered by an escalation rule", and a stop "because the
# user account is limited or disabled due to Prohibited Use Policy (PUP) violations". Every
# other value of this open enum (MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, IMAGE_OTHER,
# NO_IMAGE, values added later) passes through lowercased, never as "stop", and calls under any
# non-STOP reason keep that reason: a failed or filtered turn reads neither as a completed turn
# nor as calls to dispatch.
const _GEMINI_FILTER_REASONS = ("SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
                                "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "ESCALATION",
                                "PUP_LIMITED_DISABLED")

_gemini_finish_reason(fr::AbstractString, has_calls::Bool)::String =
    fr == "STOP"                 ? (has_calls ? TOOL_CALLS : STOP) :
    fr == "MAX_TOKENS"           ? "length" :
    fr in _GEMINI_FILTER_REASONS ? CONTENT_FILTER : lowercase(fr)

# Gemini usageMetadata → neutral TokenUsage (UsageMetadata, https://ai.google.dev/api/generate-content):
# promptTokenCount already counts a cached prefix (cachedContentTokenCount is a subset of it), and
# totalTokenCount is prompt + tool-use prompt + thoughts + candidates. The tool-use prompt
# (toolUsePromptTokenCount) is input the model read, so it is billed as prompt; thoughts bill at
# the output rate → completion = candidates + thoughts. The prompt's audio is the AUDIO count of
# promptTokensDetails and toolUsePromptTokensDetails. The billed tier is the reply's
# x-gemini-service-tier header (`header_tier`), the only place Gemini 4 Argon names it (observed
# 2026-10-22), else usageMetadata's serviceTier: Google names the header as the sign of a
# downgraded Priority request, and "Downgraded requests are billed at the standard rate"
# (https://ai.google.dev/gemini-api/docs/priority-inference).
function _gemini_usage(u, header_tier::Union{String,Nothing}=nothing)::Union{TokenUsage,Nothing}
    u isa AbstractDict || return nothing
    _i(x) = x isa Integer ? Int(x) : 0
    prompt   = _i(get(u, "promptTokenCount", 0)) + _i(get(u, "toolUsePromptTokenCount", 0))
    cand     = _i(get(u, "candidatesTokenCount", 0))
    thoughts = _i(get(u, "thoughtsTokenCount", 0))
    cached   = _i(get(u, "cachedContentTokenCount", 0))
    total    = _i(get(u, "totalTokenCount", prompt + cand + thoughts))
    audio    = _modality_tokens(get(u, "promptTokensDetails", nothing), "AUDIO", "tokenCount") +
               _modality_tokens(get(u, "toolUsePromptTokensDetails", nothing), "AUDIO", "tokenCount")
    TokenUsage(prompt_tokens = prompt, completion_tokens = cand + thoughts,
        total_tokens = total, cached_tokens = cached, reasoning_tokens = thoughts, prompt_audio_tokens = audio,
        service_tier = something(header_tier, _wire_string(u, "serviceTier"), Some(nothing)))
end

# The tokens a list of per-modality counts gives `modality`: generateContent's ModalityTokenCount
# entries ({"modality": "AUDIO", "tokenCount": n}) or the Interactions API's ModalityTokens
# ({"modality": "audio", "tokens": n}).
function _modality_tokens(counts, modality::String, key::String)::Int
    n = 0
    for c in (counts isa AbstractVector ? counts : ())
        c isa AbstractDict && get(c, "modality", nothing) == modality || continue
        k = get(c, key, 0)
        k isa Integer && (n += Int(k))
    end
    n
end

# The tier a reply's head names (https://ai.google.dev/gemini-api/docs/priority-inference: "monitor
# the x-gemini-service-tier header ... to detect if requests are being frequently downgraded").
function _gemini_header_tier(resp::HTTP.Response)::Union{String,Nothing}
    v = HTTP.header(resp, "x-gemini-service-tier", "")
    isempty(v) ? nothing : String(v)
end

# A usage written back as the usageMetadata `_gemini_usage` reads it as, its audio count included.
function _gemini_usage_metadata(u::TokenUsage)::Dict{String,Any}
    d = Dict{String,Any}("promptTokenCount" => u.prompt_tokens, "cachedContentTokenCount" => u.cached_tokens,
                         "candidatesTokenCount" => u.completion_tokens - u.reasoning_tokens,
                         "thoughtsTokenCount" => u.reasoning_tokens, "totalTokenCount" => u.total_tokens,
                         "promptTokensDetails" => Any[Dict{String,Any}("modality" => "AUDIO",
                                                                       "tokenCount" => u.prompt_audio_tokens)])
    isnothing(u.service_tier) || (d["serviceTier"] = u.service_tier)
    d
end

# The refusal of a filtered candidate: its finishMessage, Google's own account of why generation
# stopped, which is "Optional", else the stop named. A language flag, an escalation rule and a
# Prohibited Use Policy stop are not the safety filter, so each is named with the FinishReason
# enum's definition of it (https://ai.google.dev/api/generate-content#FinishReason).
const _GEMINI_STOP_DEFINITIONS = Dict{String,String}(
    "LANGUAGE" => "The response candidate content was flagged for using an unsupported language.",
    "ESCALATION" => "Request was filtered by an escalation rule.",
    "PUP_LIMITED_DISABLED" => "Token generation stopped because the user account is limited or disabled due to " *
                              "Prohibited Use Policy (PUP) violations.")
function _gemini_refusal(cand)::String
    cand isa AbstractDict || return "Model response blocked by safety filter."
    message = _wire_string(cand, "finishMessage")
    isnothing(message) || return message
    stop = _wire_string(cand, "finishReason")
    definition = isnothing(stop) ? nothing : get(_GEMINI_STOP_DEFINITIONS, stop, nothing)
    isnothing(definition) ? "Model response blocked by safety filter." : "$stop: $definition"
end

# The reply names the model that served it in modelVersion: a deprecated id Google routes to its
# successor answers as that successor, which is the model priced. A reply that ends at the
# per-request cap of a long generation carries the token that continues it (`continuation_token`).
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
    # A candidate without a finishReason reports none, so none is invented for it.
    finish = fr_raw isa AbstractString ? _gemini_finish_reason(fr_raw, !isempty(tool_calls)) : nothing
    usage = _gemini_usage(get(data, "usageMetadata", nothing), _gemini_header_tier(resp))
    txt = takestring!(text)
    msg = if !isempty(tool_calls)
        Message(role=RoleAssistant, content=(isempty(txt) ? nothing : txt),
                tool_calls=tool_calls, finish_reason=finish, provider_content=pc)
    elseif finish == CONTENT_FILTER && isempty(txt)
        Message(role=RoleAssistant, refusal_message=_gemini_refusal(cand), finish_reason=finish, provider_content=pc)
    else
        # A well-formed candidate that yields no text is a real, legitimate turn:
        # thinking models routinely spend the whole completion budget on thought
        # tokens (finishReason MAX_TOKENS, zero visible parts). Report the empty
        # turn the provider actually sent — substituting prose here would inject
        # content nobody generated into the reply and into the next request.
        Message(role=RoleAssistant, content=txt, finish_reason=finish, provider_content=pc)
    end
    (; message=msg, usage, model=_wire_string(data, "modelVersion"),
       continuation_token=_wire_string(cand, "continuationToken"))
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
    # Every chunk names the serving model, as the unstreamed reply does.
    model = _wire_string(ev, "modelVersion")
    isnothing(model) || (state.model = model)
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
            fr isa AbstractString &&
                (state.finish_reason = _gemini_finish_reason(fr, !isempty(state.tool_calls)))
            # The latest token wins: a long stream also sends checkpoint tokens on its way.
            token = _wire_string(cand, "continuationToken")
            isnothing(token) || (state.continuation_token = token)
            # A filter stop with no answer text is a refusal, recorded once: the chunks
            # after the filtered one (trailing usage, a repeated candidate) add nothing.
            if state.finish_reason == CONTENT_FILTER && position(state.content) == 0 &&
               position(state.refusal) == 0
                print(state.refusal, _gemini_refusal(cand))
            end
        end
    end
    u = get(ev, "usageMetadata", nothing)
    u isa AbstractDict && (state.usage = _gemini_usage(u, state.service_tier))
    :continue
end

# A stream's tier is also in its response head, read before the first chunk (`state.service_tier`),
# so the streamed usage names the tier the reply not streamed names.
_stream_head!(::Union{Type{GEMINIServiceEndpoint},Type{GEMINIOpenAIServiceEndpoint}}, state::StreamState,
              head::HTTP.Response)::Nothing = (state.service_tier = _gemini_header_tier(head); nothing)

# The compatible endpoint's reply reads as the OpenAI wire's, its billed tier header first, as on
# the native wire: the response head's x-gemini-service-tier, else the reply's service_tier — a
# stream's head outranks every chunk's (`_chunk_tier!`). Neither named a tier in replies to flex
# and priority requests on 2026-10-22, streamed or not, so such a reply reports none.
function decode_response(::Type{GEMINIOpenAIServiceEndpoint}, resp::HTTP.Response)
    d = extract_message(resp)
    tier = _gemini_header_tier(resp)
    usage = isnothing(tier) || isnothing(d.usage) ? d.usage : TokenUsage(; _usage_fields(d.usage)..., service_tier=tier)
    merge(d, (; usage))
end
_chunk_tier!(::Type{GEMINIOpenAIServiceEndpoint}, state::StreamState, tier::String)::Nothing =
    (isnothing(state.service_tier) && (state.service_tier = tier); nothing)

# A reply that names no tier was billed at the one its request asked for, so the result records
# that one: "flex" prices at the Flex rates, and "priority" at the Priority rates — an upper bound,
# since Google bills a downgraded Priority request at Standard and this endpoint does not report
# the downgrade. A tier the reply names is kept.
_result_usage(::Type{GEMINIOpenAIServiceEndpoint}, chat::Chat, u::TokenUsage)::TokenUsage =
    isnothing(u.service_tier) ? TokenUsage(; _usage_fields(u)..., service_tier=chat.service_tier) : u

# ─── Long decode: one turn in several requests ───────────────────────────────
# Gemini 4 Argon generates past one request's output cap in hops: a reply that reaches the cap
# with tokens left ends with finishReason CONTINUATION and a continuationToken, and the next hop
# is "the same request again with the token" — the same contents, system instruction, tools and
# generation config, with no partial output added — sending "only the latest token"
# (https://ai.google.dev/gemini-api/docs/generate-content/long-decode). A reply that stops at
# MAX_TOKENS carries a token too, but continuing it needs a larger max_output_tokens: that turn
# ends. "One hop returns up to 65,536 output tokens. Non-streaming REST requests return up to
# 32,768." (the same page), which sizes the requests a turn may take (`_request_cap`).
# Continuation is exercised against a local mock; a live CONTINUATION reply was not observed.
_request_output_cap(::Type{GEMINIServiceEndpoint}, stream::Bool)::Int = stream ? 65_536 : 32_768

function _continuation_body(::Type{GEMINIServiceEndpoint}, body::AbstractString,
                            finish::Union{String,Nothing}, token::Union{String,Nothing})::Union{String,Nothing}
    (finish == "continuation" && !isnothing(token)) || return nothing
    request = JSON.parse(body; dicttype=Dict{String,Any})
    request["continuationToken"] = token
    JSON.json(request)
end

function _gemini_first_candidate(reply)::Union{Dict{String,Any},Nothing}
    candidates = reply isa AbstractDict ? get(reply, "candidates", nothing) : nothing
    candidates isa AbstractVector && !isempty(candidates) || return nothing
    c = first(candidates)
    c isa Dict{String,Any} ? c : nothing
end

function _gemini_candidate_parts(cand::AbstractDict)::Vector{Any}
    content = get(cand, "content", nothing)
    parts = content isa AbstractDict ? get(content, "parts", nothing) : nothing
    parts isa AbstractVector ? Vector{Any}(parts) : Any[]
end

_gemini_reply_usage(d::AbstractDict, resp::HTTP.Response) =
    _gemini_usage(get(d, "usageMetadata", nothing), _gemini_header_tier(resp))

# One reply for a turn sent in hops, joined as Google's REST loop joins it — "Combine the parts
# from all hops into one model turn" — and billed as it says: "Add each field across hops". The
# later reply's candidate gives the finish reason, refusal text and next token, the latest reply
# names the served model, and the usage is the hops' summed as a streamed turn's is
# (`_add_usage`). A later reply without a candidate is its own answer: decoding it reports why.
function _join_hops(::Type{GEMINIServiceEndpoint}, earlier::HTTP.Response, later::HTTP.Response)::HTTP.Response
    parsed_a = JSON.parse(earlier.body; dicttype=Dict{String,Any})
    parsed_b = JSON.parse(later.body; dicttype=Dict{String,Any})
    first_a, first_b = _gemini_first_candidate(parsed_a), _gemini_first_candidate(parsed_b)
    (isnothing(first_a) || isnothing(first_b)) && return later
    a, b = parsed_a::Dict{String,Any}, parsed_b::Dict{String,Any}      # each holds a candidate
    candidate = copy(first_b)
    candidate["content"] = Dict{String,Any}("role" => "model",
        "parts" => vcat(_gemini_candidate_parts(first_a), _gemini_candidate_parts(first_b)))
    joined = copy(b)
    joined["candidates"] = Any[candidate]
    usage = _add_usage(_gemini_reply_usage(a, earlier), _gemini_reply_usage(b, later))
    isnothing(usage) ? delete!(joined, "usageMetadata") : (joined["usageMetadata"] = _gemini_usage_metadata(usage))
    model = something(_wire_string(b, "modelVersion"), _wire_string(a, "modelVersion"), Some(nothing))
    isnothing(model) || (joined["modelVersion"] = model)
    HTTP.Response(200, later.headers, Vector{UInt8}(JSON.json(joined)))
end

# ─── Long thinking: longer default bounds ─────────────────────────────────────
# Gemini 4 Argon thinks for minutes on hard problems: the server ends a non-streaming call after
# 10 minutes and a stream after 20, and Google advises "Set your HTTP read timeout to 20 minutes.
# A 3-minute or 5-minute timeout stops the request early."
# (https://ai.google.dev/gemini-api/docs/generate-content/long-decode). A Gemini 4 call (a model
# id starting `gemini-4-`) whose caller set no RequestConfig in any channel takes those 20
# minutes per attempt, between stream bytes and in all.
const _GEMINI_LONG_THINKING_CONFIG = RequestConfig(request_timeout=1200.0, stream_idle_timeout=1200.0,
                                                   total_deadline=1200.0)
_model_default_config(::Union{Type{GEMINIServiceEndpoint},Type{GEMINIOpenAIServiceEndpoint}},
                      model::AbstractString)::Union{RequestConfig,Nothing} =
    startswith(model, "gemini-4-") ? _GEMINI_LONG_THINKING_CONFIG : nothing

# ─── Cost ─────────────────────────────────────────────────────────────────────

# A billed tier's prices as multiples of a model's standard rates, from the Flex and Priority
# tables of https://ai.google.dev/gemini-api/docs/pricing (2026-10-22): input and output bill at
# half the standard rates on Flex and at 1.8 times on Priority (Gemini 3.8 Flash $1.35/$6.75 for
# $0.75/$3.75, Gemini 4 Argon $3.60/$18.00 for $2.00/$10.00), and so does a cache hit, except on
# the models whose table lists another cache-hit price: `cache` holds those, as multiples of the
# model's standard cache-hit rate.
const _GEMINI_TIER_PRICES = Dict{String,@NamedTuple{multiple::Float64, cache::Dict{String,Float64}}}(
    "flex" => (multiple=0.5, cache=Dict{String,Float64}(
        # "Same as Standard", or the standard figure
        "gemini-3.1-pro-preview" => 1.0, "gemini-3.1-pro-preview-customtools" => 1.0,
        "gemini-3-flash-preview" => 1.0,
        "gemini-2.5-pro" => 1.0, "gemini-2.5-flash" => 1.0, "gemini-2.5-flash-lite" => 1.0,
        "gemini-3.5-flash-lite" => 0.02 / 0.03)),          # $0.02 for the standard $0.03
    "priority" => (multiple=1.8, cache=Dict{String,Float64}(
        "gemini-3.5-flash-lite" => 0.05 / 0.03)))          # $0.05 for the standard $0.03

# The multiples of the standard rates that the tier a reply reports costs: of the input and
# output rates, and of the cache-hit rate. "standard" (or "unspecified", its default) or no tier
# is the standard rates; a tier the page gives no price for is estimated at them, and says so
# once per model and tier.
function _gemini_tier_multiples(model::String, tier::Union{String,Nothing})::NTuple{2,Float64}
    (isnothing(tier) || tier in ("standard", "unspecified")) && return (1.0, 1.0)
    prices = get(_GEMINI_TIER_PRICES, tier, nothing)
    if isnothing(prices)
        @warn "Gemini's pricing page lists no price for this service tier; its cost is estimated at the " *
              "standard rates" model tier _id = Symbol("unilm_unpriced_tier_", model, "_", tier) maxlog = 1
        return (1.0, 1.0)
    end
    (prices.multiple, get(prices.cache, model, prices.multiple))
end

# The audio input rate of each model on https://ai.google.dev/gemini-api/docs/pricing (2026-10-22),
# scaled on Flex and Priority as the table scales its text rates. The page spells input modalities
# out wherever their rates differ ("$0.25 (text / image / video) $0.50 (audio)" on 3.1 Flash-Lite)
# and where audio is excluded (an image model's "(text/image/video)"), so an input price that names
# no modality, or names audio among the others, is every modality's. A model with an audio price of
# its own maps to it, in USD per token at the standard tier; a model with its one input price maps
# to `nothing`: audio bills at the price row's input rate, so a `pricing=` row, or a price change
# such as Robotics-ER 2 Preview's on 2027-01-01, moves both. Audio on a model without an entry — an
# image model, an id the page does not list — leaves the call unpriced, as a model without a price
# row is.
const _GEMINI_AUDIO_INPUT_RATES = Dict{String,Union{Float64,Nothing}}(
    "gemini-4-argon"                     => nothing,      # its one input price, naming no modality
    "gemini-3.8-flash"                   => nothing,      # likewise
    "gemini-3.7-flash"                   => nothing,      # routed to and priced as gemini-3.8-flash
    "gemini-3.6-flash"                   => nothing,      # its one input price, naming no modality
    "gemini-3.5-flash"                   => nothing,      # routed to and priced as gemini-3.6-flash
    "gemini-3.5-flash-lite"              => nothing,      # "(text / image / video / audio)"
    "gemini-3.1-pro-preview"             => nothing,      # its one input price, naming no modality
    "gemini-3.1-pro-preview-customtools" => nothing,      # priced as 3.1 Pro Preview
    "gemini-2.5-pro"                     => nothing,      # its one input price, naming no modality
    "gemini-robotics-er-2-preview"       => nothing,      # "(text / image / video / audio)"
    "gemini-3.1-flash-lite"              => 0.50 / 1_000_000,   # shuts down 2027-05-07
    "gemini-3-flash-preview"             => 1.00 / 1_000_000,
    "gemini-2.5-flash"                   => 1.00 / 1_000_000,
    "gemini-2.5-flash-lite"              => 0.30 / 1_000_000)

# Whether `model`'s table states an audio rate of its own, which may differ from its text rate.
_gemini_own_audio_rate(model::AbstractString)::Bool = !isnothing(get(_GEMINI_AUDIO_INPUT_RATES, model, nothing))

# The audio rate a usage's audio input bills at, `rates` being the model's price row, or `nothing`
# where none applies: the model has no entry above; the reply also counts audio output, which no
# table here prices; or the call also read the cache on a model with an audio price of its own,
# whose cached-audio price the page lists apart from the cached-text one while a usage does not
# say how many cached tokens are audio. A one-price model's cache price names no modality either,
# so its cache hits bill at the row's cached rate whatever they hold.
function _gemini_audio_rate(model::String, u::TokenUsage, rates::PriceRow)::Union{Float64,Nothing}
    (u.completion_audio_tokens > 0 || !haskey(_GEMINI_AUDIO_INPUT_RATES, model)) && return nothing
    own = _GEMINI_AUDIO_INPUT_RATES[model]
    isnothing(own) && return rates.input
    u.cached_tokens > 0 ? nothing : own
end

# The models whose table bills a prompt over 200K tokens at long-context rates, the whole request:
# twice the input and cache-hit rates and 1.5 times the output rate, on every tier (pricing page,
# 2026-10-22). 3.1 Pro Preview lists input "$2.00, prompts <= 200k tokens $4.00, prompts > 200k
# tokens", output $12.00 / $18.00, cache hits $0.20 / $0.40, and Priority $7.20 / $32.40 / $0.72;
# its custom-tools endpoint is priced "Same as Gemini 3.1 Pro Preview"; 2.5 Pro lists $1.25 /
# $2.50, $10.00 / $15.00, $0.125 / $0.25. A -latest alias is priced as the model that served it.
const _GEMINI_LONG_CONTEXT_MODELS = ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools", "gemini-2.5-pro")
const _GEMINI_SHORT_CONTEXT = 200_000

# The multiples of the input and cache-hit rates, and of the output rate, that a prompt of
# `prompt_tokens` costs on `model`. A turn Gemini continues in several requests is priced from
# its summed usage, so its prompt is the requests' prompts summed, not one request's: that is
# right only while no model that continues has long-context rates — today only Gemini 4 Argon
# continues, and its table has none.
_gemini_context_multiples(model::String, prompt_tokens::Int)::NTuple{2,Float64} =
    prompt_tokens > _GEMINI_SHORT_CONTEXT && model in _GEMINI_LONG_CONTEXT_MODELS ? (2.0, 1.5) : (1.0, 1.0)

# What Gemini bills a call, on the native and the OpenAI-compatible endpoint alike
# (https://ai.google.dev/gemini-api/docs/pricing): prompts are cached implicitly, so a cache hit
# bills at the context-caching rate and every other prompt token, a tool-use prompt included, at
# the input rate — an audio one at the model's audio rate (`_gemini_audio_rate`), the call
# unpriced where none applies — with no premium for writing the cache; output, thinking
# included, bills at the output rate; a long prompt at its long-context multiples
# (`_gemini_context_multiples`); and each rate at the billed tier's multiple of it
# (`_gemini_tier_multiples`).
function _billed_cost(service::Union{Type{GEMINIServiceEndpoint},GEMINIServiceEndpoint,
                                     Type{GEMINIOpenAIServiceEndpoint},GEMINIOpenAIServiceEndpoint},
                      model::String, u::TokenUsage, rates::PriceRow)::Float64
    audio_rate = _gemini_audio_rate(model, u, rates)
    _counts_audio(u) && isnothing(audio_rate) && return _unpriced_audio(service, model)
    audio = min(u.prompt_audio_tokens, u.prompt_tokens)
    cached = min(u.cached_tokens, u.prompt_tokens)
    rate, cache = _gemini_tier_multiples(model, u.service_tier)
    input, output = _gemini_context_multiples(model, u.prompt_tokens)
    rate * (input * ((u.prompt_tokens - cached - audio) * rates.input + audio * something(audio_rate, 0.0)) +
            output * u.completion_tokens * rates.output) + cache * input * cached * rates.cached_input
end
