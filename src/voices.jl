# ============================================================================
# Custom voices: OpenAI's custom voices and voice consents (/v1/audio/voices, /v1/audio/voice_consents) and
# Gemini's Voices API (/v1beta/voices). A created voice's id is what `speak` takes as `voice` on its service.
# ============================================================================

# How a service manages voices: OpenAI's routes (create and delete; consents) or Gemini's (create, list, get, delete).
abstract type _VoicesWire end
struct _OpenAIVoices <: _VoicesWire end
struct _GeminiVoices <: _VoicesWire end
_voices_wire(_) = _OpenAIVoices()
_voices_wire(::Type{GEMINIServiceEndpoint}) = _GeminiVoices()

_voices_url(service)::String = _api_base_url(service) * "/v1/audio/voices"
_voices_url(::Type{GEMINIServiceEndpoint})::String = GEMINI_NATIVE_BASE * "/voices"
_voice_consents_url(service)::String = _api_base_url(service) * "/v1/audio/voice_consents"

"""
    Voice

A voice [`speak`](@ref) can use, as [`create_voice`](@ref), [`get_voice`](@ref) and
[`list_voices`](@ref) return it:
- `id`: the voice's id, which `speak` takes as `voice` on the service that made it — OpenAI's
  `voice_…`; Gemini's `voice_…` for a stored custom voice, and a prebuilt voice's catalog id (such
  as `"achernar"`). `nothing` for a Gemini voice replicated with `store = false`, which has a `key`.
- `name`: OpenAI's `name`; Gemini's `display_name` (`nothing` when it has none).
- `type`: how it was made: OpenAI's `"audio_sample"` or `"prompt"`; Gemini's `"replicated"`,
  `"prompted"` or `"prebuilt"`.
- `created_at`: OpenAI's creation time, in Unix seconds; Gemini reports none (`nothing`).
- `key`: a Gemini stateless replication key (`voicekey_…`), which `speak` takes as `voice`; else `nothing`.
- `raw`: the voice as the provider sent it — OpenAI's `prompt`; Gemini's `expire_time`, `model`,
  `language_code`, `prompted`, `sample_audio` (a base64 WAV preview of a prompted voice) and `usage`.
"""
@kwdef struct Voice
    id::Union{String,Nothing}
    name::Union{String,Nothing} = nothing
    type::String
    created_at::Union{Int,Nothing} = nothing
    key::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end
# The raw voice can hold a preview of a megabyte or more: shown by its keys.
Base.show(io::IO, v::Voice) = print(io, "Voice(id=", repr(v.id), ", name=", repr(v.name), ", type=", repr(v.type),
    ", created_at=", repr(v.created_at), ", key=", repr(v.key), ", raw=<", length(v.raw), " keys>)")

"""
    VoiceList

A page of [`Voice`](@ref)s from [`list_voices`](@ref): Gemini's custom voices, newest first, then its
prebuilt catalog. `has_more` signals a further page, whose cursor is `raw["next_page_token"]` (pass it
as `after`); `raw` holds the page as Gemini sent it.
"""
@kwdef struct VoiceList
    data::Vector{Voice}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VoiceConsent

An OpenAI voice consent: the recording of a speaker consenting to a custom voice of theirs, which
[`create_voice`](@ref) names by `id` (`consent = "cons_…"`). `name` is its label, `language` the BCP 47
tag of the consent phrase, `created_at` its Unix time; `raw` the object as OpenAI sent it.
"""
@kwdef struct VoiceConsent
    id::String
    name::String
    language::String
    created_at::Int
    raw::Dict{String,Any} = Dict{String,Any}()
end

"A page of [`VoiceConsent`](@ref)s from [`list_voice_consents`](@ref); `has_more` signals a further page, which starts after `raw[\"last_id\"]` (pass it as `after`)."
@kwdef struct VoiceConsentList
    data::Vector{VoiceConsent}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful [`create_voice`](@ref)/[`get_voice`](@ref) result wrapping a [`Voice`](@ref)."
@kwdef struct VoiceSuccess <: LLMRequestResponse; response::Voice; end
"Successful [`list_voices`](@ref) result wrapping a [`VoiceList`](@ref)."
@kwdef struct VoiceListSuccess <: LLMRequestResponse; response::VoiceList; end
"Successful [`create_voice_consent`](@ref)/[`get_voice_consent`](@ref)/[`update_voice_consent`](@ref) result wrapping a [`VoiceConsent`](@ref)."
@kwdef struct VoiceConsentSuccess <: LLMRequestResponse; response::VoiceConsent; end
"Successful [`list_voice_consents`](@ref) result wrapping a [`VoiceConsentList`](@ref)."
@kwdef struct VoiceConsentListSuccess <: LLMRequestResponse; response::VoiceConsentList; end
"""
    VoiceDeleteSuccess

Successful [`delete_voice`](@ref)/[`delete_voice_consent`](@ref) result: the service confirmed the
removal of `id`. `object` is OpenAI's record type (`"audio.voice.deleted"`, or `"audio.voice_consent"` for a
consent); Gemini answers an empty object, so there `id` is the voice requested and `object` is `nothing`.
"""
@kwdef struct VoiceDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; object::Union{String,Nothing} = nothing; end
"Voices API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct VoiceFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Voices API call that produced no usable reply (transport failure, timeout, cancel, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel."
@kwdef struct VoiceCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Replies ──────────────────────────────────────────────────────────────────

_voice_field(d::AbstractDict, key::String, ::Type{T}) where {T} = _audio_field(d, key, T, "voice")
_voice_required(d::AbstractDict, key::String, ::Type{T}) where {T} = _audio_required(d, key, T, "voice")

# OpenAI's VoiceResource requires `object` ("audio.voice"), `id`, `name`, `type` and `created_at`.
function _parse_voice(::_OpenAIVoices, d::AbstractDict)::Voice
    object = _voice_required(d, "object", String)
    object == "audio.voice" || throw(ArgumentError("the reply is a $(repr(object)), not an \"audio.voice\""))
    Voice(id=_voice_required(d, "id", String), name=_voice_required(d, "name", String),
          type=_voice_required(d, "type", String), created_at=_voice_required(d, "created_at", Int),
          raw=Dict{String,Any}(d))
end
# Gemini's Voice requires `type`; `id` is unset for a voice replicated with store false, which has a `key` instead,
# and a voice with neither is none speak could use.
function _parse_voice(::_GeminiVoices, d::AbstractDict)::Voice
    id, key = _voice_field(d, "id", String), _voice_field(d, "key", String)
    isnothing(id) && isnothing(key) && throw(ArgumentError("the Gemini voice has neither an id nor a key"))
    Voice(id=id, name=_voice_field(d, "display_name", String), type=_voice_required(d, "type", String),
          key=key, raw=Dict{String,Any}(d))
end

# Gemini's ListVoices: `voices` and `next_page_token`, each left out when empty (`{}` is an empty last page).
function _voice_page(wire::_GeminiVoices, d::AbstractDict)::VoiceList
    token = _voice_field(d, "next_page_token", String)
    VoiceList(data=Voice[_parse_voice(wire, v) for v in something(_voice_field(d, "voices", AbstractVector), Any[])],
              has_more=!(isnothing(token) || isempty(token)), raw=Dict{String,Any}(d))
end

# OpenAI's VoiceConsentResource requires `object` ("audio.voice_consent"), `id`, `name`, `language` and `created_at`.
function _parse_voice_consent(d::AbstractDict)::VoiceConsent
    object = _audio_required(d, "object", String, "voice consent")
    object == "audio.voice_consent" ||
        throw(ArgumentError("the reply is a $(repr(object)), not an \"audio.voice_consent\""))
    VoiceConsent(id=_audio_required(d, "id", String, "voice consent"), name=_audio_required(d, "name", String, "voice consent"),
                 language=_audio_required(d, "language", String, "voice consent"),
                 created_at=_audio_required(d, "created_at", Int, "voice consent"), raw=Dict{String,Any}(d))
end

# A delete's reply. OpenAI's VoiceDeletedResource (`object` "audio.voice.deleted") and VoiceConsentDeletedResource
# (`object` "audio.voice_consent", as the reference gives it) require `id`, `object` and `deleted`. Gemini's
# voices.delete answers `{}`: its 200 is the confirmation, of the voice requested.
function _voice_deleted(d::AbstractDict, object::String)::VoiceDeleteSuccess
    got = _audio_required(d, "object", String, "delete reply")
    got == object || throw(ArgumentError("the delete reply is a $(repr(got)) record, not the $(repr(object)) it answers with"))
    VoiceDeleteSuccess(id=_audio_required(d, "id", String, "delete reply"), deleted=_confirm_deleted(d), object=got)
end

# ─── Requests ─────────────────────────────────────────────────────────────────

# OpenAI takes audio samples and consent recordings of at most 10 MiB ("Maximum size is 10 MiB").
const _VOICE_AUDIO_MAX = 10 * 1024^2
function _voice_audio_file(verb::String, what::String, path::AbstractString; max::Union{Int,Nothing}=nothing)::String
    isfile(path) || throw(ArgumentError("$verb: $what not found: $path"))
    isnothing(max) || filesize(path) <= max || throw(ArgumentError(
        "$verb: OpenAI takes a $what of at most 10 MiB (got $(filesize(path)) bytes)"))
    String(path)
end

const _VOICE_PITCHES = ("low", "medium", "high")
# Gemini's descriptive fields of a created voice, by the keyword each comes from.
const _GEMINI_VOICE_FIELDS = (:language => "language_code", :region => "region_code", :accent => "accent",
                              :persona => "persona", :context => "context", :gender => "gender", :pitch => "pitch",
                              :description => "description")

# The options a call sets, of those named.
_set_options(opts::NamedTuple, names)::Vector{String} =
    String[String(n) for n in names if !isnothing(getfield(opts, n))]

function _refuse_voice_options(verb::String, provider::String, opts::NamedTuple, names, fix::String)::Nothing
    set = _set_options(opts, names)
    isempty(set) || throw(ArgumentError("$verb: $provider takes no $(join(set, ", ")): $fix"))
    nothing
end

# Exactly one way to make a voice: from a description (`prompt`) or from a recording (`audio_sample`).
function _voice_source(verb::String, prompt, audio_sample)::Symbol
    isnothing(prompt) == isnothing(audio_sample) && throw(ArgumentError(
        "$verb: give a prompt (a voice designed from a description) or an audio_sample (a voice replicated from a " *
        "recording), one of the two"))
    isnothing(prompt) || isempty(strip(prompt)) && throw(ArgumentError("$verb: the prompt is blank"))
    isnothing(prompt) ? :sample : :prompt
end

# OpenAI's CreateVoicePromptRequest (JSON: type "prompt", name 1–256 characters, prompt and script_hint 1–2000, not
# blank, model) or CreateVoiceFromConsentRequest (multipart: type "audio_sample", name, audio_sample, consent).
function _create_voice_request(::_OpenAIVoices, service, opts::NamedTuple)
    verb = "create_voice"
    _refuse_voice_options(verb, "OpenAI", opts, (:consent_audio, :base_voice, :store, first.(_GEMINI_VOICE_FIELDS)...),
        "those are Gemini's (service=GEMINIServiceEndpoint)")
    source = _voice_source(verb, opts.prompt, opts.audio_sample)
    name = something(opts.name, "")
    1 <= length(name) <= 256 || throw(ArgumentError("$verb: OpenAI names a voice with 1 to 256 characters (got $(length(name)))"))
    if source === :prompt
        isnothing(opts.consent) || throw(ArgumentError("$verb: a consent goes with an audio_sample, not a prompt"))
        length(opts.prompt) <= 2000 || throw(ArgumentError("$verb: OpenAI takes a prompt of at most 2000 characters"))
        hint = opts.script_hint
        isnothing(hint) || (!isempty(strip(hint)) && length(hint) <= 2000) || throw(ArgumentError(
            "$verb: the script_hint is 1 to 2000 characters and not blank"))
        body = Dict{String,Any}("type" => "prompt", "name" => name, "prompt" => opts.prompt)
        isnothing(hint) || (body["script_hint"] = hint)
        isnothing(opts.model) || (body["model"] = opts.model)
        return auth_header(service), JSON.json(body)
    end
    isnothing(opts.script_hint) || throw(ArgumentError("$verb: a script_hint goes with a prompt, not an audio_sample"))
    consent = something(opts.consent, "")
    isempty(strip(consent)) && throw(ArgumentError(
        "$verb: a voice from an audio_sample names its speaker's consent recording: consent = \"cons_…\" " *
        "(create_voice_consent makes one)"))
    sample = _voice_audio_file(verb, "audio_sample", opts.audio_sample; max=_VOICE_AUDIO_MAX)
    isnothing(opts.model) || throw(ArgumentError("$verb: OpenAI takes a model for a prompt voice only"))
    fields = Pair{String,Any}["type" => "audio_sample", "name" => name, "consent" => consent,
        "audio_sample" => HTTP.Multipart(basename(sample), IOBuffer(read(sample)), _mime_for(sample))]
    auth_header_multipart(service), HTTP.Form(fields)
end

# Gemini's CreateVoice: `store` (true unless set: "stateful storage enabled by default") and the `voice`, a prompted one
# (`prompted.input`, an optional `base_voice` to edit) or a replicated one (`replicated.source_audio` and
# `consent_audio`, each the audio's MIME type and base64 bytes), with its optional model, display name and
# descriptive fields (https://ai.google.dev/gemini-api/docs/voice-design, /voice-replication).
function _create_voice_request(::_GeminiVoices, service, opts::NamedTuple)
    verb = "create_voice"
    _refuse_voice_options(verb, "Gemini", opts, (:consent, :script_hint),
        "Gemini takes the consent recording itself (consent_audio) and no script hint")
    source = _voice_source(verb, opts.prompt, opts.audio_sample)
    isnothing(opts.pitch) || opts.pitch in _VOICE_PITCHES || throw(ArgumentError(
        "$verb: pitch is \"low\", \"medium\" or \"high\" (got $(repr(opts.pitch)))"))
    voice = Dict{String,Any}()
    isnothing(opts.model) || (voice["model"] = opts.model)
    isnothing(opts.name) || (voice["display_name"] = opts.name)
    for (kw, field) in _GEMINI_VOICE_FIELDS
        v = getfield(opts, kw)
        isnothing(v) || (voice[field] = v)
    end
    if source === :prompt
        isnothing(opts.consent_audio) || throw(ArgumentError("$verb: consent_audio goes with an audio_sample, not a prompt"))
        voice["type"] = "prompted"
        voice["prompted"] = Dict{String,Any}("input" => opts.prompt,
            (isnothing(opts.base_voice) ? () : ("base_voice" => opts.base_voice,))...)
    else
        isnothing(opts.base_voice) || throw(ArgumentError("$verb: a base_voice goes with a prompt, not an audio_sample"))
        isnothing(opts.consent_audio) && throw(ArgumentError(
            "$verb: Gemini replicates a voice from an audio_sample and the speaker's consent recording: pass consent_audio"))
        audio(path, what) = (p = _voice_audio_file(verb, what, path);
            Dict{String,Any}("mime_type" => _mime_for(p), "data" => base64encode(read(p))))
        voice["type"] = "replicated"
        voice["replicated"] = Dict{String,Any}("source_audio" => audio(opts.audio_sample, "audio_sample"),
                                               "consent_audio" => audio(opts.consent_audio, "consent_audio"))
    end
    auth_header(service), JSON.json(Dict{String,Any}("store" => something(opts.store, true), "voice" => voice))
end

# One exchange: a 200's JSON to `decode`, another status the VoiceFailure, anything thrown the VoiceCallError. A
# single bounded attempt: a create that timed out may still have made the voice, and a second would make another.
function _voice_call(decode::Function, method::String, url::String, headers, body, config, cancel)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = isnothing(body) ? _http(method, url, headers; cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)) :
            _http(method, url, headers, body; cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 ? decode(_json_object(resp.body)) : _failure(VoiceFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(VoiceCallError, e)
    end
end

const _CANCEL_DOC = "Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VoiceCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried)."

"""
    create_voice(; name=nothing, prompt=nothing, audio_sample=nothing, service=OPENAIServiceEndpoint, …)
        -> VoiceSuccess | VoiceFailure | VoiceCallError

Create a custom voice, from a description (`prompt`) or from a recording of the speaker
(`audio_sample`, a path), one of the two, and return it as a [`Voice`](@ref) whose `id` (or
Gemini's stateless `key`) is what [`speak`](@ref) takes as `voice` on that service.

On `OPENAIServiceEndpoint` ([custom voices](https://developers.openai.com/api/docs/guides/custom-voices)):
`name` (1 to 256 characters) is required. A `prompt` (at most 2,000 characters) with an optional
`script_hint` (the text the voice speaks while it is made) and `model` (`"auto"` unless named) makes a
prompt voice, which OpenAI supports "only in Live, not in Realtime or the speech endpoint": a Realtime
session refused one with "Prompt-based voices are not supported by the Realtime API" (2026-11-17), so
`speak` cannot use it. An `audio_sample` (at most 10 MiB) with `consent`, the id of the speaker's
[`VoiceConsent`](@ref), makes a voice from the recording; that access is "limited to eligible
customers", and an organization without it gets HTTP 404 "Your organization does not have access to
this endpoint" as a [`VoiceFailure`](@ref).

On `GEMINIServiceEndpoint` ([voice design](https://ai.google.dev/gemini-api/docs/voice-design),
[voice replication](https://ai.google.dev/gemini-api/docs/voice-replication)): a `prompt` designs a
voice (`base_voice` names a voice it edits); an `audio_sample` with `consent_audio`, a recording of
the speaker reading the consent phrase, replicates one. `store` (default `true`) keeps the voice in the
project, under an id `voice_…` (200 per project, one year); `store = false` replicates a voice without
storing it and returns its `key` (`voicekey_…`, seven days). `name` is its display name; `model`,
`language` (a BCP-47 tag), `region`, `accent`, `persona`, `context`, `gender`, `pitch` (`"low"`,
`"medium"` or `"high"`) and `description` describe it. A designed voice's reply carries a WAV preview
(`raw["sample_audio"]`) and its token usage (`raw["usage"]`), which UniLM does not price.

An option the service does not take, a missing name on OpenAI, both or neither of `prompt` and
`audio_sample`, a missing consent and a missing file throw `ArgumentError` before any request.
A create is a single attempt, never retried: one that timed out may still have made the voice.
$_CANCEL_DOC
"""
function create_voice(; name::Union{AbstractString,Nothing}=nothing, prompt::Union{AbstractString,Nothing}=nothing,
                      audio_sample::Union{AbstractString,Nothing}=nothing, consent::Union{AbstractString,Nothing}=nothing,
                      consent_audio::Union{AbstractString,Nothing}=nothing, model::Union{AbstractString,Nothing}=nothing,
                      script_hint::Union{AbstractString,Nothing}=nothing, base_voice::Union{AbstractString,Nothing}=nothing,
                      store::Union{Bool,Nothing}=nothing, language::Union{AbstractString,Nothing}=nothing,
                      region::Union{AbstractString,Nothing}=nothing, accent::Union{AbstractString,Nothing}=nothing,
                      persona::Union{AbstractString,Nothing}=nothing, context::Union{AbstractString,Nothing}=nothing,
                      gender::Union{AbstractString,Nothing}=nothing, pitch::Union{AbstractString,Nothing}=nothing,
                      description::Union{AbstractString,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                      config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :voices, "Voices API")
    wire = _voices_wire(service)
    opts = (; name, prompt, audio_sample, consent, consent_audio, model, script_hint, base_voice, store, language,
            region, accent, persona, context, gender, pitch, description)
    headers, body = _create_voice_request(wire, service, opts)
    _voice_call(d -> VoiceSuccess(response=_parse_voice(wire, d)), "POST", _voices_url(service), headers, body,
                config, cancel)
end

# OpenAI's API has no voice listing or lookup: its custom voices are created and deleted only.
function _refuse_openai_voice_read(::_OpenAIVoices, verb::String)
    throw(ArgumentError("$verb: OpenAI's API has no voice $(verb == "list_voices" ? "listing" : "lookup") (its custom " *
        "voices are created and deleted only); Gemini's Voices API has one: service=GEMINIServiceEndpoint"))
end
_refuse_openai_voice_read(::_GeminiVoices, ::String) = nothing

# A list filter's values: one string, or several (Gemini matches any of them).
_filter_values(::Nothing)::Vector{String} = String[]
_filter_values(v::AbstractString)::Vector{String} = [String(v)]
_filter_values(v::AbstractVector{<:AbstractString})::Vector{String} = String.(v)

"""
    list_voices(; service=GEMINIServiceEndpoint-only, limit=nothing, after=nothing, type=nothing, language=nothing,
                region=nothing, accent=nothing, gender=nothing, pitch=nothing, persona=nothing, context=nothing,
                search=nothing) -> VoiceListSuccess | VoiceFailure | VoiceCallError

List Gemini's voices, a page at a time: the project's custom voices, newest first, then the
prebuilt catalog (the 30 studio voices and the Extended Voice Library), each a [`Voice`](@ref) whose
`id` [`speak`](@ref) takes as `voice`. `limit` is the page's size (Gemini's default 50, at most
1,000), `after` the `raw["next_page_token"]` of the page before. The filters take a string or a list
of strings (a voice matching any of them): `type` (`"prebuilt"`, `"prompted"`, `"replicated"`),
`language` (BCP-47 tags), `region`, `accent`, `gender`, `pitch`, `persona` and `context`; `search`
matches display names and descriptions ([voice library](https://ai.google.dev/gemini-api/docs/speech-generation#voice-library)).
OpenAI's API lists no voices: `service = OPENAIServiceEndpoint` (the default of the voice verbs)
throws `ArgumentError` naming `GEMINIServiceEndpoint`, as does a `limit` outside 1–1,000.
$_CANCEL_DOC
"""
function list_voices(; service::ServiceEndpointSpec=OPENAIServiceEndpoint, limit::Union{Int,Nothing}=nothing,
                     after::Union{AbstractString,Nothing}=nothing, type=nothing, language=nothing, region=nothing,
                     accent=nothing, gender=nothing, pitch=nothing, persona=nothing, context=nothing,
                     search::Union{AbstractString,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing,
                     cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :voices, "Voices API")
    wire = _voices_wire(service)
    _refuse_openai_voice_read(wire, "list_voices")
    isnothing(limit) || 1 <= limit <= 1000 || throw(ArgumentError(
        "list_voices: Gemini's page holds 1 to 1000 voices (got limit=$limit)"))
    query = String[]
    isnothing(limit) || push!(query, "page_size=$limit")
    isnothing(after) || push!(query, "page_token=" * _uripart(after))
    for (key, v) in ("type" => type, "language_code" => language, "region_code" => region, "accent" => accent,
                     "gender" => gender, "pitch" => pitch, "persona" => persona, "context" => context)
        append!(query, (key * "=" * _uripart(x) for x in _filter_values(v)))
    end
    isnothing(search) || push!(query, "search=" * _uripart(search))
    url = _voices_url(service) * (isempty(query) ? "" : "?" * join(query, "&"))
    _voice_call(d -> VoiceListSuccess(response=_voice_page(wire, d)), "GET", url, auth_header(service), nothing,
                config, cancel)
end

"""
    get_voice(voice_id; service=OPENAIServiceEndpoint) -> VoiceSuccess | VoiceFailure | VoiceCallError

A stored custom voice of Gemini's (`voice_…`), with its preview for a designed voice. Gemini looks up
stored voices only: a prebuilt voice's id answers HTTP 404. OpenAI's API has no voice lookup:
`service = OPENAIServiceEndpoint` throws `ArgumentError` naming `GEMINIServiceEndpoint`.
$_CANCEL_DOC
"""
function get_voice(voice_id::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                   config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :voices, "Voices API")
    wire = _voices_wire(service)
    _refuse_openai_voice_read(wire, "get_voice")
    _voice_call(d -> VoiceSuccess(response=_parse_voice(wire, d)), "GET", _voices_url(service) * "/" * _uripart(voice_id),
                auth_header(service), nothing, config, cancel)
end

"""
    delete_voice(voice_id; service=OPENAIServiceEndpoint) -> VoiceDeleteSuccess | VoiceFailure | VoiceCallError

Delete a custom voice: OpenAI's (its consent recording stays) or a stored one of Gemini's. A voice
already deleted, or never made, answers HTTP 404, a [`VoiceFailure`](@ref).
$_CANCEL_DOC
"""
function delete_voice(voice_id::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                      config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :voices, "Voices API")
    wire = _voices_wire(service)
    decode = wire isa _GeminiVoices ? (_ -> VoiceDeleteSuccess(id=String(voice_id), deleted=true)) :
                                      (d -> _voice_deleted(d, "audio.voice.deleted"))
    _voice_call(decode, "DELETE", _voices_url(service) * "/" * _uripart(voice_id), auth_header(service), nothing,
                config, cancel)
end

# ─── OpenAI voice consents ────────────────────────────────────────────────────

# Gemini takes the consent recording inside the voice it replicates; only OpenAI keeps consents apart.
function _consents_service(service, verb::String)
    validate_capability(service, :voices, "Voices API")
    _voices_wire(service) isa _OpenAIVoices || throw(ArgumentError(
        "$verb: voice consents are OpenAI's; Gemini takes the consent recording inside the voice it replicates " *
        "(create_voice(…; audio_sample, consent_audio))"))
    service
end

"""
    create_voice_consent(recording; name, language, service=OPENAIServiceEndpoint)
        -> VoiceConsentSuccess | VoiceFailure | VoiceCallError

Upload the recording (a path, at most 10 MiB) of a speaker reading OpenAI's consent phrase, under the
label `name`, `language` being the BCP 47 tag of the phrase (`"en-US"`); the [`VoiceConsent`](@ref)'s
`id` is the `consent` of [`create_voice`](@ref). OpenAI's custom voices from recordings are limited to
eligible customers: an organization without access gets HTTP 404, a [`VoiceFailure`](@ref). A blank
name or language, a missing or too large file, and a service other than OpenAI throw `ArgumentError`
before any request. A single attempt, never retried.
$_CANCEL_DOC
"""
function create_voice_consent(recording::AbstractString; name::AbstractString, language::AbstractString,
                              service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                              config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    _consents_service(service, "create_voice_consent")
    for (k, v) in (("name", name), ("language", language))
        isempty(strip(v)) && throw(ArgumentError("create_voice_consent: the $k is blank"))
    end
    path = _voice_audio_file("create_voice_consent", "recording", recording; max=_VOICE_AUDIO_MAX)
    form = HTTP.Form(Pair{String,Any}["name" => String(name), "language" => String(language),
        "recording" => HTTP.Multipart(basename(path), IOBuffer(read(path)), _mime_for(path))])
    _voice_call(d -> VoiceConsentSuccess(response=_parse_voice_consent(d)), "POST", _voice_consents_url(service),
                auth_header_multipart(service), form, config, cancel)
end

"""
    list_voice_consents(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)
        -> VoiceConsentListSuccess | VoiceFailure | VoiceCallError

OpenAI's voice consents, a page at a time: `limit` 1 to 100 (OpenAI's default 20), `after` the id of
the last consent of the page before (`raw["last_id"]`). A `limit` out of range throws `ArgumentError`.
$_CANCEL_DOC
"""
function list_voice_consents(; limit::Union{Int,Nothing}=nothing, after::Union{AbstractString,Nothing}=nothing,
                             service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                             config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    _consents_service(service, "list_voice_consents")
    isnothing(limit) || 1 <= limit <= 100 || throw(ArgumentError(
        "list_voice_consents: limit is 1 to 100 (got $limit)"))
    url = _voice_consents_url(service) * _vs_list_query("list_voice_consents", "limit" => limit, "after" => after)
    _voice_call("GET", url, auth_header(service), nothing, config, cancel) do d
        page, has_more = _vs_page(d)
        VoiceConsentListSuccess(response=VoiceConsentList(data=VoiceConsent[_parse_voice_consent(c) for c in page],
                                                          has_more=has_more, raw=Dict{String,Any}(d)))
    end
end

"""
    get_voice_consent(consent_id; service=OPENAIServiceEndpoint) -> VoiceConsentSuccess | VoiceFailure | VoiceCallError

An OpenAI voice consent by id.
$_CANCEL_DOC
"""
function get_voice_consent(consent_id::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                           config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    _consents_service(service, "get_voice_consent")
    _voice_call(d -> VoiceConsentSuccess(response=_parse_voice_consent(d)), "GET",
                _voice_consents_url(service) * "/" * _uripart(consent_id), auth_header(service), nothing, config, cancel)
end

"""
    update_voice_consent(consent_id; name, service=OPENAIServiceEndpoint)
        -> VoiceConsentSuccess | VoiceFailure | VoiceCallError

Relabel an OpenAI voice consent (its `name`, the only field OpenAI updates); a blank name throws
`ArgumentError` before any request.
$_CANCEL_DOC
"""
function update_voice_consent(consent_id::AbstractString; name::AbstractString,
                              service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                              config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    _consents_service(service, "update_voice_consent")
    isempty(strip(name)) && throw(ArgumentError("update_voice_consent: the name is blank"))
    _voice_call(d -> VoiceConsentSuccess(response=_parse_voice_consent(d)), "POST",
                _voice_consents_url(service) * "/" * _uripart(consent_id), auth_header(service),
                JSON.json(Dict("name" => name)), config, cancel)
end

"""
    delete_voice_consent(consent_id; service=OPENAIServiceEndpoint) -> VoiceDeleteSuccess | VoiceFailure | VoiceCallError

Delete an OpenAI voice consent; `object` of the result is `"audio.voice_consent"`, as OpenAI's
reference gives its deletion record.
$_CANCEL_DOC
"""
function delete_voice_consent(consent_id::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                              config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    _consents_service(service, "delete_voice_consent")
    _voice_call(d -> _voice_deleted(d, "audio.voice_consent"), "DELETE",
                _voice_consents_url(service) * "/" * _uripart(consent_id), auth_header(service), nothing, config, cancel)
end
