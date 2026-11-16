# ============================================================================
# Provider model listings: the models OpenAI, Anthropic and Gemini serve the
# authenticated key, with their token limits, capabilities and lifecycle dates.
#   OpenAI     GET /v1/models      https://developers.openai.com/api/reference/resources/models/methods/list
#   Anthropic  GET /v1/models      https://platform.claude.com/docs/en/api/models/list
#   Gemini     GET /v1beta/models  https://ai.google.dev/api/models
# Anthropic and Gemini answer in pages, and every page is fetched. One model by id is
# GET <listing>/{model} on all three; OpenAI deletes a fine-tuned model with
# DELETE /v1/models/{model}.
# ============================================================================

"""
    ProviderModel

One model a provider lists ([`list_models`](@ref) on [`OPENAIServiceEndpoint`](@ref),
[`ANTHROPICServiceEndpoint`](@ref) or [`GEMINIServiceEndpoint`](@ref)), or looks up by id
([`retrieve_model`](@ref)):

- `id::String`: the name a request passes as `model` — OpenAI's and Anthropic's `id`,
  Gemini's `name` without its `models/` prefix;
- `display_name::Union{String,Nothing}`: Anthropic's `display_name`, Gemini's
  `displayName`; `nothing` on OpenAI;
- `created::Union{DateTime,Nothing}`, in UTC: OpenAI's `created`, Anthropic's `created_at` —
  `nothing` where Anthropic gives the Unix epoch, its value for an unknown release date —
  and `nothing` on Gemini;
- `input_token_limit`, `output_token_limit` (`Union{Int,Nothing}`): Anthropic's
  `max_input_tokens` and `max_tokens` (the largest `max_tokens` a request may set),
  Gemini's `inputTokenLimit` and `outputTokenLimit`; `nothing` on OpenAI;
- `capabilities::Dict{String,Any}`: Anthropic's `capabilities` object as sent (thinking
  types, effort levels, input kinds, tools, …), Gemini's `supportedGenerationMethods` and
  `thinking`; empty on OpenAI;
- `deprecated::Union{Date,Nothing}`: the UTC date of Anthropic's `deprecated_at`, set once a
  model is deprecated; `nothing` otherwise, and on OpenAI and Gemini, whose listings carry
  no deprecation date;
- `retires::Union{Date,Nothing}`: OpenAI's `shutdown_date`, the UTC date of Anthropic's
  `retires_at` (a schedule Anthropic may revise until the model retires); `nothing` while
  none is announced, and on Gemini;
- `raw::Dict{String,Any}`: the entry as the provider sent it.
"""
struct ProviderModel
    id::String
    display_name::Union{String,Nothing}
    created::Union{DateTime,Nothing}
    input_token_limit::Union{Int,Nothing}
    output_token_limit::Union{Int,Nothing}
    capabilities::Dict{String,Any}
    deprecated::Union{Date,Nothing}
    retires::Union{Date,Nothing}
    raw::Dict{String,Any}
end

function Base.show(io::IO, m::ProviderModel)
    print(io, "ProviderModel(", repr(m.id))
    isnothing(m.display_name) || print(io, ", ", repr(m.display_name))
    isnothing(m.deprecated) || print(io, ", deprecated ", m.deprecated)
    isnothing(m.retires) || print(io, ", retires ", m.retires)
    print(io, ")")
end

"Successful [`list_models`](@ref) result on OpenAI, Anthropic or Gemini: the [`ProviderModel`](@ref)s of every page, in the order listed, and `raw`, the parsed body of each page in the order fetched."
struct ModelListSuccess <: LLMRequestResponse
    models::Vector{ProviderModel}
    raw::Vector{Dict{String,Any}}
end
"Successful [`retrieve_model`](@ref) result: `model`, the [`ProviderModel`](@ref) the provider returned, decoded as its listing entry is; its `raw` is the reply."
struct ModelSuccess <: LLMRequestResponse
    model::ProviderModel
end
"Successful [`delete_model`](@ref) result on OpenAI (`DELETE /v1/models/{model}`), a fine-tuned model's deletion: the `id` the reply names, and `deleted`, which the reply confirmed (`true`)."
@kwdef struct ModelDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Error result of [`list_models`](@ref) and [`retrieve_model`](@ref), and of a fine-tuned model's deletion, on OpenAI, Anthropic or Gemini: HTTP `status`, the raw `response` body, and the `request_id` the provider sent (`x-request-id`/`request-id` header), if any. An error status on any page is the listing's failure; an unknown model is a 404."
@kwdef struct ModelListFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"[`list_models`](@ref) or [`retrieve_model`](@ref) call, or a fine-tuned model's deletion, on OpenAI, Anthropic or Gemini that produced no result (a missing key, no connection, a timeout, a cancel, or a reply that is not the documented listing, model or deletion); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel."
@kwdef struct ModelListCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── The three wires ─────────────────────────────────────────────────────────

# How a provider lists its models: where the first page is, how a page reads, and where the
# next one starts. An endpoint with no listing of its own lists through its System One
# protocol (`_model_listing` in system_one.jl).
abstract type _ModelListing end
struct _OpenAIModels <: _ModelListing end
struct _AnthropicModels <: _ModelListing end
struct _GeminiModels <: _ModelListing end

_model_listing(::Type{OPENAIServiceEndpoint}) = _OpenAIModels()
_model_listing(::_AnthropicEndpoint) = _AnthropicModels()
_model_listing(::Type{GEMINIServiceEndpoint}) = _GeminiModels()

# The first page; a function so tests can route their own endpoint types to a local server.
_models_url(::Type{OPENAIServiceEndpoint})::String = _api_base_url(OPENAIServiceEndpoint) * "/v1/models"
_models_url(::_AnthropicEndpoint)::String = ANTHROPIC_BASE_URL * "/v1/models"
_models_url(::Type{GEMINIServiceEndpoint})::String = GEMINI_NATIVE_BASE * "/models"

# The headers of every page: the endpoint's own, except on Anthropic, whose Models API
# reference names `anthropic-version` and `anthropic-workspace-id` for GET /v1/models, and no
# `anthropic-beta` or user-profile header
# (https://platform.claude.com/docs/en/api/models/list, Headers): an Anthropic listing sends the
# key, the API version and the endpoint's workspace, never its user profile or its betas.
_models_headers(service) = auth_header(service)
_models_headers(service::_AnthropicEndpoint)::Vector{Pair{String,String}} =
    _anthropic_base_headers(_anthropic_endpoint(service); profile=false)

# A page's entries, and the cursor of the page after it (`nothing` on the last page). OpenAI
# lists every model on one page; Anthropic pages by the last id on the page (`has_more`,
# `last_id` → `after_id`), Gemini by an opaque token (`nextPageToken` → `pageToken`), which it
# leaves out on the last page, as it leaves out an empty `models` (proto3 JSON omits a field
# at its default value).
_page_entries(::Union{_OpenAIModels,_AnthropicModels}, page::Dict{String,Any}) = _listing_required(page, "data", AbstractVector)
_page_entries(::_GeminiModels, page::Dict{String,Any}) = something(_listing_field(page, "models", AbstractVector), [])

_next_cursor(::_OpenAIModels, ::Dict{String,Any})::Nothing = nothing
function _next_cursor(::_AnthropicModels, page::Dict{String,Any})::Union{String,Nothing}
    _listing_required(page, "has_more", Bool) || return nothing
    @something(_listing_field(page, "last_id", String),
               throw(ArgumentError("the model listing has more pages (\"has_more\": true) and no \"last_id\" to continue from")))
end
_next_cursor(::_GeminiModels, page::Dict{String,Any})::Union{String,Nothing} =
    (t = _listing_field(page, "nextPageToken", String); isnothing(t) || isempty(t) ? nothing : t)

# Every page asks for the largest page the API serves, so a listing takes as few requests as it
# can: Anthropic's `limit` "Ranges from `1` to `1000`"
# (https://platform.claude.com/docs/en/api/models/list), and Gemini's ListModels "returns at most
# 1000 models per page", where "all other parameters provided to `ListModels` must match the call
# that provided the page token" (pageSize, https://ai.google.dev/api/models). OpenAI lists every
# model on one page.
_first_page(::_OpenAIModels, url::String)::String = url
_first_page(::_AnthropicModels, url::String)::String = url * "?limit=1000"
_first_page(::_GeminiModels, url::String)::String = url * "?pageSize=1000"

_page_url(l::_AnthropicModels, url::String, cursor::String)::String = _first_page(l, url) * "&after_id=" * _uripart(cursor)
_page_url(l::_GeminiModels, url::String, cursor::String)::String = _first_page(l, url) * "&pageToken=" * _uripart(cursor)
# A listing whose pages name no cursor (OpenAI's) has no page after the first.
_page_url(listing::_ModelListing, ::String, ::String) =
    throw(ArgumentError("$(nameof(typeof(listing))) names no page after the first"))

# ─── Entries ─────────────────────────────────────────────────────────────────
# A field may be absent or null where the provider's reference allows it; any other JSON type
# than the documented one is a malformed listing — a call error, never a guess.

function _listing_field(d::AbstractDict, key::String, ::Type{T})::Union{T,Nothing} where {T}
    v = get(d, key, nothing)
    isnothing(v) || v isa T || throw(ArgumentError("model listing field $(repr(key)) holds a $(typeof(v)), not the documented $T"))
    v
end

_listing_required(d::AbstractDict, key::String, ::Type{T}) where {T} =
    @something(_listing_field(d, key, T), throw(ArgumentError("the model listing has no $(repr(key))")))

# The name a request passes as `model`: never blank.
function _listing_id(d::AbstractDict)::String
    id = _listing_required(d, "id", String)
    isempty(id) && throw(ArgumentError("the model listing has an empty \"id\""))
    id
end

# OpenAI's `shutdown_date`: a calendar date, YYYY-MM-DD.
const _YMD = Dates.DateFormat("yyyy-mm-dd")
_calendar_date(s::AbstractString)::Union{Date,Nothing} =
    occursin(r"^\d{4}-\d{2}-\d{2}$", s) ? tryparse(Date, s, _YMD) : nothing

# Anthropic's timestamps: RFC 3339 date-times (section 5.6: a "T", an optional fraction, then
# "Z" or a ±hh:mm offset; "t" and "z" may be lower case), read as the UTC instant they name, to
# the millisecond. `nothing` for any other text, a field out of range, or a leap second, which
# `DateTime` cannot hold.
const _RFC3339 = r"^(\d{4}-\d{2}-\d{2})[Tt]([01]\d|2[0-3]):([0-5]\d):([0-5]\d)(?:\.(\d+))?(?:[Zz]|([+-])([01]\d|2[0-3]):([0-5]\d))$"

function _rfc3339_utc(s::AbstractString)::Union{DateTime,Nothing}
    m = match(_RFC3339, s)
    isnothing(m) && return nothing
    day = tryparse(Date, something(m[1]), _YMD)
    isnothing(day) && return nothing
    int(i::Int) = parse(Int, something(m[i]))
    ms = isnothing(m[5]) ? 0 : parse(Int, rpad(first(something(m[5]), 3), 3, '0'))
    t = DateTime(day) + Dates.Hour(int(2)) + Dates.Minute(int(3)) + Dates.Second(int(4)) + Dates.Millisecond(ms)
    isnothing(m[6]) && return t
    offset = Dates.Minute(60 * int(7) + int(8))
    something(m[6]) == "+" ? t - offset : t + offset
end

function _listing_date(d::AbstractDict, key::String)::Union{Date,Nothing}
    s = _listing_field(d, key, String)
    isnothing(s) && return nothing
    @something(_calendar_date(s), throw(ArgumentError("model listing field $(repr(key)) is not a YYYY-MM-DD date: $(repr(s))")))
end

function _listing_instant(d::AbstractDict, key::String)::Union{DateTime,Nothing}
    s = _listing_field(d, key, String)
    isnothing(s) && return nothing
    @something(_rfc3339_utc(s), throw(ArgumentError("model listing field $(repr(key)) is not an RFC 3339 date-time: $(repr(s))")))
end

_utc_date(::Nothing) = nothing
_utc_date(t::DateTime)::Date = Date(t)

# "May be set to an epoch value if the release date is unknown" (Anthropic's reference).
const _UNIX_EPOCH = DateTime(1970)
_release(t::Union{DateTime,Nothing}) = t == _UNIX_EPOCH ? nothing : t

function _provider_model(::_OpenAIModels, e::Dict{String,Any})::ProviderModel
    created = _listing_field(e, "created", Int)
    ProviderModel(_listing_id(e), nothing, isnothing(created) ? nothing : Dates.unix2datetime(created), nothing, nothing,
                  Dict{String,Any}(), nothing, _listing_date(e, "shutdown_date"), e)
end

_provider_model(::_AnthropicModels, e::Dict{String,Any})::ProviderModel =
    ProviderModel(_listing_id(e), _listing_field(e, "display_name", String), _release(_listing_instant(e, "created_at")),
                  _listing_field(e, "max_input_tokens", Int), _listing_field(e, "max_tokens", Int),
                  Dict{String,Any}(something(_listing_field(e, "capabilities", AbstractDict), Dict{String,Any}())),
                  _utc_date(_listing_instant(e, "deprecated_at")), _utc_date(_listing_instant(e, "retires_at")), e)

# Gemini's Model resource says what a model supports in two fields: the generation methods it
# serves and whether it thinks.
function _gemini_model_capabilities(e::Dict{String,Any})::Dict{String,Any}
    caps = Dict{String,Any}()
    methods = _listing_field(e, "supportedGenerationMethods", AbstractVector)
    if !isnothing(methods)
        all(m -> m isa String, methods) || throw(ArgumentError(
            "model listing field \"supportedGenerationMethods\" holds a value that is not a string"))
        caps["supportedGenerationMethods"] = methods
    end
    thinking = _listing_field(e, "thinking", Bool)
    isnothing(thinking) || (caps["thinking"] = thinking)
    caps
end

function _provider_model(::_GeminiModels, e::Dict{String,Any})::ProviderModel
    name = _listing_required(e, "name", String)
    id = chopprefix(name, "models/")
    (id == name || isempty(id)) && throw(ArgumentError("Gemini model name $(repr(name)) is not of the form models/{model}"))
    ProviderModel(id, _listing_field(e, "displayName", String), nothing, _listing_field(e, "inputTokenLimit", Int),
                  _listing_field(e, "outputTokenLimit", Int), _gemini_model_capabilities(e), nothing, nothing, e)
end

function _listing_page(resp::HTTP.Response)::Dict{String,Any}
    page = JSON.parse(resp.body; dicttype=Dict{String,Any})
    page isa Dict{String,Any} || throw(ArgumentError("the model listing is not a JSON object"))
    page
end

function _page_models(listing::_ModelListing, page::Dict{String,Any})::Vector{ProviderModel}
    entries = _page_entries(listing, page)
    all(e -> e isa AbstractDict, entries) || throw(ArgumentError("a model listing entry is not a JSON object"))
    ProviderModel[_provider_model(listing, Dict{String,Any}(e)) for e in entries]
end

# ─── The listing ─────────────────────────────────────────────────────────────

# Every page through the shared retry seam, one `total_deadline` for them all; an error status
# on any page is the failure. A cursor the listing names twice would fetch the same pages
# again and again, so it ends the call.
function _list_models(listing::_ModelListing, service, config::Union{Nothing,RequestConfig},
                      cancel::Union{Nothing,CancelToken})
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()
    try
        first_page = _models_url(service)
        headers = _models_headers(service)
        models = ProviderModel[]; pages = Dict{String,Any}[]; cursors = Set{String}()
        url = _first_page(listing, first_page)
        while true
            resp = _http_with_retries(cfg, t0, "GET", url, headers; cancel=tok)
            resp.status == 200 || return _failure(ModelListFailure, resp)
            page = _listing_page(resp)
            push!(pages, page)
            append!(models, _page_models(listing, page))
            cursor = _next_cursor(listing, page)
            isnothing(cursor) && return ModelListSuccess(models, pages)
            cursor in cursors && throw(ArgumentError("the model listing named the page cursor $(repr(cursor)) twice"))
            push!(cursors, cursor)
            url = _page_url(listing, first_page, cursor)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(ModelListCallError, e)
    end
end

# ─── One model by id ─────────────────────────────────────────────────────────

# A model is a path segment below its provider's listing: GET /v1/models/{model} on OpenAI and
# Anthropic, GET /v1beta/models/{model} on Gemini (its `name` without `models/`).
_model_url(service, id::AbstractString)::String = _models_url(service) * "/" * _uripart(id)

function _model_reply(resp::HTTP.Response)::Dict{String,Any}
    d = JSON.parse(resp.body; dicttype=Dict{String,Any})
    d isa Dict{String,Any} || throw(ArgumentError("the model reply is not a JSON object"))
    d
end

"""
    retrieve_model(id; service=OPENAIServiceEndpoint, config=nothing, cancel=nothing)

One model by its id, as the provider describes it: `GET /v1/models/{model}` on
[`OPENAIServiceEndpoint`](@ref) and [`ANTHROPICServiceEndpoint`](@ref) (the bare type or an
instance, which sends its workspace, as [`list_models`](@ref) does), `GET /v1beta/models/{model}`
on [`GEMINIServiceEndpoint`](@ref). Returns a [`ModelSuccess`](@ref) whose `model` is the
[`ProviderModel`](@ref) the provider's listing would hold for it — Anthropic resolves an alias to
the dated id it names (`"claude-opus-4-5"` is `"claude-opus-4-5-20251101"`) — a
[`ModelListFailure`](@ref) (HTTP 404 for a model the key cannot use, a retired one among them), or
a [`ModelListCallError`](@ref), which is also what a missing key gives: it names the variable.

`config` and `cancel` work as for [`list_models`](@ref): the lookup rides the shared retry seam.
An empty `id` throws `ArgumentError` before any request, and so does an endpoint that looks up no
model by id: TypeSafe and Ollama, which list theirs ([`model_info`](@ref) shows an installed
Ollama model), and every endpoint without the `:models` capability. A lookup is not recorded by
[`with_recorded_answers`](@ref): inside a scope it goes to the provider, as outside one.

```julia
r = retrieve_model("claude-haiku-5-5"; service=ANTHROPICServiceEndpoint)
issuccess(r) && r.model.output_token_limit   # 128000
```
"""
retrieve_model(id::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
               config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _retrieve_model(_model_listing(service), service, id, config, cancel)

function _retrieve_model(listing::_ModelListing, service, id::AbstractString, config::Union{Nothing,RequestConfig},
                         cancel::Union{Nothing,CancelToken})
    isempty(id) && throw(ArgumentError("retrieve_model needs a model id; list_models lists them"))
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()
    try
        resp = _http_with_retries(cfg, t0, "GET", _model_url(service, id), _models_headers(service); cancel=tok)
        resp.status == 200 || return _failure(ModelListFailure, resp)
        ModelSuccess(_provider_model(listing, _model_reply(resp)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(ModelListCallError, e)
    end
end

# An endpoint with no listing of its own (whose `_model_listing` is its System One protocol)
# looks up no model by id: refused by name, after the capability check every verb makes.
function _retrieve_model(::Any, service, ::AbstractString, ::Union{Nothing,RequestConfig}, ::Union{Nothing,CancelToken})
    name = _service_name(service)
    _capability_declared(service) || throw(ArgumentError("Model lookup is not supported by $name: it declares no capabilities"))
    validate_capability(service, :models, "Model lookup")
    throw(ArgumentError("Model lookup is not supported by $name: it looks up no model by id; " * _lookup_instead(service)))
end
_lookup_instead(_)::String = "list_models lists its models"
_lookup_instead(::OllamaEndpoint)::String = "model_info shows an installed model, and list_models lists them"

# ─── A fine-tuned model's deletion (OpenAI) ──────────────────────────────────

# OpenAI's DELETE /v1/models/{model} deletes a fine-tuned model ("You must have the Owner role in
# your organization", 403 otherwise; 404 for a model the key cannot see or that the endpoint cannot
# delete). One attempt, as the other delete verbs make; the reply must name the model and confirm.
function _delete_model(service, id::AbstractString, config::Union{Nothing,RequestConfig},
                       cancel::Union{Nothing,CancelToken})
    validate_capability(service, :fine_tuning, "Model deletion")
    _delete_model(_model_listing(service), service, id, config, cancel)
end

_delete_model(::Any, service, ::AbstractString, ::Union{Nothing,RequestConfig}, ::Union{Nothing,CancelToken}) =
    throw(ArgumentError("Model deletion is not supported by $(_service_name(service)): OpenAI's Models API deletes a " *
                        "fine-tuned model by id; send it to OPENAIServiceEndpoint"))

function _delete_model(::_OpenAIModels, service, id::AbstractString, config::Union{Nothing,RequestConfig},
                       cancel::Union{Nothing,CancelToken})
    isempty(id) && throw(ArgumentError("delete_model needs the id of a fine-tuned model (\"ft:…\")"))
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("DELETE", _model_url(service, id), _models_headers(service); cfg, remaining=_remaining_s(cfg, t0),
                     cancel=_resolve_cancel(cancel))
        resp.status == 200 || return _failure(ModelListFailure, resp)
        d = _model_reply(resp)
        ModelDeleteSuccess(id=_listing_id(d), deleted=_confirm_deleted(d))
    catch e
        e isa InterruptException && rethrow()
        _callerr(ModelListCallError, e)
    end
end
