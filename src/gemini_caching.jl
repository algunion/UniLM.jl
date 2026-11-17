# ============================================================================
# Gemini context caching — explicit caches (cachedContents) on the Gemini API
# (https://ai.google.dev/gemini-api/docs/caching, https://ai.google.dev/api/caching). A cache holds a
# prompt's start — system instruction, contents, tools and tool config — which a generateContent request
# names in `cachedContent` (a Chat's `cached_content`) and reads at the cache-hit rate. Each reply shape
# below is one the API returned on 2026-11-17.
# ============================================================================

# ─── Gemini resources: names, addresses, replies (gemini_file_search.jl shares them) ─────────────────

# A Gemini resource as the API names it, `<collection>/<id>`, from that name or its id: the id is one path
# segment, neither empty nor a dot segment, which a URL would resolve away.
_gemini_segment(s::AbstractString)::Bool = !isempty(s) && !occursin('/', s) && !(s in (".", ".."))

function _gemini_resource_name(collection::String, name::AbstractString, what::String)::String
    id = chopprefix(name, collection * "/")
    _gemini_segment(id) || throw(ArgumentError(
        "$what names a resource as \"$collection/<id>\" or by its id (got $(repr(name)))"))
    collection * "/" * id
end

# Whether `name` has a resource name's shape: the literal segments as written, an id where `shape` holds `nothing`.
_gemini_shaped(name::AbstractString, shape::Tuple)::Bool =
    (segs = split(name, '/'); length(segs) == length(shape) &&
     all(((s, p),) -> isnothing(p) ? _gemini_segment(s) : s == p, zip(segs, shape)))

const _CACHE_SHAPE = ("cachedContents", nothing)

# The API's address; a function of the endpoint, so tests can route one to a local server. A resource's URL
# escapes each segment of its name.
_gemini_api_root(::Type{GEMINIServiceEndpoint})::String = GEMINI_NATIVE_BASE
_gemini_path(name::AbstractString)::String = join((_uripart(s) for s in split(name, '/')), '/')
_gemini_url(service, name::AbstractString, suffix::String="")::String =
    _gemini_api_root(service) * "/" * _gemini_path(name) * suffix
_gemini_query(params::Pair...)::String = (q = _query(params...); isempty(q) ? "" : "?" * join(q, "&"))

# A reply is the documented object or a malformed one: a field the API always sends is never read with a
# default, and a field of another JSON type is never coerced. proto3 JSON leaves a field out at its default
# value — a zero count, an empty list, `false` — which reads as that default. Each throw becomes the verb's
# call error.
_gemini_object(x, what::String)::Dict{String,Any} =
    x isa Dict{String,Any} ? x : throw(ArgumentError("$what is not a JSON object"))

function _gemini_field(d::AbstractDict, key::String, ::Type{T}, what::String) where {T}
    v = get(d, key, nothing)
    isnothing(v) || v isa T || throw(ArgumentError("$what's $(repr(key)) is a JSON $(_json_type(v)), not the documented type"))
    v
end
_gemini_required(d::AbstractDict, key::String, ::Type{T}, what::String) where {T} =
    @something(_gemini_field(d, key, T, what), throw(ArgumentError("$what has no $(repr(key))")))

function _gemini_time(d::AbstractDict, key::String, what::String)::DateTime
    s = _gemini_required(d, key, String, what)
    @something(_rfc3339_utc(s), throw(ArgumentError("$what's $(repr(key)) is not an RFC 3339 date-time: $(repr(s))")))
end

# An int64, which proto3 JSON sends as a string.
function _gemini_int64(d::AbstractDict, key::String, what::String)::Int
    s = _gemini_field(d, key, String, what)
    isnothing(s) ? 0 : @something(tryparse(Int, s), throw(ArgumentError("$what's $(repr(key)) is no integer: $(repr(s))")))
end

function _gemini_reply_name(d::AbstractDict, shape::Tuple, what::String)::String
    name = _gemini_required(d, "name", String, what)
    _gemini_shaped(name, shape) || throw(ArgumentError("$what's name is not $(_shape_text(shape)): $(repr(name))"))
    name
end
_shape_text(shape::Tuple)::String = join((something(p, "<id>") for p in shape), '/')

# ─── The cache a Chat names ──────────────────────────────────────────────────

# Explicit caching "can be manually enabled on Gemini 3 and 2.5 models ...; not supported for Gemini 4 models"
# (https://ai.google.dev/gemini-api/docs/caching); the API answered a cache for gemini-4-hydrogen with HTTP 400
# "Explicit caching is not supported for models/gemini-4-hydrogen." (observed 2026-11-17).
function _gemini_explicit_cache_model(model::AbstractString, what::String)::Nothing
    startswith(model, "gemini-4-") && throw(ArgumentError("$what: $model takes no explicit cache: Google " *
        "caches explicitly on Gemini 3 and 2.5 models, \"not supported for Gemini 4 models\", which cache " *
        "implicitly (https://ai.google.dev/gemini-api/docs/caching); use a Gemini 3 or 2.5 model"))
    nothing
end

# ─── Parsed objects ──────────────────────────────────────────────────────────

"""
    CacheObject

A Gemini context cache, as [`create_cache`](@ref), [`retrieve_cache`](@ref), [`update_cache`](@ref) and
[`list_caches`](@ref) return it; `raw` holds the cache as Gemini sent it.

- `name::String`: `"cachedContents/<id>"`, what a [`Chat`](@ref)'s `cached_content` and the other cache
  verbs take (the verbs take the id alone too).
- `model::String`: the model the cache was made for, without its `models/` prefix: "Cached content can be
  only used with model it was created for".
- `display_name::Union{String,Nothing}`: the name it was given; `""` when none was.
- `total_tokens::Int`: `usageMetadata.totalTokenCount`, the tokens the cache holds, which its storage bills.
- `create_time`, `update_time`, `expire_time` (`DateTime`, UTC): when it was made, last updated, and
  expires; Gemini deletes it then.

A reply without one of these fields is no cache: the call returns its [`CacheCallError`](@ref). The
contents a cache holds are not returned: "It's not possible to retrieve or view cached content".
"""
@kwdef struct CacheObject
    name::String
    model::String
    display_name::Union{String,Nothing} = nothing
    total_tokens::Int
    create_time::DateTime
    update_time::DateTime
    expire_time::DateTime
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    CacheList

A page of [`CacheObject`](@ref)s from [`list_caches`](@ref); `has_more` signals a further page, which
starts at `raw["nextPageToken"]`, and `raw` holds the page as sent.
"""
@kwdef struct CacheList
    data::Vector{CacheObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

# A Chat names a cache by its object too (api.jl stores the name).
_cache_reference(::Nothing) = nothing
_cache_reference(name::AbstractString)::String = _gemini_resource_name("cachedContents", name, "cached_content")
_cache_reference(cache::CacheObject)::String = cache.name

# ─── Result types ────────────────────────────────────────────────────────────

"Successful [`create_cache`](@ref), [`retrieve_cache`](@ref) or [`update_cache`](@ref) result wrapping a [`CacheObject`](@ref)."
@kwdef struct CacheSuccess <: LLMRequestResponse; response::CacheObject; end
"Successful [`list_caches`](@ref) result wrapping a [`CacheList`](@ref)."
@kwdef struct CacheListSuccess <: LLMRequestResponse; response::CacheList; end
"Successful [`delete_cache`](@ref) result: Gemini deleted the cache `name` (`deleted` is always `true`); its reply is empty, so `name` is the cache the call named."
@kwdef struct CacheDeleteSuccess <: LLMRequestResponse; name::String; deleted::Bool; end
"Context caching error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent, if any. A cache that is gone, or another project's, is Gemini's 403 \"CachedContent not found (or permission denied)\"."
@kwdef struct CacheFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Context caching call that produced no usable reply (a missing key, transport failure, timeout, cancel, or a 200 that is not the documented cache); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel."
@kwdef struct CacheCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Replies ─────────────────────────────────────────────────────────────────

function _parse_cache(d::Dict{String,Any})::CacheObject
    w = "the cache"
    model = _gemini_required(d, "model", String, w)
    startswith(model, "models/") || throw(ArgumentError("the cache's model is not models/<model>: $(repr(model))"))
    usage = _gemini_required(d, "usageMetadata", AbstractDict, w)
    CacheObject(name=_gemini_reply_name(d, _CACHE_SHAPE, w), model=chopprefix(model, "models/"),
                display_name=_gemini_field(d, "displayName", String, w),
                total_tokens=_gemini_required(usage, "totalTokenCount", Int, "the cache's usageMetadata"),
                create_time=_gemini_time(d, "createTime", w), update_time=_gemini_time(d, "updateTime", w),
                expire_time=_gemini_time(d, "expireTime", w), raw=d)
end

_cache_success(x)::CacheSuccess = CacheSuccess(response=_parse_cache(_gemini_object(x, "the cache reply")))

# A page: `cachedContents` and `nextPageToken`, each left out when empty, so `{}` is an empty last page.
function _cache_page(x)::CacheListSuccess
    d = _gemini_object(x, "the cache listing")
    token = _gemini_field(d, "nextPageToken", String, "the cache listing")
    caches = something(_gemini_field(d, "cachedContents", AbstractVector, "the cache listing"), [])
    CacheListSuccess(response=CacheList(data=CacheObject[_parse_cache(_gemini_object(c, "a listed cache")) for c in caches],
                                        has_more=!(isnothing(token) || isempty(token)), raw=d))
end

# One cache call: a 200's body decoded, another status the failure, anything thrown the call error.
_cache_call(decode::Function, method::String, url::String, service, config, cancel; body=nothing) =
    _vs_json_call(decode, CacheFailure, CacheCallError, method, url, service, config, cancel; body)

# ─── Requests ────────────────────────────────────────────────────────────────

# A cache's expiration, the API's oneof: a TTL in whole seconds, a google-duration ("300s"), or an expiry, an
# RFC 3339 UTC time; "If not set, the TTL defaults to 1 hour", and "There are no minimum or maximum bounds on
# the TTL" (https://ai.google.dev/gemini-api/docs/caching). Both at once the API answered with HTTP 400
# ("oneof field 'expiration' is already set", observed 2026-11-17).
const _RFC3339_MS = Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.sss")

function _cache_expiration(verb::String, ttl::Union{Integer,Nothing}, expire_time::Union{DateTime,Nothing})::Dict{Symbol,Any}
    isnothing(ttl) || isnothing(expire_time) || throw(ArgumentError(
        "$verb: set ttl or expire_time, not both: a cache's expiration is one of them, and Gemini answers both with HTTP 400"))
    isnothing(expire_time) || return Dict{Symbol,Any}(:expireTime => Dates.format(expire_time, _RFC3339_MS) * "Z")
    isnothing(ttl) && return Dict{Symbol,Any}()
    ttl > 0 || throw(ArgumentError("$verb: ttl is the cache's lifetime in seconds, > 0 (got $ttl)"))
    Dict{Symbol,Any}(:ttl => string(ttl, "s"))
end

# The Chat fields a cache holds: its model, its messages — the system instruction and the contents — and its tools
# with their config. A field that shapes a reply belongs to the requests that name the cache.
const _CACHE_CHAT_FIELDS = (:service, :model, :messages, :history, :tools, :tool_choice, :parallel_tool_calls,
                            :_cumulative_cost)

function _cache_body(chat::Chat, ttl, expire_time, display_name)::Dict{Symbol,Any}
    verb = "create_cache"
    _gemini_explicit_cache_model(chat.model, verb)
    isnothing(chat.cached_content) || throw(ArgumentError(
        "$verb: a cache holds contents, not another cache (a CachedContent names none); unset cached_content"))
    set = Symbol[f for f in fieldnames(Chat) if !(f in _CACHE_CHAT_FIELDS) && !isnothing(getfield(chat, f))]
    isempty(set) || throw(ArgumentError("$verb: a cache holds a Chat's model, messages, tools and tool_choice; " *
        "$(join(set, ", ")) shape a reply, and belong to the Chat that names the cache: unset them here"))
    # "Maximum 128 Unicode characters" (CachedContent.displayName).
    isnothing(display_name) || length(display_name) <= 128 || throw(ArgumentError(
        "$verb: display_name takes at most 128 characters (got $(length(display_name)))"))
    for m in chat.messages, a in something(m.attachments, Attachment[])
        _gemini_check_part(a, chat.model)
    end
    sysinstr, contents = _gemini_contents(chat.messages)
    body = Dict{Symbol,Any}(:model => "models/" * chat.model)
    isempty(contents) || (body[:contents] = contents)
    isnothing(sysinstr) || (body[:systemInstruction] = Dict(:parts => [Dict(:text => sysinstr)]))
    isnothing(chat.tools) || (body[:tools] = [Dict(:functionDeclarations => [_gemini_tool(t) for t in chat.tools])])
    (tc = _gemini_tool_config(chat)) === nothing || (body[:toolConfig] = tc)
    isnothing(display_name) || (body[:displayName] = display_name)
    merge!(body, _cache_expiration(verb, ttl, expire_time))
end

"""
    create_cache(chat::Chat; ttl=nothing, expire_time=nothing, display_name=nothing, config=nothing, cancel=nothing)

Cache the start of a prompt on Gemini (cachedContents.create): `chat`'s system messages (the system
instruction), its other messages (the contents, attachments included), its `tools` and `tool_choice`
(the tool config), for `chat.model`. A request then names the cache with a [`Chat`](@ref)'s
`cached_content` — the result's `name` or the `CacheObject` itself — and holds only what follows: no
system message, no tools, no tool_choice. Returns [`CacheSuccess`](@ref), [`CacheFailure`](@ref) or
[`CacheCallError`](@ref).

- `ttl`: the cache's lifetime in seconds; or `expire_time`, a UTC `DateTime` — one of them, else
  `ArgumentError`. Unset, Gemini keeps a cache for an hour.
- `display_name`: at most 128 characters.

Throws `ArgumentError` before any request when `chat.service` lacks `:context_caching`
(`GEMINIServiceEndpoint` has it), for a Gemini 4 model, which Google does not cache explicitly, and for a
`chat` that sets a field a cache does not hold — `max_tokens`, `reasoning_effort`, `stream`,
`response_format`, `cached_content` and the rest — which belong to the `Chat` that names the cache.
Gemini refuses a cache below the model's minimum size (HTTP 400 "Cached content is too small. …
min_total_token_count=1024" on `gemini-3.8-flash`, 2026-11-17).

A cache bills its storage by the hour ([Context caching](@ref gemini_caching)), which
[`estimated_cost`](@ref) does not price; a reply that reads it bills the cached tokens at the model's
cache-hit rate. `config` bounds the call (a single attempt: a create is never retried); `cancel`
(default: the ambient [`with_cancel`](@ref) token) ends it with a [`UniLMCancelled`](@ref) in `cause`.
"""
function create_cache(chat::Chat; ttl::Union{Integer,Nothing}=nothing, expire_time::Union{DateTime,Nothing}=nothing,
                      display_name::Union{AbstractString,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing,
                      cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(chat.service, :context_caching, "Context caching")
    body = _cache_body(chat, ttl, expire_time, display_name)
    _cache_call(_cache_success, "POST", _gemini_url(chat.service, "cachedContents"), chat.service, config, cancel; body)
end

"""
    retrieve_cache(name; service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

A cache's metadata (cachedContents.get), by its name `"cachedContents/<id>"` or its id. Returns
[`CacheSuccess`](@ref), [`CacheFailure`](@ref) — a cache that expired or was deleted is Gemini's 403 — or
[`CacheCallError`](@ref). A malformed name throws `ArgumentError` before any request.
"""
function retrieve_cache(name::AbstractString; service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                        config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :context_caching, "Context caching")
    url = _gemini_url(service, _cache_reference(name))
    _cache_call(_cache_success, "GET", url, service, config, cancel)
end

"""
    list_caches(; limit=nothing, after=nothing, service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

A page of the project's caches (cachedContents.list): `limit` is the page size (`pageSize`; Gemini serves
at most 1000, "values above 1000 will be coerced to 1000"), `after` the previous page's
`raw["nextPageToken"]`. Returns [`CacheListSuccess`](@ref), [`CacheFailure`](@ref) or
[`CacheCallError`](@ref). A `limit` below 1 throws `ArgumentError` before any request.
"""
function list_caches(; limit::Union{Integer,Nothing}=nothing, after::Union{AbstractString,Nothing}=nothing,
                     service::ServiceEndpointSpec=GEMINIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
                     cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :context_caching, "Context caching")
    isnothing(limit) || limit >= 1 || throw(ArgumentError("list_caches: limit is a page size >= 1 (got $limit)"))
    url = _gemini_url(service, "cachedContents") * _gemini_query("pageSize" => limit, "pageToken" => after)
    _cache_call(_cache_page, "GET", url, service, config, cancel)
end

"""
    update_cache(name; ttl=nothing, expire_time=nothing, service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

Move a cache's expiration (cachedContents.patch): a new `ttl` in seconds from now, or a new
`expire_time` (UTC) — one of them, else `ArgumentError` before any request: "only expiration is
updatable". Returns [`CacheSuccess`](@ref) with the updated cache, [`CacheFailure`](@ref) or
[`CacheCallError`](@ref).
"""
function update_cache(name::AbstractString; ttl::Union{Integer,Nothing}=nothing, expire_time::Union{DateTime,Nothing}=nothing,
                      service::ServiceEndpointSpec=GEMINIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
                      cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :context_caching, "Context caching")
    body = _cache_expiration("update_cache", ttl, expire_time)
    isempty(body) && throw(ArgumentError(
        "update_cache: only a cache's expiration updates (\"only expiration is updatable\"): set ttl or expire_time"))
    url = _gemini_url(service, _cache_reference(name)) * _gemini_query("updateMask" => String(only(keys(body))))
    _cache_call(_cache_success, "PATCH", url, service, config, cancel; body)
end

"""
    delete_cache(name; service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

Delete a cache before it expires (cachedContents.delete), by its name or id. Returns
[`CacheDeleteSuccess`](@ref), [`CacheFailure`](@ref) or [`CacheCallError`](@ref).
"""
function delete_cache(name::AbstractString; service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                      config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :context_caching, "Context caching")
    full = _cache_reference(name)
    # The API answers an empty message: its 200 is the confirmation, and the cache deleted the one named.
    _cache_call(x -> (_gemini_object(x, "the delete reply"); CacheDeleteSuccess(name=full, deleted=true)),
                "DELETE", _gemini_url(service, full), service, config, cancel)
end
