# ============================================================================
# OpenAI Containers API — /v1/containers. Sandboxed compute for the Code
# Interpreter tool; expires after idle. Files added via multipart or by the id
# of an uploaded file, and listed, read, downloaded and deleted.
# ============================================================================

"""
    ContainerObject

A code-interpreter container: `id`, `status`, and `name`; `raw` holds the
unparsed JSON response.
"""
@kwdef struct ContainerObject
    id::String
    status::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    ContainerFileObject

A file inside a container, as [`add_container_file`](@ref), [`retrieve_container_file`](@ref) and
[`list_container_files`](@ref) return it: `id`,
`container_id`, `path` (where the code interpreter sees the file), `bytes`,
`source` (e.g. `"user"`, `"assistant"`) and `created_at`; `raw` holds the unparsed
JSON response.
"""
@kwdef struct ContainerFileObject
    id::String
    container_id::String
    path::String
    bytes::Union{Int,Nothing} = nothing
    source::Union{String,Nothing} = nothing
    created_at::Union{Int,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    ContainerList

A page of container records (raw JSON dicts) from [`list_containers`](@ref);
`has_more` signals that further pages are available.
"""
@kwdef struct ContainerList
    data::Vector{Dict{String,Any}}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

# The memory of a container, as OpenAI's reference lists it (`"1g"` its default); the Responses
# code_interpreter tool's auto container takes the same values.
const _CONTAINER_MEMORY_LIMITS = ("1g", "4g", "16g", "64g")
function _check_memory_limit(verb::String, memory_limit::Union{String,Nothing})
    isnothing(memory_limit) || memory_limit in _CONTAINER_MEMORY_LIMITS || throw(ArgumentError(
        "$verb: memory_limit is $(join(repr.(_CONTAINER_MEMORY_LIMITS), ", ", " or ")) (got $(repr(memory_limit)))"))
    return memory_limit
end

"""
    ContainerFileList

A page of [`ContainerFileObject`](@ref)s from [`list_container_files`](@ref); `has_more` signals
that further pages are available.
"""
@kwdef struct ContainerFileList
    data::Vector{ContainerFileObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    ContainerNetworkPolicy

The outbound network access of a container [`create_container`](@ref) makes:
[`ContainerNetworkDisabled`](@ref) or [`ContainerNetworkAllowlist`](@ref).
"""
abstract type ContainerNetworkPolicy end

"""
    ContainerNetworkDisabled()

No outbound network access for the container, sent as `{"type": "disabled"}`.
"""
struct ContainerNetworkDisabled <: ContainerNetworkPolicy end
JSON.lower(::ContainerNetworkDisabled) = Dict{Symbol,Any}(:type => "disabled")

"""
    ContainerDomainSecret(domain, name, value)

A secret OpenAI injects into the container's requests to `domain`, one of its allowlisted
domains, under `name`. Each of the three is a non-empty string, as OpenAI's reference requires,
else `ArgumentError`. The `value` is never printed.
"""
struct ContainerDomainSecret
    domain::String
    name::String
    value::String
    function ContainerDomainSecret(domain::AbstractString, name::AbstractString, value::AbstractString)
        for (field, v) in (("domain", domain), ("name", name), ("value", value))
            isempty(v) && throw(ArgumentError(
                "ContainerDomainSecret: $field is empty; OpenAI's reference requires at least one character"))
        end
        new(domain, name, value)
    end
end
Base.show(io::IO, s::ContainerDomainSecret) =
    print(io, "ContainerDomainSecret(", repr(s.domain), ", ", repr(s.name), ", <value not shown>)")
JSON.lower(s::ContainerDomainSecret) = Dict{Symbol,Any}(:domain => s.domain, :name => s.name, :value => s.value)

"""
    ContainerNetworkAllowlist(allowed_domains; domain_secrets=ContainerDomainSecret[])

Outbound network access for the container to `allowed_domains` only (at least one, else
`ArgumentError`), with optional [`ContainerDomainSecret`](@ref)s for them; sent as
`{"type": "allowlist", "allowed_domains": […], "domain_secrets": […]}`, without
`domain_secrets` when there are none.
"""
struct ContainerNetworkAllowlist <: ContainerNetworkPolicy
    allowed_domains::Vector{String}
    domain_secrets::Vector{ContainerDomainSecret}
    function ContainerNetworkAllowlist(allowed_domains::Vector{String};
                                       domain_secrets::Vector{ContainerDomainSecret}=ContainerDomainSecret[])
        isempty(allowed_domains) && throw(ArgumentError(
            "ContainerNetworkAllowlist: allowed_domains is empty; list at least one domain, " *
            "or use ContainerNetworkDisabled() for no network access"))
        new(allowed_domains, domain_secrets)
    end
end
function JSON.lower(p::ContainerNetworkAllowlist)
    d = Dict{Symbol,Any}(:type => "allowlist", :allowed_domains => p.allowed_domains)
    isempty(p.domain_secrets) || (d[:domain_secrets] = p.domain_secrets)
    d
end

"""
    ContainerSkill

A skill [`create_container`](@ref) loads into the container: a [`SkillReference`](@ref) to a skill
made with [`create_skill`](@ref), or an [`InlineSkill`](@ref), whose zip goes with the request.
"""
abstract type ContainerSkill end

"""
    SkillReference(skill_id; version=nothing)

A skill made with OpenAI's Skills API ([`create_skill`](@ref)), for [`create_container`](@ref)'s
`skills`: its id (1 to 64 characters) and `version`, a positive integer (or its digits) or
`"latest"`; without one, the skill's default version. Anything else is an `ArgumentError`. Sent as
`{"type": "skill_reference", "skill_id": …, "version": …}`.
"""
struct SkillReference <: ContainerSkill
    skill_id::String
    version::Union{String,Nothing}
    function SkillReference(skill_id::AbstractString; version::Union{AbstractString,Integer,Nothing}=nothing)
        1 <= length(skill_id) <= 64 || throw(ArgumentError(
            "SkillReference: skill_id has 1 to 64 characters (got $(length(skill_id)))"))
        v = isnothing(version) ? nothing : string(version)
        isnothing(v) || v == "latest" || occursin(r"^[1-9][0-9]*\z", v) || throw(ArgumentError(
            "SkillReference: version is a positive integer or \"latest\" (got $(repr(version)))"))
        new(skill_id, v)
    end
end
function JSON.lower(s::SkillReference)
    d = Dict{Symbol,Any}(:type => "skill_reference", :skill_id => s.skill_id)
    isnothing(s.version) || (d[:version] = s.version)
    d
end

"""
    InlineSkill(zip; name, description)

A skill sent with [`create_container`](@ref)'s request rather than made beforehand with
[`create_skill`](@ref): `zip` is the path of a `.zip` of the skill's folder (its SKILL.md at the
root), or the zip's bytes. Sent as `{"type": "inline", "name": …, "description": …, "source":
{"type": "base64", "media_type": "application/zip", "data": …}}`, the zip base64-encoded. A path
that is no `.zip` file, and bytes that do not start as a zip archive does (a local file header,
`PK\\x03\\x04`), are an `ArgumentError`. Across a container's inline skills OpenAI takes at most 50 MiB
of zip: `create_container` refuses more before sending. Printed, it shows its name and its size, not
its bytes.
"""
struct InlineSkill <: ContainerSkill
    name::String
    description::String
    zip::Vector{UInt8}
    function InlineSkill(zip::AbstractVector{UInt8}; name::AbstractString, description::AbstractString)
        length(zip) >= 4 && zip[begin:begin+3] == b"PK\x03\x04" || throw(ArgumentError(
            "InlineSkill: the bytes are no zip archive, which starts with a local file header (PK\\x03\\x04); " *
            "pass a .zip of the skill's folder"))
        new(name, description, Vector{UInt8}(zip))
    end
end
function InlineSkill(path::AbstractString; name::AbstractString, description::AbstractString)
    isfile(path) && lowercase(splitext(path)[2]) == ".zip" || throw(ArgumentError(
        "InlineSkill: $path is no .zip file; zip the skill's folder, or upload the folder with create_skill " *
        "and pass a SkillReference"))
    InlineSkill(read(path); name, description)
end
Base.show(io::IO, s::InlineSkill) = print(io, "InlineSkill(", repr(s.name), ", ", length(s.zip), " bytes)")
JSON.lower(s::InlineSkill) = Dict{Symbol,Any}(:type => "inline", :name => s.name, :description => s.description,
    :source => Dict{Symbol,Any}(:type => "base64", :media_type => "application/zip", :data => base64encode(s.zip)))

# Across a list's inline skills "decoded ZIP data must total at most 50 MiB and Base64 text must total at most
# 70,254,592 characters, including line breaks" (OpenAI's API specification, November 2026); UniLM's base64 has
# no line breaks.
function _check_inline_skills(skills::AbstractVector{<:ContainerSkill})::Nothing
    inline = InlineSkill[s for s in skills if s isa InlineSkill]
    bytes = sum(s -> length(s.zip), inline; init=0)
    bytes <= 50 * 2^20 && sum(s -> 4 * cld(length(s.zip), 3), inline; init=0) <= 70_254_592 && return nothing
    throw(ArgumentError("create_container: OpenAI takes at most 50 MiB ($(50 * 2^20) bytes) of zip data across a " *
        "container's inline skills (got $bytes bytes in $(length(inline))); upload a large skill with create_skill " *
        "and pass a SkillReference"))
end

"Successful Containers API result: a [`ContainerObject`](@ref) from create/retrieve, or the [`ContainerFileObject`](@ref) that [`add_container_file`](@ref) created or [`retrieve_container_file`](@ref) read."
@kwdef struct ContainerSuccess <: LLMRequestResponse; response::Union{ContainerObject,ContainerFileObject}; end
"Successful [`list_containers`](@ref) result wrapping a [`ContainerList`](@ref)."
@kwdef struct ContainerListSuccess <: LLMRequestResponse; response::ContainerList; end
"Successful [`list_container_files`](@ref) result wrapping a [`ContainerFileList`](@ref)."
@kwdef struct ContainerFileListSuccess <: LLMRequestResponse; response::ContainerFileList; end
"Successful [`delete_container`](@ref) or [`delete_container_file`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of `id`."
@kwdef struct ContainerDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Containers API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct ContainerFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Containers API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct ContainerCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# OpenAI's reference requires a container's `status`: a reply without it is malformed (a
# KeyError, the verb's *CallError), not one whose status is unknown.
_parse_container(d::AbstractDict) = ContainerObject(id=d["id"], status=d["status"], name=get(d, "name", nothing), raw=Dict{String,Any}(d))
_parse_container_file(d::AbstractDict) = ContainerFileObject(id=d["id"], container_id=d["container_id"], path=d["path"],
    bytes=get(d, "bytes", nothing), source=get(d, "source", nothing), created_at=get(d, "created_at", nothing), raw=Dict{String,Any}(d))
_cont_call(decode::Function, method::String, path::String, service, config, cancel; body=nothing) =
    _vs_json_call(decode, ContainerFailure, ContainerCallError, method,
                  _api_base_url(service) * CONTAINERS_PATH * path, service, config, cancel; body)
_cont_resp(resp) = resp.status == 200 ?
    ContainerSuccess(response=_parse_container(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
    _failure(ContainerFailure, resp)

"""
    create_container(; name, file_ids=nothing, expires_after=nothing, memory_limit=nothing,
                     network_policy=nothing, skills=nothing, service=OPENAIServiceEndpoint)

Create a container. `memory_limit` is `"1g"` (OpenAI's default), `"4g"`, `"16g"` or `"64g"`,
another value an `ArgumentError` before anything is sent; `network_policy` a
[`ContainerNetworkPolicy`](@ref), its outbound access; `skills` the [`ContainerSkill`](@ref)s it
loads: [`SkillReference`](@ref)s to skills made with [`create_skill`](@ref), and
[`InlineSkill`](@ref)s, more than 50 MiB of zip across them an `ArgumentError` before anything is
sent. Returns `ContainerSuccess`, `ContainerFailure`, or `ContainerCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function create_container(; name::String, file_ids::Union{Vector{String},Nothing}=nothing,
    expires_after::Union{AbstractDict,Nothing}=nothing, memory_limit::Union{String,Nothing}=nothing,
    network_policy::Union{ContainerNetworkPolicy,Nothing}=nothing,
    skills::Union{AbstractVector{<:ContainerSkill},Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    _check_memory_limit("create_container", memory_limit)
    isnothing(skills) || _check_inline_skills(skills)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:name => name)
        !isnothing(file_ids) && (d[:file_ids] = file_ids)
        !isnothing(expires_after) && (d[:expires_after] = expires_after)
        !isnothing(memory_limit) && (d[:memory_limit] = memory_limit)
        !isnothing(network_policy) && (d[:network_policy] = network_policy)
        !isnothing(skills) && (d[:skills] = skills)
        _cont_resp(_http("POST", _api_base_url(service) * CONTAINERS_PATH, auth_header(service),
            JSON.json(d); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(ContainerCallError, e)
    end
end

"""
    retrieve_container(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function retrieve_container(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _cont_resp(_http("GET", _api_base_url(service) * CONTAINERS_PATH * "/" * _uripart(id), auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(ContainerCallError, e)
    end
end

"""
    list_containers(; limit=nothing, order=nothing, after=nothing, name=nothing, service=OPENAIServiceEndpoint)

List containers, a page at a time: `order` is `"asc"` or `"desc"` (OpenAI's default) by creation
time, another an `ArgumentError` before anything is sent; `after` is the id that ends the page
before, and `name` keeps the containers of that name.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_containers(; limit::Union{Int,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    after::Union{String,Nothing}=nothing, name::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    query = _vs_list_query("list_containers", "limit" => limit, "order" => order, "after" => after, "name" => name)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * CONTAINERS_PATH * query
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 || return _failure(ContainerFailure, resp)
        data = JSON.parse(resp.body; dicttype=Dict{String,Any})
        page, has_more = _vs_page(data)
        ContainerListSuccess(response=ContainerList(data=Vector{Dict{String,Any}}(page), has_more=has_more, raw=data))
    catch e
        e isa InterruptException && rethrow()
        _callerr(ContainerCallError, e)
    end
end

"""
    delete_container(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function delete_container(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("DELETE", _api_base_url(service) * CONTAINERS_PATH * "/" * _uripart(id), auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 || return _failure(ContainerFailure, resp)
        _vs_deletion(ContainerDeleteSuccess, JSON.parse(resp.body; dicttype=Dict{String,Any}), "container.deleted")
    catch e
        e isa InterruptException && rethrow()
        _callerr(ContainerCallError, e)
    end
end

"""
    add_container_file(container_id, path; service=OPENAIServiceEndpoint)  (multipart upload)

Upload the file at `path` into the container. A success is a `ContainerSuccess` whose
`response` is the created [`ContainerFileObject`](@ref). A `path` that is no file throws
`ArgumentError` before any request, as in [`upload_file`](@ref).

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function add_container_file(container_id::String, path::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    isfile(path) || throw(ArgumentError("file not found: $path"))
    cfg = _resolve_config(config); t0 = time_ns()
    tok = _resolve_cancel(cancel)
    try
        # A cancelled token reads no file: the form holds the whole file, read before `_http` checks the token
        # (and outside the request's time bound).
        iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
        form = HTTP.Form(["file" => HTTP.Multipart(basename(path), IOBuffer(read(path)), _mime_for(path))])
        url = _api_base_url(service) * CONTAINERS_PATH * "/" * _uripart(container_id) * "/files"
        resp = _http("POST", url, auth_header_multipart(service), form; cfg, remaining=_remaining_s(cfg, t0), cancel=tok)
        resp.status == 200 || return _failure(ContainerFailure, resp)
        ContainerSuccess(response=_parse_container_file(JSON.parse(resp.body; dicttype=Dict{String,Any})))
    catch e
        e isa InterruptException && rethrow()
        _callerr(ContainerCallError, e)
    end
end

"""
    add_container_file(container_id; file_id, service=OPENAIServiceEndpoint)

Copy a file uploaded through the Files API ([`upload_file`](@ref)) into the container, by its
id, sent as JSON. A success is a `ContainerSuccess` whose `response` is the created
[`ContainerFileObject`](@ref).

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function add_container_file(container_id::String; file_id::String, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    _cont_call(d -> ContainerSuccess(response=_parse_container_file(d)), "POST", "/" * _uripart(container_id) * "/files",
               service, config, cancel; body=Dict(:file_id => file_id))
end

"""
    list_container_files(container_id; limit=nothing, order=nothing, after=nothing, service=OPENAIServiceEndpoint)

List the files in a container — those added and those its code wrote — a page at a time: `order`
is `"asc"` or `"desc"` (OpenAI's default) by creation time, another an `ArgumentError` before
anything is sent, and `after` the id that ends the page before. Returns
`ContainerFileListSuccess`, `ContainerFailure`, or `ContainerCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_container_files(container_id::String; limit::Union{Int,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    query = _vs_list_query("list_container_files", "limit" => limit, "order" => order, "after" => after)
    _cont_call("GET", "/" * _uripart(container_id) * "/files" * query, service, config, cancel) do d
        page, has_more = _vs_page(d)
        ContainerFileListSuccess(response=ContainerFileList(
            data=ContainerFileObject[_parse_container_file(f) for f in page], has_more=has_more, raw=Dict{String,Any}(d)))
    end
end

"""
    retrieve_container_file(container_id, file_id; service=OPENAIServiceEndpoint)

Read the record of a file in a container: a `ContainerSuccess` whose `response` is its
[`ContainerFileObject`](@ref), `ContainerFailure`, or `ContainerCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function retrieve_container_file(container_id::String, file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    _cont_call(d -> ContainerSuccess(response=_parse_container_file(d)), "GET",
               "/" * _uripart(container_id) * "/files/" * _uripart(file_id), service, config, cancel)
end

"""
    container_file_content(container_id, file_id; service=OPENAIServiceEndpoint)

Download a file in a container — one the code wrote, say — as a [`FileContentSuccess`](@ref) of
its bytes, which [`save_file_content`](@ref) writes to disk; else `ContainerFailure` or
`ContainerCallError`. OpenAI discards a container's files when it expires.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function container_file_content(container_id::String, file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * CONTAINERS_PATH * "/" * _uripart(container_id) * "/files/" * _uripart(file_id) * "/content"
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 ? FileContentSuccess(content=Vector{UInt8}(resp.body)) : _failure(ContainerFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(ContainerCallError, e)
    end
end

"""
    delete_container_file(container_id, file_id; service=OPENAIServiceEndpoint)

Delete a file in a container. Returns `ContainerDeleteSuccess` once the service confirms the
removal, `ContainerFailure`, or `ContainerCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `ContainerCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function delete_container_file(container_id::String, file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :containers, "Containers API")
    _cont_call(d -> _vs_deletion(ContainerDeleteSuccess, d, "container.file.deleted"), "DELETE",
               "/" * _uripart(container_id) * "/files/" * _uripart(file_id), service, config, cancel)
end
