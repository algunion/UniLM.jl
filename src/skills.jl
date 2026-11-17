# ============================================================================
# Skills API — upload a skill (a folder whose root holds a SKILL.md, or a .zip of one), version
# it, read it back, download it and delete it, on each provider's own Skills API:
#   OpenAI     /v1/skills  https://developers.openai.com/api/docs/guides/tools-skills
#   Anthropic  /v1/skills  https://platform.claude.com/docs/en/api/skills/create
# An OpenAI skill loads into a container by reference (`SkillReference`, containers.jl).
# ============================================================================

const _SKILLS_PATH = "/v1/skills"
const _SK = "Skills API"

# ─── The provider wires ──────────────────────────────────────────────────────
# Where a provider keeps its skills, how its records read, and which options it takes. An endpoint
# not routed here speaks OpenAI's wire.
abstract type _SkillsWire end
struct _OpenAISkills <: _SkillsWire end
struct _AnthropicSkills <: _SkillsWire end

_skills_wire(service) = _OpenAISkills()
_skills_wire(::_AnthropicEndpoint) = _AnthropicSkills()

_skills_provider(::_OpenAISkills) = "OpenAI"
_skills_provider(::_AnthropicSkills) = "Anthropic"

# The Skills API's address; a function, so tests can route an endpoint to a local server.
_skills_url(service)::String = _api_base_url(service) * _SKILLS_PATH
_skills_url(::_AnthropicEndpoint)::String = ANTHROPIC_BASE_URL * _SKILLS_PATH

# The headers of a Skills request: the endpoint's own on OpenAI's wire. Anthropic's Skills reference
# names `anthropic-version` and `anthropic-workspace-id` (https://platform.claude.com/docs/en/api/skills/create,
# Headers), and its SDK (v2.9.0) sends no beta from `client.skills` or, unless asked, `client.beta.skills`:
# the API "is out of beta and needs no beta header", and `skills-2025-10-02` would select the beta's
# response shapes (https://platform.claude.com/docs/en/build-with-claude/skills-guide). An Anthropic Skills
# request sends the key, the API version and the endpoint's workspace, never its user profile or its betas.
_skills_headers(service) = auth_header(service)
_skills_headers(service::_AnthropicEndpoint)::Vector{Pair{String,String}} =
    _anthropic_base_headers(_anthropic_endpoint(service); profile=false)
# An upload's form sets its own Content-Type, with the boundary.
_skills_upload_headers(service) = auth_header_multipart(service)
_skills_upload_headers(service::_AnthropicEndpoint) = _skills_headers(service)

# The type tag of each record: OpenAI's `object`, Anthropic's `type`.
_skills_tag(::_OpenAISkills) = "object"
_skills_tag(::_AnthropicSkills) = "type"
_skills_kinds(::_OpenAISkills) =
    (skill="skill", version="skill.version", deleted="skill.deleted", version_deleted="skill.version.deleted")
_skills_kinds(::_AnthropicSkills) =
    (skill="skill", version="skill_version", deleted="skill_deleted", version_deleted="skill_version_deleted")

function _check_skills_kind(w::_SkillsWire, d::AbstractDict, kind::String)::Nothing
    got = _reply_required(d, _skills_tag(w), String, _SK)
    got == kind || throw(ArgumentError("the $_SK reply is a $(repr(got)) record, not the $(repr(kind)) this call answers with"))
    nothing
end

# ─── Objects ──────────────────────────────────────────────────────────────────

"""
    SkillObject

A skill stored by a provider's Skills API, as [`create_skill`](@ref), [`retrieve_skill`](@ref),
[`update_skill`](@ref) and [`list_skills`](@ref) return it; `raw` holds it as the provider sent it.

- `id::String`: the id the other Skills verbs take, and [`SkillReference`](@ref) on OpenAI.
- `name::String`: OpenAI's `name`; Anthropic's `display_name`, from the SKILL.md's `name` unless
  `create_skill` set one.
- `description::Union{String,Nothing}`: OpenAI's `description`; `nothing` on Anthropic, whose
  versions carry theirs.
- `default_version::Union{String,Nothing}`: OpenAI's `default_version`, the version a reference
  without one loads; `nothing` on Anthropic, which has none.
- `latest_version::String`: the newest version — OpenAI's `latest_version`, Anthropic's
  `latest_version_id`.
- `created_at::Int`: the creation time in Unix seconds — OpenAI's `created_at`; Anthropic's, an
  RFC 3339 time, to the second.

The rest is in `raw`: Anthropic's `source` (`{"type": "custom"}` for a skill of yours) and
`updated_at`. A reply without a field the provider requires, or naming another kind of record, is
no skill: the call returns a [`SkillCallError`](@ref).
"""
@kwdef struct SkillObject
    id::String
    name::String
    description::Union{String,Nothing} = nothing
    default_version::Union{String,Nothing} = nothing
    latest_version::String
    created_at::Int
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    SkillVersionObject

One version of a skill, as [`create_skill_version`](@ref), [`retrieve_skill_version`](@ref) and
[`list_skill_versions`](@ref) return it; `raw` holds it as the provider sent it.

- `id::String`: the version's id.
- `skill_id::String`: the skill's id.
- `version::String`: what the version verbs, [`skill_content`](@ref) and a reference take — OpenAI's
  `version` number (`"1"`, `"2"`, …), Anthropic's `id`.
- `name::String`, `description::String`: from the version's SKILL.md.
- `created_at::Int`: the creation time in Unix seconds.
"""
@kwdef struct SkillVersionObject
    id::String
    skill_id::String
    version::String
    name::String
    description::String
    created_at::Int
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    SkillList

A page of [`SkillObject`](@ref)s from [`list_skills`](@ref). `has_more` signals that further pages
are available, and `after` is the cursor to pass as `after` for the next page — OpenAI's
`last_id`, Anthropic's `next_page` —, `nothing` on the last page; `raw` holds the page as sent.
"""
@kwdef struct SkillList
    data::Vector{SkillObject}
    has_more::Bool
    after::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    SkillVersionList

A page of [`SkillVersionObject`](@ref)s from [`list_skill_versions`](@ref), read as a
[`SkillList`](@ref) is.
"""
@kwdef struct SkillVersionList
    data::Vector{SkillVersionObject}
    has_more::Bool
    after::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

# ─── Result types ─────────────────────────────────────────────────────────────

"Successful [`create_skill`](@ref), [`retrieve_skill`](@ref) or [`update_skill`](@ref) result wrapping a [`SkillObject`](@ref)."
@kwdef struct SkillSuccess <: LLMRequestResponse; response::SkillObject; end
"Successful [`list_skills`](@ref) result wrapping a [`SkillList`](@ref)."
@kwdef struct SkillListSuccess <: LLMRequestResponse; response::SkillList; end
"Successful [`create_skill_version`](@ref) or [`retrieve_skill_version`](@ref) result wrapping a [`SkillVersionObject`](@ref)."
@kwdef struct SkillVersionSuccess <: LLMRequestResponse; response::SkillVersionObject; end
"Successful [`list_skill_versions`](@ref) result wrapping a [`SkillVersionList`](@ref)."
@kwdef struct SkillVersionListSuccess <: LLMRequestResponse; response::SkillVersionList; end
"Successful [`delete_skill`](@ref) or [`delete_skill_version`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of `id`, the id its reply names."
@kwdef struct SkillDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Skills API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct SkillFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Skills API call that produced no usable reply (transport failure, timeout, cancel, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel."
@kwdef struct SkillCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Decoding ─────────────────────────────────────────────────────────────────

# An RFC 3339 time the reference requires, in Unix seconds.
function _skills_time(d::AbstractDict, key::String)::Int
    s = _reply_required(d, key, String, _SK)
    t = @something(_rfc3339_utc(s), throw(ArgumentError("$_SK reply field $(repr(key)) is not an RFC 3339 date-time: $(repr(s))")))
    floor(Int, Dates.datetime2unix(t))
end

_skill_req(d::AbstractDict, key::String, ::Type{T}=String) where {T} = _reply_required(d, key, T, _SK)

# OpenAI's SkillResource requires every field read here; Anthropic's Skill its id, display_name,
# latest_version_id and created_at.
function _parse_skill(w::_OpenAISkills, d::AbstractDict)::SkillObject
    _check_skills_kind(w, d, _skills_kinds(w).skill)
    SkillObject(id=_skill_req(d, "id"), name=_skill_req(d, "name"), description=_skill_req(d, "description"),
                default_version=_skill_req(d, "default_version"), latest_version=_skill_req(d, "latest_version"),
                created_at=_skill_req(d, "created_at", Int), raw=Dict{String,Any}(d))
end
function _parse_skill(w::_AnthropicSkills, d::AbstractDict)::SkillObject
    _check_skills_kind(w, d, _skills_kinds(w).skill)
    SkillObject(id=_skill_req(d, "id"), name=_skill_req(d, "display_name"), latest_version=_skill_req(d, "latest_version_id"),
                created_at=_skills_time(d, "created_at"), raw=Dict{String,Any}(d))
end

# A version: OpenAI's path takes its `version` number, Anthropic's its `id`.
function _parse_skill_version(w::_SkillsWire, d::AbstractDict)::SkillVersionObject
    _check_skills_kind(w, d, _skills_kinds(w).version)
    id = _skill_req(d, "id")
    SkillVersionObject(id=id, skill_id=_skill_req(d, "skill_id"), version=w isa _OpenAISkills ? _skill_req(d, "version") : id,
                       name=_skill_req(d, "name"), description=_skill_req(d, "description"),
                       created_at=w isa _OpenAISkills ? _skill_req(d, "created_at", Int) : _skills_time(d, "created_at"),
                       raw=Dict{String,Any}(d))
end

# A page's entries, whether more follow, and the cursor of the next page. OpenAI's list requires `data`,
# `has_more` and `last_id` (null on an empty page); Anthropic's has `data` and `next_page`, its opaque
# cursor, null (or empty, as its SDK reads it) on the last page.
function _skills_page(::_OpenAISkills, d::AbstractDict)
    more = _skill_req(d, "has_more", Bool)
    (_skill_req(d, "data", AbstractVector), more, more ? _skill_req(d, "last_id") : nothing)
end
function _skills_page(::_AnthropicSkills, d::AbstractDict)
    next = _reply_field(d, "next_page", String, _SK)
    after = isnothing(next) || isempty(next) ? nothing : next
    (_skill_req(d, "data", AbstractVector), !isnothing(after), after)
end

# A deletion record: OpenAI's names its `object` and its `id` and confirms `"deleted": true`; Anthropic's
# names its `type` and the `id`, its 200 the confirmation.
_skills_deleted(::_OpenAISkills, d::AbstractDict, kind::String) = _vs_deletion(SkillDeleteSuccess, d, kind)
function _skills_deleted(w::_AnthropicSkills, d::AbstractDict, kind::String)
    _check_skills_kind(w, d, kind)
    SkillDeleteSuccess(id=_skill_req(d, "id"), deleted=true)
end

# ─── Uploads ──────────────────────────────────────────────────────────────────

# OpenAI takes at most 500 files in a skill (CreateSkillBody and CreateSkillVersionBody: `maxItems: 500`);
# Anthropic names no count.
_check_skill_file_count(::_AnthropicSkills, ::String, ::Int)::Nothing = nothing
_check_skill_file_count(::_OpenAISkills, verb::String, n::Int)::Nothing =
    n <= 500 ? nothing : throw(ArgumentError("$verb: OpenAI takes at most 500 files in a skill (got $n)"))

# The part a single zip goes as: OpenAI's guide uploads it as `files`, Anthropic's SDK sends every file
# as a `files[]` part.
_skill_zip_part(::_OpenAISkills) = "files"
_skill_zip_part(::_AnthropicSkills) = "files[]"

const _SkillPart = @NamedTuple{name::String, filename::String, path::String, type::String}

# The parts of an upload, checked before anything is read or sent: a folder's files, each a `files[]` part
# named by its path under the folder's name — both providers take a skill as the files of one top-level
# folder whose root holds its SKILL.md — or one `.zip` of such a folder.
function _skill_parts(verb::String, w::_SkillsWire, path::AbstractString)::Vector{_SkillPart}
    if isdir(path)
        any(n -> lowercase(n) == "skill.md" && isfile(joinpath(path, n)), readdir(path)) || throw(ArgumentError(
            "$verb: the folder $path has no SKILL.md at its root, which a skill's folder holds"))
        root = last(splitpath(abspath(path)))
        files = sort!([joinpath(dir, f) for (dir, _, fs) in walkdir(path) for f in fs])
        _check_skill_file_count(w, verb, length(files))
        _SkillPart[(name="files[]", filename=join([root; splitpath(relpath(f, path))], "/"), path=f, type=_mime_for(f))
                   for f in files]
    elseif isfile(path) && lowercase(splitext(path)[2]) == ".zip"
        _SkillPart[(name=_skill_zip_part(w), filename=basename(path), path=String(path), type="application/zip")]
    else
        throw(ArgumentError("$verb: $path is no skill folder or .zip file"))
    end
end

_skill_form(parts::Vector{_SkillPart}, fields::Vector{Pair{String,Any}}) =
    HTTP.Form(append!(copy(fields),
        Pair{String,Any}[p.name => HTTP.Multipart(p.filename, IOBuffer(read(p.path)), p.type) for p in parts]))

# ─── The exchange ─────────────────────────────────────────────────────────────

# An option the provider does not take, refused by name before anything is sent.
_skills_refuse(verb::String, w::_SkillsWire, option::String, why::String) =
    throw(ArgumentError("$verb: $(_skills_provider(w))'s Skills API takes no $option$why; unset $option"))

# A route below /v1/skills; an empty id or version would name another route, so it is refused.
function _skill_route(verb::String, skill_id::AbstractString)::String
    isempty(skill_id) && throw(ArgumentError("$verb needs a skill id; list_skills lists them"))
    "/" * _uripart(skill_id)
end
function _skill_version_route(verb::String, skill_id::AbstractString, version::Union{AbstractString,Integer})::String
    v = string(version)
    isempty(v) && throw(ArgumentError("$verb needs a version; list_skill_versions lists them"))
    _skill_route(verb, skill_id) * "/versions/" * _uripart(v)
end

# One exchange with the Skills API: `route` below its base, sent with `json` as its body, or with the form of
# `upload` and `fields` (read only once the token is checked: a cancelled create reads no file); a 200 goes to
# `decode`, another status is the SkillFailure, and anything thrown the SkillCallError.
function _skills_call(decode::Function, method::String, route::String, service, config::Union{Nothing,RequestConfig},
                      cancel::Union{Nothing,CancelToken}; json::Union{AbstractDict,Nothing}=nothing,
                      upload::Union{Vector{_SkillPart},Nothing}=nothing, fields::Vector{Pair{String,Any}}=Pair{String,Any}[])
    cfg = _resolve_config(config); t0 = time_ns(); tok = _resolve_cancel(cancel)
    try
        url = _skills_url(service) * route
        resp = if !isnothing(upload)
            iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
            form = _skill_form(upload, fields)
            _http(method, url, _skills_upload_headers(service), form; cfg, remaining=_remaining_s(cfg, t0), cancel=tok)
        elseif !isnothing(json)
            _http(method, url, _skills_headers(service), JSON.json(json); cfg, remaining=_remaining_s(cfg, t0), cancel=tok)
        else
            _http(method, url, _skills_headers(service); cfg, remaining=_remaining_s(cfg, t0), cancel=tok)
        end
        resp.status == 200 ? decode(resp) : _failure(SkillFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(SkillCallError, e)
    end
end

_skill_success(w::_SkillsWire) = r -> SkillSuccess(response=_parse_skill(w, _json_object(r.body)))
_skill_version_success(w::_SkillsWire) = r -> SkillVersionSuccess(response=_parse_skill_version(w, _json_object(r.body)))

# ─── Requests ─────────────────────────────────────────────────────────────────

"""
    create_skill(path; display_name=nothing, service=OPENAIServiceEndpoint)

Upload a skill: `path` is its folder, whose root holds its SKILL.md (the skill's `name` and
`description` in its front matter, then its instructions), or a `.zip` of that folder. A folder goes
as one part per file — every file in it, its subfolders' too — each named by its path under the
folder's name; a zip as one part. Returns [`SkillSuccess`](@ref), [`SkillFailure`](@ref) or
[`SkillCallError`](@ref).

- `display_name` (Anthropic only): the skill's label, up to 255 characters on one line; Anthropic
  takes the SKILL.md's `name` without one.

Refused with `ArgumentError` before anything is sent: a `path` that is no folder or `.zip` file, a
folder without a SKILL.md at its root, more than 500 files on OpenAI, `display_name` on OpenAI, and a
`display_name` longer than 255 characters or of more than one line.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single
bounded attempt; `max_attempts` does not apply): like every create, an upload is never sent again.

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the
call cancellable: once the token is cancelled it returns a `SkillCallError` whose `cause` is a
[`UniLMCancelled`](@ref) — without reading a file or sending anything when the token was already
cancelled, at once when mid-request.
"""
function create_skill(path::AbstractString; display_name::Union{AbstractString,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    isnothing(display_name) || _check_display_name(w, display_name)
    parts = _skill_parts("create_skill", w, path)
    fields = isnothing(display_name) ? Pair{String,Any}[] : Pair{String,Any}["display_name" => String(display_name)]
    _skills_call(_skill_success(w), "POST", "", service, config, cancel; upload=parts, fields)
end

# Anthropic's `display_name`: "Human-readable, single-line label for the Skill. Maximum 255 characters."
function _check_display_name(::_AnthropicSkills, name::AbstractString)::Nothing
    length(name) <= 255 && !occursin(r"[\r\n]", name) && return nothing
    throw(ArgumentError("create_skill: display_name is a single line of at most 255 characters (got $(length(name)))"))
end
_check_display_name(w::_OpenAISkills, ::AbstractString) =
    _skills_refuse("create_skill", w, "display_name", ": an OpenAI skill is named by its SKILL.md")

"""
    list_skills(; limit=nothing, after=nothing, order=nothing, source=nothing, service=OPENAIServiceEndpoint)

List the skills, a page at a time: a [`SkillListSuccess`](@ref) whose [`SkillList`](@ref) holds the
page's skills and the `after` of the next one.

- `limit`: the page's size (OpenAI 0 to 100; Anthropic 1 to 1000, default 20).
- `after`: where the page starts, the previous page's `after`.
- `order` (OpenAI only): `"asc"` or `"desc"` by creation time.
- `source` (Anthropic only): `"custom"` for your skills, `"anthropic"` for Anthropic's.

An `order` outside the two, `order` on Anthropic or `source` on OpenAI is an `ArgumentError` before
anything is sent. `config` and `cancel` work as for [`create_skill`](@ref).
"""
function list_skills(; limit::Union{Int,Nothing}=nothing, after::Union{AbstractString,Nothing}=nothing,
    order::Union{AbstractString,Nothing}=nothing, source::Union{AbstractString,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    query = _skills_list_query("list_skills", w, limit, after, order, source)
    _skills_call("GET", query, service, config, cancel) do r
        d = _json_object(r.body)
        entries, more, next = _skills_page(w, d)
        SkillListSuccess(response=SkillList(data=SkillObject[_parse_skill(w, e) for e in entries],
                                            has_more=more, after=next, raw=d))
    end
end

# A list's query: OpenAI's `limit`, `order` and `after`; Anthropic's `limit`, `page` (the cursor `after`
# carries) and, for skills, `source`.
function _skills_list_query(verb::String, w::_OpenAISkills, limit, after, order, source)::String
    isnothing(source) || _skills_refuse(verb, w, "source", ", which only Anthropic's takes")
    _vs_list_query(verb, "limit" => limit, "order" => order, "after" => after)
end
function _skills_list_query(verb::String, w::_AnthropicSkills, limit, after, order, source)::String
    isnothing(order) || _skills_refuse(verb, w, "order", ", which only OpenAI's takes")
    params = _query("limit" => limit, "page" => after, "source" => source)
    isempty(params) ? "" : "?" * join(params, "&")
end

"""
    retrieve_skill(skill_id; service=OPENAIServiceEndpoint)

The skill `skill_id`: [`SkillSuccess`](@ref), [`SkillFailure`](@ref) (a 404 for an unknown id) or
[`SkillCallError`](@ref). An empty id is an `ArgumentError` before anything is sent, as in every
Skills verb. `config` and `cancel` work as for [`create_skill`](@ref).
"""
function retrieve_skill(skill_id::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    _skills_call(_skill_success(_skills_wire(service)), "GET", _skill_route("retrieve_skill", skill_id), service, config, cancel)
end

"""
    update_skill(skill_id; default_version, service=OPENAIServiceEndpoint)

Set the version of skill `skill_id` that a reference without one loads (OpenAI only): `default_version`
is a version number, `2` or `"2"`, sent as a string. Returns the skill as a [`SkillSuccess`](@ref).
Anthropic has no default version — a request names a version or `"latest"` —, so on
`ANTHROPICServiceEndpoint` the call is an `ArgumentError` before anything is sent.
"""
function update_skill(skill_id::AbstractString; default_version::Union{AbstractString,Integer},
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    w isa _AnthropicSkills && _skills_refuse("update_skill", w, "default_version",
        ": an Anthropic skill has no default version, and a request names a version or \"latest\"")
    route = _skill_route("update_skill", skill_id)
    _skills_call(_skill_success(w), "POST", route, service, config, cancel;
                 json=Dict("default_version" => string(default_version)))
end

"""
    delete_skill(skill_id; service=OPENAIServiceEndpoint)

Delete skill `skill_id` and every version of it: a [`SkillDeleteSuccess`](@ref) when the provider's
deletion record names it — OpenAI's a `"skill.deleted"` object saying `"deleted": true`, Anthropic's a
`"skill_deleted"` —, the [`SkillCallError`](@ref) otherwise.
"""
function delete_skill(skill_id::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    _skills_call(r -> _skills_deleted(w, _json_object(r.body), _skills_kinds(w).deleted), "DELETE",
                 _skill_route("delete_skill", skill_id), service, config, cancel)
end

"""
    skill_content(skill_id; version=nothing, service=OPENAIServiceEndpoint)

Download a skill as the zip of its files: a [`FileContentSuccess`](@ref) of its bytes, which
[`save_file_content`](@ref) writes to disk, else [`SkillFailure`](@ref) or [`SkillCallError`](@ref).
`version` is the version to download (a [`SkillVersionObject`](@ref)'s `version`); without one, OpenAI
serves the skill's default version. Anthropic serves a version only: there, a call without `version`
is an `ArgumentError` before anything is sent.
"""
function skill_content(skill_id::AbstractString; version::Union{AbstractString,Integer,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    isnothing(version) && w isa _AnthropicSkills && throw(ArgumentError(
        "skill_content: Anthropic serves the zip of a version only: pass version, a SkillVersionObject's version " *
        "(the skill's latest_version is its newest)"))
    route = isnothing(version) ? _skill_route("skill_content", skill_id) :
                                 _skill_version_route("skill_content", skill_id, version)
    _skills_call(r -> FileContentSuccess(content=Vector{UInt8}(r.body)), "GET", route * "/content", service, config, cancel)
end

"""
    create_skill_version(skill_id, path; default=nothing, service=OPENAIServiceEndpoint)

Upload a new version of skill `skill_id`, from `path` as [`create_skill`](@ref) takes it: the whole
skill, whose SKILL.md names the skill's name. Returns a [`SkillVersionSuccess`](@ref).

- `default` (OpenAI only): `true` makes the new version the skill's default.

`default` on Anthropic, whose newest version is the skill's `latest_version`, is an `ArgumentError`
before anything is sent, as are the paths `create_skill` refuses.
"""
function create_skill_version(skill_id::AbstractString, path::AbstractString; default::Union{Bool,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    isnothing(default) || w isa _OpenAISkills || _skills_refuse("create_skill_version", w, "default",
        ": its newest version is the skill's latest_version, which a request names as \"latest\"")
    route = _skill_route("create_skill_version", skill_id) * "/versions"
    parts = _skill_parts("create_skill_version", w, path)
    fields = isnothing(default) ? Pair{String,Any}[] : Pair{String,Any}["default" => string(default)]
    _skills_call(_skill_version_success(w), "POST", route, service, config, cancel; upload=parts, fields)
end

"""
    list_skill_versions(skill_id; limit=nothing, after=nothing, order=nothing, service=OPENAIServiceEndpoint)

List the versions of skill `skill_id`, a page at a time, as [`list_skills`](@ref) lists skills (`order`
on OpenAI only, by version number; Anthropic lists the newest first): a
[`SkillVersionListSuccess`](@ref).
"""
function list_skill_versions(skill_id::AbstractString; limit::Union{Int,Nothing}=nothing,
    after::Union{AbstractString,Nothing}=nothing, order::Union{AbstractString,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    route = _skill_route("list_skill_versions", skill_id) * "/versions"
    query = _skills_list_query("list_skill_versions", w, limit, after, order, nothing)
    _skills_call("GET", route * query, service, config, cancel) do r
        d = _json_object(r.body)
        entries, more, next = _skills_page(w, d)
        SkillVersionListSuccess(response=SkillVersionList(
            data=SkillVersionObject[_parse_skill_version(w, e) for e in entries], has_more=more, after=next, raw=d))
    end
end

"""
    retrieve_skill_version(skill_id, version; service=OPENAIServiceEndpoint)

The version `version` of skill `skill_id` (a [`SkillVersionObject`](@ref)'s `version`; Anthropic takes
`"latest"` too): a [`SkillVersionSuccess`](@ref). OpenAI may answer a version just made with a 404 for
a moment ("A newly uploaded version can briefly return 404").
"""
function retrieve_skill_version(skill_id::AbstractString, version::Union{AbstractString,Integer};
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    _skills_call(_skill_version_success(_skills_wire(service)), "GET",
                 _skill_version_route("retrieve_skill_version", skill_id, version), service, config, cancel)
end

"""
    delete_skill_version(skill_id, version; service=OPENAIServiceEndpoint)

Delete one version of skill `skill_id`: a [`SkillDeleteSuccess`](@ref) when the deletion record names it,
read as [`delete_skill`](@ref) reads one. OpenAI refuses to delete the default version while others
remain, and deleting the last deletes the skill; Anthropic refuses to delete a skill's only version.
"""
function delete_skill_version(skill_id::AbstractString, version::Union{AbstractString,Integer};
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :skills, _SK)
    w = _skills_wire(service)
    _skills_call(r -> _skills_deleted(w, _json_object(r.body), _skills_kinds(w).version_deleted), "DELETE",
                 _skill_version_route("delete_skill_version", skill_id, version), service, config, cancel)
end

# ─── A skill into a container ─────────────────────────────────────────────────

"""
    SkillReference(skill::SkillObject; version=nothing)
    SkillReference(version::SkillVersionObject)

The reference [`create_container`](@ref) takes to a skill made with [`create_skill`](@ref) on OpenAI:
the skill at `version` (its default without one), or the version given.
"""
SkillReference(s::SkillObject; version::Union{AbstractString,Integer,Nothing}=nothing) = SkillReference(s.id; version)
SkillReference(v::SkillVersionObject) = SkillReference(v.skill_id; version=v.version)
