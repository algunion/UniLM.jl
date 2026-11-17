# test/skills.jl — the Skills family (create_skill and the verbs beside it) on each provider's wire, served by
# `_with_scripted` (platform_seam_fixtures.jl).

# OpenAI's Skills wire at the scripted endpoint (SkillsProbe, platform_seam_fixtures.jl).
const _OS = SkillsProbe{UniLM._OpenAISkills}

# A skill folder: its SKILL.md at the root, a file beside it and one in a subfolder.
function _skill_dir(root::String="csv-sum")
    dir = joinpath(mktempdir(), root)
    mkpath(joinpath(dir, "scripts"))
    write(joinpath(dir, "SKILL.md"), "---\nname: csv-sum\ndescription: Sums a CSV column\n---\nSum the column.\n")
    write(joinpath(dir, "notes.txt"), "n")
    write(joinpath(dir, "scripts", "sum.py"), "print(1)")
    dir
end
# The bytes of a zip archive start with a local file header's signature.
const _ZIP_BYTES = UInt8[0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x00]

# A skill and a version as OpenAI's reference shows them (SkillResource, SkillVersionResource).
const _OA_SKILL = """{"id":"skill_abc","object":"skill","name":"csv-sum","description":"Sums a CSV column",
    "created_at":1794859809,"default_version":"1","latest_version":"2"}"""
const _OA_VERSION = """{"object":"skill.version","id":"skillver_abc","skill_id":"skill_abc","version":"2",
    "created_at":1794859810,"name":"csv-sum","description":"Sums a CSV column"}"""

# A multipart part as HTTP.jl writes it.
_part(name, filename, type, content) =
    "Content-Disposition: form-data; name=\"$name\"; filename=\"$filename\"\r\nContent-Type: $type\r\n\r\n$content\r\n"

# OpenAI's directory upload: one `files[]` part per file, each named by its path under the folder's name;
# a zip goes as the one `files` part (https://developers.openai.com/api/docs/guides/tools-skills).
# Refuted if: a part goes under another name, a path loses its folder, a zip goes as `files[]`, or the
# SkillResource decodes to other values.
@testset "OpenAI create_skill: a folder as files[] parts under its name, a zip as files; the skill it answers" begin
    dir = _skill_dir()
    r, seen = _with_scripted(() -> create_skill(dir; service=_OS), (_, _) -> _json(200, _OA_SKILL))
    req = only(seen)
    @test (req.method, req.target) == ("POST", "/v1/skills")
    for p in (_part("files[]", "csv-sum/SKILL.md", "text/markdown", read(joinpath(dir, "SKILL.md"), String)),
              _part("files[]", "csv-sum/notes.txt", "text/plain", "n"),
              _part("files[]", "csv-sum/scripts/sum.py", "application/octet-stream", "print(1)"))
        @test occursin(p, req.body)
    end
    @test count("name=\"files[]\"", req.body) == 3 && !occursin("display_name", req.body)
    @test r isa SkillSuccess
    s = r.response
    @test (s.id, s.name, s.description, s.default_version, s.latest_version, s.created_at) ==
          ("skill_abc", "csv-sum", "Sums a CSV column", "1", "2", 1794859809)
    @test s.raw["object"] == "skill"
    # A trailing slash names the same folder.
    _, seen = _with_scripted(() -> create_skill(dir * "/"; service=_OS), (_, _) -> _json(200, _OA_SKILL))
    @test occursin("filename=\"csv-sum/SKILL.md\"", only(seen).body)
    zip = joinpath(mktempdir(), "csv-sum.zip")
    write(zip, _ZIP_BYTES)
    _, seen = _with_scripted(() -> create_skill(zip; service=_OS), (_, _) -> _json(200, _OA_SKILL))
    @test occursin(_part("files", "csv-sum.zip", "application/zip", String(copy(_ZIP_BYTES))), only(seen).body)
    @test count("Content-Disposition", only(seen).body) == 1
end

# Refused before anything is sent: what neither provider takes as a skill, and what OpenAI does not take.
# Refuted if: any of these reaches the endpoint, or the message does not name the verb and the fix.
@testset "create_skill refuses, before sending, a path that is no skill folder or zip, and what OpenAI does not take" begin
    dir = _skill_dir()
    bare = mktempdir()
    write(joinpath(bare, "README.md"), "x")
    txt = joinpath(mktempdir(), "skill.txt")
    write(txt, "x")
    many = mktempdir()
    write(joinpath(many, "SKILL.md"), "x")
    foreach(i -> write(joinpath(many, "f$i.txt"), "x"), 1:500)
    err, seen = _with_scripted((_, _) -> _json(200, _OA_SKILL)) do
        map([() -> create_skill("/no/such/skill"; service=_OS), () -> create_skill(bare; service=_OS),
             () -> create_skill(txt; service=_OS), () -> create_skill(dir; display_name="CSV", service=_OS),
             () -> create_skill(many; service=_OS), () -> create_skill_version("skill_abc", many; service=_OS)]) do f
            try f(); nothing catch e; e end
        end
    end
    @test isempty(seen) && all(e -> e isa ArgumentError, err)
    @test err[1].msg == "create_skill: /no/such/skill is no skill folder or .zip file"
    @test err[2].msg == "create_skill: the folder $bare has no SKILL.md at its root, which a skill's folder holds"
    @test err[3].msg == "create_skill: $txt is no skill folder or .zip file"
    @test err[4].msg == "create_skill: OpenAI's Skills API takes no display_name: an OpenAI skill is named by its " *
                        "SKILL.md; unset display_name"
    @test err[5].msg == "create_skill: OpenAI takes at most 500 files in a skill (got 501)"
    @test err[6].msg == "create_skill_version: OpenAI takes at most 500 files in a skill (got 501)"
end

# OpenAI's ListSkills and ListSkillVersions: `limit`, `order` (asc or desc) and `after`; a SkillListResource
# requires `data`, `has_more`, `first_id` and `last_id`, the cursor of the next page.
# Refuted if: the query differs, an order outside the enum or Anthropic's `source` is sent, a page without
# `data` or `has_more` decodes, or `after` is not the last id while has_more is true.
@testset "OpenAI list_skills and list_skill_versions: limit, order, after; the page and its cursor" begin
    page(item, more, last) = """{"object":"list","data":[$item],"first_id":"x","last_id":$last,"has_more":$more}"""
    r, seen = _with_scripted(() -> list_skills(; limit=2, order="asc", after=_HOSTILE_ID, service=_OS),
                             (_, _) -> _json(200, page(_OA_SKILL, true, "\"skill_abc\"")))
    @test only(seen).target == "/v1/skills?limit=2&order=asc&after=$_HOSTILE_ENC"
    @test r isa SkillListSuccess && r.response isa SkillList && r.response.has_more && r.response.after == "skill_abc"
    @test only(r.response.data).id == "skill_abc" && r.response.raw["first_id"] == "x"
    done = _answered(() -> list_skills(; service=_OS), 200; body=page(_OA_SKILL, false, "\"skill_abc\""))
    @test done isa SkillListSuccess && !done.response.has_more && isnothing(done.response.after)
    v, seen = _with_scripted(() -> list_skill_versions("skill_abc"; limit=1, order="desc", after="skillver_a", service=_OS),
                             (_, _) -> _json(200, page(_OA_VERSION, false, "null")))
    @test only(seen).target == "/v1/skills/skill_abc/versions?limit=1&order=desc&after=skillver_a"
    @test v isa SkillVersionListSuccess && v.response isa SkillVersionList && !v.response.has_more
    @test only(v.response.data).version == "2"
    for body in ("{}", """{"object":"list","data":[]}""", """{"object":"list","has_more":false}""",
                 page(_OA_SKILL, true, "null"))
        @test _answered(() -> list_skills(; service=_OS), 200; body) isa SkillCallError
    end
    @test _answered(() -> list_skill_versions("s"; service=_OS), 200; body=page(_OA_SKILL, false, "null")) isa SkillCallError
    err, seen = _with_scripted((_, _) -> _json(200, "{}")) do
        map([() -> list_skills(; order="up", service=_OS), () -> list_skills(; source="custom", service=_OS),
             () -> list_skill_versions("s"; order="up", service=_OS)]) do f
            try f(); nothing catch e; e end
        end
    end
    @test isempty(seen) && all(e -> e isa ArgumentError, err)
    @test err[1].msg == "list_skills: order is \"asc\" or \"desc\" (got \"up\")"
    @test err[2].msg == "list_skills: OpenAI's Skills API takes no source, which only Anthropic's takes; unset source"
    @test err[3].msg == "list_skill_versions: order is \"asc\" or \"desc\" (got \"up\")"
end

# OpenAI's GetSkill, UpdateSkillDefaultVersion (a JSON body), DeleteSkill and the version routes.
# Refuted if: a method or a route differs, an id leaves its path segment, the default version goes as a
# number, or a deletion record of another kind, without its id or without "deleted": true reports a delete.
@testset "OpenAI retrieve, update, delete; a version's create, retrieve and delete" begin
    E = _HOSTILE_ENC
    dir = _skill_dir()
    sent = last(_with_scripted((_, _) -> _json(200, "{}")) do
        retrieve_skill(_HOSTILE_ID; service=_OS)
        update_skill(_HOSTILE_ID; default_version=2, service=_OS)
        delete_skill(_HOSTILE_ID; service=_OS)
        create_skill_version(_HOSTILE_ID, dir; default=true, service=_OS)
        retrieve_skill_version(_HOSTILE_ID, _HOSTILE_ID; service=_OS)
        delete_skill_version(_HOSTILE_ID, 2; service=_OS)
        skill_content(_HOSTILE_ID; service=_OS)
        skill_content(_HOSTILE_ID; version=_HOSTILE_ID, service=_OS)
    end)
    @test [(r.method, r.target) for r in sent] ==
          [("GET", "/v1/skills/$E"), ("POST", "/v1/skills/$E"), ("DELETE", "/v1/skills/$E"),
           ("POST", "/v1/skills/$E/versions"), ("GET", "/v1/skills/$E/versions/$E"), ("DELETE", "/v1/skills/$E/versions/2"),
           ("GET", "/v1/skills/$E/content"), ("GET", "/v1/skills/$E/versions/$E/content")]
    @test JSON.parse(sent[2].body) == Dict("default_version" => "2")
    @test occursin("Content-Disposition: form-data; name=\"default\"\r\n\r\ntrue\r\n", sent[4].body)
    @test occursin("filename=\"csv-sum/SKILL.md\"", sent[4].body)
    _, seen = _with_scripted(() -> create_skill_version("skill_abc", dir; service=_OS), (_, _) -> _json(200, _OA_VERSION))
    @test !occursin("name=\"default\"", only(seen).body)
    g = _answered(() -> retrieve_skill("skill_abc"; service=_OS), 200; body=_OA_SKILL)
    @test g isa SkillSuccess && g.response.latest_version == "2"
    u = _answered(() -> update_skill("skill_abc"; default_version="2", service=_OS), 200; body=_OA_SKILL)
    @test u isa SkillSuccess && u.response.default_version == "1"
    cv = _answered(() -> create_skill_version("skill_abc", dir; service=_OS), 200; body=_OA_VERSION)
    @test cv isa SkillVersionSuccess
    ver = cv.response
    @test (ver.id, ver.skill_id, ver.version, ver.name, ver.description, ver.created_at) ==
          ("skillver_abc", "skill_abc", "2", "csv-sum", "Sums a CSV column", 1794859810)
    got = _answered(() -> retrieve_skill_version("skill_abc", "2"; service=_OS), 200; body=_OA_VERSION).response
    @test (got.id, got.version, got.raw) == (ver.id, ver.version, ver.raw)
    del(body) = _answered(() -> delete_skill("skill_abc"; service=_OS), 200; body)
    @test del("""{"object":"skill.deleted","deleted":true,"id":"skill_abc"}""") == SkillDeleteSuccess(id="skill_abc", deleted=true)
    for body in ("""{"object":"skill.deleted","deleted":false,"id":"skill_abc"}""", """{"object":"skill.deleted","id":"skill_abc"}""",
                 """{"object":"skill.deleted","deleted":true}""", """{"object":"skill.version.deleted","deleted":true,"id":"skill_abc"}""")
        @test del(body) isa SkillCallError
    end
    vdel(body) = _answered(() -> delete_skill_version("skill_abc", "1"; service=_OS), 200; body)
    @test vdel("""{"object":"skill.version.deleted","deleted":true,"id":"skillver_1","version":"1"}""") ==
          SkillDeleteSuccess(id="skillver_1", deleted=true)
    @test vdel("""{"object":"skill.deleted","deleted":true,"id":"skill_abc"}""") isa SkillCallError
end

# A reply without a field OpenAI's reference requires, of another JSON type, or of another kind is no skill.
# Refuted if: such a reply decodes as a success.
@testset "OpenAI: a skill or a version without what the reference requires is a SkillCallError" begin
    full = JSON.parse(_OA_SKILL)
    for key in ("id", "object", "name", "description", "created_at", "default_version", "latest_version")
        r = _answered(() -> retrieve_skill("skill_abc"; service=_OS), 200; body=JSON.json(delete!(copy(full), key)))
        @test r isa SkillCallError && occursin(key, r.error)
    end
    @test _answered(() -> retrieve_skill("s"; service=_OS), 200; body=JSON.json(merge(full, Dict("created_at" => "1")))) isa SkillCallError
    r = _answered(() -> retrieve_skill("s"; service=_OS), 200; body=_OA_VERSION)
    @test r isa SkillCallError && occursin("\"skill.version\"", r.error)
    fullv = JSON.parse(_OA_VERSION)
    for key in ("id", "object", "skill_id", "version", "created_at", "name", "description")
        r = _answered(() -> retrieve_skill_version("s", "2"; service=_OS), 200; body=JSON.json(delete!(copy(fullv), key)))
        @test r isa SkillCallError && occursin(key, r.error)
    end
    @test _answered(() -> retrieve_skill_version("s", "2"; service=_OS), 200; body=_OA_SKILL) isa SkillCallError
end

# A skill's zip, at the skill's default version or at a version, as file_content returns a file's bytes.
# Refuted if: the bytes differ, the save is not the Files one, or an error status is not a SkillFailure with its id.
@testset "skill_content: the zip's bytes, saved as a Files download is; a failure keeps its request id" begin
    c = first(_with_scripted(() -> skill_content("skill_abc"; version="2", service=_OS),
                             (_, _) -> HTTP.Response(200, ["Content-Type" => "application/zip"], _ZIP_BYTES)))
    @test c isa FileContentSuccess && c.content == _ZIP_BYTES
    path = tempname()
    try
        @test save_file_content(c, path) == path && read(path) == _ZIP_BYTES
    finally
        rm(path; force=true)
    end
    r = _answered(() -> skill_content("skill_x"; service=_OS), 404; headers=["x-request-id" => "req_skill"])
    @test r isa SkillFailure && r.status == 404 && r.request_id == "req_skill"
end

# Refuted if: an empty id or version is sent (it would name another route), a verb ignores its config or its
# cancel, a cancelled create reads its files, cancel is not the last keyword, or an endpoint without :skills sends.
@testset "the Skills verbs: empty ids, the config seam, cancel, the capability" begin
    dir = _skill_dir()
    calls = [(; kw...) -> create_skill(dir; service=_OS, kw...), (; kw...) -> list_skills(; service=_OS, kw...),
             (; kw...) -> retrieve_skill("s"; service=_OS, kw...), (; kw...) -> update_skill("s"; default_version="1", service=_OS, kw...),
             (; kw...) -> delete_skill("s"; service=_OS, kw...), (; kw...) -> skill_content("s"; service=_OS, kw...),
             (; kw...) -> create_skill_version("s", dir; service=_OS, kw...), (; kw...) -> list_skill_versions("s"; service=_OS, kw...),
             (; kw...) -> retrieve_skill_version("s", "1"; service=_OS, kw...), (; kw...) -> delete_skill_version("s", "1"; service=_OS, kw...)]
    for call in calls
        @test _seam_timeout(call(config=_TINY_DEADLINE), SkillCallError)
    end
    tok = cancel!(CancelToken())
    for call in calls
        r, seen = _with_scripted(() -> call(cancel=tok), (_, _) -> _json(200, "{}"))
        @test r isa SkillCallError && r.cause isa UniLMCancelled && isempty(seen)
        # The ambient token too.
        r, seen = _with_scripted(() -> with_cancel(() -> call(), tok), (_, _) -> _json(200, "{}"))
        @test r isa SkillCallError && r.cause isa UniLMCancelled && isempty(seen)
    end
    # A cancelled create reads no file: a read of this unreadable one would be its SystemError.
    locked = _skill_dir()
    chmod(joinpath(locked, "notes.txt"), 0o000)
    try
        r, seen = _with_scripted(() -> create_skill(locked; service=_OS, cancel=tok), (_, _) -> _json(200, "{}"))
        @test (typeof(r), typeof(r.cause), length(seen)) == (SkillCallError, UniLMCancelled, 0)
    finally
        chmod(joinpath(locked, "notes.txt"), 0o600)
    end
    for f in (create_skill, list_skills, retrieve_skill, update_skill, delete_skill, skill_content, create_skill_version,
              list_skill_versions, retrieve_skill_version, delete_skill_version), m in methods(f)
        @test (f, last(filter(k -> !endswith(String(k), "..."), Base.kwarg_decl(m)))) == (f, :cancel)
    end
    err, seen = _with_scripted((_, _) -> _json(200, "{}")) do
        map([() -> retrieve_skill(""; service=_OS), () -> delete_skill(""; service=_OS),
             () -> retrieve_skill_version("s", ""; service=_OS), () -> delete_skill_version("", "1"; service=_OS),
             () -> skill_content("s"; version="", service=_OS)]) do f
            try f(); nothing catch e; e end
        end
    end
    @test isempty(seen) && all(e -> e isa ArgumentError, err)
    @test err[1].msg == "retrieve_skill needs a skill id; list_skills lists them"
    @test err[3].msg == "retrieve_skill_version needs a version; list_skill_versions lists them"
    @test err[5].msg == "skill_content needs a version; list_skill_versions lists them"
    @test has_capability(OPENAIServiceEndpoint, :skills)
    ds = DeepSeekEndpoint("k")
    for f in (() -> create_skill(dir; service=ds), () -> list_skills(; service=ds), () -> retrieve_skill("s"; service=ds),
              () -> delete_skill("s"; service=ds), () -> skill_content("s"; service=ds), () -> list_skill_versions("s"; service=ds))
        e = try f(); nothing catch x; x end
        @test e isa ArgumentError && startswith(e.msg, "Skills API is not supported by DeepSeekEndpoint")
    end
end

# A skill made with create_skill loads into a container by reference.
# Refuted if: the reference names another id or version, or an OpenAI version that is no number passes.
@testset "SkillReference of a skill, at its default or a version, and of a version" begin
    s = _answered(() -> retrieve_skill("skill_abc"; service=_OS), 200; body=_OA_SKILL).response
    v = _answered(() -> retrieve_skill_version("skill_abc", "2"; service=_OS), 200; body=_OA_VERSION).response
    @test JSON.lower(SkillReference(s)) == Dict(:type => "skill_reference", :skill_id => "skill_abc")
    @test JSON.lower(SkillReference(s; version="latest"))[:version] == "latest"
    @test JSON.lower(SkillReference(v)) == Dict(:type => "skill_reference", :skill_id => "skill_abc", :version => "2")
end

# ─── Anthropic ───────────────────────────────────────────────────────────────

const _AS = SkillsProbe{UniLM._AnthropicSkills}

# A skill and a version as Anthropic's reference shows them (Skill, SkillVersion).
const _AN_SKILL = """{"id":"skill_01JAbcdefghijklmnopqrstuvw","created_at":"2026-11-16T20:10:09.427722Z","display_name":"CSV sum",
    "latest_version_id":"skver_01AbCdEfGhIjKlMnOpQrStUv","source":{"type":"custom"},"type":"skill",
    "updated_at":"2026-11-16T20:10:09.427722Z"}"""
const _AN_VERSION = """{"id":"skver_01AbCdEfGhIjKlMnOpQrStUv","created_at":"2026-11-16T20:10:10Z","description":"Sums a CSV column",
    "name":"csv-sum","skill_id":"skill_01JAbcdefghijklmnopqrstuvw","type":"skill_version"}"""

# Anthropic's create: `files[]` parts in one top-level folder, a zip as one `files[]` part, and `display_name`
# (https://platform.claude.com/docs/en/api/skills/create); its Skill: `display_name`, `latest_version_id`, an
# RFC 3339 `created_at`, no description or default version.
# Refuted if: a part or the label goes under another name, the label's limits are not checked before sending,
# or a field is read from the wrong key, to the wrong type or a shifted time.
@testset "Anthropic create_skill: files[] parts and display_name; the Skill it answers" begin
    dir = _skill_dir()
    r, seen = _with_scripted(() -> create_skill(dir; display_name="CSV sum", service=_AS), (_, _) -> _json(200, _AN_SKILL))
    req = only(seen)
    @test (req.method, req.target) == ("POST", "/v1/skills")
    @test occursin("Content-Disposition: form-data; name=\"display_name\"\r\n\r\nCSV sum\r\n", req.body)
    @test occursin(_part("files[]", "csv-sum/SKILL.md", "text/markdown", read(joinpath(dir, "SKILL.md"), String)), req.body)
    @test occursin(_part("files[]", "csv-sum/scripts/sum.py", "application/octet-stream", "print(1)"), req.body)
    @test count("name=\"files[]\"", req.body) == 3
    s = r.response
    @test r isa SkillSuccess
    @test (s.id, s.name, s.description, s.default_version, s.latest_version, s.created_at) ==
          ("skill_01JAbcdefghijklmnopqrstuvw", "CSV sum", nothing, nothing, "skver_01AbCdEfGhIjKlMnOpQrStUv", 1794859809)
    @test s.raw["source"] == Dict("type" => "custom") && s.raw["updated_at"] == "2026-11-16T20:10:09.427722Z"
    zip = joinpath(mktempdir(), "csv-sum.zip")
    write(zip, _ZIP_BYTES)
    _, seen = _with_scripted(() -> create_skill(zip; service=_AS), (_, _) -> _json(200, _AN_SKILL))
    @test occursin(_part("files[]", "csv-sum.zip", "application/zip", String(copy(_ZIP_BYTES))), only(seen).body)
    @test !occursin("display_name", only(seen).body)
    @test _answered(() -> create_skill(dir; display_name="x"^255, service=_AS), 200; body=_AN_SKILL) isa SkillSuccess
    err, seen = _with_scripted((_, _) -> _json(200, _AN_SKILL)) do
        map(n -> try create_skill(dir; display_name=n, service=_AS); nothing catch e; e end, ["x"^256, "CSV\nsum"])
    end
    @test isempty(seen)
    @test err[1] isa ArgumentError && err[1].msg == "create_skill: display_name is a single line of at most 255 characters (got 256)"
    @test err[2] isa ArgumentError && startswith(err[2].msg, "create_skill: display_name is a single line")
end

# Anthropic's lists: `limit`, the `page` cursor and, for skills, `source`; `data` and `next_page`, null or
# empty on the last page (its SDK ends a listing on either).
# Refuted if: `after` goes as anything but `page`, OpenAI's order is sent, a page without `data` decodes, or
# has_more and after do not follow `next_page`.
@testset "Anthropic list_skills and list_skill_versions: limit, page, source; data and next_page" begin
    page(item, next) = """{"data":[$item],"next_page":$next}"""
    r, seen = _with_scripted(() -> list_skills(; limit=2, after=_HOSTILE_ID, source="custom", service=_AS),
                             (_, _) -> _json(200, page(_AN_SKILL, "\"page_2\"")))
    @test only(seen).target == "/v1/skills?limit=2&page=$_HOSTILE_ENC&source=custom"
    @test r isa SkillListSuccess && r.response.has_more && r.response.after == "page_2"
    @test only(r.response.data).name == "CSV sum"
    for next in ("null", "\"\"")
        l = _answered(() -> list_skills(; service=_AS), 200; body=page(_AN_SKILL, next))
        @test l isa SkillListSuccess && !l.response.has_more && isnothing(l.response.after)
    end
    @test _answered(() -> list_skills(; service=_AS), 200; body="""{"next_page":null}""") isa SkillCallError
    v, seen = _with_scripted(() -> list_skill_versions("skill_1"; limit=1, after="p", service=_AS),
                             (_, _) -> _json(200, page(_AN_VERSION, "null")))
    @test only(seen).target == "/v1/skills/skill_1/versions?limit=1&page=p"
    @test v isa SkillVersionListSuccess && only(v.response.data).version == "skver_01AbCdEfGhIjKlMnOpQrStUv"
    err, seen = _with_scripted((_, _) -> _json(200, "{}")) do
        map([() -> list_skills(; order="asc", service=_AS), () -> list_skill_versions("s"; order="asc", service=_AS)]) do f
            try f(); nothing catch e; e end
        end
    end
    @test isempty(seen)
    @test err[1].msg == "list_skills: Anthropic's Skills API takes no order, which only OpenAI's takes; unset order"
    @test err[2].msg == "list_skill_versions: Anthropic's Skills API takes no order, which only OpenAI's takes; unset order"
end

# Anthropic's version routes take the version's id, its download the version's only (there is no skill-level
# content route), and it has no default version to set.
# Refuted if: a route differs, a version reads as anything but its id, a deletion record of another type or
# without its id reports a delete, or update_skill, a default or a download without a version is sent.
@testset "Anthropic retrieve, delete, versions and their zip; what it does not take" begin
    E = _HOSTILE_ENC
    dir = _skill_dir()
    sent = last(_with_scripted((_, _) -> _json(200, "{}")) do
        retrieve_skill(_HOSTILE_ID; service=_AS)
        delete_skill(_HOSTILE_ID; service=_AS)
        create_skill_version(_HOSTILE_ID, dir; service=_AS)
        retrieve_skill_version(_HOSTILE_ID, "latest"; service=_AS)
        delete_skill_version(_HOSTILE_ID, _HOSTILE_ID; service=_AS)
        skill_content(_HOSTILE_ID; version=_HOSTILE_ID, service=_AS)
    end)
    @test [(r.method, r.target) for r in sent] ==
          [("GET", "/v1/skills/$E"), ("DELETE", "/v1/skills/$E"), ("POST", "/v1/skills/$E/versions"),
           ("GET", "/v1/skills/$E/versions/latest"), ("DELETE", "/v1/skills/$E/versions/$E"),
           ("GET", "/v1/skills/$E/versions/$E/content")]
    @test count("name=\"files[]\"", sent[3].body) == 3 && !occursin("name=\"default\"", sent[3].body)
    v = _answered(() -> create_skill_version("skill_1", dir; service=_AS), 200; body=_AN_VERSION)
    @test v isa SkillVersionSuccess
    @test (v.response.id, v.response.skill_id, v.response.version, v.response.name, v.response.description,
           v.response.created_at) == ("skver_01AbCdEfGhIjKlMnOpQrStUv", "skill_01JAbcdefghijklmnopqrstuvw",
                                      "skver_01AbCdEfGhIjKlMnOpQrStUv", "csv-sum", "Sums a CSV column", 1794859810)
    del(body) = _answered(() -> delete_skill("skill_1"; service=_AS), 200; body)
    @test del("""{"id":"skill_1","type":"skill_deleted"}""") == SkillDeleteSuccess(id="skill_1", deleted=true)
    @test del("""{"id":"skill_1","type":"skill_version_deleted"}""") isa SkillCallError
    @test del("""{"type":"skill_deleted"}""") isa SkillCallError && del("""{"id":"skill_1"}""") isa SkillCallError
    vdel = _answered(() -> delete_skill_version("skill_1", "skver_1"; service=_AS), 200; body="""{"id":"skver_1","type":"skill_version_deleted"}""")
    @test vdel == SkillDeleteSuccess(id="skver_1", deleted=true)
    c = first(_with_scripted(() -> skill_content("skill_1"; version="skver_1", service=_AS),
                             (_, _) -> HTTP.Response(200, ["Content-Type" => "application/zip"], _ZIP_BYTES)))
    @test c isa FileContentSuccess && c.content == _ZIP_BYTES
    err, seen = _with_scripted((_, _) -> _json(200, _AN_SKILL)) do
        map([() -> update_skill("skill_1"; default_version="1", service=_AS), () -> skill_content("skill_1"; service=_AS),
             () -> create_skill_version("skill_1", dir; default=true, service=_AS)]) do f
            try f(); nothing catch e; e end
        end
    end
    @test isempty(seen) && all(e -> e isa ArgumentError, err)
    @test err[1].msg == "update_skill: Anthropic's Skills API takes no default_version: an Anthropic skill has no default " *
                        "version, and a request names a version or \"latest\"; unset default_version"
    @test err[2].msg == "skill_content: Anthropic serves the zip of a version only: pass version, a SkillVersionObject's " *
                        "version (the skill's latest_version is its newest)"
    @test err[3].msg == "create_skill_version: Anthropic's Skills API takes no default: its newest version is the skill's " *
                        "latest_version, which a request names as \"latest\"; unset default"
    # A SkillReference is OpenAI's: an Anthropic version id is no OpenAI version.
    @test_throws ArgumentError SkillReference(v.response)
end

# A Skill or a version without a field Anthropic's reference requires, of another type or kind, or with a
# `created_at` that is no RFC 3339 time is no skill.
# Refuted if: such a reply decodes as a success.
@testset "Anthropic: a Skill or a version without what the reference requires is a SkillCallError" begin
    full = JSON.parse(_AN_SKILL)
    for key in ("id", "type", "display_name", "latest_version_id", "created_at")
        r = _answered(() -> retrieve_skill("s"; service=_AS), 200; body=JSON.json(delete!(copy(full), key)))
        @test r isa SkillCallError && occursin(key, r.error)
    end
    r = _answered(() -> retrieve_skill("s"; service=_AS), 200; body=JSON.json(merge(full, Dict("created_at" => "yesterday"))))
    @test r isa SkillCallError && occursin("RFC 3339", r.error)
    @test _answered(() -> retrieve_skill("s"; service=_AS), 200; body=_AN_VERSION) isa SkillCallError
    fullv = JSON.parse(_AN_VERSION)
    for key in ("id", "type", "skill_id", "name", "description", "created_at")
        r = _answered(() -> retrieve_skill_version("s", "v"; service=_AS), 200; body=JSON.json(delete!(copy(fullv), key)))
        @test r isa SkillCallError && occursin(key, r.error)
    end
end

# An Anthropic endpoint whose Skills requests `_with_scripted` serves, with the headers that endpoint sends.
struct _AnthropicSkillsAt <: UniLM.ServiceEndpoint
    endpoint::UniLM._AnthropicEndpoint
end
UniLM._skills_wire(::_AnthropicSkillsAt) = UniLM._AnthropicSkills()
UniLM._skills_url(::_AnthropicSkillsAt) = _url_probe_base[] * "/v1/skills"
UniLM._skills_headers(s::_AnthropicSkillsAt) = UniLM._skills_headers(s.endpoint)
UniLM._skills_upload_headers(s::_AnthropicSkillsAt) = UniLM._skills_upload_headers(s.endpoint)
UniLM.provider_capabilities(::_AnthropicSkillsAt) = Set([:skills])

# Anthropic's Skills reference names `anthropic-version` and `anthropic-workspace-id`; the SDK (v2.9.0) sends no
# beta for skills, and `skills-2025-10-02` would select the beta's response shapes.
# Refuted if: an Anthropic Skills request sends the endpoint's betas or its user profile, or leaves out its key,
# its API version or its workspace — from the type or from an instance.
@testset "an Anthropic Skills request sends the key, the version and the workspace, never the user profile or the betas" begin
    dir = _skill_dir()
    withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
        base = ["Authorization" => "Bearer sk-ant-local-0000", "anthropic-version" => "2026-10-21"]
        every = ANTHROPICServiceEndpoint(betas=["skills-2025-10-02"], user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9",
                                         workspace_id="wrkspc_011CZkZaBF1tNoB5wlCeusgy")
        for (service, want) in ((ANTHROPICServiceEndpoint, base), (ANTHROPICServiceEndpoint(), base),
                                (every, [base; "anthropic-workspace-id" => "wrkspc_011CZkZaBF1tNoB5wlCeusgy"]))
            @test (@inferred UniLM._skills_headers(service)) == want
            @test UniLM._skills_upload_headers(service) == want
            sent = Dict{String,String}[]
            _with_scripted((_, req) -> (push!(sent, Dict(lowercase(String(k)) => String(v) for (k, v) in req.headers));
                                        _json(200, _AN_SKILL))) do
                create_skill(dir; service=_AnthropicSkillsAt(service))
                retrieve_skill("skill_1"; service=_AnthropicSkillsAt(service))
            end
            for (i, h) in enumerate(sent)
                @test Dict(k => v for (k, v) in h if k == "authorization" || startswith(k, "anthropic-")) ==
                      Dict(lowercase(k) => v for (k, v) in want)
                @test i == 1 ? startswith(h["content-type"], "multipart/form-data; boundary=") : !haskey(h, "content-type")
            end
        end
    end
    @test has_capability(ANTHROPICServiceEndpoint, :skills) && has_capability(ANTHROPICServiceEndpoint(), :skills)
    @test UniLM._skills_wire(ANTHROPICServiceEndpoint()) == UniLM._AnthropicSkills()
    @test UniLM._skills_url(ANTHROPICServiceEndpoint) == "https://api.anthropic.com/v1/skills"
    @test UniLM._skills_url(OPENAIServiceEndpoint) == "https://api.openai.com/v1/skills"
end
