# test/recorded_answers.jl — recorded System One answers, zero-spend.
#
# One local mock plays the service for the whole file. It answers every question
# of a System One request — so `ask`, `nl_dispatch` and `@branch` all run against
# it — and the models listing, numbers its replies through the request-id header,
# and counts the requests it receives. Recording runs against it with a sentinel
# API key; the replay tests run after it is closed and with TYPESAFE_API_KEY
# unset, so a request that went for the network could not be sent.

using Sockets, Base64

const _RA_KEY = "ts-sentinel-3f9c2a71"   # must appear in no recording
const _RA_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)
const _RA_ROOT = mktempdir()
_ra_dir(name::AbstractString) = joinpath(_RA_ROOT, name)

const _RA_HITS = Threads.Atomic{Int}(0)

# The mock's answer to one question of request number `n`: the LAST Choice option
# (a replayed dispatch cannot pass by landing on the default first one), and
# numbers that carry `n`, so no two recordings look alike — a Score's mass moved from its
# first level to its last by n/10⁵, and its score the expectation that makes.
function _ra_answer(q::AbstractDict, n::Int)::Dict{String,Any}
    kind = q["type"]
    if kind == "choice"
        options = collect(keys(q["criteria"]))
        Dict{String,Any}("type" => kind, "choice" => last(options), "confidence" => 0.9,
            "probabilities" => Dict{String,Any}(
                o => (o == last(options) ? 0.9 : 0.1 / max(length(options) - 1, 1)) for o in options))
    elseif kind == "score"
        levels = q["criteria"]
        k, shift = length(levels), n / 100_000
        p = k == 1 ? [1.0] : [1 / k + (i == 1 ? -shift : i == k ? shift : 0.0) for i in 1:k]
        Dict{String,Any}("type" => kind, "score" => sum((i - 1) * p[i] for i in 1:k), "confidence" => 0.5,
            "legend" => Dict{String,Any}(string(i - 1) => levels[i] for i in eachindex(levels)),
            "probabilities" => Dict{String,Any}(string(i - 1) => p[i] for i in 1:k))
    else
        Dict{String,Any}("type" => kind, "noul" => n / 1000)
    end
end

function _ra_handler(req)
    n = Threads.atomic_add!(_RA_HITS, 1) + 1
    reply(status, payload) = HTTP.Response(status,
        ["Content-Type" => "application/json", "x-typesafe-request-id" => "req_$n"],
        Vector{UInt8}(JSON.json(payload)))
    String(req.target) == "/v1/models" && return reply(200, Dict("models" => [Dict(
        "name" => "jev-latest", "description" => "listing $n", "release_date" => "2026-09-10T18:38:01+00:00")]))
    body = JSON.parse(String(copy(req.body)))
    body["state"] == "fail" && return reply(422, Dict("detail" => [Dict(
        "loc" => ["body", "state"], "msg" => "rejected by the mock", "type" => "value_error")]))
    reply(200, Dict("model" => "jev-1.13.0",
                    "answers" => Dict(name => _ra_answer(q, n) for (name, q) in body["questions"]),
                    "usage" => Dict("input_tokens" => 100 + n, "output_tokens" => 3)))
end

# Probe an ephemeral port, then serve on it; the close-then-rebind window can race.
function _ra_start()
    for _ in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        server = try
            HTTP.serve!(_ra_handler, "127.0.0.1", port; verbose=false)
        catch e
            e isa Base.IOError || rethrow()
            continue
        end
        return (server = server, url = "http://127.0.0.1:$port")
    end
    error("could not bind an ephemeral port for the recorded-answers mock")
end
const _RA_MOCK = _ra_start()

# The real TYPESAFEServiceEndpoint, pointed at the mock: with the sentinel key
# (recording), or with no key at all (replay — nothing could be sent).
_ra_online(f::Function) =
    withenv(f, UniLM.TYPESAFE_API_KEY => _RA_KEY, UniLM.TYPESAFE_BASE_URL_ENV => _RA_MOCK.url)
_ra_offline(f::Function) =
    withenv(f, UniLM.TYPESAFE_API_KEY => nothing, UniLM.TYPESAFE_BASE_URL_ENV => _RA_MOCK.url)

_ra_request(state::AbstractString="My payouts have been failing for 3 days.";
            options::Vector{String}=["billing", "shipping", "other"]) =
    SystemOneRequest(state, ["team" => choice("Which team should handle this ticket?", options),
                             "urgency" => score("How urgent is this ticket?", ["Can wait", "This week", "Today"]),
                             "upset" => noul("Is the customer upset?")]; model="jev-latest")

# The documented key, computed here from its definition.
_ra_key(method::AbstractString, path::AbstractString, body::AbstractString="") =
    bytes2hex(UniLM.SHA.sha256(string(method, " ", path, "\n", body)))
_ra_file(dir::AbstractString, r::SystemOneRequest) =
    joinpath(dir, _ra_key("POST", "/v1/systemone", JSON.json(r)) * ".json")
const _RA_MODELS_FILE = _ra_key("GET", "/v1/models") * ".json"

_ra_route(::nl"the customer was charged twice", t) = :billing
_ra_route(::nl"the parcel is late", t) = :shipping

_ra_branch(state::AbstractString) = @branch state config=_RA_CFG begin
    "billing"  => :billing
    "shipping" => :shipping
end

function _ra_calls()
    r = ask(_ra_request(); config=_RA_CFG)
    d = nl_dispatch(_ra_route, "Where is my parcel?"; config=_RA_CFG)
    b = _ra_branch("I was charged twice.")
    m = list_models(; config=_RA_CFG)
    (ask = r, dispatch = d, branch = b, models = m)
end

# Everything a caller reads off a result.
_ra_seen(r::SystemOneSuccess) =
    (r.response.model, r.response.usage, r.response.request_id, r.response.raw,
     Dict(k => (typeof(a), a.raw) for (k, a) in answers(r)))
_ra_seen(m::TypeSafeModelsSuccess) =
    ([(c.name, c.description, c.release_date, c.raw) for c in m.models], m.raw)

_ra_caught(f::Function) = try f(); nothing catch e; e end

# ─── With the service reachable ──────────────────────────────────────────────

@testset "recorded answers — :record writes one readable file per request and no credential" begin
    dir = _ra_dir("record")
    a, b = _ra_request(), _ra_request("Where is my parcel?")
    live = _ra_online() do
        with_recorded_answers(dir; mode=:record) do
            (a1 = ask(a; config=_RA_CFG), a2 = ask(a; config=_RA_CFG), b = ask(b; config=_RA_CFG),
             models = list_models(; config=_RA_CFG))
        end
    end
    @test all(issuccess, live)
    files = readdir(dir; join=true)
    @test length(files) == 3                          # the repeated request is one recording
    @test isfile(_ra_file(dir, a)) && isfile(_ra_file(dir, b)) && isfile(joinpath(dir, _RA_MODELS_FILE))

    text = read(_ra_file(dir, a), String)
    @test startswith(text, "{\n  \"request\": {\n    \"method\": \"POST\",")   # 2-space indent
    rec = JSON.parse(text)
    @test collect(keys(rec)) == ["request", "response", "recorded_at"]
    @test rec["request"]["path"] == "/v1/systemone"
    @test rec["request"]["body"] == JSON.parse(JSON.json(a))
    @test rec["response"]["status"] == 200
    # The second call replaced the first one's recording.
    @test rec["response"]["request_id"] == live.a2.response.request_id != live.a1.response.request_id
    @test rec["response"]["body"] == live.a2.response.raw

    stamp = match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$", rec["recorded_at"])
    @test stamp !== nothing
    @test abs(UniLM._utc_epoch_seconds((parse(Int, something(c)) for c in stamp.captures)...) - time()) < 120
    @test UniLM._rfc3339_utc(0) == "1970-01-01T00:00:00Z"
    @test UniLM._rfc3339_utc(784111777) == "1994-11-06T08:49:37Z"
    @test UniLM._rfc3339_utc(951782400) == "2000-02-29T00:00:00Z"

    listing = JSON.parse(read(joinpath(dir, _RA_MODELS_FILE), String))
    @test listing["request"]["method"] == "GET" && listing["request"]["body"] === nothing
    @test listing["response"]["body"] == live.models.raw

    # Neither the key nor any header is ever written.
    written = join(read.(files, String))
    @test !contains(written, _RA_KEY)
    @test !contains(lowercase(written), "bearer") && !contains(lowercase(written), "authorization")
    # Readable like any file written in place, not private like the temporary file it was.
    @test Sys.iswindows() || all(f -> filemode(f) & 0o777 == 0o644, files)
end

@testset "recorded answers — :record_missing calls the service only for what is missing" begin
    dir = _ra_dir("missing")
    a, b = _ra_request("recorded first"), _ra_request("recorded later")
    _ra_online() do
        with_recorded_answers(() -> ask(a; config=_RA_CFG), dir; mode=:record)
        before = _RA_HITS[]
        with_recorded_answers(dir; mode=:record_missing) do
            @test issuccess(ask(a; config=_RA_CFG))   # recorded: replayed
            @test issuccess(ask(b; config=_RA_CFG))   # missing: sent, then recorded
            @test issuccess(ask(b; config=_RA_CFG))   # recorded a moment ago: replayed
        end
        @test _RA_HITS[] - before == 1
    end
    @test isfile(_ra_file(dir, b))
end

@testset "recorded answers — :record replaces an unreadable recording, :record_missing refuses to" begin
    dir = mkpath(_ra_dir("replace"))
    req = _ra_request("replace me")
    file = _ra_file(dir, req)
    write(file, "{not json")
    _ra_online() do
        before = _RA_HITS[]
        e = with_recorded_answers(() -> _ra_caught(() -> ask(req; config=_RA_CFG)), dir; mode=:record_missing)
        @test e isa ReplayMissError && startswith(e.reason, "unreadable recording: ")
        @test _RA_HITS[] == before && read(file, String) == "{not json"   # nothing sent, nothing written
        r = with_recorded_answers(() -> ask(req; config=_RA_CFG), dir; mode=:record)
        @test r isa SystemOneSuccess && _RA_HITS[] == before + 1
        @test JSON.parse(read(file, String))["response"]["body"] == r.response.raw
        @test _ra_seen(with_recorded_answers(() -> ask(req; config=_RA_CFG), dir)) == _ra_seen(r)
    end
end

@testset "recorded answers — a non-200 is returned as usual and never recorded" begin
    dir = _ra_dir("failure")
    r = _ra_online(() -> with_recorded_answers(() -> ask(_ra_request("fail"); config=_RA_CFG), dir; mode=:record))
    @test r isa SystemOneFailure && r.status == 422
    @test contains(r.message, "rejected by the mock")
    @test !isfile(_ra_file(dir, _ra_request("fail")))
end

@testset "recorded answers — a paid answer that cannot be written is thrown, never returned" begin
    dir = _ra_dir("unwritable")
    calls = (() -> ask(_ra_request("unwritable"); config=_RA_CFG),
             () -> list_models(; config=_RA_CFG),
             () -> nl_dispatch(_ra_route, "unwritable"; config=_RA_CFG),
             () -> nl_classify("unwritable", (billing = "the customer was charged twice", shipping = "the parcel is late");
                               config=_RA_CFG),
             () -> _ra_branch("unwritable"))
    _ra_online() do
        for mode in (:record, :record_missing), call in calls
            before = _RA_HITS[]
            e = with_recorded_answers(dir; mode) do
                rm(dir; force=true, recursive=true)
                write(dir, "")   # the checked directory is a file by the time the answer arrives
                _ra_caught(call)
            end
            # The lost write, with its I/O error inside: not a call error a fallback would
            # take for the service's.
            @test e isa RecordingWriteError && e.cause isa Union{Base.IOError,SystemError}
            @test dirname(e.file) == abspath(dir) && endswith(e.file, ".json")
            msg = sprint(showerror, e)
            @test contains(msg, e.file) && contains(msg, "billed") && contains(msg, sprint(showerror, e.cause))
            @test _RA_HITS[] == before + 1
            rm(dir)
        end
    end
end

@testset "recorded answers — twenty requests recorded at once are twenty whole files" begin
    dir = _ra_dir("concurrent")
    results = _ra_online() do
        with_recorded_answers(dir; mode=:record) do
            asyncmap(i -> ask(_ra_request("ticket $i"); config=_RA_CFG), 1:20; ntasks=8)
        end
    end
    @test all(issuccess, results)
    @test length(readdir(dir)) == 20 && all(endswith(".json"), readdir(dir))   # no temporary file left
    @test all(1:20) do i
        rec = JSON.parse(read(_ra_file(dir, _ra_request("ticket $i")), String))
        rec["request"]["body"]["state"] == "ticket $i" &&
            rec["response"]["request_id"] == results[i].response.request_id
    end
end

@testset "recorded answers — outside any scope, calls go to the service" begin
    dir = _ra_dir("outside")
    _ra_online() do
        with_recorded_answers(() -> ask(_ra_request("outside"); config=_RA_CFG), dir; mode=:record)
        before = _RA_HITS[]
        @test issuccess(with_recorded_answers(() -> ask(_ra_request("outside"); config=_RA_CFG), dir))
        @test _RA_HITS[] == before                   # replayed inside the scope
        @test issuccess(ask(_ra_request("outside"); config=_RA_CFG))
        @test _RA_HITS[] == before + 1               # sent once the scope is gone
    end
end

# ─── With the service gone ───────────────────────────────────────────────────

@testset "recorded answers — :replay needs no key and no service" begin
    dir = _ra_dir("replay")
    live = _ra_online(() -> with_recorded_answers(_ra_calls, dir; mode=:record))
    close(_RA_MOCK.server)
    before = _RA_HITS[]
    replayed = _ra_offline(() -> with_recorded_answers(_ra_calls, dir))
    @test _RA_HITS[] == before
    @test replayed.ask isa SystemOneSuccess
    @test _ra_seen(replayed.ask) == _ra_seen(live.ask)
    @test replayed.ask["team"].probabilities == live.ask["team"].probabilities
    @test replayed.ask["urgency"].legend == live.ask["urgency"].legend
    @test replayed.dispatch === live.dispatch === :shipping
    @test replayed.branch === live.branch === :shipping
    @test replayed.models isa TypeSafeModelsSuccess
    @test _ra_seen(replayed.models) == _ra_seen(live.models)
end

@testset "recorded answers — a miss is thrown out of ask, never returned as a call error" begin
    dir = _ra_dir("replay")
    unrecorded = _ra_request("never recorded")
    e = _ra_offline(() -> with_recorded_answers(() -> _ra_caught(() -> ask(unrecorded; config=_RA_CFG)), dir))
    @test e isa ReplayMissError
    @test e.dir == abspath(dir) && e.method == "POST" && e.path == "/v1/systemone"
    @test e.body == JSON.json(unrecorded)
    @test e.key == _ra_key("POST", "/v1/systemone", e.body)
    @test e.reason == "no recording"
    msg = sprint(showerror, e)
    @test contains(msg, abspath(dir)) && contains(msg, first(e.key, 12)) && !contains(msg, e.key)
    @test contains(msg, "\"team\", \"upset\", \"urgency\"")
    @test contains(msg, "TYPESAFE_API_KEY") && contains(msg, ":record_missing")

    _ra_offline() do
        with_recorded_answers(dir) do
            @test_throws ReplayMissError nl_dispatch(_ra_route, "never recorded"; config=_RA_CFG)
            @test_throws ReplayMissError _ra_branch("never recorded")
            # The same question with its options reordered is a different request.
            @test_throws ReplayMissError ask(_ra_request(; options=["other", "shipping", "billing"]);
                                             config=_RA_CFG)
        end
        empty = mkpath(_ra_dir("empty"))
        listing = with_recorded_answers(() -> _ra_caught(() -> list_models(; config=_RA_CFG)), empty)
        @test listing isa ReplayMissError && listing.key * ".json" == _RA_MODELS_FILE
        @test !contains(sprint(showerror, listing), "asking")
    end
end

@testset "recorded answers — an unreadable recording is thrown like a miss, never returned" begin
    req = _ra_request("unreadable")
    response(body; status=200, id="req_x") =
        JSON.json(Dict("response" => Dict("status" => status, "request_id" => id, "body" => body)))
    # Each file exists under the request's key, and none of them is a recording.
    cases = [
        ("{not json",                                     "invalid JSON"),
        ("",                                              "invalid JSON"),
        ("""["a list"]""",                                "no \"response\" object"),
        ("""{"request": {}}""",                           "no \"response\" object"),
        (response(Dict("model" => "m"); status=500),      "status is 500"),
        (JSON.json(Dict("response" => Dict("body" => Dict()))), "status is nothing"),
        (response("a string"),                            "\"body\" is not a JSON object"),
        (response(nothing),                               "\"body\" is not a JSON object"),
        (response(Dict("model" => "m"); id=7),            "\"request_id\" is neither a string nor null"),
    ]
    _ra_offline() do
        for (i, (text, why)) in enumerate(cases)
            dir = mkpath(_ra_dir("unreadable-$i"))
            file = _ra_file(dir, req)
            write(file, text)
            # Thrown out of ask, not returned: a call error here would read as a service failure.
            e = with_recorded_answers(() -> _ra_caught(() -> ask(req; config=_RA_CFG)), dir)
            @test e isa ReplayMissError
            @test startswith(e.reason, "unreadable recording: ") && contains(e.reason, why)
            msg = sprint(showerror, e)
            @test contains(msg, file) && contains(msg, "Delete") && contains(msg, ":record_missing")
            @test contains(msg, "\"team\", \"upset\", \"urgency\"") && !contains(msg, "\n")
            # :record_missing reports it the same way instead of recording over it.
            @test with_recorded_answers(() -> _ra_caught(() -> ask(req; config=_RA_CFG)), dir;
                                        mode=:record_missing) isa ReplayMissError
            @test read(file, String) == text
        end
        # The listing, and the verbs built on ask, surface it the same way.
        dir = mkpath(_ra_dir("unreadable-listing"))
        write(joinpath(dir, _RA_MODELS_FILE), response("a string"))
        listing = with_recorded_answers(() -> _ra_caught(() -> list_models(; config=_RA_CFG)), dir)
        @test listing isa ReplayMissError && startswith(listing.reason, "unreadable recording: ")
        dispatch() = with_recorded_answers(() -> _ra_caught(() -> nl_dispatch(_ra_route, "unreadable";
                                                                               config=_RA_CFG)), dir)
        missed = dispatch()          # its miss names the key its recording would have
        @test missed isa ReplayMissError && missed.reason == "no recording"
        write(joinpath(dir, missed.key * ".json"), "{not json")
        again = dispatch()
        @test again isa ReplayMissError && startswith(again.reason, "unreadable recording: ")

        # A well-formed recording whose body the decoder rejects is the service's payload,
        # not the recording's: the same call error a live 200 with that body returns.
        dir = mkpath(_ra_dir("undecodable"))
        write(_ra_file(dir, req), response(Dict("model" => "m"); id="req_broken"))
        write(joinpath(dir, _RA_MODELS_FILE), response(Dict("data" => []); id="req_broken"))
        r = with_recorded_answers(() -> ask(req; config=_RA_CFG), dir)
        @test r isa SystemOneCallError && r.request_id == "req_broken" && contains(r.error, "answers")
        m = with_recorded_answers(() -> list_models(; config=_RA_CFG), dir)
        @test m isa SystemOneCallError && m.request_id == "req_broken" && contains(m.error, "models")
    end
end

@testset "recorded answers — the scope reaches spawned tasks, and scopes nest" begin
    dir = _ra_dir("replay")
    req = _ra_request()
    recorded = JSON.parse(read(_ra_file(dir, req), String))["response"]
    _ra_offline() do
        with_recorded_answers(dir) do
            spawned = fetch(Threads.@spawn ask(req; config=_RA_CFG))
            @test spawned isa SystemOneSuccess && spawned.response.raw == recorded["body"]
            mapped = asyncmap(_ -> ask(req; config=_RA_CFG), 1:4; ntasks=4)
            @test all(r -> r isa SystemOneSuccess && r.response.request_id == recorded["request_id"], mapped)
        end
        # An inner :record scope's service is the outer :replay scope, so it records
        # the replayed answer with no key and no service to reach.
        copied_to = _ra_dir("copy")
        r = with_recorded_answers(dir) do
            with_recorded_answers(() -> ask(req; config=_RA_CFG), copied_to; mode=:record)
        end
        @test r isa SystemOneSuccess
        copied = JSON.parse(read(_ra_file(copied_to, req), String))["response"]
        @test copied["body"] == recorded["body"] && copied["request_id"] == recorded["request_id"]
        # Outside every scope the call goes for the service, which a keyless process cannot reach.
        out = ask(req; config=_RA_CFG)
        @test out isa SystemOneCallError && contains(out.error, "TYPESAFE_API_KEY")
    end
end

@testset "recorded answers — a cancelled token ends the call before any replay" begin
    dir = _ra_dir("replay")
    tok = cancel!(CancelToken())
    out = _ra_offline() do
        with_recorded_answers(dir) do
            [ask(_ra_request(); config=_RA_CFG, cancel=tok),                 # recorded
             ask(_ra_request("never recorded"); config=_RA_CFG, cancel=tok), # would be a miss
             list_models(; config=_RA_CFG, cancel=tok),
             with_cancel(() -> ask(_ra_request(); config=_RA_CFG), tok)]      # ambient token
        end
    end
    @test all(r -> r isa SystemOneCallError && r.cause isa UniLMCancelled && r.cause.source === :token, out)
    # The same typed result the live path returns.
    live = ask(_ra_request(); config=_RA_CFG, cancel=tok)
    @test (typeof(live), live.status, live.request_id, typeof(live.cause)) ==
          (typeof(out[1]), out[1].status, out[1].request_id, typeof(out[1].cause))
end

@testset "recorded answers — a bad mode or a bad directory fails before f runs" begin
    ran = Ref(false)
    @test_throws ArgumentError with_recorded_answers(() -> (ran[] = true), _ra_dir("replay"); mode=:bogus)
    @test_throws ArgumentError with_recorded_answers(() -> (ran[] = true), _ra_dir("absent"))
    # A recording mode creates the directory and writes a file into it before f runs,
    # so a directory that could not hold an answer fails before one is paid for.
    @test with_recorded_answers(() -> :done, _ra_dir("absent"); mode=:record) === :done
    @test isdir(_ra_dir("absent")) && isempty(readdir(_ra_dir("absent")))
    file = _ra_dir("a-file")
    write(file, "not a directory")
    for dir in (file, joinpath(file, "below")), mode in (:record, :record_missing)
        e = _ra_caught(() -> with_recorded_answers(() -> (ran[] = true), dir; mode))
        @test e isa ArgumentError && contains(e.msg, repr(dir))
        @test contains(e.msg, dir == file ? "not a directory" : "EEXIST")
    end
    readonly = mkpath(_ra_dir("read-only"))
    chmod(readonly, 0o555)
    try
        # Root, and Windows, write into it anyway: the check is only observable where the mode holds.
        refused = try touch(joinpath(readonly, "probe")); false catch; true end
        if refused
            e = _ra_caught(() -> with_recorded_answers(() -> (ran[] = true), readonly; mode=:record))
            @test e isa ArgumentError && contains(e.msg, repr(readonly)) && contains(e.msg, "Permission denied")
        end
    finally
        chmod(readonly, 0o755)
    end
    @test !ran[]
end

# ─── The LLM verbs ───────────────────────────────────────────────────────────
#
# A second mock plays the language-model providers — OpenAI Chat Completions and
# Responses (and its lifecycle routes), Anthropic Messages, Gemini generateContent,
# embeddings, FIM completions — each with its own request-id header (Gemini sends
# none), beside headers no recording may keep.
# `_RLHere` runs a provider's own `auth_header`, encoder and decoder against it and
# moves only the origin of the provider's URL, so the path and any query reach the
# mock as they would reach the provider. Each key variable holds a sentinel while
# recording and is removed for replay, where the mock must see no request.

const _RL_KEYS = Dict(UniLM.OPENAI_API_KEY => "sk-openai-sentinel-5b1e",
                      UniLM.ANTHROPIC_API_KEY => "sk-ant-sentinel-77c2",
                      UniLM.GEMINI_API_KEY => "gemini-sentinel-0d9a",
                      "AZURE_OPENAI_API_KEY" => "azure-sentinel-4f3b")
const _RL_DEEPSEEK_KEY = "sk-deepseek-sentinel-91aa"
const _RL_QUERY_KEY = "query-sentinel-6e0c"
const _RL_SECRETS = [collect(values(_RL_KEYS)); _RL_DEEPSEEK_KEY; _RL_QUERY_KEY]
const _RL_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)

const _RL_HITS = Threads.Atomic{Int}(0)
const _RL_SEEN = String[]            # each request's target, headers and body, one string per request
const _RL_LOCK = ReentrantLock()

_rl_undecodable(raw) = occursin("undecodable", raw)

# Turn 1 of a tool loop (the last message is the user's) calls `add`; a later turn answers.
function _rl_chat_reply(b, n, raw)
    _rl_undecodable(raw) && return Dict("id" => "chatcmpl-$n", "object" => "chat.completion")  # no choices
    calls = haskey(b, "tools") && b["messages"][end]["role"] == "user"
    msg = calls ? Dict("role" => "assistant", "content" => nothing, "tool_calls" => [Dict("id" => "call_$n",
                       "type" => "function", "function" => Dict("name" => "add", "arguments" => "{\"a\":2,\"b\":3}"))]) :
                  Dict("role" => "assistant", "content" => "reply $n")
    Dict("id" => "chatcmpl-$n", "object" => "chat.completion", "model" => b["model"],
         "choices" => [Dict("index" => 0, "message" => msg, "finish_reason" => calls ? "tool_calls" : "stop")],
         "usage" => Dict("prompt_tokens" => 100 + n, "completion_tokens" => 10 + n, "total_tokens" => 110 + 2n))
end

function _rl_responses_reply(b, n, raw)
    calls = haskey(b, "tools") && b["input"] isa AbstractString
    output = calls ? [Dict("type" => "function_call", "id" => "fc_$n", "call_id" => "call_$n", "name" => "add",
                           "arguments" => "{\"a\":2,\"b\":3}", "status" => "completed")] :
                     [Dict("type" => "message", "id" => "msg_$n", "role" => "assistant", "status" => "completed",
                           "content" => [Dict("type" => "output_text", "text" => "reply $n", "annotations" => [])])]
    Dict("id" => "resp_$n", "object" => "response", "status" => _rl_undecodable(raw) ? "failed" : "completed",
         "model" => b["model"], "output" => output,
         "usage" => Dict("input_tokens" => 100 + n, "output_tokens" => 10 + n, "total_tokens" => 110 + 2n))
end

_rl_anthropic_reply(b, n, raw) = _rl_undecodable(raw) ? Dict("type" => "message") :
    Dict("id" => "msg_$n", "type" => "message", "role" => "assistant", "model" => b["model"],
         "content" => [Dict("type" => "text", "text" => "reply $n")], "stop_reason" => "end_turn",
         "usage" => Dict("input_tokens" => 100 + n, "output_tokens" => 10 + n))

# A request that asks for a long generation gets one hop of it (`_rl_gemini_hop`).
function _rl_gemini_reply(b, n, raw, path)
    hops = get(_RL_HOPS, _rl_variant(raw), nothing)
    isnothing(hops) || return _rl_gemini_hop(b, match(r"/models/([^/:]+):", path)[1], hops)
    Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "reply $n")]),
                               "finishReason" => "STOP", "index" => 0)],
         "usageMetadata" => Dict("promptTokenCount" => 100 + n, "candidatesTokenCount" => 10 + n,
                                 "totalTokenCount" => 110 + 2n),
         "modelVersion" => "gemini-3.8-flash", "responseId" => "gem_$n")
end

function _rl_embeddings_reply(b, n, raw)
    inputs = b["input"] isa AbstractString ? [b["input"]] : b["input"]
    Dict("object" => "list", "model" => b["model"],
         "data" => [Dict("object" => "embedding", "index" => i - 1, "embedding" => [n / 8, i / 8, 0.5])
                    for i in eachindex(inputs)],
         "usage" => Dict("prompt_tokens" => 3 + n, "total_tokens" => 3 + n))
end

# The Responses lifecycle routes: /v1/responses/{id}[/route], and the two without an id.
function _rl_lifecycle_reply(method, path, n)
    segments = split(path, '/')                     # "", "v1", "responses", the id or a route, [a route]
    id, route = segments[4], last(segments)
    usage = Dict("input_tokens" => 100 + n, "output_tokens" => 10 + n, "total_tokens" => 110 + 2n)
    route == "input_tokens" && return Dict("object" => "response.input_tokens", "input_tokens" => 100 + n)
    route == "compact" && return Dict("id" => "resp_$n", "object" => "response.compaction", "usage" => usage,
        "output" => [Dict("id" => "cmp_$n", "type" => "compaction", "encrypted_content" => "enc_$n")])
    route == "input_items" && return Dict("object" => "list", "first_id" => "msg_$n", "last_id" => "msg_$n",
        "has_more" => false, "data" => [Dict("id" => "msg_$n", "type" => "message", "role" => "user",
                                             "content" => [Dict("type" => "input_text", "text" => "hi")])])
    method == "DELETE" && return Dict("id" => id, "object" => "response", "deleted" => true)
    Dict("id" => id, "object" => "response", "status" => route == "cancel" ? "cancelled" : "completed",
         "model" => "gpt-5.4-mini", "output" => [], "usage" => usage)
end

# FIM on the completions route answers in the completions shape: a choice's `text`.
_rl_fim_reply(b, n) = Dict("id" => "cmpl-$n", "object" => "text_completion", "model" => b["model"],
    "choices" => [Dict("index" => 0, "text" => "fill $n", "finish_reason" => "stop")],
    "usage" => Dict("prompt_tokens" => 100 + n, "completion_tokens" => 10 + n, "total_tokens" => 110 + 2n))

# ─── Streams, as each wire frames them ───────────────────────────────────────
# OpenAI Chat Completions SSE (DeepSeek's too), Responses SSE, Anthropic Messages SSE,
# native Gemini SSE (streaming is the URL's method there, not a body flag), Gemini
# Interactions SSE, and Ollama's newline-delimited JSON. A prompt can ask for a variant
# of a stream with `variant:<name>`.

_rl_sse(event, data) = (isnothing(event) ? "" : "event: $event\n") * "data: " * JSON.json(data) * "\n\n"
_rl_variant(raw) = (m = match(r"variant:([a-z_]+)", raw); isnothing(m) ? nothing : Symbol(m[1]))

# Streams a test compares byte for byte with their recording, so they carry no request
# number: CRLF line ends, and a last line the server never ends with a newline.
const _RL_STREAM_BYTES = Dict(
    :crlf => "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"carried\"}}]}\r\n\r\n" *
             "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n",
    :no_final_newline => "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"unterminated\"}}]}\n\n" *
                         "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" *
                         "data: [DONE]")
_rl_mock_bytes(variant::Symbol) = _RL_STREAM_BYTES[variant]

# A truncated stream has no finish and no `[DONE]`; an in-band error arrives on the 200 stream.
function _rl_chat_stream(b, n, variant)
    haskey(_RL_STREAM_BYTES, variant) && return _RL_STREAM_BYTES[variant]
    chunk(delta, finish) = _rl_sse(nothing, Dict("id" => "c$n", "object" => "chat.completion.chunk",
        "model" => b["model"], "choices" => [Dict("index" => 0, "delta" => delta, "finish_reason" => finish)]))
    text = chunk(Dict("role" => "assistant", "content" => "streamed $n"), nothing)
    variant === :truncated && return text
    variant === :inband_error && return text * _rl_sse(nothing, Dict("error" => Dict(
        "message" => "the mock failed mid-stream", "type" => "server_error"))) * "data: [DONE]\n\n"
    text * chunk(Dict(), "stop") * "data: [DONE]\n\n"
end

function _rl_responses_stream(b, n)
    reply = Dict("id" => "resp_$n", "object" => "response", "status" => "completed", "model" => b["model"],
        "output" => [Dict("type" => "message", "id" => "msg_$n", "role" => "assistant", "status" => "completed",
                          "content" => [Dict("type" => "output_text", "text" => "streamed $n", "annotations" => [])])],
        "usage" => Dict("input_tokens" => 100 + n, "output_tokens" => 10 + n, "total_tokens" => 110 + 2n))
    _rl_sse("response.created", Dict("type" => "response.created",
                                     "response" => merge(reply, Dict("status" => "in_progress", "output" => [])))) *
    _rl_sse("response.output_text.delta", Dict("type" => "response.output_text.delta", "delta" => "streamed $n")) *
    _rl_sse("response.completed", Dict("type" => "response.completed", "response" => reply))
end

_rl_anthropic_stream(b, n) =
    _rl_sse("message_start", Dict("type" => "message_start", "message" => Dict("id" => "msg_$n", "type" => "message",
        "role" => "assistant", "model" => b["model"], "content" => [],
        "usage" => Dict("input_tokens" => 100 + n, "output_tokens" => 1)))) *
    _rl_sse("content_block_start", Dict("type" => "content_block_start", "index" => 0,
                                        "content_block" => Dict("type" => "text", "text" => ""))) *
    _rl_sse("content_block_delta", Dict("type" => "content_block_delta", "index" => 0,
                                        "delta" => Dict("type" => "text_delta", "text" => "streamed $n"))) *
    _rl_sse("content_block_stop", Dict("type" => "content_block_stop", "index" => 0)) *
    _rl_sse("message_delta", Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "end_turn"),
                                  "usage" => Dict("output_tokens" => 10 + n))) *
    _rl_sse("message_stop", Dict("type" => "message_stop"))

# `variant:hops` and `variant:two_hops` ask for a long generation continued over three requests and
# over two, streamed or not: each reply names its hop, and every one reports the same usage.
const _RL_HOPS = Dict(:hops => 3, :two_hops => 2)
function _rl_gemini_hop(b, model, hops::Int)
    token = get(b, "continuationToken", nothing)
    hop = token isa String ? parse(Int, chopprefix(token, "tok-")) + 1 : 1
    ends = hop < hops ? Dict("finishReason" => "CONTINUATION", "continuationToken" => "tok-$hop") :
                        Dict("finishReason" => "STOP")
    Dict("candidates" => [merge(Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "hop $hop. ")]),
                                     "index" => 0), ends)],
         "usageMetadata" => Dict("promptTokenCount" => 100, "candidatesTokenCount" => 10, "totalTokenCount" => 110),
         "modelVersion" => model)
end

function _rl_gemini_stream(b, n, model, variant)
    haskey(_RL_HOPS, variant) && return _rl_sse(nothing, _rl_gemini_hop(b, model, _RL_HOPS[variant]))
    usage = Dict("promptTokenCount" => 100 + n, "candidatesTokenCount" => 10 + n, "totalTokenCount" => 110 + 2n)
    part(text) = Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => text)]), "index" => 0)
    _rl_sse(nothing, Dict("candidates" => [part("streamed $n")], "modelVersion" => model)) *
    _rl_sse(nothing, Dict("candidates" => [Dict("finishReason" => "STOP", "index" => 0)], "usageMetadata" => usage,
                          "modelVersion" => model))
end

function _rl_interactions_stream(b, n)
    done = Dict("id" => "int_$n", "status" => "completed", "model" => b["model"],
                "usage" => Dict("total_input_tokens" => 100 + n, "total_output_tokens" => 10 + n, "total_tokens" => 110 + 2n))
    _rl_sse("interaction.created", Dict("interaction" => merge(done, Dict("status" => "in_progress")))) *
    _rl_sse("step.start", Dict("index" => 0, "step" => Dict("type" => "model_output"))) *
    _rl_sse("step.delta", Dict("index" => 0, "delta" => Dict("type" => "text", "text" => "streamed $n"))) *
    _rl_sse("step.stop", Dict("index" => 0)) *
    _rl_sse("interaction.completed", Dict("interaction" => done)) * "event: done\ndata: [DONE]\n\n"
end

_rl_ollama_stream(b, n) =
    JSON.json(Dict("model" => b["model"], "message" => Dict("role" => "assistant", "content" => "streamed $n"),
                   "done" => false)) * "\n" *
    JSON.json(Dict("model" => b["model"], "message" => Dict("role" => "assistant", "content" => ""), "done" => true,
                   "done_reason" => "stop", "prompt_eval_count" => 100 + n, "eval_count" => 10 + n)) * "\n"

function _rl_stream(http::HTTP.Stream, path, b, n, raw)
    variant = _rl_variant(raw)
    gemini = match(r"/models/([^/:]+):streamGenerateContent$", path)
    id, type, bytes =
        endswith(path, "/messages") ? ("request-id" => "req_$n", "text/event-stream", _rl_anthropic_stream(b, n)) :
        !isnothing(gemini) ? (nothing, "text/event-stream", _rl_gemini_stream(b, n, gemini[1], variant)) :
        endswith(path, "/interactions") ? (nothing, "text/event-stream", _rl_interactions_stream(b, n)) :
        endswith(path, "/responses") ? ("x-request-id" => "req_$n", "text/event-stream", _rl_responses_stream(b, n)) :
        path == "/api/chat" ? (nothing, "application/x-ndjson", _rl_ollama_stream(b, n)) :
        ("x-request-id" => "req_$n", "text/event-stream", _rl_chat_stream(b, n, variant))
    isnothing(id) || HTTP.setheader(http, id)
    HTTP.setheader(http, "Content-Type" => type)
    HTTP.startwrite(http)
    write(http, bytes)
end

# A recordings directory that passes the scope's check, then is a file by the time a reply
# arrives: the mock serves `after` requests as usual, then replaces it as it serves the next,
# so the recording of that request's paid reply fails.
const _RL_DOOMED = Ref(("", 0))
_rl_unwritable_dir(; after::Int=0) = (dir = mktempdir(_RA_ROOT); @lock(_RL_LOCK, _RL_DOOMED[] = (dir, after)); dir)
function _rl_doom!()
    dir = @lock _RL_LOCK begin
        d, k = _RL_DOOMED[]
        _RL_DOOMED[] = k > 0 ? (d, k - 1) : ("", 0)
        k > 0 ? "" : d
    end
    isempty(dir) || (rm(dir; recursive=true); write(dir, ""))
    nothing
end

function _rl_serve(http::HTTP.Stream)
    req = http.message
    raw = String(read(http))
    n = Threads.atomic_add!(_RL_HITS, 1) + 1
    @lock _RL_LOCK push!(_RL_SEEN, join([String(req.target); [string(k, ": ", v) for (k, v) in req.headers]; raw], "\n"))
    _rl_doom!()
    b = isempty(raw) ? Dict{String,Any}() : JSON.parse(raw; dicttype=Dict{String,Any})   # a GET, DELETE or cancel sends none
    path = String(HTTP.URI(String(req.target)).path)
    HTTP.setstatus(http, 200)
    HTTP.setheader(http, "set-cookie" => "session=cookie-sentinel")      # never recorded
    HTTP.setheader(http, "openai-organization" => "org-sentinel")        # never recorded
    # Native Gemini names the tier that billed a reply in its response head alone.
    tier = get(b, "serviceTier", nothing)
    tier isa String && HTTP.setheader(http, "x-gemini-service-tier" => tier)
    (get(b, "stream", false) === true || endswith(path, ":streamGenerateContent")) &&
        return _rl_stream(http, path, b, n, raw)
    id, reply = endswith(path, "/messages") ? ("request-id" => "req_$n", _rl_anthropic_reply(b, n, raw)) :
                occursin(":generateContent", path) ? (nothing, _rl_gemini_reply(b, n, raw, path)) :
                endswith(path, "/responses") ? ("x-request-id" => "req_$n", _rl_responses_reply(b, n, raw)) :
                startswith(path, "/v1/responses/") ? ("x-request-id" => "req_$n", _rl_lifecycle_reply(req.method, path, n)) :
                endswith(path, "/embeddings") ? ("x-request-id" => "req_$n", _rl_embeddings_reply(b, n, raw)) :
                endswith(path, "/v1/completions") ? ("x-request-id" => "req_$n", _rl_fim_reply(b, n)) :
                ("x-request-id" => "req_$n", _rl_chat_reply(b, n, raw))     # a prefix completion too
    isnothing(id) || HTTP.setheader(http, id)
    HTTP.setheader(http, "Content-Type" => "application/json")
    HTTP.startwrite(http)
    write(http, JSON.json(reply))
end

const _RL_MOCK = let server = HTTP.listen!(_rl_serve, "127.0.0.1", 0; verbose=false)
    (server = server, url = "http://127.0.0.1:$(HTTP.port(server))")
end

# A provider's own wire and credentials, with the origin of its URL moved to the mock.
struct _RLHere{S} <: UniLM.ServiceEndpoint
    real::S
end
_rl_here(url::String) = replace(url, r"^https?://[^/]+" => _RL_MOCK.url)
UniLM.get_url(s::_RLHere, chat::Chat) = _rl_here(UniLM.get_url(s.real, chat))
UniLM.get_url(s::_RLHere, emb::Embeddings) = _rl_here(UniLM.get_url(s.real, emb))
UniLM.get_url(s::_RLHere, r::Respond) = _rl_here(UniLM.get_url(s.real, r))
UniLM.auth_header(s::_RLHere) = UniLM.auth_header(s.real)
UniLM.encode_request(s::_RLHere, chat::Chat) = UniLM.encode_request(s.real, chat)
UniLM.decode_response(s::_RLHere, resp::HTTP.Response) = UniLM.decode_response(s.real, resp)
UniLM.encode_agentic(s::_RLHere, r::Respond) = UniLM.encode_agentic(s.real, r)
UniLM.decode_agentic(s::_RLHere, resp::HTTP.Response) = UniLM.decode_agentic(s.real, resp)
UniLM._agentic_url(s::_RLHere) = _rl_here(UniLM._agentic_url(s.real))
UniLM._resolve_base_url(s::_RLHere) = _rl_here(UniLM._resolve_base_url(s.real))
UniLM.provider_capabilities(s::_RLHere) = UniLM.provider_capabilities(s.real)
UniLM.get_url(s::_RLHere, fim::FIMCompletion) = _rl_here(UniLM.get_url(s.real, fim))
UniLM._prefix_complete_url(s::_RLHere) = _rl_here(UniLM._prefix_complete_url(s.real))
# The stream seams: without them a wrapped provider's every streamed line would be dropped,
# its response head unread and its usage priced at the standard rates.
UniLM.decode_agentic_stream(s::_RLHere, chunk::String, state::UniLM.AgenticStreamState) =
    UniLM.decode_agentic_stream(s.real, chunk, state)
UniLM._stream_head!(s::_RLHere, state::UniLM.StreamState, head::HTTP.Response) = UniLM._stream_head!(s.real, state, head)
UniLM._billed_cost(s::_RLHere, model::String, u::TokenUsage, rates::UniLM.PriceRow) =
    UniLM._billed_cost(s.real, model, u, rates)
UniLM._continuation_body(s::_RLHere, body, finish, token) = UniLM._continuation_body(s.real, body, finish, token)
UniLM._join_hops(s::_RLHere, earlier::HTTP.Response, later::HTTP.Response) = UniLM._join_hops(s.real, earlier, later)
# How long the provider's event handler takes per event: zero, except in the test that slows it.
const _RL_PARSE_DELAY = Ref(0.0)
function UniLM.handle_sse_event!(s::_RLHere, event::AbstractString, payload::AbstractString, state::UniLM.StreamState)
    _RL_PARSE_DELAY[] > 0 && sleep(_RL_PARSE_DELAY[])
    UniLM.handle_sse_event!(s.real, event, payload, state)
end

# A backend that authenticates in the query string, as Google's `?key=` form does.
struct _RLQueryKey <: UniLM.OpenAIWireEndpoint end
UniLM.get_url(::_RLQueryKey, ::Chat) = _RL_MOCK.url * "/v1/chat/completions?key=" * _RL_QUERY_KEY
UniLM.auth_header(::_RLQueryKey) = ["Content-Type" => "application/json"]

# Azure's URL is configuration, not a credential: replay needs it to name the request.
const _RL_AZURE = ("AZURE_OPENAI_BASE_URL" => _RL_MOCK.url, "AZURE_OPENAI_API_VERSION" => "2024-10-21",
                   "AZURE_OPENAI_DEPLOY_NAME_GPT_5_4_MINI" => "dep-mini")
_rl_online(f::Function) = withenv(f, _RL_KEYS..., _RL_AZURE...)
_rl_offline(f::Function) = withenv(f, (k => nothing for k in keys(_RL_KEYS))..., _RL_AZURE...)

_rl_chat(service; model, prompt="Say hello.", kws...) =
    Chat(; service, model, messages=[Message(Val(:system), "Be brief."), Message(Val(:user), prompt)], kws...)

const _RL_OPENAI = _RLHere(OPENAIServiceEndpoint)
const _RL_ADD = Tool(func=FunctionSignature(name="add", description="Add two numbers",
    parameters=Dict("type" => "object", "required" => ["a", "b"],
                    "properties" => Dict("a" => Dict("type" => "number"), "b" => Dict("type" => "number")))))
_rl_add(name, args) = string(args["a"] + args["b"])
_rl_loop_chat() = _rl_chat(_RL_OPENAI; model="gpt-5.4-mini", prompt="Add 2 and 3.", tools=[_RL_ADD])
_rl_loop_respond() = Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input="Add 2 and 3.", tools=[_RL_ADD])

# Everything a caller reads off a result.
_rl_seen(r::LLMSuccess) = (JSON.json(r.message), r.message.content, r.message.finish_reason, r.usage,
                           estimated_cost(r), cumulative_cost(r.self), JSON.json(r.self.messages))
_rl_seen(r::ResponseSuccess) = (output_text(r), r.response.id, r.response.status, r.response.model,
                                r.response.usage, r.response.raw, estimated_cost(r))
_rl_seen(r::EmbeddingSuccess) = (embedding_vectors(r), r.usage, r.raw, estimated_cost(r))
_rl_seen(r::LLMCallError) = (r.request_id, r.status, r.error, typeof(r.cause), JSON.json(r.self.messages))
_rl_seen(r::ResponseFailure) = (r.request_id, r.status, JSON.parse(r.response))
_rl_seen(r::ToolLoopResult) = (r.completed, r.turns_used, r.llm_error, _rl_seen(r.response),
    [(o.tool_name, o.arguments, o.success, o.result.result) for o in r.tool_calls])

# name, call, the path the provider is sent, the id header it answers with, the result type
const _RL_CASES = [
    ("openai-chat", () -> chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini"); config=_RL_CFG),
     "/v1/chat/completions", "x-request-id", LLMSuccess),
    ("openai-responses", () -> respond(Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input="Say hello.");
                                       config=_RL_CFG),
     "/v1/responses", "x-request-id", ResponseSuccess),
    ("openai-embeddings", () -> embeddingrequest!(Embeddings(["hello", "world"]; service=_RL_OPENAI,
                                                             model="text-embedding-3-small", dimensions=3);
                                                  config=_RL_CFG),
     "/v1/embeddings", "x-request-id", EmbeddingSuccess),
    ("anthropic", () -> chatrequest!(_rl_chat(_RLHere(ANTHROPICServiceEndpoint); model="claude-haiku-4-5");
                                     config=_RL_CFG),
     "/v1/messages", "request-id", LLMSuccess),
    ("gemini", () -> chatrequest!(_rl_chat(_RLHere(GEMINIServiceEndpoint); model="gemini-3.8-flash"); config=_RL_CFG),
     "/v1beta/models/gemini-3.8-flash:generateContent", nothing, LLMSuccess),
    ("gemini-compat", () -> chatrequest!(_rl_chat(_RLHere(GEMINIOpenAIServiceEndpoint); model="gemini-3.8-flash");
                                         config=_RL_CFG),
     "/v1beta/openai/chat/completions", "x-request-id", LLMSuccess),
    ("deepseek", () -> chatrequest!(_rl_chat(_RLHere(DeepSeekEndpoint(_RL_DEEPSEEK_KEY)); model="deepseek-flash");
                                    config=_RL_CFG),
     "/v1/chat/completions", "x-request-id", LLMSuccess),
    ("azure", () -> chatrequest!(_rl_chat(AZUREServiceEndpoint; model="gpt-5.4-mini"); config=_RL_CFG),
     "/openai/deployments/dep-mini/chat/completions", "x-request-id", LLMSuccess),
    ("query-key", () -> chatrequest!(_rl_chat(_RLQueryKey(); model="gpt-5.4-mini"); config=_RL_CFG),
     "/v1/chat/completions", "x-request-id", LLMSuccess),
    # 200s the decoder refuses: the id reaches the caller only through the restored header.
    ("openai-undecodable", () -> chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", prompt="undecodable");
                                              config=_RL_CFG),
     "/v1/chat/completions", "x-request-id", LLMCallError),
    ("responses-failed", () -> respond(Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input="undecodable");
                                       config=_RL_CFG),
     "/v1/responses", "x-request-id", ResponseFailure),
    ("anthropic-undecodable", () -> chatrequest!(_rl_chat(_RLHere(ANTHROPICServiceEndpoint); model="claude-haiku-4-5",
                                                          prompt="undecodable"); config=_RL_CFG),
     "/v1/messages", "request-id", LLMCallError),
]

@testset "recorded LLM answers — each provider's reply replays like the live one, with no credential on disk" begin
    @testset "$name" for (name, call, path, header, T) in _RL_CASES
        dir = _ra_dir("llm-" * name)
        before = _RL_HITS[]
        live = _rl_online(() -> with_recorded_answers(call, dir; mode=:record))
        @test live isa T
        @test _RL_HITS[] == before + 1
        sent = @lock _RL_LOCK last(_RL_SEEN)
        @test any(s -> contains(sent, s), _RL_SECRETS)          # the credential went to the mock...
        file = only(readdir(dir; join=true))
        text = read(file, String)
        @test !any(s -> contains(text, s), _RL_SECRETS)         # ...and into no recording
        @test !contains(text, "key=") && !contains(text, "api-version") && !contains(text, "sk-")
        @test !any(h -> contains(lowercase(text), h),
                   ("bearer", "authorization", "api-key", "x-goog-api-key", "cookie", "organization"))
        rec = JSON.parse(text)
        @test rec["request"]["method"] == "POST" && rec["request"]["path"] == path
        res = rec["response"]
        if isnothing(header)
            @test res["request_id"] === nothing && !haskey(res, "request_id_header")
            @test isempty(UniLM._replayed(file).headers)
        else
            @test res["request_id"] == "req_$(before + 1)" && res["request_id_header"] == header
            # That one header and no other. Header names are case-insensitive, and HTTP.jl
            # 2.7 capitalises them on construction ("X-Request-Id") where 2.8 does not.
            replayed = UniLM._replayed(file)
            @test HTTP.header(replayed, header) == res["request_id"]
            @test [lowercase(String(first(h))) for h in replayed.headers] == [header]
        end
        # Replayed with no key: nothing reaches the mock, and the caller reads the same result.
        hits = _RL_HITS[]
        replayed = _rl_offline(() -> with_recorded_answers(call, dir))
        @test _RL_HITS[] == hits
        @test replayed isa T && _rl_seen(replayed) == _rl_seen(live)
        T === LLMSuccess && @test estimated_cost(live) > 0 && cumulative_cost(live.self) > 0
        T <: Union{LLMCallError,ResponseFailure} && @test replayed.request_id == "req_$(before + 1)"
    end
end

@testset "recorded LLM answers — a credential in the body is sent, and never keyed, written or reported" begin
    dir = _ra_dir("llm-mcp")
    mcp(token; input="Say hello.") = Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input=input,
        tools=[MCPTool(server_label="crm", server_url="https://mcp.example.com/sse", authorization="oauth-$token",
                       headers=Dict("Authorization" => "Bearer header-$token")),
               Dict("type" => "mcp", "server_label" => "drive", "connector_id" => "connector_googledrive",
                    "authorization" => "oauth-$token")])
    live = _rl_online(() -> with_recorded_answers(() -> respond(mcp("sentinel-1"); config=_RL_CFG), dir; mode=:record))
    @test live isa ResponseSuccess
    sent = @lock _RL_LOCK last(_RL_SEEN)
    @test contains(sent, "oauth-sentinel-1") && contains(sent, "header-sentinel-1")   # the service gets them...
    text = read(only(readdir(dir; join=true)), String)
    @test !contains(text, "sentinel-1")                                                  # ...no recording does
    tools = JSON.parse(text)["request"]["body"]["tools"]
    @test [t["authorization"] for t in tools] == ["<redacted>", "<redacted>"] && tools[1]["headers"] == "<redacted>"
    @test tools[1]["server_url"] == "https://mcp.example.com/sse" && tools[2]["connector_id"] == "connector_googledrive"

    # The key never saw a token: another one replays the same recording, with no key and no service.
    hits = _RL_HITS[]
    replayed = _rl_offline(() -> with_recorded_answers(() -> respond(mcp("sentinel-2"); config=_RL_CFG), dir))
    @test _RL_HITS[] == hits && _rl_seen(replayed) == _rl_seen(live)
    # A miss carries the body as it was keyed.
    miss = _rl_offline(() -> with_recorded_answers(() -> _ra_caught(() -> respond(mcp("sentinel-3"; input="Bye."))), dir))
    @test miss isa ReplayMissError && miss.key == _ra_key("POST", "/v1/responses", miss.body)
    @test !contains(repr(miss), "sentinel") && contains(miss.body, "\"authorization\":\"<redacted>\"")

    # Gemini Interactions: an MCP server's headers, and the search keys of a retrieval tool.
    gemini = Respond(service=_RLHere(GEMINIServiceEndpoint), model="gemini-3.8-flash", input="Say hello.",
        tools=[Dict("type" => "mcp_server", "name" => "crm", "url" => "https://mcp.example.com/mcp",
                    "headers" => Dict("Authorization" => "Bearer sentinel-4")),
               Dict("type" => "retrieval", "retrieval_types" => ["exa_ai_search", "parallel_ai_search"],
                    "exa_ai_search_config" => Dict("api_key" => "sentinel-5"),
                    "parallel_ai_search_config" => Dict("api_key" => "sentinel-6"))])
    g = _rl_offline(() -> with_recorded_answers(() -> _ra_caught(() -> respond(gemini)), mkpath(_ra_dir("llm-mcp-gemini"))))
    @test g isa ReplayMissError && g.path == "/v1beta/interactions" && !contains(repr(g), "sentinel")
    server, retrieval = JSON.parse(g.body)["tools"]
    @test server["headers"] == "<redacted>" && server["url"] == "https://mcp.example.com/mcp"
    @test retrieval["exa_ai_search_config"]["api_key"] == retrieval["parallel_ai_search_config"]["api_key"] == "<redacted>"
    @test retrieval["retrieval_types"] == ["exa_ai_search", "parallel_ai_search"]

    # A body with no credential is keyed byte for byte, as it always was: never re-encoded.
    for plain in ("""{"tools": [{"type": "function", "name": "f"}], "temperature": 0.10}""",
                  """{"tools": [{"type": "mcp", "server_label": "crm"}], "temperature": 0.10}""",
                  "not JSON, though it mentions \"tools\"")
        @test UniLM._redacted(plain) == plain
    end
end

@testset "recorded answers — an image a body carries as a data URL is keyed and written as its digest" begin
    png, jpeg = base64encode(UInt8[0x89, 0x50, 0x4e, 0x47, 0x01, 0x02]), base64encode(UInt8[0xff, 0xd8, 0x03])
    body = JSON.json(JSON.Object{String,Any}(
        "messages" => [JSON.Object{String,Any}("role" => "user", "content" => [
            JSON.Object{String,Any}("type" => "text", "text" => "an image follows; its URL holds ;base64,"),
            JSON.Object{String,Any}("type" => "image_url", "image_url" => JSON.Object{String,Any}("url" => "data:image/png;base64," * png))])],
        "input" => Any["data:image/jpeg;name=photo.jpg;base64," * jpeg, "data:text/plain,plain", 7]))
    keyed = UniLM._redacted(body)
    sha(s) = bytes2hex(UniLM.SHA.sha256(s))
    parsed = JSON.parse(keyed)
    @test parsed["messages"][1]["content"][2]["image_url"]["url"] == "data:image/png;sha256," * sha(png)
    @test parsed["input"] == Any["data:image/jpeg;name=photo.jpg;sha256," * sha(jpeg), "data:text/plain,plain", 7]
    @test parsed["messages"][1]["content"][1]["text"] == "an image follows; its URL holds ;base64,"   # not a data URL
    @test !contains(keyed, png) && !contains(keyed, jpeg)
    @test UniLM._redacted(keyed) == keyed                    # a digest is not a data URL: keying is idempotent
    # A body whose text merely mentions the marker is keyed byte for byte, never re-encoded.
    for plain in ("""{"note": "a ;base64, b", "temperature": 0.10}""", "not JSON ;base64, at all")
        @test UniLM._redacted(plain) == plain
    end
end

@testset "recorded answers — the committed recordings hold no data URL, so every key stays as it was" begin
    files = readdir(joinpath(pkgdir(UniLM), "docs", "recorded_answers"); join=true)
    @test length(files) >= 146
    for f in files
        body = JSON.parse(read(f, String))["request"]["body"]
        isnothing(body) && continue                         # a GET has none
        text = JSON.json(body)
        @test !contains(text, ";base64,") && UniLM._redacted(text) == text
    end
end

@testset "recorded LLM answers — an image attachment is keyed and written as its digest, never its bytes" begin
    images = joinpath(@__DIR__, "fixtures", "typesafe", "images")
    red, blue = (ImageAttachment(read(joinpath(images, f))) for f in ("red-300.png", "blue-300.png"))
    pic(img) = chatrequest!(Chat(service=_RL_OPENAI, model="gpt-5.4-mini",
                                 messages=[Message(Val(:user), "What colour is this?", img)]); config=_RL_CFG)
    dir = _ra_dir("llm-image")
    live = _rl_online(() -> with_recorded_answers(() -> pic(red), dir; mode=:record))
    @test live isa LLMSuccess
    b64 = base64encode(red.data)
    @test contains(@lock(_RL_LOCK, last(_RL_SEEN)), b64)              # the provider gets the image
    text = read(only(readdir(dir; join=true)), String)
    @test !contains(text, b64) && !contains(text, "base64,")
    @test contains(text, "data:image/png;sha256," * bytes2hex(UniLM.SHA.sha256(b64)))
    hits = _RL_HITS[]
    replayed = _rl_offline(() -> with_recorded_answers(() -> pic(red), dir))
    @test _RL_HITS[] == hits && _rl_seen(replayed) == _rl_seen(live)
    miss = _rl_offline(() -> with_recorded_answers(() -> _ra_caught(() -> pic(blue)), dir))
    @test miss isa ReplayMissError && !contains(miss.body, "base64,") &&
          contains(miss.body, "data:image/png;sha256," * bytes2hex(UniLM.SHA.sha256(base64encode(blue.data))))
end

const _RL_DEEPSEEK = _RLHere(DeepSeekEndpoint(_RL_DEEPSEEK_KEY))
_rl_prefix_chat() = push!(_rl_chat(_RL_DEEPSEEK; model="deepseek-flash", prompt="Write f."),
                          Message(role=UniLM.RoleAssistant, content="def f("))
# What a caller reads off a lifecycle, FIM or prefix result: a parsed JSON answer as it is.
_rl_answer(d::AbstractDict) = d
_rl_answer(r::FIMSuccess) = (fim_text(r), r.response.usage, r.response.model, r.response.raw, estimated_cost(r))
_rl_answer(r) = _rl_seen(r)
# Every result these verbs return for a failed call.
const _RL_FAILURES = Union{ResponseCallError,ResponseFailure,LLMCallError,LLMFailure,FIMCallError,FIMFailure}
# The Responses lifecycle verbs on OpenAI, FIM and prefix completion on DeepSeek: one request each.
const _RL_LIFECYCLE = (
    get_response = () -> get_response("resp_rl"; service=_RL_OPENAI),
    list_input_items = () -> list_input_items("resp_rl"; limit=2, service=_RL_OPENAI),
    count_input_tokens = () -> count_input_tokens(; input="hi", service=_RL_OPENAI),
    compact_response = () -> compact_response(; input="hi", service=_RL_OPENAI),
    cancel_response = () -> cancel_response("resp_rl"; service=_RL_OPENAI),
    delete_response = () -> delete_response("resp_rl"; service=_RL_OPENAI),
    fim_complete = () -> fim_complete("def f("; suffix="): pass", service=_RL_DEEPSEEK, model="deepseek-flash"),
    prefix_complete = () -> prefix_complete(_rl_prefix_chat()))

@testset "recorded answers — the Responses lifecycle and FIM verbs record once and replay offline" begin
    dir = _ra_dir("llm-lifecycle")
    calls() = map(call -> call(), _RL_LIFECYCLE)
    live = _rl_online(() -> with_recorded_answers(calls, dir; mode=:record))
    hits = _RL_HITS[]
    replayed = _rl_offline(() -> with_recorded_answers(calls, dir))
    @test _RL_HITS[] == hits                                   # nothing reached the server
    @test length(readdir(dir)) == 8
    @test map(typeof, replayed) == map(typeof, live)
    @test !any(r -> r isa _RL_FAILURES, replayed)
    @test map(_rl_answer, replayed) == map(_rl_answer, live)  # what a caller reads, not only its type
    # The page a listing asks for is part of the key: another limit, order or cursor is another request.
    for page in ((; limit=3), (; limit=2, order="asc"), (; limit=2, after="msg_x"))
        miss = _rl_offline(() -> with_recorded_answers(
            () -> _ra_caught(() -> list_input_items("resp_rl"; page..., service=_RL_OPENAI)), dir))
        @test miss isa ReplayMissError && miss.path == "/v1/responses/resp_rl/input_items"
    end
end

@testset "recorded answers — a lifecycle or FIM verb throws its miss and its lost write, never returns them" begin
    empty, dir = mkpath(_ra_dir("llm-lifecycle-empty")), _ra_dir("llm-lifecycle-unwritable")
    @testset "$name" for (name, call) in pairs(_RL_LIFECYCLE)
        @test _rl_offline(() -> with_recorded_answers(() -> _ra_caught(call), empty)) isa ReplayMissError
        before = _RL_HITS[]
        e = _rl_online(() -> with_recorded_answers(dir; mode=:record) do
            rm(dir; force=true, recursive=true)
            write(dir, "")   # the checked directory is a file by the time the reply arrives
            _ra_caught(call)
        end)
        @test e isa RecordingWriteError && _RL_HITS[] == before + 1   # sent once, and the paid reply's loss is thrown
        rm(dir)
    end
end

@testset "recorded LLM answers — a two-turn tool loop records two files and replays them offline" begin
    for (name, loop) in (("chat", () -> tool_loop!(_rl_loop_chat(), _rl_add; config=_RL_CFG)),
                         ("respond", () -> tool_loop(_rl_loop_respond(), _rl_add; config=_RL_CFG)),
                         ("callable", () -> tool_loop!(_rl_loop_chat(); tools=[CallableTool(_RL_ADD, _rl_add)],
                                                       config=_RL_CFG)))
        dir = _ra_dir("llm-loop-" * name)
        before = _RL_HITS[]
        live = _rl_online(() -> with_recorded_answers(loop, dir; mode=:record))
        @test live.completed && live.turns_used == 2 && _RL_HITS[] == before + 2
        @test length(readdir(dir)) == 2
        @test [o.result.result for o in live.tool_calls] == ["5"]
        replayed = _rl_offline(() -> with_recorded_answers(loop, dir))
        @test _RL_HITS[] == before + 2
        @test _rl_seen(replayed) == _rl_seen(live)
    end
end

# ─── Streamed calls ──────────────────────────────────────────────────────────

# A streamed call run to its result, with the text deltas its callback received.
function _rl_streamed(start::Function)
    deltas = String[]
    r = fetch(start((d, close) -> d isa String && push!(deltas, d)))
    (r, deltas)
end
_rl_stream_chat(service, model; kws...) =
    _rl_streamed(cb -> chatrequest!(_rl_chat(service; model, stream=true, kws...); config=_RL_CFG, callback=cb))
_rl_openai_stream() = _rl_stream_chat(_RL_OPENAI, "gpt-5.4-mini")
function _rl_respond_stream()
    deltas = String[]
    task = respond("Say hello."; service=_RL_OPENAI, model="gpt-5.4-mini", config=_RL_CFG) do chunk, close
        chunk isa String && push!(deltas, chunk)
    end
    (fetch(task), deltas)
end

const _RL_STREAM_CASES = [
    ("openai-chat", _rl_openai_stream),
    ("responses", _rl_respond_stream),
    ("anthropic", () -> _rl_stream_chat(_RLHere(ANTHROPICServiceEndpoint), "claude-haiku-4-5")),
    ("gemini", () -> _rl_stream_chat(_RLHere(GEMINIServiceEndpoint), "gemini-3.8-flash")),
    ("interactions", () -> _rl_streamed(cb -> respond(Respond(service=_RLHere(GEMINIServiceEndpoint),
        model="gemini-3.8-flash", input="Say hello.", stream=true); config=_RL_CFG, callback=cb))),
    ("ollama", () -> _rl_stream_chat(OllamaEndpoint(base_url=_RL_MOCK.url), "gemma4:e4b")),
    ("deepseek", () -> _rl_stream_chat(_RL_DEEPSEEK, "deepseek-flash")),
]

_rl_text(r::LLMSuccess) = r.message.content
_rl_text(r::ResponseSuccess) = output_text(r)

# A streamed chat that asks the mock for a variant of its stream. A `:stopped` call sets its close
# flag from the final message's callback, and a `:callback_throws` call throws there.
_rl_stream_case(case::Symbol) = _rl_streamed() do cb
    callback = case === :stopped ? (d, close) -> (cb(d, close); d isa Message && (close[] = true)) :
               case === :callback_throws ? (d, close) -> (cb(d, close); d isa Message && error("the callback failed")) :
               cb
    chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", prompt="variant:$case", stream=true);
                 config=_RL_CFG, callback)
end

# Gemini Flex, which the mock names in the response head alone, and a reply's cost at the standard rates.
_rl_gemini_flex_stream() = _rl_stream_chat(_RLHere(GEMINIServiceEndpoint), "gemini-3.8-flash"; service_tier="flex")
_rl_gemini_flex_generate() =
    (chatrequest!(_rl_chat(_RLHere(GEMINIServiceEndpoint); model="gemini-3.8-flash", service_tier="flex");
                  config=_RL_CFG), String[])
_rl_standard_cost(r::LLMSuccess) = estimated_cost(LLMSuccess(; r.message, r.self, r.model,
    usage=TokenUsage(; UniLM._usage_fields(token_usage(r))..., service_tier=nothing)))

_rl_gemini_continuation_stream() = _rl_stream_chat(_RLHere(GEMINIServiceEndpoint), "gemini-4-argon"; prompt="variant:hops")
_rl_hops(r::LLMSuccess) = count("hop ", r.message.content)

# With no callback no user code runs, so only the rethrow keeps a lost write from being sent again.
const _RL_RETRYING = UniLM.RequestConfig(max_attempts=3, total_deadline=30.0)
_rl_openai_stream_nocb() =
    fetch(chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", stream=true); config=_RL_RETRYING))
_rl_respond_stream_nocb() =
    fetch(respond(Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input="Say hello.", stream=true); config=_RL_RETRYING))
_rl_unwrap(e) = e isa TaskFailedException ? _rl_unwrap(e.task.exception) :
                e isa CompositeException ? _rl_unwrap(first(e.exceptions)) :
                e isa CapturedException ? _rl_unwrap(e.ex) : e

# A replayed stream whose callback cancels the call at its first delta.
function _rl_cancelled_replay()
    tok = CancelToken()
    fetch(chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", stream=true); config=_RL_CFG, cancel=tok,
                       callback=(d, close) -> d isa String && cancel!(tok)))
end

const _RA_COMMITTED = joinpath(pkgdir(UniLM), "docs", "recorded_answers")

@testset "recorded LLM answers — a streamed call records once and replays offline with the same deltas" begin
    @testset "$name" for (name, call) in _RL_STREAM_CASES   # OpenAI chat, Responses (do-block), Anthropic,
        dir = _ra_dir("llm-stream-$name")                   # Gemini native, Interactions, Ollama NDJSON, DeepSeek
        live, live_deltas = _rl_online(() -> with_recorded_answers(call, dir; mode=:record))
        @test only(JSON.parsefile.(joinpath.(dir, readdir(dir))))["response"]["stream"] isa Vector
        hits = _RL_HITS[]
        replayed, replayed_deltas = _rl_offline(() -> with_recorded_answers(call, dir))
        @test _RL_HITS[] == hits
        @test !isempty(live_deltas) && join(replayed_deltas) == join(live_deltas)    # the deltas, not none of them
        @test _rl_text(replayed) == _rl_text(live)
    end
end
@testset "recorded LLM answers — a streamed miss is thrown by the verb itself, before any task runs" begin
    empty = mkpath(_ra_dir("llm-stream-miss"))
    @test_throws ReplayMissError _rl_offline(() -> with_recorded_answers(empty) do
        chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", stream=true); config=_RL_CFG)
    end)
    # respond too, a Respond with stream=true and a do-block callback alike
    @test_throws ReplayMissError _rl_offline(() -> with_recorded_answers(empty) do
        respond(Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input="Say hello.", stream=true); config=_RL_CFG)
    end)
    @test_throws ReplayMissError _rl_offline(() -> with_recorded_answers(empty) do
        respond((chunk, close) -> nothing, "Say hello."; service=_RL_OPENAI, model="gpt-5.4-mini", config=_RL_CFG)
    end)
end
@testset "recorded LLM answers — an in-band error, a truncated, a stopped or a callback-failed stream is never recorded" begin
    for case in (:inband_error, :truncated, :stopped, :callback_throws)
        dir = _ra_dir("llm-stream-unrecorded-$case")
        _rl_online(() -> with_recorded_answers(() -> _rl_stream_case(case), dir; mode=:record))
        @test isempty(readdir(dir))
    end
    # A respond stream its callback stops — here as the whole response arrives — is not recorded either.
    dir = _ra_dir("llm-respond-stream-stopped")
    stopped = _rl_online(() -> with_recorded_answers(dir; mode=:record) do
        fetch(respond("Say hello."; service=_RL_OPENAI, model="gpt-5.4-mini", config=_RL_CFG) do chunk, close
            chunk isa UniLM.ResponseObject && (close[] = true)
        end)
    end)
    @test stopped isa ResponseSuccess && isempty(readdir(dir))
end
@testset "recorded LLM answers — a replayed Gemini reply keeps its tier, so it costs what the live one cost" begin
    for (name, call) in (("stream", _rl_gemini_flex_stream), ("generate", _rl_gemini_flex_generate))
        dir = _ra_dir("llm-tier-$name")                     # the mock names the tier only in the response head
        live, _ = _rl_online(() -> with_recorded_answers(call, dir; mode=:record))
        replayed, _ = _rl_offline(() -> with_recorded_answers(call, dir))
        @test token_usage(live).service_tier == "flex"
        @test estimated_cost(live) < _rl_standard_cost(live)     # flex is really priced as flex
        @test estimated_cost(replayed) == estimated_cost(live)
    end
end
@testset "recorded LLM answers — streams through nested scopes; the inner scope records what the outer replayed" begin
    outer, inner = _ra_dir("llm-stream-outer"), _ra_dir("llm-stream-inner")
    _rl_online(() -> with_recorded_answers(_rl_openai_stream, outer; mode=:record))
    _rl_offline(() -> with_recorded_answers(outer) do
        with_recorded_answers(_rl_openai_stream, inner; mode=:record_missing)
    end)
    @test readdir(inner) == readdir(outer)
end
@testset "recorded answers — an old recording without kept headers still replays as today" begin
    for f in readdir(_RA_COMMITTED)
        j = JSON.parsefile(joinpath(_RA_COMMITTED, f))["response"]
        haskey(j, "headers") && continue                       # written with its kept headers: a newer recording
        r = UniLM._replayed(joinpath(_RA_COMMITTED, f))
        @test r.status == j["status"] && JSON.parse(String(r.body)) == j["body"]
        # A recording that names no header is System One's (the first header `_replayed` knows).
        j["request_id"] === nothing ||
            @test HTTP.header(r, get(j, "request_id_header", "x-typesafe-request-id")) == j["request_id"]
    end
end
@testset "recorded LLM answers — CRLF lines and a missing final newline replay byte for byte" begin
    for variant in (:crlf, :no_final_newline)
        dir = _ra_dir("llm-stream-$variant")
        live, _ = _rl_online(() -> with_recorded_answers(() -> _rl_stream_case(variant), dir; mode=:record))
        @test join(only(JSON.parsefile.(joinpath.(dir, readdir(dir))))["response"]["stream"], "\n") == _rl_mock_bytes(variant)
        replayed, _ = _rl_offline(() -> with_recorded_answers(() -> _rl_stream_case(variant), dir))
        @test _rl_text(replayed) == _rl_text(live)
    end
end
@testset "recorded LLM answers — a Gemini continuation records and replays every hop" begin
    dir = _ra_dir("llm-stream-hops")
    live, _ = _rl_online(() -> with_recorded_answers(_rl_gemini_continuation_stream, dir; mode=:record))
    @test length(readdir(dir)) == _rl_hops(live)
    replayed, _ = _rl_offline(() -> with_recorded_answers(_rl_gemini_continuation_stream, dir))
    @test _rl_text(replayed) == _rl_text(live)
end
@testset "recorded LLM answers — a paid streamed reply that cannot be written fails loudly and is sent once" begin
    for call in (_rl_openai_stream_nocb, _rl_respond_stream_nocb)       # chatrequest! and respond, no callback
        hits = _RL_HITS[]
        err = _rl_online(() -> _ra_caught(() -> with_recorded_answers(call, _rl_unwritable_dir(); mode=:record)))
        @test _rl_unwrap(err) isa RecordingWriteError                    # through fetch's TaskFailedException
        @test _RL_HITS[] == hits + 1                                     # RequestConfig(max_attempts=3): no resend
    end
end
@testset "recorded LLM answers — cancelling a replayed stream ends it with UniLMCancelled" begin
    src = _ra_dir("llm-stream-cancel-src")
    _rl_online(() -> with_recorded_answers(_rl_openai_stream, src; mode=:record))
    r = _rl_offline(() -> with_recorded_answers(() -> _rl_cancelled_replay(), src))
    @test r isa LLMCallError && r.cause isa UniLMCancelled
end
@testset "recorded LLM answers — a later hop's miss reaches the caller at fetch, as its task's failure" begin
    dir = _ra_dir("llm-stream-hops-gap")
    _rl_online(() -> with_recorded_answers(_rl_gemini_continuation_stream, dir; mode=:record))
    second = only(f for f in readdir(dir; join=true)
                  if get(JSON.parsefile(f)["request"]["body"], "continuationToken", nothing) == "tok-1")
    rm(second)
    err = _rl_offline(() -> with_recorded_answers(() -> _ra_caught(_rl_gemini_continuation_stream), dir))
    @test err isa TaskFailedException && _rl_unwrap(err) isa ReplayMissError
end
@testset "recorded LLM answers — a miss or lost write of a call in a stream callback reaches the caller at fetch" begin
    # A recorded call's miss, or its paid answer's lost recording, is thrown out of the verb and never
    # absorbed as a service failure — also when a stream's callback made that call, bare or inside the
    # TaskFailedException a fetched task gives it.
    src = _ra_dir("llm-stream-callback-miss")
    _rl_online(() -> with_recorded_answers(_rl_openai_stream, src; mode=:record))
    _rl_online(() -> with_recorded_answers(_rl_respond_stream, src; mode=:record))
    inner() = chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", prompt="Not recorded."); config=_RL_CFG)
    chat_stream(cb) = chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", stream=true); config=_RL_CFG, callback=cb)
    respond_stream(cb) = respond(cb, "Say hello."; service=_RL_OPENAI, model="gpt-5.4-mini", config=_RL_CFG)
    missed = (d, close) -> d isa String && inner()
    fetched = (d, close) -> d isa String && fetch(Threads.@spawn inner())
    for start in (chat_stream, respond_stream), (callback, thrown) in ((missed, ReplayMissError),
                                                                        (fetched, TaskFailedException))
        e = _rl_offline(() -> with_recorded_answers(() -> _ra_caught(() -> fetch(start(callback))), src))
        @test e isa TaskFailedException && e.task.exception isa thrown && _rl_unwrap(e) isa ReplayMissError
    end
    # The stream and the callback's call are both answered; the call's recording cannot be written.
    for start in (chat_stream, respond_stream)
        dir = _rl_unwritable_dir(after=1)
        hits = _RL_HITS[]
        e = _rl_online(() -> _ra_caught(() -> fetch(start((d, close) -> d isa String &&
                                                          with_recorded_answers(inner, dir; mode=:record)))))
        @test e isa TaskFailedException && e.task.exception isa RecordingWriteError
        @test _RL_HITS[] == hits + 2                                     # nothing sent again
        ispath(dir) && rm(dir; recursive=true)
    end
end

# A raw peer that answers every request with `bytes` as one chunk of a chunked body cut before its
# end — the last chunk without its final CRLF, then a close — so the read that would end the body
# fails after the stream's bytes; `posts` counts the requests.
function _rl_cut_stream_server(bytes::String)
    listener = Sockets.listen(Sockets.localhost, 0)
    url = "http://127.0.0.1:$(Int(Sockets.getsockname(listener)[2]))"
    posts = Threads.Atomic{Int}(0)
    function drain(sock)                     # the request: its head, then its body, sized or chunked
        req = ""
        while (he = findfirst("\r\n\r\n", req)) === nothing
            b = readavailable(sock); isempty(b) && return; req *= String(b)
        end
        head = SubString(req, 1, last(he))
        len = match(r"(?i)content-length:\s*(\d+)", head)
        whole() = occursin(r"(?i)transfer-encoding:\s*chunked", head) ? occursin("0\r\n\r\n", req) :
                  isnothing(len) || sizeof(req) >= last(he) + parse(Int, len.captures[1])
        while !whole()
            b = readavailable(sock); isempty(b) && return; req *= String(b)
        end
    end
    Threads.@spawn try
        while true
            sock = Sockets.accept(listener)
            Threads.atomic_add!(posts, 1)
            try
                drain(sock)
                write(sock, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nx-request-id: req_cut\r\n" *
                            "Transfer-Encoding: chunked\r\n\r\n")
                write(sock, string(string(sizeof(bytes); base=16), "\r\n", bytes, "\r\n", "0\r\n"))
            finally
                close(sock)
            end
        end
    catch                                    # the listener closed: the test is over
    end
    (; listener, url, posts)
end

@testset "recorded LLM answers — a stream whose connection fails after its final event is recorded; one cut before it is not" begin
    # A reply is recorded once it is whole — its final event arrived — and no stop ended its reading,
    # whatever the connection does after that event. A stream cut before it is not recorded, though
    # the call keeps the turn its finish reason completed.
    chat_bytes = _rl_chat_stream(Dict("model" => "mock"), 1, nothing)       # through `data: [DONE]`
    respond_bytes = _rl_responses_stream(Dict("model" => "mock"), 1)        # through `response.completed`
    cfg = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)
    chat_call(url) = fetch(chatrequest!(Chat(service=GenericOpenAIEndpoint(url, ""), model="mock", stream=true,
        messages=[Message(Val(:system), "Be brief."), Message(Val(:user), "Say hello.")]); config=cfg))
    respond_call(url) = fetch(respond(Respond(service=GenericOpenAIEndpoint(url, ""), model="mock",
                                              input="Say hello.", stream=true); config=cfg))
    for (name, call, bytes) in (("chat", chat_call, chat_bytes), ("respond", respond_call, respond_bytes))
        srv = _rl_cut_stream_server(bytes)
        dir = _ra_dir("llm-stream-cut-after-final-$name")
        try
            live = with_recorded_answers(() -> call(srv.url), dir; mode=:record)
            @test live isa Union{LLMSuccess,ResponseSuccess} && srv.posts[] == 1
            @test length(readdir(dir)) == 1                                  # recorded, as a clean stream is
            replayed = with_recorded_answers(() -> call(srv.url), dir)
            @test srv.posts[] == 1 && _rl_text(replayed) == _rl_text(live)
        finally
            close(srv.listener)
        end
    end
    srv = _rl_cut_stream_server(String(chopsuffix(chat_bytes, "data: [DONE]\n\n")))   # the finish chunk, then the cut
    dir = _ra_dir("llm-stream-cut-before-final")
    try
        live = with_recorded_answers(() -> chat_call(srv.url), dir; mode=:record)
        @test live isa LLMSuccess && live.message.finish_reason == "stop"   # the turn is kept …
        @test isempty(readdir(dir))                                         # … and not recorded
    finally
        close(srv.listener)
    end
end
@testset "recorded LLM answers — a turn that ends on its second hop's miss or lost write bills its first, streamed or not" begin
    # One turn of two requests, unstreamed and streamed. Its second request misses on replay, or
    # its second reply's recording cannot be written: either way the first reply was billed, and
    # the chat's total adds it the same on both paths.
    turn(stream) = _rl_chat(_RLHere(GEMINIServiceEndpoint); model="gemini-4-argon", prompt="variant:two_hops", stream)
    send(chat) = (r = chatrequest!(chat; config=_RL_CFG); r isa Task ? fetch(r) : r)
    function missed(stream)
        dir = _ra_dir("llm-hops-billed-miss-$stream")
        _rl_online(() -> with_recorded_answers(() -> send(turn(stream)), dir; mode=:record))
        rm(only(f for f in readdir(dir; join=true) if haskey(JSON.parsefile(f)["request"]["body"], "continuationToken")))
        chat = turn(stream)
        @test _rl_unwrap(_rl_offline(() -> with_recorded_answers(() -> _ra_caught(() -> send(chat)), dir))) isa ReplayMissError
        cumulative_cost(chat)
    end
    function lost(stream)
        chat = turn(stream)
        e = _rl_online(() -> _ra_caught(() -> with_recorded_answers(() -> send(chat), _rl_unwritable_dir(after=1); mode=:record)))
        @test _rl_unwrap(e) isa RecordingWriteError
        cumulative_cost(chat)
    end
    for case in (missed, lost)
        unstreamed, streamed = case(false), case(true)
        @test unstreamed > 0
        @test streamed == unstreamed
    end
    # A user's interrupt from the callback during the second reply: the first was billed, as above.
    chat = turn(true)
    interrupt = (d, close) -> d isa String && occursin("hop 2", d) && throw(InterruptException())
    e = _rl_online(() -> _ra_caught(() -> fetch(chatrequest!(chat; config=_RL_CFG, callback=interrupt))))
    @test e isa TaskFailedException && _rl_unwrap(e) isa InterruptException
    @test cumulative_cost(chat) == missed(false)
end
@testset "recorded LLM answers — outside a scope a stream reads its API key first, as it always did" begin
    tok = CancelToken()
    cancel!(tok)
    r = _rl_offline(() -> fetch(chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", stream=true);
                                             config=_RL_CFG, cancel=tok)))
    @test r isa LLMCallError
    @test r.cause isa KeyError                               # the missing key, not the stop
end
@testset "recorded LLM answers — a replayed stream arms no idle guard, so a slow parse never times it out" begin
    dir = _ra_dir("llm-stream-slow")
    live, _ = _rl_online(() -> with_recorded_answers(_rl_openai_stream, dir; mode=:record))
    tight = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0, stream_idle_timeout=0.05)
    replayed = try
        _RL_PARSE_DELAY[] = 0.25                             # five idle bounds without a read, per event
        _rl_offline(() -> with_recorded_answers(dir) do
            fetch(chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini", stream=true); config=tight))
        end)
    finally
        _RL_PARSE_DELAY[] = 0.0
    end
    @test replayed isa LLMSuccess && _rl_text(replayed) == _rl_text(live)
end

@testset "recorded LLM answers — a paid reply that cannot be written is thrown, never returned" begin
    dir = _ra_dir("llm-unwritable")
    calls = (() -> chatrequest!(_rl_chat(_RL_OPENAI; model="gpt-5.4-mini"); config=_RL_CFG),
             () -> respond(Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input="Say hello."); config=_RL_CFG),
             () -> embeddingrequest!(Embeddings("hello"; service=_RL_OPENAI, model="text-embedding-3-small",
                                                dimensions=3); config=_RL_CFG),
             () -> tool_loop!(_rl_loop_chat(), _rl_add; config=_RL_CFG),
             () -> tool_loop(_rl_loop_respond(), _rl_add; config=_RL_CFG))
    _rl_online() do
        for mode in (:record, :record_missing), call in calls
            before = _RL_HITS[]
            e = with_recorded_answers(dir; mode) do
                rm(dir; force=true, recursive=true)
                write(dir, "")   # the checked directory is a file by the time the reply arrives
                _ra_caught(call)
            end
            # The lost write — not a call error, and not a loop that goes on.
            @test e isa RecordingWriteError && e.cause isa Union{Base.IOError,SystemError}
            @test dirname(e.file) == abspath(dir)
            @test _RL_HITS[] == before + 1
            rm(dir)
        end
    end
    # A dispatcher's own paid request whose recording is lost stops the loop too: as a
    # tool error the model would read it and the loop would go on without the recording.
    lost = (name, args) -> with_recorded_answers(dir; mode=:record) do
        rm(dir; force=true, recursive=true)
        write(dir, "")
        embeddingrequest!(Embeddings("hello"; service=_RL_OPENAI, model="text-embedding-3-small", dimensions=3);
                          config=_RL_CFG)
        "5"
    end
    _rl_online() do
        for (loop, turn) in ((n -> tool_loop!(_rl_loop_chat(), lost; config=_RL_CFG, tool_concurrency=n), "/chat/completions"),
                             (n -> tool_loop(_rl_loop_respond(), lost; config=_RL_CFG, tool_concurrency=n), "/responses")),
            n in (1, 2)
            before = _RL_HITS[]
            e = _ra_caught(() -> loop(n))
            @test e isa RecordingWriteError && dirname(e.file) == abspath(dir)
            # The first turn and the dispatcher's request were sent; no second turn followed.
            @test _RL_HITS[] == before + 2
            @test contains(@lock(_RL_LOCK, _RL_SEEN[end - 1]), turn)
            rm(dir)
        end
    end
end

@testset "recorded LLM answers — a miss escapes the verbs and the tool loops, never a call error" begin
    empty = mkpath(_ra_dir("llm-empty"))
    tools = [CallableTool(_RL_ADD, _rl_add)]
    hits = _RL_HITS[]
    _rl_offline() do
        with_recorded_answers(empty) do
            chat = _rl_chat(_RL_OPENAI; model="gpt-5.4-mini")
            e = _ra_caught(() -> chatrequest!(chat; config=_RL_CFG))
            @test e isa ReplayMissError && length(chat) == 2            # nothing appended
            @test e.dir == abspath(empty) && e.method == "POST" && e.path == "/v1/chat/completions"
            @test e.body == UniLM.encode_request(_RL_OPENAI, chat) && e.reason == "no recording"
            @test e.key == _ra_key("POST", "/v1/chat/completions", e.body)
            msg = sprint(showerror, e)
            @test contains(msg, "POST /v1/chat/completions") && contains(msg, "for model \"gpt-5.4-mini\"")
            @test contains(msg, "access to the provider") && contains(msg, ":record_missing")
            @test !contains(msg, "TYPESAFE_API_KEY") && !contains(msg, "asking") && !contains(msg, "\n")

            @test_throws ReplayMissError chatrequest!(; service=_RL_OPENAI, model="gpt-5.4-mini",
                                                      systemprompt="Be brief.", userprompt="Say hello.")
            @test_throws ReplayMissError respond(Respond(service=_RL_OPENAI, model="gpt-5.4-mini", input="Say hello."))
            @test_throws ReplayMissError respond("Say hello."; service=_RL_OPENAI, model="gpt-5.4-mini")
            @test_throws ReplayMissError embeddingrequest!(Embeddings("hello"; service=_RL_OPENAI,
                                                                      model="text-embedding-3-small"))
            loop_chat = _rl_loop_chat()
            @test_throws ReplayMissError tool_loop!(loop_chat, _rl_add)
            @test length(loop_chat) == 2
            @test_throws ReplayMissError tool_loop!(_rl_loop_chat(); tools)
            @test_throws ReplayMissError tool_loop(_rl_loop_respond(), _rl_add)
            @test_throws ReplayMissError tool_loop(Respond(service=_RL_OPENAI, model="gpt-5.4-mini",
                                                           input="Add 2 and 3.", tools=tools))
            @test_throws ReplayMissError tool_loop("Add 2 and 3."; tools, service=_RL_OPENAI, model="gpt-5.4-mini")

            # A model in the path, not the body: the request line names it.
            g = _ra_caught(() -> chatrequest!(_rl_chat(_RLHere(GEMINIServiceEndpoint); model="gemini-3.8-flash")))
            @test g isa ReplayMissError && g.path == "/v1beta/models/gemini-3.8-flash:generateContent"
            @test !contains(sprint(showerror, g), "for model")
            # A credential in the query reaches neither the key nor the message.
            q = _ra_caught(() -> chatrequest!(_rl_chat(_RLQueryKey(); model="mock")))
            @test q isa ReplayMissError && q.path == "/v1/chat/completions"
            @test !contains(sprint(showerror, q), _RL_QUERY_KEY) && !contains(sprint(showerror, q), "?")
        end
    end
    @test _RL_HITS[] == hits && isempty(readdir(empty))
end

@testset "recorded LLM answers — in a replayed loop, a dispatcher's own miss still escapes" begin
    # Only the loop's first turn is recorded: the dispatcher's `ask` has no recording.
    dir = _ra_dir("llm-dispatch-miss")
    _rl_online(() -> with_recorded_answers(() -> chatrequest!(_rl_loop_chat(); config=_RL_CFG), dir; mode=:record))
    jev = (name, args) -> ask("the ticket", "urgent" => noul("Is it urgent?"))
    e = _rl_offline(() -> with_recorded_answers(() -> _ra_caught(() -> tool_loop!(_rl_loop_chat(), jev)), dir))
    @test e isa ReplayMissError && e.path == "/v1/systemone"
end

@testset "recorded LLM answers — a fetched stream's lost write or miss stops a tool loop and answers an MCP client -32603" begin
    # A dispatcher or an MCP tool handler that fetches a streamed call gets its lost write, or a
    # later hop's miss, inside the TaskFailedException `fetch` raises (out of an @sync block,
    # inside a CompositeException; out of an asyncmap, inside a CapturedException). As a tool
    # error the model would read it, a local path with it, and go on; a loop throws that
    # exception whole instead, and an MCP server logs it and answers the generic -32603.
    gap = _ra_dir("llm-dispatch-stream-gap")             # a continued stream without its second hop
    _rl_online(() -> with_recorded_answers(_rl_gemini_continuation_stream, gap; mode=:record))
    rm(only(f for f in readdir(gap; join=true)
            if get(JSON.parsefile(f)["request"]["body"], "continuationToken", nothing) == "tok-1"))
    lost_dir = _ra_dir("llm-dispatch-stream-lost")
    lost = (name, args) -> with_recorded_answers(lost_dir; mode=:record) do
        rm(lost_dir; force=true, recursive=true)
        write(lost_dir, "")                              # a file by the time the streamed reply arrives
        _rl_openai_stream()
        "5"
    end
    missed = (name, args) -> (with_recorded_answers(_rl_gemini_continuation_stream, gap); "5")
    synced = function (name, args)                       # the same miss out of an @sync block
        @sync @async missed(name, args)
        "5"
    end
    mapped = (name, args) -> only(asyncmap(_ -> missed(name, args), 1:1))   # and out of an asyncmap
    _rl_online() do
        for (dispatcher, wrapper, cause, sent) in ((lost, TaskFailedException, RecordingWriteError, 2),
                                                   (missed, TaskFailedException, ReplayMissError, 1),
                                                   (synced, CompositeException, ReplayMissError, 1),
                                                   (mapped, CapturedException, ReplayMissError, 1)), n in (1, 2)
            chat = _rl_loop_chat()
            for loop in (() -> tool_loop!(chat, dispatcher; config=_RL_CFG, tool_concurrency=n),
                         () -> tool_loop(_rl_loop_respond(), dispatcher; config=_RL_CFG, tool_concurrency=n))
                before = _RL_HITS[]
                e = _ra_caught(loop)
                @test e isa wrapper && _rl_unwrap(e) isa cause   # what the dispatcher threw, whole
                @test _RL_HITS[] == before + sent       # the first turn (and the lost stream); no second turn
                ispath(lost_dir) && rm(lost_dir; recursive=true)
            end
            @test length(chat) == 2                     # the chat loop's turn is rolled back
        end
        for (handler, dir) in ((lost, lost_dir), (missed, gap), (synced, gap), (mapped, gap))
            server = MCPServer("fetched-stream", "1.0.0")
            register_tool!(server, "add", "Adds two numbers", Dict{String,Any}("type" => "object"),
                           args -> handler("add", args))
            req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 23, "method" => "tools/call",
                                   "params" => Dict{String,Any}("name" => "add", "arguments" => Dict{String,Any}()))
            resp = @test_logs (:error,) match_mode=:any UniLM._dispatch_guarded(server, req)
            err = get(resp, "error", Dict{String,Any}())
            @test !haskey(resp, "result") && get(err, "code", nothing) == -32603 &&
                  get(err, "message", nothing) == "Internal error"
            @test !contains(JSON.json(resp), dir) && !contains(JSON.json(resp), "recording")   # no local path
            ispath(lost_dir) && rm(lost_dir; recursive=true)
        end
    end
end

@testset "recorded answers — the id header is kept by name, and a recording without one is System One's" begin
    source = JSON.parse(read(_ra_file(_ra_dir("replay"), _ra_request()), String))
    @test source["response"]["request_id_header"] == "x-typesafe-request-id"
    id = source["response"]["request_id"]
    # As written before the header's name was kept (and before the kept headers).
    old = JSON.parse(JSON.json(source))
    delete!(old["response"], "request_id_header")
    delete!(old["response"], "headers")
    dir = mkpath(_ra_dir("old-format"))
    write(_ra_file(dir, _ra_request()), JSON.json(old))
    r = _ra_offline(() -> with_recorded_answers(() -> ask(_ra_request(); config=_RA_CFG), dir))
    @test r isa SystemOneSuccess && r.response.request_id == id
    # A name outside the three is not a recording.
    for name in ("authorization", "X-Request-Id", 7)
        old["response"]["request_id_header"] = name
        write(_ra_file(dir, _ra_request()), JSON.json(old))
        e = _ra_offline(() -> with_recorded_answers(() -> _ra_caught(() -> ask(_ra_request(); config=_RA_CFG)), dir))
        @test e isa ReplayMissError && startswith(e.reason, "unreadable recording: ") &&
              contains(e.reason, "request_id_header")
    end
end

@testset "recorded answers — a request keyed on more than its body files the extra with it" begin
    # A System One backend keys its provider options (or image parts) after the body; the
    # exchange is driven directly here, with a stand-in for the network.
    dir = mkpath(_ra_dir("key-extra"))
    body = JSON.json(_ra_request("keyed on more"))
    extra = "long_context=true"
    answered = HTTP.Response(200, ["x-typesafe-request-id" => "req_extra"],
                             Vector{UInt8}("""{"model":"m","answers":{},"usage":{}}"""))
    calls = Ref(0)
    exchange(mode::Symbol, key_extra::String) = with_recorded_answers(dir; mode) do
        UniLM._recorded_exchange(() -> (calls[] += 1; answered), "POST", "/v1/systemone", body, time_ns(), nothing;
                                 key_extra)
    end
    @test exchange(:record, extra).status == 200 && calls[] == 1
    file = joinpath(dir, _ra_key("POST", "/v1/systemone", body * "\n" * extra) * ".json")
    rec = JSON.parse(read(file, String))
    @test rec["request"]["key_extra"] == extra && rec["request"]["body"] == JSON.parse(body)
    # Replayed under the same extra; with none, or another, it is another request.
    @test exchange(:replay, extra).status == 200 && calls[] == 1
    miss = _ra_caught(() -> exchange(:replay, ""))
    @test miss isa ReplayMissError && miss.body == body && miss.key == _ra_key("POST", "/v1/systemone", body)
    other = _ra_caught(() -> exchange(:replay, "long_context=false"))
    @test other isa ReplayMissError && other.body == body * "\nlong_context=false"
    @test other.key == _ra_key("POST", "/v1/systemone", other.body)
    @test contains(sprint(showerror, other), "asking \"team\", \"upset\", \"urgency\"")   # its questions still named
    # With no extra, nothing in the file names one: every earlier key and file stays as it was.
    exchange(:record, "")
    plain = JSON.parse(read(joinpath(dir, _ra_key("POST", "/v1/systemone", body) * ".json"), String))
    @test !haskey(plain["request"], "key_extra")
end

close(_RL_MOCK.server)

@testset "raw base64 media digests like a data URL" begin
    img = ImageAttachment(joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png"))
    wav = AudioAttachment(joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav"))
    b64 = UniLM.Base64.base64encode(img.data)
    @test UniLM._media_digest(img) == UniLM._image_digest("data:image/png;base64," * b64)
    @test startswith(UniLM._media_digest(wav), "data:audio/wav;sha256,")
    @test length(UniLM._media_digest(img)) == length("data:image/png;sha256,") + 64
    mp3 = AudioAttachment(Vector{UInt8}("ID3" * "\0"^7))
    for (clip, mime) in ((wav, "audio/wav"), (mp3, "audio/mpeg"))
        @test UniLM._media_digest(clip) == UniLM._image_digest("data:$mime;base64," * UniLM.Base64.base64encode(clip.data))
    end
    # A format with no media type of its own is an error, never filed as another format.
    @test_throws ArgumentError UniLM._audio_mime("flac")
    # Every format the sniffer can return (each `return "<format>"` in its source) has a media type.
    file, line = functionloc(only(methods(UniLM._sniff_audio_format)))
    returned(ex) = ex isa Expr ? (ex.head === :return && length(ex.args) == 1 && ex.args[1] isa String ?
        String[ex.args[1]] : mapreduce(returned, vcat, ex.args; init=String[])) : String[]
    formats = unique(returned(Meta.parse(join(readlines(file)[line:end], '\n'), 1)[1]))
    @test "wav" in formats && "mp3" in formats                                  # the source was read
    @test all(f -> startswith(UniLM._audio_mime(f), "audio/"), formats)
end
