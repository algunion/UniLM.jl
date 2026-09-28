# test/recorded_answers.jl — recorded System One answers, zero-spend.
#
# One local mock plays the service for the whole file. It answers every question
# of a System One request — so `ask`, `nl_dispatch` and `@branch` all run against
# it — and the models listing, numbers its replies through the request-id header,
# and counts the requests it receives. Recording runs against it with a sentinel
# API key; the replay tests run after it is closed and with TYPESAFE_API_KEY
# unset, so a request that went for the network could not be sent.

using Sockets

const _RA_KEY = "ts-sentinel-3f9c2a71"   # must appear in no recording
const _RA_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)
const _RA_ROOT = mktempdir()
_ra_dir(name::AbstractString) = joinpath(_RA_ROOT, name)

const _RA_HITS = Threads.Atomic{Int}(0)

# The mock's answer to one question of request number `n`: the LAST Choice option
# (a replayed dispatch cannot pass by landing on the default first one), and
# numbers that carry `n`, so no two recordings look alike.
function _ra_answer(q::AbstractDict, n::Int)::Dict{String,Any}
    kind = q["type"]
    if kind == "choice"
        options = collect(keys(q["criteria"]))
        Dict{String,Any}("type" => kind, "choice" => last(options), "confidence" => 0.9,
            "probabilities" => Dict{String,Any}(
                o => (o == last(options) ? 0.9 : 0.1 / max(length(options) - 1, 1)) for o in options))
    elseif kind == "score"
        levels = q["criteria"]
        Dict{String,Any}("type" => kind, "score" => 1 + n / 1000, "confidence" => 0.5,
            "legend" => Dict{String,Any}(string(i - 1) => levels[i] for i in eachindex(levels)),
            "probabilities" => Dict{String,Any}(string(i - 1) => 1 / length(levels) for i in eachindex(levels)))
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
             () -> _ra_branch("unwritable"))
    _ra_online() do
        for mode in (:record, :record_missing), call in calls
            before = _RA_HITS[]
            e = with_recorded_answers(dir; mode) do
                rm(dir; force=true, recursive=true)
                write(dir, "")   # the checked directory is a file by the time the answer arrives
                _ra_caught(call)
            end
            # The write's own I/O error, not a call error a fallback would take for the service's.
            @test e isa Union{Base.IOError,SystemError}
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
