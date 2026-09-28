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
# Responses, Anthropic Messages, Gemini generateContent, embeddings — each with its
# own request-id header (Gemini sends none), beside headers no recording may keep.
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

_rl_gemini_reply(b, n, raw) = Dict(
    "candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "reply $n")]),
                          "finishReason" => "STOP", "index" => 0)],
    "usageMetadata" => Dict("promptTokenCount" => 100 + n, "candidatesTokenCount" => 10 + n,
                            "totalTokenCount" => 110 + 2n),
    "modelVersion" => "gemini-3.8-flash", "responseId" => "gem_$n")

function _rl_embeddings_reply(b, n, raw)
    inputs = b["input"] isa AbstractString ? [b["input"]] : b["input"]
    Dict("object" => "list", "model" => b["model"],
         "data" => [Dict("object" => "embedding", "index" => i - 1, "embedding" => [n / 8, i / 8, 0.5])
                    for i in eachindex(inputs)],
         "usage" => Dict("prompt_tokens" => 3 + n, "total_tokens" => 3 + n))
end

function _rl_serve(http::HTTP.Stream)
    req = http.message
    raw = String(read(http))
    n = Threads.atomic_add!(_RL_HITS, 1) + 1
    @lock _RL_LOCK push!(_RL_SEEN, join([String(req.target); [string(k, ": ", v) for (k, v) in req.headers]; raw], "\n"))
    b = JSON.parse(raw; dicttype=Dict{String,Any})
    path = String(HTTP.URI(String(req.target)).path)
    HTTP.setstatus(http, 200)
    HTTP.setheader(http, "set-cookie" => "session=cookie-sentinel")      # never recorded
    HTTP.setheader(http, "openai-organization" => "org-sentinel")        # never recorded
    if get(b, "stream", false) === true
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        chunk(delta, finish) = "data: " * JSON.json(Dict("id" => "c$n", "object" => "chat.completion.chunk",
            "choices" => [Dict("index" => 0, "delta" => delta, "finish_reason" => finish)])) * "\n\n"
        write(http, chunk(Dict("role" => "assistant", "content" => "streamed $n"), nothing) *
                    chunk(Dict(), "stop") * "data: [DONE]\n\n")
        return
    end
    id, reply = endswith(path, "/messages") ? ("request-id" => "req_$n", _rl_anthropic_reply(b, n, raw)) :
                occursin(":generateContent", path) ? (nothing, _rl_gemini_reply(b, n, raw)) :
                endswith(path, "/responses") ? ("x-request-id" => "req_$n", _rl_responses_reply(b, n, raw)) :
                endswith(path, "/embeddings") ? ("x-request-id" => "req_$n", _rl_embeddings_reply(b, n, raw)) :
                ("x-request-id" => "req_$n", _rl_chat_reply(b, n, raw))
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
            @test UniLM._replayed(file).headers == [header => res["request_id"]]
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

@testset "recorded LLM answers — a streamed call is never recorded: it reaches the service as outside a scope" begin
    streamed() = fetch(chatrequest!(_rl_chat(GenericOpenAIEndpoint(_RL_MOCK.url, ""); model="gpt-5.4-mini",
                                             stream=true); config=_RL_CFG))
    empty = mkpath(_ra_dir("llm-stream-replay"))
    recording = _ra_dir("llm-stream-record")
    before = _RL_HITS[]
    outside = streamed()
    inside = [with_recorded_answers(streamed, empty),
              with_recorded_answers(streamed, recording; mode=:record),
              with_recorded_answers(streamed, recording; mode=:record_missing)]
    @test _RL_HITS[] == before + 4
    @test isempty(readdir(empty)) && isempty(readdir(recording))
    for (k, r) in enumerate([outside; inside])
        @test r isa LLMSuccess && r.message.content == "streamed $(before + k)" && r.message.finish_reason == "stop"
    end
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
            @test contains(msg, "the provider's API key") && contains(msg, ":record_missing")
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

@testset "recorded answers — the id header is kept by name, and a recording without one is System One's" begin
    source = JSON.parse(read(_ra_file(_ra_dir("replay"), _ra_request()), String))
    @test source["response"]["request_id_header"] == "x-typesafe-request-id"
    id = source["response"]["request_id"]
    # As written before the header's name was kept.
    old = JSON.parse(JSON.json(source))
    delete!(old["response"], "request_id_header")
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

close(_RL_MOCK.server)
