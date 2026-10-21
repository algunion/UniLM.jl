# Ollama model management: list_models/model_info/running_models decoding, load and
# unload over /api/chat, and the /api/pull NDJSON stream with its time and cancel
# contracts. Every exchange is with a local mock server; the fixtures in
# test/fixtures/ollama/ are live captures from Ollama 0.34.4 (`tags_0406.json`,
# `ps_0406.json` and `show_embeddinggemma2.json`: 0.40.6). Offline.

using Sockets

_om_fixture(name::AbstractString) = read(joinpath(@__DIR__, "fixtures", "ollama", name), String)
_om_line(d) = JSON.json(d) * "\n"
const _OM_CFG = RequestConfig(max_attempts=1, total_deadline=30.0)

# A local Ollama stand-in: `serve(n, req, http)` answers the n-th request (`req` =
# method, target, body) on the raw server stream, at its own pace. Returns the server,
# its base URL, and the requests seen, in arrival order.
function _om_mock(serve::Function)
    seen = @NamedTuple{method::String, target::String, body::String}[]
    guard = ReentrantLock()
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        server = try
            HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
                head = HTTP.startread(http)
                req = (method=String(head.method), target=String(head.target), body=String(read(http)))
                n = @lock guard (push!(seen, req); length(seen))
                serve(n, req, http)
            end
        catch
            attempt == 5 && rethrow()
            continue
        end
        return server, "http://127.0.0.1:$port", seen
    end
end

function _om_send(http, status::Int, body::AbstractString)
    HTTP.setstatus(http, status)
    HTTP.setheader(http, "Content-Type" => "application/json")
    HTTP.startwrite(http)
    write(http, body)
end

# NDJSON lines, each written and flushed on its own, `gap` seconds apart (separate reads).
function _om_stream(http, lines; status::Int=200, gap::Float64=0.02)
    HTTP.setstatus(http, status)
    HTTP.setheader(http, "Content-Type" => "application/x-ndjson")
    HTTP.startwrite(http)
    for l in lines
        write(http, l)
        flush(http)
        gap > 0 && sleep(gap)
    end
end

# The fixture server: /api/tags, /api/ps, and /api/show for the two captured models.
function _om_fixtures(_, req, http)
    req.target == "/api/tags" && return _om_send(http, 200, _om_fixture("tags.json"))
    req.target == "/api/ps" && return _om_send(http, 200, _om_fixture("ps.json"))
    model = JSON.parse(req.body)["model"]
    model == "gemma4:latest" && return _om_send(http, 200, _om_fixture("show_gemma4.json"))
    model == "embeddinggemma:latest" && return _om_send(http, 200, _om_fixture("show_embeddinggemma.json"))
    _om_send(http, 404, JSON.json(Dict("error" => "model '$model' not found")))
end

const _OM_PULL = [
    _om_line(Dict("status" => "pulling manifest")),
    # Ollama omits a zero "completed" (omitempty), as in the first report of a layer.
    (_om_line(filter(kv -> kv[2] != 0, Dict{String,Any}("status" => "pulling 4e30e2665218",
                   "digest" => "sha256:4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448",
                   "total" => 7162405886, "completed" => c))) for c in (0, 3_000_000_000, 7162405886))...,
    _om_line(Dict("status" => "verifying sha256 digest")),
    _om_line(Dict("status" => "writing manifest")),
    _om_line(Dict("status" => "success"))]

# Refuted if: a field is read from the wrong key or decoded to the wrong value/type, a
# verb reaches the wrong route, or the one-line `show` drops the name, size or capabilities.
@testset "fixtures decode into the typed structs" begin
    server, base, seen = _om_mock(_om_fixtures)
    try
        ep = OllamaEndpoint(base_url=base)
        r = list_models(; service=ep, config=_OM_CFG)
        @test r isa OllamaSuccess{Vector{OllamaModel}} && issuccess(r)
        @test seen[end].method == "GET" && seen[end].target == "/api/tags"
        @test [m.name for m in r.response] ==
              ["gemma4:e2b", "gemma4:31b", "qwen2.5-coder:1.5b-base", "embeddinggemma:latest", "gemma4:latest"]
        m = r.response[1]
        @test (m.size, m.digest, m.modified_at) ==
              (7162405886, "7fbdbf8f5e45a75bb122155ed546e765b4d9c53a1285f62fd9f506baa1c5a47e",
               "2026-09-29T10:05:41.659835542+03:00")
        @test (m.family, m.parameter_size, m.quantization, m.context_length) == ("gemma4", "5.1B", "Q4_K_M", 131072)
        @test m.capabilities == [:completion, :vision, :audio, :tools, :thinking]
        @test m.raw == JSON.parse(_om_fixture("tags.json"))["models"][1]
        emb = r.response[4]
        @test (emb.family, emb.context_length, emb.capabilities) == ("gemma3", 2048, [:embedding])
        @test r.response[3].capabilities == [:completion, :insert]
        @test sprint(show, m) == "OllamaModel(\"gemma4:e2b\", 7.2 GB, [:completion, :vision, :audio, :tools, :thinking])"
        @test sprint(show, emb) == "OllamaModel(\"embeddinggemma:latest\", 0.6 GB, [:embedding])"

        i = model_info("gemma4:latest"; service=ep, config=_OM_CFG)
        @test i isa OllamaSuccess{OllamaModelInfo}
        @test seen[end].method == "POST" && seen[end].target == "/api/show"
        @test JSON.parse(seen[end].body) == Dict("model" => "gemma4:latest")
        g = i.response
        @test g.name == "gemma4:latest"
        @test g.capabilities == [:completion, :vision, :audio, :tools, :thinking]
        @test g.context_length == 131072
        @test g.thinking_levels == [false, true] && g.thinking_default === true
        @test g.parameters == Dict("temperature" => 1, "top_k" => 64, "top_p" => 0.95)
        @test g.parameters["top_k"] isa Int && g.parameters["top_p"] isa Float64
        @test (g.family, g.parameter_size, g.quantization) == ("gemma4", "8.0B", "Q4_K_M")
        @test g.raw["requires"] == "0.20.0"
        e = model_info("embeddinggemma:latest"; service=ep, config=_OM_CFG).response
        @test e.context_length == 2048 && e.capabilities == [:embedding]
        @test isempty(e.thinking_levels) && isnothing(e.thinking_default)       # cannot think
        @test e.parameters == Dict("num_batch" => 2048, "num_ctx" => 2048)
        @test (e.family, e.parameter_size, e.quantization) == ("gemma3", "307.58M", "BF16")

        p = running_models(; service=ep, config=_OM_CFG)
        @test p isa OllamaSuccess{Vector{OllamaRunningModel}}
        @test seen[end].method == "GET" && seen[end].target == "/api/ps"
        loaded = only(p.response)
        @test (loaded.name, loaded.size, loaded.size_vram, loaded.context_length, loaded.expires_at) ==
              ("gemma4:e2b", 8141469121, 8141469121, 131072, "2026-09-29T10:11:00.036845+03:00")
        @test sprint(show, loaded) == "OllamaRunningModel(\"gemma4:e2b\", 8.1 GB, 8.1 GB in VRAM, context 131072)"

        f = model_info("gemma4:nope"; service=ep, config=_OM_CFG)
        @test f isa OllamaFailure && f.status == 404 && !issuccess(f)
        @test JSON.parse(f.response) == Dict("error" => "model 'gemma4:nope' not found")
    finally
        close(server)
    end
end

# Refuted if: a repeated parameter keeps only one value, a quoted string keeps its
# quotes, or a thinking model with named levels (gpt-oss) does not decode.
@testset "show: parameters text and named thinking levels" begin
    @test UniLM._decode_parameters("stop                           \"<start_of_turn>\"\n" *
                                  "stop                           \"<end_of_turn>\"\nnum_ctx 4096\ntemperature 0.7") ==
          Dict("stop" => ["<start_of_turn>", "<end_of_turn>"], "num_ctx" => 4096, "temperature" => 0.7)
    @test UniLM._decode_parameters("stop \"a\\nb\"\nuse_mmap false\nseed \"42\"") ==
          Dict("stop" => "a\nb", "use_mmap" => false, "seed" => "42")          # a quoted number stays text
    @test isempty(UniLM._decode_parameters(""))
    @test_throws ArgumentError UniLM._decode_parameters("orphan")
    reply = Dict("capabilities" => ["completion", "tools", "thinking"], "parameters" => "temperature 1",
                 "model_info" => Dict("general.architecture" => "gptoss", "gptoss.context_length" => 131072),
                 "thinking" => Dict("values" => ["low", "medium", "high"], "default" => "medium"))
    server, base, _ = _om_mock((_, _, http) -> _om_send(http, 200, JSON.json(reply)))
    try
        i = model_info("gpt-oss:20b"; service=OllamaEndpoint(base_url=base), config=_OM_CFG).response
        @test i.thinking_levels == ["low", "medium", "high"] && i.thinking_default == "medium"
        @test i.context_length == 131072
        @test isnothing(i.family) && isnothing(i.quantization)                  # no details: nothing, not ""
    finally
        close(server)
    end
end

# Refuted if: a missing or mistyped required field decodes to a default (0, "", nothing)
# instead of failing, or a verb returns a struct built from part of a reply.
@testset "malformed replies are call errors, never half-filled structs" begin
    tags = JSON.parse(_om_fixture("tags.json"))
    ps = JSON.parse(_om_fixture("ps.json"))
    without(d, k) = filter(kv -> kv[1] != k, d)
    bodies = [JSON.json(Dict("models" => [tags["models"][1], without(tags["models"][2], "name")])),
              JSON.json(Dict("models" => [without(tags["models"][1], "size")])),
              JSON.json(Dict("models" => [merge(tags["models"][1], Dict("size" => "7 GB"))])),
              JSON.json(Dict("models" => [merge(tags["models"][1], Dict("capabilities" => [1]))])),
              JSON.json(Dict("tags" => [])), "[]", "not json",
              JSON.json(Dict("models" => [without(ps["models"][1], "size_vram")])),
              JSON.json(Dict("models" => [without(ps["models"][1], "expires_at")])),
              JSON.json(Dict("thinking" => Dict("values" => [false, 2]))),
              JSON.json(Dict("parameters" => 7))]
    server, base, _ = _om_mock((n, _, http) -> _om_send(http, 200, bodies[n]))
    try
        ep = OllamaEndpoint(base_url=base)
        calls = [fill(() -> list_models(; service=ep, config=_OM_CFG), 7);
                 fill(() -> running_models(; service=ep, config=_OM_CFG), 2);
                 fill(() -> model_info("m"; service=ep, config=_OM_CFG), 2)]
        for (k, call) in enumerate(calls)
            r = call()
            @test r isa OllamaCallError && isnothing(r.status) && r.cause isa Exception
            k == 1 && @test occursin("\"name\"", r.error)
            k == 2 && @test occursin("\"size\"", r.error)
            k == 3 && @test occursin("\"size\"", r.error) && occursin("String", r.error)
            k == 8 && @test occursin("\"size_vram\"", r.error)
            k == 9 && @test occursin("\"expires_at\"", r.error)
        end
    finally
        close(server)
    end
end

# Refuted if: a load drops the endpoint's keep_alive/options/runner (or sends them in the
# wrong wire form), an unload carries anything but keep_alive 0 and the endpoint's runner,
# a reply that is not the one asked for (done_reason) reads as success, or local misuse
# reaches the network. /api/show reports a chat model, so every load is an empty chat.
@testset "load and unload: request bodies and the reply each expects" begin
    reasons = ["load", "load", "load", "unload", "stop", "load", "load", "unload"]
    chats = Ref(0)
    server, base, seen = _om_mock((_, req, http) -> req.target == "/api/show" ?
        _om_send(http, 200, JSON.json(Dict("capabilities" => ["completion"]))) :
        _om_send(http, 200, JSON.json(Dict(
        "model" => "gemma4:e2b", "created_at" => "2026-09-29T10:00:00Z",
        "message" => Dict("role" => "assistant", "content" => ""), "done" => true, "done_reason" => reasons[chats[] += 1]))))
    chat(k) = filter(r -> r.target == "/api/chat", seen)[k]                          # the k-th load or unload
    try
        ep = OllamaEndpoint(base_url=base, keep_alive=Inf, num_ctx=8192, num_gpu=99)
        r = load_model("gemma4:e2b"; service=ep, config=_OM_CFG)
        @test r isa OllamaSuccess{Nothing} && isnothing(r.response)
        @test [(x.method, x.target) for x in seen] == [("POST", "/api/show"), ("POST", "/api/chat")]
        @test JSON.parse(chat(1).body) == Dict("model" => "gemma4:e2b", "messages" => [], "stream" => false,
                                               "keep_alive" => -1, "options" => Dict("num_ctx" => 8192, "num_gpu" => 99))
        @test load_model("gemma4:e2b"; service=OllamaEndpoint(base_url=base), config=_OM_CFG) isa OllamaSuccess
        @test JSON.parse(chat(2).body) == Dict("model" => "gemma4:e2b", "messages" => [], "stream" => false)
        @test load_model("gemma4:e2b"; service=OllamaEndpoint(base_url=base, keep_alive=600, shift=false), config=_OM_CFG) isa OllamaSuccess
        @test JSON.parse(chat(3).body)["keep_alive"] == 600
        @test JSON.parse(chat(3).body)["shift"] === false    # a runner setting: the chat that follows must not reload
        u = unload_model("gemma4:e2b"; service=ep, config=_OM_CFG)
        @test u isa OllamaSuccess{Nothing}
        @test JSON.parse(chat(4).body) == Dict("model" => "gemma4:e2b", "messages" => [], "stream" => false, "keep_alive" => 0)
        bad = load_model("gemma4:e2b"; service=ep, config=_OM_CFG)                      # answered "stop"
        @test bad isa OllamaCallError && occursin("done_reason \"stop\"", bad.error) && bad.cause isa ArgumentError
        bad = unload_model("gemma4:e2b"; service=ep, config=_OM_CFG)                    # answered "load"
        @test bad isa OllamaCallError && occursin("\"unload\" was expected", bad.error)
        # The endpoint's runner goes with a load and an unload; without one, none is sent.
        pinned = OllamaEndpoint(base_url=base, runner=:ggml)
        @test load_model("gemma4:e2b"; service=pinned, config=_OM_CFG) isa OllamaSuccess
        @test unload_model("gemma4:e2b"; service=pinned, config=_OM_CFG) isa OllamaSuccess
        @test JSON.parse(chat(7).body)["runner"] == "ggml" && JSON.parse(chat(8).body)["runner"] == "ggml"
        @test !haskey(JSON.parse(chat(2).body), "runner") && !haskey(JSON.parse(chat(4).body), "runner")
        @test count(x -> x.target == "/api/show", seen) == 5                             # one per load, none per unload
        # Local misuse throws before any request.
        n = length(seen)
        @test_throws ArgumentError load_model("gemma4:e2b"; service=OllamaEndpoint(base_url=base, keep_alive=0))
        for verb in (model_info, load_model, unload_model, pull_model), name in ("", "  ")
            @test_throws ArgumentError verb(name; service=ep)
        end
        @test length(seen) == n
    finally
        close(server)
    end
end

# Refuted if: reports arrive out of order, lose digest/completed/total, the call does not
# end at "success", an undecodable line kills the pull (or goes uncounted), or the request
# is not {"model": name} to /api/pull.
@testset "pull: progress in order, then success" begin
    lines = [_OM_PULL[1:2]; "{not json}\n"; _OM_PULL[3:end]]
    server, base, seen = _om_mock((n, _, http) -> _om_stream(http, n == 1 ? lines : _OM_PULL))
    try
        ep = OllamaEndpoint(base_url=base)
        got = OllamaPullProgress[]
        before = UniLM._SSE_DROPPED_LINES[]
        r = @test_logs (:warn, r"undecodable") match_mode=:any pull_model("gemma4:e2b"; service=ep, progress=p -> push!(got, p))
        @test r isa OllamaSuccess{Nothing} && issuccess(r)
        @test UniLM._SSE_DROPPED_LINES[] - before == 1
        @test seen[1].method == "POST" && seen[1].target == "/api/pull"
        @test JSON.parse(seen[1].body) == Dict("model" => "gemma4:e2b")
        @test [p.status for p in got] == ["pulling manifest", "pulling 4e30e2665218", "pulling 4e30e2665218",
                                          "pulling 4e30e2665218", "verifying sha256 digest", "writing manifest", "success"]
        @test [p.completed for p in got[2:4]] == [0, 3_000_000_000, 7162405886]
        @test all(p -> p.total == 7162405886 && startswith(p.digest, "sha256:4e30e2665218"), got[2:4])
        @test all(p -> isnothing(p.digest) && isnothing(p.completed) && isnothing(p.total), got[[1, 5, 6, 7]])
        @test pull_model("gemma4:e2b"; service=ep) isa OllamaSuccess{Nothing}             # no progress callback
    finally
        close(server)
    end
end

# Refuted if: a failure the server reports on the 200 stream, a stream that ends early,
# a non-200 status or a throwing callback reads as success, or loses its message/cause.
@testset "pull: failures are typed results" begin
    boom = ErrorException("progress bar broke")
    scripts = [(200, [_OM_PULL[1], _om_line(Dict("error" => "pull model manifest: file does not exist"))]),
               (200, _OM_PULL[1:3]),                                              # EOF before success
               (400, [JSON.json(Dict("error" => "invalid model name"))]),
               (200, _OM_PULL)]
    server, base, seen = _om_mock((n, _, http) -> _om_stream(http, scripts[n][2]; status=scripts[n][1]))
    try
        ep = OllamaEndpoint(base_url=base)
        r = pull_model("gemma4:nope"; service=ep)
        @test r isa OllamaCallError && r.error == "pull model manifest: file does not exist" && isnothing(r.status)
        r = pull_model("gemma4:e2b"; service=ep)
        @test r isa OllamaCallError && occursin("ended before success", r.error)
        r = pull_model("bad name"; service=ep)
        @test r isa OllamaFailure && r.status == 400 && JSON.parse(r.response) == Dict("error" => "invalid model name")
        calls = Ref(0)
        r = pull_model("gemma4:e2b"; service=ep, progress=_ -> (calls[] += 1; calls[] == 2 && throw(boom)))
        @test r isa OllamaCallError && r.cause === boom && occursin("progress bar broke", r.error)
        @test calls[] == 2                                                        # nothing runs after the throw
        @test length(seen) == 4                                                   # one attempt each
    finally
        close(server)
    end
end

# Refuted if: a cancel mid-download waits for the server, a silent server outlives
# stream_idle_timeout, a server that never answers outlives request_timeout, time in
# `progress` counts as wire idleness, or request_timeout/total_deadline cut a healthy
# download that outlasts them. Timings are measured from the triggering event.
@testset "pull: cancel, timeouts, and a long healthy download" begin
    release = Threads.Atomic{Bool}(false)
    hold() = timedwait(() -> release[], 20.0)
    scripts = [http -> (_om_stream(http, _OM_PULL[1:2]); hold()),                 # 1: two lines, then holds
               http -> (_om_stream(http, _OM_PULL[1:1]); hold()),                 # 2: one line, then mute
               http -> hold(),                                                    # 3: never sends headers
               http -> _om_stream(http, _OM_PULL; gap=0.1),                       # 4: one read per line
               http -> _om_stream(http, _OM_PULL; gap=0.25)]                      # 5: ~1.75 s in all
    server, base, seen = _om_mock((n, _, http) -> scripts[n](http))
    try
        ep = OllamaEndpoint(base_url=base)
        tok = CancelToken()
        second = Threads.Atomic{Bool}(false)
        t = Threads.@spawn (pull_model("gemma4:e2b"; service=ep, cancel=tok,
                                       progress=p -> p.status == "pulling 4e30e2665218" && (second[] = true)), time_ns())
        @test timedwait(() -> second[], 20.0) === :ok
        cancelled_at = time_ns()
        cancel!(tok)
        @test timedwait(() -> istaskdone(t), 20.0) === :ok
        r, done_at = fetch(t)
        @test r isa OllamaCallError && r.cause isa UniLMCancelled && r.cause.source === :token
        @test (done_at - cancelled_at) / 1e9 < 2.0

        last_byte = Ref(UInt64(0))
        r = pull_model("gemma4:e2b"; service=ep, config=RequestConfig(stream_idle_timeout=0.5),
                       progress=_ -> (last_byte[] = time_ns()))
        waited = (time_ns() - last_byte[]) / 1e9
        @test r isa OllamaCallError && r.cause isa UniLMTimeout && r.cause.phase === :stream_idle
        @test waited < 3.0

        t0 = time_ns()
        r = pull_model("gemma4:e2b"; service=ep, config=RequestConfig(request_timeout=0.5))
        @test r isa OllamaCallError && r.cause isa UniLMTimeout && r.cause.phase === :request
        @test (time_ns() - t0) / 1e9 < 2.0
        release[] = true

        slow = pull_model("gemma4:e2b"; service=ep, config=RequestConfig(stream_idle_timeout=0.5),
                          progress=p -> p.status == "pulling manifest" && sleep(1.5))
        @test slow isa OllamaSuccess{Nothing}
        long = pull_model("gemma4:e2b"; service=ep,
                          config=RequestConfig(request_timeout=0.5, total_deadline=1.0, stream_idle_timeout=5.0))
        @test long isa OllamaSuccess{Nothing}
        @test length(seen) == 5
    finally
        release[] = true
        close(server)
    end
end

# Refuted if: a pre-cancelled token (explicit or ambient) lets a request out, or a verb
# reports a pre-cancel as anything but UniLMCancelled.
@testset "a cancelled token sends nothing" begin
    server, base, seen = _om_mock(_om_fixtures)
    try
        ep = OllamaEndpoint(base_url=base)
        tok = cancel!(CancelToken())
        verbs = [c -> list_models(; service=ep, cancel=c), c -> model_info("m"; service=ep, cancel=c),
                 c -> running_models(; service=ep, cancel=c), c -> load_model("m"; service=ep, cancel=c),
                 c -> unload_model("m"; service=ep, cancel=c), c -> pull_model("m"; service=ep, cancel=c)]
        for verb in verbs, r in (verb(tok), with_cancel(() -> verb(nothing), tok))
            @test r isa OllamaCallError && r.cause isa UniLMCancelled
        end
        @test isempty(seen)
    finally
        close(server)
    end
end

# Refuted if: an unreachable server's error lacks the Ollama remedy on any verb.
@testset "unreachable server: every verb says how to fix it" begin
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2]); close(tcp)                         # nothing listens here now
    ep = OllamaEndpoint(base_url="http://127.0.0.1:$port")
    for r in (list_models(; service=ep, config=_OM_CFG), model_info("m"; service=ep, config=_OM_CFG),
              running_models(; service=ep, config=_OM_CFG), load_model("m"; service=ep, config=_OM_CFG),
              unload_model("m"; service=ep, config=_OM_CFG), pull_model("m"; service=ep, config=_OM_CFG))
        @test r isa OllamaCallError
        @test occursin("no Ollama server answered at http://127.0.0.1:$port", r.error) && occursin("ollama serve", r.error)
    end
end

# Refuted if: routing by service type sends a TypeSafe or unsupported endpoint down the
# Ollama path (or the reverse).
@testset "list_models routes by service" begin
    @test UniLM.provider_capabilities(OllamaEndpoint()) ⊇ Set([:models])
    @test_throws ArgumentError list_models(; service=OPENAIServiceEndpoint)       # no models listing
    withenv(UniLM.TYPESAFE_API_KEY => nothing) do                                   # TypeSafe path, no network
        @test list_models() isa SystemOneCallError
    end
end

# Refuted if: a runner or format the server reports is dropped or mistyped (a runner is a
# Symbol; details in /api/tags and /api/show, the entry itself in /api/ps, else its
# details), an empty detail decodes to "" instead of nothing, or a reply without a runner
# gets one guessed.
@testset "runner and format (Ollama 0.40); an empty detail is not reported" begin
    fx(f) = JSON.parse(_om_fixture(f); dicttype=Dict{String,Any})
    ms = Dict(m.name => m for m in UniLM._decode_model.(UniLM._ollama_entries(fx("tags_0406.json"))))
    @test ms["gemma4:e4b"].runner === :mlx && ms["gemma4:e4b"].format == "safetensors"
    @test isnothing(ms["gemma4:e4b"].family) && isnothing(ms["gemma4:e4b"].parameter_size)   # "" in the reply
    @test isnothing(ms["gemma4:e4b"].quantization)                      # "quantization_level": "" in the capture
    d = deepcopy(ms["gemma4:e4b"].raw); d["details"]["runner"] = ""
    @test isnothing(UniLM._decode_model(d).runner)
    @test ms["kev:0.8b-q8_0"].runner === :llamacpp && ms["kev:0.8b-q8_0"].format == "gguf"
    ps = UniLM._decode_running.(UniLM._ollama_entries(fx("ps_0406.json")))
    @test Dict(r.name => r.runner for r in ps) == Dict("tev1:0.8b" => :mlx, "kev:0.8b-q8_0" => :llamacpp)
    psent(top, det) = Dict{String,Any}("name" => "m", "size" => 1, "size_vram" => 1, "expires_at" => "t",
                                       "runner" => top, "details" => Dict{String,Any}("runner" => det))
    @test UniLM._decode_running(psent(nothing, "ggml")).runner === :ggml   # none on the entry: details
    @test UniLM._decode_running(psent("", "mlx")).runner === :mlx          # empty on the entry: details
    @test isnothing(UniLM._decode_running(psent("", "")).runner)
    info = UniLM._decode_show("embeddinggemma-2:740m-q8_0", fx("show_embeddinggemma2.json"))
    @test info.runner === :llamacpp && info.format == "gguf" && :embedding in info.capabilities
    # The Ollama 0.34 capture reports `format` ("gguf") and no `runner`: not reported is nothing, not a guess.
    @test all(m -> isnothing(m.runner) && m.format == "gguf", UniLM._decode_model.(UniLM._ollama_entries(fx("tags.json"))))
    @test sprint(show, ms["gemma4:e4b"]) ==
          "OllamaModel(\"gemma4:e4b\", 9.5 GB, mlx, [:completion, :vision, :audio, :tools, :thinking])"
end

@testset "delete_model: DELETE /api/delete with the name; 404 when absent; a blank name sends nothing" begin
    seen = Tuple{String,String,String}[]
    handler = req -> (push!(seen, (String(req.method), String(req.target),
                                   applicable(copy, req.body) ? String(copy(req.body)) : ""));
        length(seen) == 1 ? HTTP.Response(200, UInt8[]) :      # Ollama answers a delete with an empty 200
        HTTP.Response(404, ["Content-Type" => "application/json"], Vector{UInt8}("{\"error\":\"model 'x' not found\"}")))
    tcp = Sockets.listen(Sockets.localhost, 0); port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    server = HTTP.serve!(handler, "127.0.0.1", port; verbose=false)
    try
        ep = OllamaEndpoint(base_url="http://127.0.0.1:$port")
        @test delete_model("all-minilm:22m"; service=ep) isa OllamaSuccess{Nothing}
        @test seen[1][1:2] == ("DELETE", "/api/delete") && JSON.parse(seen[1][3]) == Dict("model" => "all-minilm:22m")
        f = delete_model("x"; service=ep, config=RequestConfig(max_attempts=1))
        @test f isa OllamaFailure && f.status == 404
        @test_throws ArgumentError delete_model("  "; service=ep)
        @test length(seen) == 2
    finally
        close(server)
    end
end

@testset "load_model: a chat model with an empty chat, any other model with an empty embeddings request" begin
    show_reply(caps) = JSON.json(Dict("capabilities" => caps, "details" => Dict("format" => "gguf"),
                                      "model_info" => Dict{String,Any}(), "parameters" => ""))
    seen = Tuple{String,String}[]
    caps = Ref(["embedding"])
    handler = req -> begin
        body = applicable(copy, req.body) ? String(copy(req.body)) : ""
        push!(seen, (String(req.target), body))
        req.target == "/api/show" && return HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(show_reply(caps[])))
        req.target == "/api/embed" && return HTTP.Response(200, ["Content-Type" => "application/json"],
                                                           Vector{UInt8}("{\"model\":\"m\",\"embeddings\":[]}"))
        HTTP.Response(200, ["Content-Type" => "application/json"],
                      Vector{UInt8}("{\"model\":\"m\",\"done\":true,\"done_reason\":\"load\",\"message\":{\"role\":\"assistant\",\"content\":\"\"}}"))
    end
    tcp = Sockets.listen(Sockets.localhost, 0); port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    server = HTTP.serve!(handler, "127.0.0.1", port; verbose=false)
    try
        ep = OllamaEndpoint(base_url="http://127.0.0.1:$port", keep_alive=600, num_ctx=2048, runner=:llamacpp)
        for kind in (["embedding"], ["decision"])
            empty!(seen); caps[] = kind
            @test load_model("m"; service=ep) isa OllamaSuccess{Nothing}
            @test [t for (t, _) in seen] == ["/api/show", "/api/embed"]
            b = JSON.parse(seen[2][2])
            @test b["input"] == [] && b["keep_alive"] == 600 && b["options"]["num_ctx"] == 2048 && b["runner"] == "llamacpp"
            @test !haskey(b, "shift")
        end
        empty!(seen); caps[] = ["completion", "tools"]
        @test load_model("m"; service=ep) isa OllamaSuccess{Nothing}
        @test [t for (t, _) in seen] == ["/api/show", "/api/chat"]                       # chat models unchanged
        # `shift` is a chat setting: an endpoint's shift goes with a chat load only.
        shifted = OllamaEndpoint(base_url="http://127.0.0.1:$port", shift=false)
        empty!(seen); caps[] = ["embedding"]
        @test load_model("m"; service=shifted) isa OllamaSuccess{Nothing} && !haskey(JSON.parse(seen[2][2]), "shift")
        empty!(seen); caps[] = ["completion"]
        @test load_model("m"; service=shifted) isa OllamaSuccess{Nothing} && JSON.parse(seen[2][2])["shift"] === false
        # Only the empty list an empty embeddings request is answered with reports the load.
        @test isnothing(UniLM._ollama_embed_loaded(Dict{String,Any}("embeddings" => [])))
        for reply in (Dict{String,Any}(), Dict{String,Any}("embeddings" => [[0.1]]), Dict{String,Any}("embeddings" => "[]"))
            @test_throws ArgumentError UniLM._ollama_embed_loaded(reply)
        end
    finally
        close(server)
    end
end

# Refuted if: load_model outlives its total_deadline because each of its two requests gets the whole budget
# (measured before the fix: a success after 3.0 s under total_deadline=2.0 with 1.2 s per request), or a breach
# is anything but a typed timeout inside an OllamaCallError.
@testset "load_model: total_deadline bounds the verb, both requests together" begin
    seen = String[]
    handler = req -> begin
        push!(seen, String(req.target))
        sleep(1.2)
        req.target == "/api/show" ?
            HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(JSON.json(Dict("capabilities" => ["completion"])))) :
            HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(JSON.json(Dict("model" => "m",
                "done" => true, "done_reason" => "load", "message" => Dict("role" => "assistant", "content" => "")))))
    end
    tcp = Sockets.listen(Sockets.localhost, 0); port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    server = HTTP.serve!(handler, "127.0.0.1", port; verbose=false)
    try
        ep = OllamaEndpoint(base_url="http://127.0.0.1:$port")
        cfg = RequestConfig(max_attempts=1, total_deadline=2.0, request_timeout=2.0)
        load_model("m"; service=ep, config=cfg)       # unmeasured: compiles the timeout path on both sides
        empty!(seen)
        t = @elapsed r = load_model("m"; service=ep, config=cfg)
        @test seen == ["/api/show", "/api/chat"]
        # The load request had only what the lookup left of the 2.0 s, not a budget of its own.
        @test r isa OllamaCallError && r.cause isa UniLMTimeout && r.cause.limit <= 0.8
        @test t < 2.6
    finally
        close(server)
    end
    # Nothing left when a request would start: it is not sent, and the call error carries the deadline breach.
    late = UniLM._ollama_call(nothing, OllamaEndpoint(base_url="http://127.0.0.1:1"), "POST", "/api/chat", "{}",
                              RequestConfig(total_deadline=1.0), nothing; t0=time_ns() - UInt64(2_000_000_000))
    @test late isa OllamaCallError && late.cause isa UniLMTimeout && late.cause.phase === :deadline
end
