@testset "Batch API — config seam wiring" begin
    @test _seam_timeout(create_batch("file_x", "/v1/chat/completions"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(retrieve_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(cancel_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(list_batches(service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(poll_batch("batch_x"; interval=0.01, timeout=0.05, service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
end

_batch_json(status::String) = JSON.json(Dict("id" => "batch_1", "status" => status))

@testset "poll_batch: a wall-clock deadline that rides out transient failures" begin
    @test_throws ArgumentError poll_batch("batch_1"; interval=0, service=SeamProbe)
    @test_throws ArgumentError poll_batch("batch_1"; interval=-1, service=SeamProbe)
    @test_throws ArgumentError poll_batch("batch_1"; interval=NaN, service=SeamProbe)
    @test_throws ArgumentError poll_batch("batch_1"; timeout=0, service=SeamProbe)

    # A 503 on GET #2 is a hiccup, not the answer.
    script = n -> n == 2 ? _json(503, "{}") : _json(200, _batch_json(n == 1 ? "in_progress" : "completed"))
    r, seen = _with_scripted((n, _) -> script(n)) do
        poll_batch("batch_1"; interval=0.01, timeout=30.0, service=URLProbe)
    end
    @test r isa BatchSuccess && r.response.status == "completed"
    @test length(seen) == 3

    # So is one GET that outlives its request_timeout (its reply is held until the poll ends).
    release = Base.Event()
    held = n -> (n == 1 && wait(release); _json(200, _batch_json("completed")))
    r2, seen2 = _with_scripted((n, _) -> held(n)) do
        try
            poll_batch("batch_1"; interval=0.01, timeout=30.0, service=URLProbe,
                       config=UniLM.RequestConfig(request_timeout=1.0))
        finally
            notify(release)
        end
    end
    @test r2 isa BatchSuccess && r2.response.status == "completed"
    @test length(seen2) == 2

    # A non-transient failure ends the poll at once.
    r3, seen3 = _with_scripted((_, _) -> _json(404, "{}")) do
        poll_batch("batch_1"; interval=0.01, timeout=30.0, service=URLProbe)
    end
    @test r3 isa BatchFailure && r3.status == 404 && length(seen3) == 1
end

@testset "poll_batch: the timeout bounds wall-clock time, not an iteration count" begin
    # Each GET takes 0.3 s. Bounded by time, the call returns within timeout + one GET +
    # interval (plus a 2 s runner-stall budget); the old iteration bound,
    # ceil(timeout/interval) = 50 GETs, took ~15 s.
    slow = (_, _) -> (sleep(0.3); _json(200, _batch_json("in_progress")))
    r, _ = _with_scripted(slow) do
        poll_batch("batch_1"; interval=0.01, timeout=0.5, service=URLProbe)   # compile the path
        get_s = @elapsed retrieve_batch("batch_1"; service=URLProbe)          # one GET, as seen here
        elapsed = @elapsed (out = poll_batch("batch_1"; interval=0.01, timeout=0.5, service=URLProbe))
        @test elapsed <= 0.5 + get_s + 0.01 + 2.0
        out
    end
    @test r isa BatchCallError
    @test r.cause isa UniLMTimeout && r.cause.phase === :deadline && r.cause.limit == 0.5
    @test isnothing(r.status)                        # a timeout carries no HTTP status
    @test r.last_observed isa BatchObject && r.last_observed.status == "in_progress"
    @test occursin("in_progress", r.error)
end

@testset "poll_batch: a cancel ends the poll at once — mid-pause, and before any GET" begin
    # A non-terminal batch keeps the poll pausing; without a wakeup on the cancel it
    # would sleep out its 30 s interval.
    hits = Threads.Atomic{Int}(0)
    busy = (_, _) -> (Threads.atomic_add!(hits, 1); _json(200, _batch_json("in_progress")))
    for via in (:keyword, :scope)
        tok = CancelToken()
        before = hits[]
        (r, after_cancel), _ = _with_scripted(busy) do
            poll() = poll_batch("batch_1"; interval=30.0, timeout=120.0, service=URLProbe,
                                cancel=(via === :keyword ? tok : nothing))
            t = Threads.@spawn ((via === :scope ? with_cancel(poll, tok) : poll()), time_ns())
            @test timedwait(() -> hits[] > before, 25.0) === :ok   # first GET answered: pausing
            sleep(0.2)
            cancelled_at = time_ns()
            cancel!(tok)
            @test timedwait(() -> istaskdone(t), 25.0) === :ok
            res, done_at = fetch(t)
            res, (done_at - cancelled_at) / 1e9
        end
        @test after_cancel < 5.0                                   # the 30 s pause is not slept out
        @test r isa BatchCallError && r.cause isa UniLMCancelled && r.cause.source === :token
        @test isnothing(r.status)
        @test r.last_observed isa BatchObject && r.last_observed.status == "in_progress"
        @test occursin("cancelled", r.error) && occursin("in_progress", r.error)
    end
    # A token cancelled up front: the typed result, and no request sent.
    r, seen = _with_scripted(busy) do
        poll_batch("batch_1"; interval=30.0, timeout=120.0, service=URLProbe, cancel=cancel!(CancelToken()))
    end
    @test r isa BatchCallError && r.cause isa UniLMCancelled && isnothing(r.last_observed)
    @test isempty(seen)
end

@testset "Batch API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_batch("batch_x"; service=URLProbe), 404; headers=["x-request-id" => "req_batch"])
    @test r isa UniLM.BatchFailure && r.request_id == "req_batch"
end

@testset "Batch API — a separator-bearing id stays one path segment" begin
    t = _recorded_targets() do
        retrieve_batch(_HOSTILE_ID; service=URLProbe)
        cancel_batch(_HOSTILE_ID; service=URLProbe)
        list_batches(; limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/batches/$_HOSTILE_ENC",
                "/v1/batches/$_HOSTILE_ENC/cancel",
                "/v1/batches?limit=2&after=$_HOSTILE_ENC"]
    g = _recorded_targets() do
        cancel_batch("batch_abc123"; service=URLProbe)
        list_batches(; after="batch_abc123", service=URLProbe)
    end
    @test g == ["/v1/batches/batch_abc123/cancel", "/v1/batches?after=batch_abc123"]
end

# The Batch object's `status` is required (OpenAI's Batch schema): a reply without it, or with another JSON type, is
# a malformed batch — a call error — and a poll on it ends at once instead of polling to its deadline.
@testset "Batch API — a reply without a string status is a call error, and poll_batch ends on it at once" begin
    for body in ("{\"id\":\"batch_1\"}", "{\"id\":\"batch_1\",\"status\":null}", "{\"id\":\"batch_1\",\"status\":3}")
        r = _answered(() -> retrieve_batch("batch_1"; service=URLProbe), 200; body)
        @test r isa BatchCallError && r.cause isa ArgumentError && occursin("\"status\"", r.error)
    end
    p, seen = _with_scripted((_, _) -> _json(200, "{\"id\":\"batch_1\"}")) do
        poll_batch("batch_1"; interval=0.01, timeout=5.0, service=URLProbe)
    end
    @test p isa BatchCallError && p.cause isa ArgumentError && length(seen) == 1
end

# A listing names its batches in `data` and whether more follow in `has_more` (both required by ListBatchesResponse):
# a 200 without either is a malformed listing, never an empty final page.
@testset "list_batches: a 200 without the documented data and has_more is a call error, not an empty page" begin
    for body in ("{}", "{\"data\":[]}", "{\"has_more\":false}", "{\"data\":{},\"has_more\":false}",
                 "{\"data\":[],\"has_more\":\"no\"}", "{\"data\":[3],\"has_more\":false}")
        r = _answered(() -> list_batches(; service=URLProbe), 200; body)
        @test r isa BatchCallError && r.cause isa ArgumentError
    end
    page = "{\"object\":\"list\",\"data\":[{\"id\":\"batch_1\",\"status\":\"completed\"}],\"has_more\":true,\"last_id\":\"batch_1\"}"
    ok = _answered(() -> list_batches(; service=URLProbe), 200; body=page)
    @test ok isa BatchListSuccess && [b.id for b in ok.response.data] == ["batch_1"] && ok.response.has_more
    @test ok.response.raw == JSON.parse(page)
end

# output_expires_after: when the batch's output and error files expire, `seconds` after their creation (the only
# anchor, created_at), "between 3600 (1 hour) and 2592000 (30 days)" (the Batch reference's BatchFileExpirationAfter).
@testset "create_batch: output_expires_after sets when the output and error files expire" begin
    created = (_, _) -> _json(200, "{\"id\":\"batch_1\",\"status\":\"validating\"}")
    r, seen = _with_scripted(created) do
        create_batch("file_in", "/v1/chat/completions"; output_expires_after=7200, service=URLProbe)
    end
    @test r isa BatchSuccess && r.response.status == "validating"
    @test JSON.parse(only(seen).body) == Dict("input_file_id" => "file_in", "endpoint" => "/v1/chat/completions",
        "completion_window" => "24h", "output_expires_after" => Dict("anchor" => "created_at", "seconds" => 7200))
    _, unset = _with_scripted(created) do
        create_batch("file_in", "/v1/chat/completions"; service=URLProbe)
    end
    @test !haskey(JSON.parse(only(unset).body), "output_expires_after")
    for s in (3600, 2_592_000)
        _, edge = _with_scripted(created) do
            create_batch("file_in", "/v1/responses"; output_expires_after=s, service=URLProbe)
        end
        @test JSON.parse(only(edge).body)["output_expires_after"]["seconds"] == s
    end
    # Out of range: refused before any request (SeamProbe's dead port would make it a call error).
    for s in (3599, 2_592_001, 0, -3600)
        err = try create_batch("file_in", "/v1/chat/completions"; output_expires_after=s, service=SeamProbe); nothing
              catch e; e end
        @test err isa ArgumentError && occursin("output_expires_after", err.msg) && occursin("3600", err.msg)
    end
end

# A batched request bills at its provider's Batch rates; its usage names the tier "batch" (Anthropic's own value for
# a Message Batches result, and the tier `batch_results` records on every result it decodes). Each figure below is
# the pricing page's Batch price, per million tokens.
using Dates: Date
using Base.ScopedValues: with
_batch_priced(service, model; kw...) = LLMSuccess(message=Message(role=RoleAssistant, content="ok"),
    self=Chat(service=service, model=model), model=model,
    usage=TokenUsage(; service_tier="batch", kw...))

@testset "Batch pricing — Anthropic: half of every rate, stacked with caching, long prompts and US inference" begin
    # Haiku 5.5 Batch: $0.05 in, $0.25 out; a cache read 0.1x and a 5-minute write 1.25x of the batch input.
    r = _batch_priced(ANTHROPICServiceEndpoint, "claude-haiku-5-5"; prompt_tokens=1000, completion_tokens=200,
                 total_tokens=1200, cached_tokens=100, cache_write_tokens=50)
    @test estimated_cost(r) ≈ (850 * 0.05 + 100 * 0.005 + 50 * 1.25 * 0.05 + 200 * 0.25) / 1e6
    # "Claude Haiku 5.5 (for prompts over 100,000 tokens)": $0.25 / $1.25.
    long = _batch_priced(ANTHROPICServiceEndpoint, "claude-haiku-5-5"; prompt_tokens=150_000, completion_tokens=10)
    @test estimated_cost(long) ≈ (150_000 * 0.25 + 10 * 1.25) / 1e6
    us = _batch_priced(ANTHROPICServiceEndpoint(), "claude-haiku-5-5"; prompt_tokens=1000, completion_tokens=100,
                  inference_geo="us")
    @test estimated_cost(us) ≈ 1.1 * (1000 * 0.05 + 100 * 0.25) / 1e6
end

@testset "Batch pricing — OpenAI: the Batch table's rates, half of Standard, where it lists the model" begin
    # gpt-6-luna Batch: $0.05 in, $0.005 cached, $0.0625 30m write, $0.125 12h write, $0.25 out.
    r = _batch_priced(OPENAIServiceEndpoint, "gpt-6-luna"; prompt_tokens=1000, completion_tokens=50, cached_tokens=200,
                 cache_write_tokens=100, cache_write_extended_tokens=40)
    @test estimated_cost(r) ≈ (700 * 0.05 + 200 * 0.005 + 60 * 0.0625 + 40 * 0.125 + 50 * 0.25) / 1e6
    # Long context (over 272K input tokens): $0.10 in, $0.375 out.
    long = _batch_priced(OPENAIServiceEndpoint, "gpt-6-luna"; prompt_tokens=300_000, completion_tokens=100)
    @test estimated_cost(long) ≈ (300_000 * 0.10 + 100 * 0.375) / 1e6
    snap = _batch_priced(OPENAIServiceEndpoint, "gpt-6-luna-2026-09-14"; prompt_tokens=1000, completion_tokens=100)
    @test estimated_cost(snap) ≈ (1000 * 0.05 + 100 * 0.25) / 1e6
    # gpt-4.1 Batch lists $1.00 / $4.00 and no cached-input price ("-"): a cache read bills as batch input.
    g41 = _batch_priced(OPENAIServiceEndpoint, "gpt-4.1"; prompt_tokens=1000, completion_tokens=100, cached_tokens=400)
    @test estimated_cost(g41) ≈ (1000 * 1.00 + 100 * 4.00) / 1e6
    # gpt-6-cyber is not in the Batch table, nor is a gpt-5.5-pro prompt over 272K tokens ("-"): the standard rates,
    # with one warning per model.
    cyber = _batch_priced(OPENAIServiceEndpoint, "gpt-6-cyber"; prompt_tokens=1000, completion_tokens=100)
    @test (@test_logs (:warn, r"Batch API rates") estimated_cost(cyber)) ≈ (1000 * 20.0 + 100 * 100.0) / 1e6
    pro = _batch_priced(OPENAIServiceEndpoint, "gpt-5.5-pro"; prompt_tokens=300_000, completion_tokens=100)
    @test (@test_logs (:warn, r"Batch API rates") estimated_cost(pro)) ≈ (300_000 * 60.0 + 100 * 270.0) / 1e6
end

@testset "Batch pricing — Gemini: each model's Batch tab, half of Standard but where it keeps the cache-hit price" begin
    on(day, r) = with(() -> estimated_cost(r), UniLM._PRICING_DATE => day)
    nov, jan = Date(2026, 11, 16), Date(2027, 1, 2)
    # Gemini 3.8 Flash Batch: $0.375 in, $0.0375 cache hit, $1.875 out, through 2026-12-31; $0.75 / $0.075 / $3.75 after.
    f38 = _batch_priced(GEMINIServiceEndpoint, "gemini-3.8-flash"; prompt_tokens=1000, completion_tokens=100, cached_tokens=200)
    @test on(nov, f38) ≈ (800 * 0.375 + 200 * 0.0375 + 100 * 1.875) / 1e6
    @test on(jan, f38) ≈ (800 * 0.75 + 200 * 0.075 + 100 * 3.75) / 1e6
    # Gemini 2.5 Flash Batch: $0.15 text / $0.50 audio in, $1.25 out, cache hits "Same as Standard" — $0.03 text, $0.1 audio.
    f25 = _batch_priced(GEMINIServiceEndpoint, "gemini-2.5-flash"; prompt_tokens=1000, prompt_audio_tokens=300,
                   cached_tokens=100, cached_audio_tokens=40, completion_tokens=50)
    @test on(nov, f25) ≈ (640 * 0.15 + 260 * 0.50 + 60 * 0.03 + 40 * 0.10 + 50 * 1.25) / 1e6
    # Gemini 3.5 Flash-Lite Batch: $0.15 in, $0.015 cache hit (Flex lists $0.02), $1.25 out.
    lite = _batch_priced(GEMINIServiceEndpoint, "gemini-3.5-flash-lite"; prompt_tokens=1000, cached_tokens=500, completion_tokens=10)
    @test on(nov, lite) ≈ (500 * 0.15 + 500 * 0.015 + 10 * 1.25) / 1e6
    # Gemini 2.5 Pro over 200K tokens, Batch: $1.25 in, $0.25 cache hit (the standard figure), $7.50 out.
    pro = _batch_priced(GEMINIServiceEndpoint, "gemini-2.5-pro"; prompt_tokens=300_000, cached_tokens=100_000, completion_tokens=1000)
    @test on(nov, pro) ≈ (200_000 * 1.25 + 100_000 * 0.25 + 1000 * 7.5) / 1e6
    # "Batch API isn't supported" on Gemini Decision 1: the standard rates, with a warning.
    d1 = _batch_priced(GEMINIServiceEndpoint, "gemini-decision-1"; prompt_tokens=1000)
    @test (@test_logs (:warn, r"Batch API rates") on(nov, d1)) ≈ 1000 * 0.10 / 1e6
end

# ── A batch endpoint speaking provider wire W, served by `_with_scripted` ──
# Its Chats encode and its replies decode as the provider's do; its batch paths hang from the scripted server.
struct BatchProbe{W} <: UniLM.ServiceEndpoint end
UniLM._batch_wire(::Type{BatchProbe{W}}) where {W} = W()
UniLM._batch_root(::Type{<:BatchProbe}) = _url_probe_base[]
UniLM._resolve_base_url(::Type{<:BatchProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{<:BatchProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{<:BatchProbe}) = Set([:chat, :batch, :files])
UniLM.default_model(::Type{<:BatchProbe}) = "probe-default"
_batch_probe(W, E) = (@eval UniLM.encode_request(::Type{BatchProbe{$W}}, c::Chat) = UniLM.encode_request($E, c);
                      @eval UniLM.decode_response(::Type{BatchProbe{$W}}, r::HTTP.Response) = UniLM.decode_response($E, r);
                      BatchProbe{W})
const _OAB = _batch_probe(UniLM._OpenAIBatches, OPENAIServiceEndpoint)
_batch_refused(f) = try f(); nothing catch e; e end
_batch_targets(seen) = [(s.method, s.target) for s in seen]

@testset "create_batch(requests) on OpenAI: Chat Completions bodies in a JSONL uploaded for purpose batch, then the batch" begin
    @test has_capability(OPENAIServiceEndpoint, :batch) && !has_capability(GEMINIOpenAIServiceEndpoint, :batch)
    luna(q; kw...) = Chat(; service=_OAB, model="gpt-6-luna", reasoning_effort="none", max_tokens=16,
                          messages=[Message(Val(:user), q)], kw...)
    reqs = ["b-2" => luna("two"), "a-1" => luna("one"; stop="x")]
    file = Dict("id" => "file-in", "object" => "file", "bytes" => 10, "created_at" => 1,
                "filename" => "unilm-batch-input.jsonl", "purpose" => "batch")
    batch = Dict("id" => "batch_1", "object" => "batch", "status" => "validating", "input_file_id" => "file-in",
                 "endpoint" => "/v1/chat/completions", "request_counts" => Dict("total" => 0, "completed" => 0, "failed" => 0))
    replies = [_json(200, JSON.json(file)), _json(200, JSON.json(batch))]
    r, seen = _with_scripted((n, _) -> replies[n]) do
        create_batch(reqs; metadata=Dict("job" => "nightly"), output_expires_after=3600)
    end
    @test r isa BatchSuccess && r.response.id == "batch_1" && r.response.input_file_id == "file-in"
    @test r.response.raw == batch && r.response.request_counts == batch["request_counts"]
    @test _batch_targets(seen) == [("POST", "/v1/files"), ("POST", "/v1/batches")]
    upload = seen[1].body
    @test occursin(r"name=\"purpose\"\r\n\r\nbatch\r\n", upload) && occursin("filename=\"unilm-batch-input.jsonl\"", upload)
    lines = [JSON.parse(l) for l in split(upload, r"\r?\n") if startswith(l, "{\"")]
    @test [l["custom_id"] for l in lines] == ["b-2", "a-1"]
    @test all(l -> l["method"] == "POST" && l["url"] == "/v1/chat/completions", lines)
    @test [l["body"] for l in lines] == [JSON.parse(UniLM.encode_request(OPENAIServiceEndpoint, c)) for (_, c) in reqs]
    @test JSON.parse(seen[2].body) == Dict("input_file_id" => "file-in", "endpoint" => "/v1/chat/completions",
        "completion_window" => "24h", "metadata" => Dict("job" => "nightly"),
        "output_expires_after" => Dict("anchor" => "created_at", "seconds" => 3600))
    # A refused upload is the batch's failure, and no batch is created.
    f, fseen = _with_scripted((_, _) -> _json(400, "{\"error\":{\"message\":\"bad file\"}}"; headers=["x-request-id" => "req_up"])) do
        create_batch(reqs)
    end
    @test f isa BatchFailure && f.status == 400 && f.request_id == "req_up" && length(fseen) == 1
end

@testset "create_batch(requests): refused before any request" begin
    c(; kw...) = Chat(; service=_OAB, model="gpt-6-luna", messages=[Message(Val(:user), "q")], kw...)
    elsewhere = Chat(service=OPENAIServiceEndpoint, model="gpt-6-luna", messages=[Message(Val(:user), "q")])
    for call in (() -> create_batch(Pair{String,Chat}[]; service=_OAB),                     # no request
                 () -> create_batch(["" => c()]),                                            # an empty custom id
                 () -> create_batch(["a" => c(), "b" => c(), "a" => c()]),                   # a repeated custom id
                 () -> create_batch(["a" => c(stream=true)]),                                # a streamed request
                 () -> create_batch(["a" => c(), "b" => elsewhere]),                         # two endpoints
                 () -> create_batch(["a" => c()]; service=OPENAIServiceEndpoint),            # not the service's Chat
                 () -> create_batch(["a" => c()]; output_expires_after=60),                  # an expiry out of range
                 () -> create_batch(["a" => c(temperature=0.5, reasoning_effort="low")]),    # a Chat its encoder refuses
                 () -> create_batch(["a" => Chat(service=GEMINIOpenAIServiceEndpoint, model="gemini-3.8-flash")]),
                 () -> delete_batch("batch_1"; service=_OAB),                                # OpenAI deletes no batch
                 () -> update_batch("batch_1"; priority=1, service=_OAB))                    # nor updates one
        @test _batch_refused(call) isa ArgumentError
    end
    msg = _batch_refused(() -> create_batch(["a" => c(), "b" => c(), "a" => c()])).msg
    @test occursin("\"a\"", msg) && occursin("unique", msg)
    @test occursin("delete_file", _batch_refused(() -> delete_batch("batch_1"; service=_OAB)).msg)
end

_batch_oa_reply(text; model="gpt-6-luna-2026-09-14") = Dict("id" => "chatcmpl-1", "object" => "chat.completion",
    "model" => model, "service_tier" => "default",
    "choices" => [Dict("index" => 0, "message" => Dict("role" => "assistant", "content" => text), "finish_reason" => "stop")],
    "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 2, "total_tokens" => 12))
_batch_jsonl(lines...) = join(JSON.json.(lines), "\n") * "\n"

@testset "batch_results on OpenAI: the output file's lines, then the error file's, each a Chat result by custom id" begin
    out = _batch_jsonl(Dict("id" => "batch_req_1", "custom_id" => "b", "error" => nothing,
                      "response" => Dict("status_code" => 200, "request_id" => "req_b", "body" => _batch_oa_reply("two"))),
                 Dict("id" => "batch_req_2", "custom_id" => "a", "error" => nothing,
                      "response" => Dict("status_code" => 400, "request_id" => "req_a",
                                         "body" => Dict("error" => Dict("message" => "bad", "type" => "invalid_request_error")))),
                 Dict("id" => "batch_req_4", "custom_id" => "d", "error" => nothing,
                      "response" => Dict("status_code" => 200, "request_id" => "req_d", "body" => Dict("choices" => []))))
    err = _batch_jsonl(Dict("id" => "batch_req_3", "custom_id" => "c", "response" => nothing,
                      "error" => Dict("code" => "batch_expired", "message" => "This request could not be executed before the completion window expired.")))
    batch = Dict("id" => "batch_1", "status" => "completed", "model" => "gpt-6-luna",
                 "output_file_id" => "file-out", "error_file_id" => "file-err")
    served = Dict("/v1/batches/batch_1" => JSON.json(batch), "/v1/files/file-out/content" => out,
                  "/v1/files/file-err/content" => err)
    r, seen = _with_scripted((_, req) -> _json(200, served[String(req.target)])) do
        batch_results("batch_1"; service=_OAB)
    end
    @test r isa BatchResultsSuccess
    @test _batch_targets(seen) == [("GET", "/v1/batches/batch_1"), ("GET", "/v1/files/file-out/content"),
                             ("GET", "/v1/files/file-err/content")]
    @test first.(r.results) == ["b", "a", "d", "c"]
    b = r.results[1].second
    @test b isa LLMSuccess && b.message.content == "two" && b.model == "gpt-6-luna-2026-09-14"
    @test b.usage.service_tier == "batch" && (b.usage.prompt_tokens, b.usage.completion_tokens) == (10, 2)
    @test b.self.service === _OAB && b.self.model == "gpt-6-luna-2026-09-14" && isempty(b.self.messages)
    a = r.results[2].second
    @test a isa LLMFailure && a.status == 400 && a.request_id == "req_a"
    @test JSON.parse(a.response)["error"]["message"] == "bad" && a.self.model == "gpt-6-luna"
    d = r.results[3].second                                     # a 200 the Chat decoder cannot read
    @test d isa LLMCallError && isnothing(d.status) && occursin("empty choices", d.error)
    c = r.results[4].second
    @test c isa LLMCallError && isnothing(c.status) && occursin("batch_expired", c.error)
    # Before the batch has an output or error file there are no results: a call error naming its status.
    pending = _answered(() -> batch_results("batch_1"; service=_OAB), 200;
                        body="{\"id\":\"batch_1\",\"status\":\"in_progress\"}")
    @test pending isa BatchCallError && pending.cause isa ArgumentError && occursin("in_progress", pending.error)
    # A file the Files API does not serve is the results' failure.
    gone, _ = _with_scripted((n, _) -> n == 1 ? _json(200, JSON.json(batch)) : _json(404, "{\"error\":{}}")) do
        batch_results("batch_1"; service=_OAB)
    end
    @test gone isa BatchFailure && gone.status == 404
    # A line without its custom id cannot be paired with a request: the reply is malformed.
    torn = Dict("/v1/batches/batch_1" => JSON.json(Dict("id" => "batch_1", "status" => "completed", "output_file_id" => "f")),
                "/v1/files/f/content" => _batch_jsonl(Dict("response" => Dict("status_code" => 200, "body" => _batch_oa_reply("x")))))
    t = first(_with_scripted((_, req) -> _json(200, torn[String(req.target)])) do
        batch_results("batch_1"; service=_OAB)
    end)
    @test t isa BatchCallError && occursin("custom_id", t.error)
end
