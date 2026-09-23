# ============================================================================
# Many calls in flight at once, against one local HTTP/1.1 mock. Every call
# carries a marker the mock echoes back, so a reply or delta routed to the wrong
# call shows up as a foreign marker. After each batch the live-task and
# open-descriptor counts must be back at the baseline taken after a warm-up
# (the gauges test/load_probe.jl uses). Fully offline (zero-spend).
# ============================================================================

using Test, HTTP, JSON, UniLM

_cc_fd_count() = length(readdir("/dev/fd"))
_cc_task_count() = length(ccall(:jl_live_tasks, Array{Any,1}, ()))

_cc_sse(s; finish=nothing) = "data: " * JSON.json(Dict("choices" => [Dict("index" => 0,
    "delta" => Dict("content" => s), "finish_reason" => finish)])) * "\n\n"

# The last user message names the reply: "plain <marker>" is a JSON reply echoing
# the marker after 0.5 s; "stream <marker> <n> <gap>" streams n deltas "<marker>:k ",
# `gap` seconds apart, then the finish reason and [DONE].
function _cc_server()
    hits = Threads.Atomic{Int}(0)
    server = HTTP.listen!("127.0.0.1", 0; verbose=false) do http::HTTP.Stream
        req = JSON.parse(String(read(http)))
        Threads.atomic_add!(hits, 1)
        words = split(req["messages"][end]["content"])
        marker = String(words[2])
        HTTP.setstatus(http, 200)
        if words[1] == "plain"
            sleep(0.5)
            HTTP.setheader(http, "Content-Type" => "application/json")
            HTTP.startwrite(http)
            write(http, JSON.json(Dict("choices" => [Dict("index" => 0, "finish_reason" => "stop",
                "message" => Dict("role" => "assistant", "content" => "echo $marker"))])))
        else
            n, gap = parse(Int, words[3]), parse(Float64, words[4])
            HTTP.setheader(http, "Content-Type" => "text/event-stream")
            HTTP.startwrite(http)
            for k in 1:n
                write(http, _cc_sse("$marker:$k ")); flush(http); sleep(gap)
            end
            write(http, _cc_sse(""; finish="stop") * "data: [DONE]\n\n")
        end
    end
    (; server, url="http://127.0.0.1:$(HTTP.port(server))", hits)
end

_cc_marker(i) = "mk-" * lpad(i, 3, '0')
_cc_text(marker, n) = join("$marker:$k " for k in 1:n)

function _cc_chat(url, prompt; stream=false)
    chat = Chat(service=GenericOpenAIEndpoint(url, ""), model="mock", stream=stream)
    push!(chat, Message(Val(:system), "echo")); push!(chat, Message(Val(:user), prompt))
    chat
end

# Run `f()` for each `i` concurrently and return the results in order.
_cc_all(f, n) = fetch.([Threads.@spawn f(i) for i in 1:n])

const _CC_CFG = RequestConfig(request_timeout=20.0, stream_idle_timeout=20.0,
                              total_deadline=60.0, max_attempts=3)

# A stream whose callback puts every delta — then the final message, tagged — on
# its own bounded channel, drained by a consumer task that sleeps `consume` seconds
# per item. Returns the result, the drained items, and whether the consumer finished.
function _cc_channel_stream(url, prompt; consume=0.0, capacity=8, cfg=_CC_CFG)
    ch = Channel{String}(capacity)
    seen = String[]
    consumer = Threads.@spawn for item in ch
        push!(seen, item)
        consume > 0 && sleep(consume)
    end
    chat = _cc_chat(url, prompt; stream=true)
    r = fetch(chatrequest!(chat; config=cfg, callback=(c, _) ->
        put!(ch, c isa Message ? "final:" * something(c.content, "") : c)))
    close(ch)
    drained = timedwait(() -> istaskdone(consumer), 25.0) === :ok
    (; r, seen, drained, chat)
end

# GC and let closing tasks finish; then the gauges must be back at the baseline
# (a per-request leak would show up once per call in the batch).
function _cc_at_baseline(base)
    GC.gc(); GC.gc(); sleep(1.0); GC.gc()
    tasks, fds = _cc_task_count() - base.tasks, _cc_fd_count() - base.fds
    ok = tasks <= 8 && fds <= 8
    ok || @warn "concurrency gauges above baseline" tasks fds
    ok
end

@testset "concurrency" begin
    srv = _cc_server()
    try
        # Warm-up: compile every path and fill the client's connection pool, so the
        # baseline is the steady state rather than the cold process.
        _cc_all(i -> chatrequest!(_cc_chat(srv.url, "plain $(_cc_marker(i))"); config=_CC_CFG), 50)
        _cc_all(i -> _cc_channel_stream(srv.url, "stream $(_cc_marker(i)) 3 0.02"), 20)
        GC.gc(); GC.gc(); sleep(1.0); GC.gc()
        base = (; tasks=_cc_task_count(), fds=_cc_fd_count())

        @testset "50 non-streaming chats on fork(chat, 50): no cross-talk, truly concurrent" begin
            forks = fork(_cc_chat(srv.url, "plain placeholder"), 50)
            for (i, c) in enumerate(forks)
                pop!(c); push!(c, Message(Val(:user), "plain $(_cc_marker(i))"))
            end
            hits0 = srv.hits[]
            started = time()
            results = _cc_all(i -> chatrequest!(forks[i]; config=_CC_CFG), 50)
            wall = time() - started
            @test all(r -> r isa LLMSuccess, results)
            @test all(i -> results[i].message.content == "echo $(_cc_marker(i))", 1:50)
            @test all(i -> last(forks[i]).content == "echo $(_cc_marker(i))", 1:50)  # each fork got its own reply
            @test srv.hits[] - hits0 == 50
            @test wall < 50 * 0.5 / 4        # the serialized sum is 25 s
            @test _cc_at_baseline(base)
        end

        @testset "20 streams into bounded channels: order, completeness, final message" begin
            outs = _cc_all(i -> _cc_channel_stream(srv.url, "stream $(_cc_marker(i)) 10 0.02"), 20)
            for (i, o) in enumerate(outs)
                text = _cc_text(_cc_marker(i), 10)
                @test o.drained && o.r isa LLMSuccess && o.r.message.content == text
                @test join(o.seen[1:end-1]) == text                  # every delta, in order, none foreign
                @test o.seen[end] == "final:" * text                 # then the final Message
                @test length(o.chat.messages) == 3
            end
            @test _cc_at_baseline(base)
        end

        @testset "slow consumers: time blocked in the callback is not wire idle time" begin
            # Capacity 1 and a consumer that takes 1.2 s per item: by the third delta the
            # callback's put! blocks ~1 s — past the 0.5 s bound plus its detection
            # window, before the terminal is read — and every stream still succeeds.
            cfg = RequestConfig(request_timeout=20.0, stream_idle_timeout=0.5,
                                total_deadline=60.0, max_attempts=1)
            outs = _cc_all(i -> _cc_channel_stream(srv.url, "stream $(_cc_marker(i)) 4 0.1";
                                                   consume=1.2, capacity=1, cfg), 5)
            for (i, o) in enumerate(outs)
                text = _cc_text(_cc_marker(i), 4)
                @test o.drained && o.r isa LLMSuccess && o.r.message.content == text
                @test join(o.seen[1:end-1]) == text
            end
            @test _cc_at_baseline(base)
        end

        @testset "cross-task cancel: cancelled streams end promptly, the rest complete" begin
            n = 10
            toks = [CancelToken() for _ in 1:n]
            first_delta = [Threads.Atomic{Bool}(false) for _ in 1:n]
            chats = [_cc_chat(srv.url, "stream $(_cc_marker(i)) 12 0.25"; stream=true) for i in 1:n]
            tasks = [Threads.@spawn begin
                         r = fetch(chatrequest!(chats[i]; config=_CC_CFG, cancel=toks[i],
                                                callback=(c, _) -> (first_delta[i][] = true)))
                         (r, time())
                     end for i in 1:n]
            @test timedwait(() -> all(a -> a[], first_delta), 25.0) === :ok
            sleep(1.0)
            cancelled_at = time()
            foreach(cancel!, toks[1:2:n])
            @test timedwait(() -> all(istaskdone, tasks), 25.0) === :ok
            for i in 1:n
                r, done_at = fetch(tasks[i])
                if isodd(i)
                    @test r isa LLMCallError && r.cause isa UniLMCancelled && r.cause.source === :token
                    @test done_at - cancelled_at < 0.5
                    @test length(chats[i].messages) == 2            # nothing committed
                else
                    @test r isa LLMSuccess && r.message.content == _cc_text(_cc_marker(i), 12)
                end
            end
            @test all(t -> isempty(t.hooks), toks)                  # no hook outlives its call
            @test _cc_at_baseline(base)
        end

        @testset "a throwing callback under concurrency: typed, never retried, isolated" begin
            n = 10
            hits0 = srv.hits[]
            results = _cc_all(n) do i
                chat = _cc_chat(srv.url, "stream $(_cc_marker(i)) 5 0.02"; stream=true)
                fetch(chatrequest!(chat; config=_CC_CFG, callback=(c, _) ->
                    (isodd(i) && c isa String && error("consumer $(_cc_marker(i)) failed"); nothing)))
            end
            for (i, r) in enumerate(results)
                if isodd(i)
                    @test r isa LLMCallError && r.cause isa ErrorException &&
                          r.cause.msg == "consumer $(_cc_marker(i)) failed"
                else
                    @test r isa LLMSuccess && r.message.content == _cc_text(_cc_marker(i), 5)
                end
            end
            @test srv.hits[] - hits0 == n                            # no retry for any of them
            @test _cc_at_baseline(base)
        end
    finally
        close(srv.server)
    end
end
