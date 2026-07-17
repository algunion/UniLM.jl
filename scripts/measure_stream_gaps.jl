#!/usr/bin/env julia
# Idle-gap falsifier for the default stream_idle_timeout (120s byte-gap).
#
# Runs one reasoning-heavy stream per provider whose API key is present, records
# the MAX inter-byte gap (the exact quantity the idle guard watches: the gap
# between consecutive readavailable returns on the raw socket, which provider
# keep-alives such as SSE comments and Anthropic `ping` events reset), and prints
# a verdict against the shipped default.
#
# NOT part of the test suite / CI — it spends real tokens and needs live keys.
# Run it before tagging a release, from an environment that holds the keys.
# Zero providers configured => it measures nothing and says so.
#
# Kill criterion: any healthy stream's max gap > 60s means the 120s default lacks
# the 2x headroom it claims — raise stream_idle_timeout (or ship Inf for streams)
# BEFORE tagging.
#
#   julia --project=. scripts/measure_stream_gaps.jl

using HTTP, JSON, Printf

const STREAM_IDLE_DEFAULT = 120.0   # RequestConfig.stream_idle_timeout default
const KILL_GAP            = 60.0    # > this ⇒ raise the default before tagging

const HARD_PROMPT = """
Work this out step by step, showing all reasoning before the final answer.
A freight train leaves city A at 06:00 traveling 72 km/h. A passenger train
leaves city B (465 km away) at 06:20 toward A at 111 km/h. A hawk starts at the
freight train at 06:20, flies 140 km/h to the passenger train, then back,
repeatedly, until the trains meet. Compute the meeting time two independent ways,
then give the total distance the hawk flies to three decimals.
"""

struct GapResult
    provider::String
    model::String
    chunks::Int
    wall_s::Float64
    max_gap_s::Float64
    status::Int
end

"Stream a POST via HTTP.open, timing the gap between raw readavailable returns."
function measure(provider::String, model::String, url::String,
                 headers::Vector{<:Pair}, body::String)::GapResult
    # Identity encoding so gzip buffering never masks the true byte cadence
    # (mirrors the library's streaming request).
    hdrs = push!(copy(headers), "Accept-Encoding" => "identity")
    chunks = 0
    max_gap = 0.0
    status = 0
    t_start = time_ns()
    last = t_start
    HTTP.open("POST", url, hdrs; status_exception=false, decompress=false) do io
        write(io, body)
        HTTP.closewrite(io)
        HTTP.startread(io)
        status = io.message.status
        if status != 200
            @warn "$provider returned HTTP $status" body=String(readavailable(io))
            return
        end
        while !eof(io)
            _ = readavailable(io)
            now = time_ns()
            gap = (now - last) / 1e9
            gap > max_gap && (max_gap = gap)
            last = now
            chunks += 1
        end
    end
    GapResult(provider, model, chunks, (time_ns() - t_start) / 1e9, max_gap, status)
end

envkey(name) = get(ENV, name, "")

function run_openai()
    key = envkey("OPENAI_API_KEY"); isempty(key) && return nothing
    model = get(ENV, "UNILM_OPENAI_MODEL", "gpt-5.5")
    body = JSON.json(Dict(
        "model" => model,
        "input" => HARD_PROMPT,
        "reasoning" => Dict("effort" => "high"),
        "stream" => true,
    ))
    measure("OpenAI (Responses)", model, "https://api.openai.com/v1/responses",
        ["Authorization" => "Bearer $key", "Content-Type" => "application/json"], body)
end

function run_anthropic()
    key = envkey("ANTHROPIC_API_KEY"); isempty(key) && return nothing
    model = get(ENV, "UNILM_ANTHROPIC_MODEL", "claude-opus-4-8")
    # Extended thinking streams thinking_delta bytes punctuated by `ping` events;
    # the ping cadence is what must keep the raw byte gap under the idle bound.
    body = JSON.json(Dict(
        "model" => model,
        "max_tokens" => 16000,
        "thinking" => Dict("type" => "enabled", "budget_tokens" => 8000),
        "stream" => true,
        "messages" => [Dict("role" => "user", "content" => HARD_PROMPT)],
    ))
    measure("Anthropic (thinking)", model, "https://api.anthropic.com/v1/messages",
        ["x-api-key" => key, "anthropic-version" => "2023-06-01",
         "Content-Type" => "application/json"], body)
end

function run_gemini()
    key = envkey("GEMINI_API_KEY"); isempty(key) && return nothing
    model = get(ENV, "UNILM_GEMINI_MODEL", "gemini-3.5-flash")
    url = "https://generativelanguage.googleapis.com/v1beta/models/$model:streamGenerateContent?alt=sse"
    body = JSON.json(Dict(
        "contents" => [Dict("role" => "user", "parts" => [Dict("text" => HARD_PROMPT)])],
    ))
    measure("Gemini (native)", model, url,
        ["x-goog-api-key" => key, "Content-Type" => "application/json"], body)
end

function main()
    results = GapResult[]
    for (name, f) in (("OPENAI_API_KEY", run_openai),
                      ("ANTHROPIC_API_KEY", run_anthropic),
                      ("GEMINI_API_KEY", run_gemini))
        try
            r = f()
            r === nothing ? @info("skipped — $name unset") : push!(results, r)
        catch e
            @error "measurement failed" exception=(e, catch_backtrace())
        end
    end
    if isempty(results)
        @warn "No providers configured — set at least one of OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY"
        return
    end

    @printf("\n  stream idle-gap measurement (default = %.0fs, kill threshold = %.0fs)\n\n",
            STREAM_IDLE_DEFAULT, KILL_GAP)
    @printf("  %-22s %-20s %7s %9s %11s   %s\n",
            "provider", "model", "chunks", "wall(s)", "max gap(s)", "verdict")
    println("  " * "-"^90)
    worst = 0.0
    for r in results
        r.status == 200 || continue
        verdict = r.max_gap_s > KILL_GAP ? "RAISE DEFAULT" : "ok"
        worst = max(worst, r.max_gap_s)
        @printf("  %-22s %-20s %7d %9.1f %11.3f   %s\n",
                r.provider, r.model, r.chunks, r.wall_s, r.max_gap_s, verdict)
    end
    println()
    if worst > KILL_GAP
        @printf("  VERDICT: a healthy stream showed a %.3fs gap > %.0fs — RAISE stream_idle_timeout (or ship Inf for streams) before tagging.\n",
                worst, KILL_GAP)
        exit(1)
    else
        @printf("  VERDICT: worst healthy gap %.3fs ≤ %.0fs — the %.0fs default keeps >=2x headroom. OK to tag.\n",
                worst, KILL_GAP, STREAM_IDLE_DEFAULT)
    end
end

main()
