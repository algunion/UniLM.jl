# ============================================================================
# P0 regression suite — falsification harness for the 2026-07-10 review
# (grounding/quality-review-2026-07-10.md). Every test states the FIXED
# contract wrapped in @test_broken. Fix PRs flip @test_broken → @test.
# An "Error (unexpected pass)" means the finding is refuted: delete the test
# and downgrade the finding. Fully offline (zero-spend).
# ============================================================================

using Sockets

# Streaming SSE mock: writes each element of `chunks` as its own flush with a
# 0.2s gap, so the client's readavailable() sees them as separate reads.
# (TCP may still coalesce under load — the affected tests note the flake
# direction: coalescing can only turn Broken into an unexpected pass, never
# into a spurious CI failure.)
function sse_mock_server(chunks::Vector{String})
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2])
    close(tcp)
    server = HTTP.listen!("127.0.0.1", port; stream=true, verbose=false) do http::HTTP.Stream
        while !eof(http)
            readavailable(http)              # drain the request body
        end
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        for c in chunks
            write(http, c)
            flush(http)
            sleep(0.2)
        end
    end
    server, "http://127.0.0.1:$port"
end

@testset "P0 regression suite (review 2026-07-10)" begin

    @testset "P0-7 fork(chat) preserves every Chat field" begin
        kwargs = Dict{Symbol,Any}(
            :model => "gpt-5.5", :history => false,
            :tools => [GPTTool(func=GPTFunctionSignature(name="f"))],
            :tool_choice => "auto", :parallel_tool_calls => true,
            :temperature => 0.5, :n => 2, :stream => false, :stop => ["x"],
            :max_tokens => 10, :max_completion_tokens => 20,
            :presence_penalty => 0.1, :response_format => ResponseFormat(),
            :frequency_penalty => 0.2, :logit_bias => Dict("50256" => -100.0),
            :user => "u", :seed => 7, :reasoning_effort => "high",
            :stream_options => Dict("include_usage" => true), :verbosity => "low",
            :store => true, :metadata => Dict("k" => "v"), :service_tier => "auto",
            :logprobs => true, :top_logprobs => 3,
            :prediction => Dict("type" => "content"), :modalities => ["text"],
            :audio => Dict("voice" => "alloy"),
            :web_search_options => Dict("search_context_size" => "low"),
            :prompt_cache_key => "pck", :safety_identifier => "sid",
        )
        chat = Chat(; kwargs...)
        forked = fork(chat)
        # messages is deepcopied (identity differs by design); the cost Ref is fresh.
        skip = (:messages, :_cumulative_cost)
        # FIXED contract: every remaining field survives a fork. Today 15 are dropped.
        @test_broken all(name -> isequal(getfield(forked, name), getfield(chat, name)),
                         setdiff(fieldnames(Chat), skip))
    end
end
