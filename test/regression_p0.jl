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
# Portability across the declared HTTP compat range ("1.9, 2"): `listen!`
# handlers receive an HTTP.Stream on both majors, so no `stream=true` kwarg
# (the 2.x major rejects it); the request body is drained with `read`
# (readavailable is undefined for server-side streams on 2.x, and a throwing
# handler turns every response into a 500).
function sse_mock_server(chunks::Vector{String})
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2])
    close(tcp)
    server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
        read(http)                           # drain the request body
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

    @testset "P0-2 chat SSE unit contracts" begin
        # (a) finish_reason=="stop" must NOT end the stream — only `data: [DONE]`
        # may. Ending early skips the trailing usage-only chunk (requests.jl:180).
        state = UniLM.StreamState()
        fb = IOBuffer()
        finish_chunk = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n"
        r = UniLM._parse_chunk(finish_chunk, state, fb)
        @test_broken r.eos == false

        # (b) the stream_options.include_usage final chunk has `"choices": []`
        # (documented). It must yield usage AND leave the carry buffer empty —
        # today choices[1] throws and the whole line is stashed in failbuff
        # (requests.jl:176,212), which later glues onto `data: [DONE]` with no
        # newline and corrupts the stream.
        state2 = UniLM.StreamState()
        fb2 = IOBuffer()
        usage_chunk = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n"
        UniLM._parse_chunk(usage_chunk, state2, fb2)
        @test_broken state2.usage !== nothing && isempty(String(take!(fb2)))
    end

    @testset "P0-1/P0-2c streamed tool call end-to-end (fragmented)" begin
        # One tool call whose arguments complete only in later reads, then a
        # usage-only chunk, then [DONE] — each in its own TCP write.
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Oslo\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9,\"total_tokens\":16}}\n\n",
            "data: [DONE]\n\n",
        ]
        server, base = sse_mock_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock",
                        stream=true,
                        tools=[GPTTool(func=GPTFunctionSignature(name="get_weather"))])
            push!(chat, Message(Val(:system), "s"))
            push!(chat, Message(Val(:user), "u"))
            fired = Ref(0)
            task = chatrequest!(chat; on_tool_call = tc -> (fired[] += 1))
            res = fetch(task)
            # P0-1: the callback must fire exactly once for the completed call.
            # (Gate at requests.jl:305 only re-checks when a NEW index appears.)
            @test_broken fired[] == 1
            # P0-2c: the fragmented usage chunk must not turn success into failure
            # (failbuff glue at requests.jl:302 destroys the [DONE] line today).
            @test_broken res isa LLMSuccess
            # Usage from the final chunk must be captured.
            @test_broken res isa LLMSuccess && res.usage !== nothing &&
                         res.usage.total_tokens == 16
        finally
            close(server)
        end
    end

    @testset "P0-15 _build_stream_message contracts" begin
        # (a) Providers emit assistant text AND tool calls in one turn (Gemini-3
        # routinely, Anthropic text-before-tool_use). The builder must keep both;
        # today the text branch is skipped whenever tool_calls exist
        # (requests.jl:220-230), while the non-streaming decoders keep both.
        state = UniLM.StreamState()
        print(state.content, "Let me check the weather.")
        state.tool_calls[0] = Dict{String,Any}(
            "id" => "call_1", "type" => "function",
            "function" => Dict{String,Any}("name" => "get_weather",
                                           "arguments" => "{\"city\":\"Oslo\"}"))
        state.finish_reason = UniLM.TOOL_CALLS
        msg = UniLM._build_stream_message(state)
        @test_broken msg.content == "Let me check the weather." &&
                     !isnothing(msg.tool_calls) && length(msg.tool_calls) == 1

        # (b) A zero-argument tool call streams arguments as "" (Anthropic
        # input_json_delta may never arrive for `{}` input). JSON.parse("")
        # throws today and destroys the whole turn.
        state2 = UniLM.StreamState()
        state2.tool_calls[0] = Dict{String,Any}(
            "id" => "call_2", "type" => "function",
            "function" => Dict{String,Any}("name" => "ping", "arguments" => ""))
        state2.finish_reason = UniLM.TOOL_CALLS
        @test_broken (UniLM._build_stream_message(state2); true)
    end
end
