# ============================================================================
# P0 regression suite — falsification harness for the 2026-07-10 review
# (grounding/quality-review-2026-07-10.md). Every test states the FIXED
# contract wrapped in @test_broken. Fix PRs flip @test_broken → @test.
# An "Error (unexpected pass)" means the finding is refuted: delete the test
# and downgrade the finding. Fully offline (zero-spend).
# ============================================================================

using Sockets

# Streaming SSE mock: writes each element of `chunks` as its own flush with a
# 0.4s gap, so the client's readavailable() sees them as separate reads.
# (TCP may still coalesce under load. Coalescing can make the driver-level
# assertions unexpectedly pass, which Test records as an Error — a CI failure
# under failfast=true (test/runtests.jl). Before treating any unexpected pass
# in a driver testset as a refutation, re-run with a larger inter-chunk gap;
# only a pass that survives fragmentation is a refutation.)
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
            sleep(0.4)
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
            # NOTE: an unexpected pass here may be TCP coalescing of chunks 5+6,
            # not a fix/refutation — re-run with a larger gap (see helper header).
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
        # (Finding #15 is filed under P1 in grounding/quality-review-2026-07-10.md,
        # folded into wave 1 because WS1 fixes it — flagged so a future coverage
        # audit doesn't miscount P0s.)
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
        @test msg.content == "Let me check the weather." &&
              !isnothing(msg.tool_calls) && length(msg.tool_calls) == 1

        # (b) A zero-argument tool call streams arguments as "" (Anthropic
        # input_json_delta may never arrive for `{}` input). JSON.parse("")
        # throws today and destroys the whole turn.
        state2 = UniLM.StreamState()
        state2.tool_calls[0] = Dict{String,Any}(
            "id" => "call_2", "type" => "function",
            "function" => Dict{String,Any}("name" => "ping", "arguments" => ""))
        state2.finish_reason = UniLM.TOOL_CALLS
        msg2 = UniLM._build_stream_message(state2)
        @test !isnothing(msg2.tool_calls) && length(msg2.tool_calls) == 1 &&
              msg2.tool_calls[1].func.name == "ping" &&
              msg2.tool_calls[1].func.arguments == Dict{String,Any}()
    end

    @testset "P0-3 Anthropic thinking block round-trip" begin
        resp_json = """
        {"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5",
         "content":[{"type":"thinking","thinking":"user wants weather","signature":"sig=="},
                    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Oslo"}}],
         "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}
        """
        # 3-arg ctor (cf. test/anthropic.jl): a String body becomes BytesBody in HTTP 2.x,
        # which decode's JSON.parse can't consume; a Vector{UInt8} body works on HTTP 1.x + 2.x.
        dec = UniLM.decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(resp_json)))
        msgs = [Message(role=UniLM.RoleUser, content="weather in Oslo?"),
                dec.message,
                Message(role=UniLM.RoleTool, content="12C", tool_call_id="toolu_1")]
        _, wire = UniLM._anthropic_messages(msgs)
        asst = wire[2][:content]
        # FIXED contract: the assistant turn opens with the thinking block,
        # signature intact (echoed verbatim). Tolerate Symbol- or String-keyed
        # blocks (verbatim echo of decoded JSON is String-keyed).
        _get(b, k) = b isa AbstractDict ? get(b, k, get(b, String(k), nothing)) : nothing
        @test_broken asst isa AbstractVector && length(asst) >= 2 &&
                     _get(asst[1], :type) == "thinking" &&
                     _get(asst[1], :signature) == "sig==" &&
                     _get(asst[end], :type) == "tool_use"
    end

    @testset "P0-4 Anthropic error event is not success" begin
        state = UniLM.StreamState()
        fb = IOBuffer()
        err_chunk = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
        UniLM.decode_stream_chunk(ANTHROPICServiceEndpoint, err_chunk, state, fb)
        # FIXED contract: the error payload is captured distinguishably on the
        # state so the driver can return LLMFailure/LLMCallError instead of
        # LLMSuccess-with-truncated-content (anthropic.jl:227 currently folds
        # `error` into plain EOS and discards the payload; the in-band error is
        # the documented 529-equivalent on an HTTP-200 stream).
        @test_broken hasproperty(state, :error) && getproperty(state, :error) !== nothing
    end

    @testset "P0-5 Interactions streaming surfaces function calls" begin
        tb = IOBuffer(); fb = IOBuffer(); le = Ref("")
        seq = "event: step.start\ndata: {\"step\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}}\n\n" *
              "event: interaction.requires_action\ndata: {\"interaction\":{\"id\":\"i_1\",\"status\":\"requires_action\",\"model\":\"gemini-3.5-flash\"}}\n\n" *
              "data: [DONE]\n\n"
        r = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, seq, tb, fb, le)
        # FIXED contract: the terminal result carries a response dict whose
        # output contains the function_call (today only step.delta text and
        # interaction.completed are handled — interactions.jl:219-228 — so a
        # tools+streaming interaction ends as :done with data=nothing and
        # _respond_stream manufactures a ResponseFailure from a 200 stream).
        out = r.data isa AbstractDict ? get(get(r.data, "response", Dict{String,Any}()), "output", Any[]) : Any[]
        @test_broken r.done == true && r.data !== nothing &&
                     any(item -> item isa AbstractDict && get(item, "type", "") == "function_call" &&
                                 get(item, "name", "") == "get_weather", out)
    end

    @testset "P0-6 Gemini id-less parallel call correlation" begin
        # FunctionCall.id is Optional in the Gemini API — two id-less parallel
        # calls currently both key tool_names[""] (gemini.jl:67,105,184),
        # last-wins, so every functionResponse is attributed to the last call.
        resp_json = """
        {"candidates":[{"content":{"role":"model","parts":[
           {"functionCall":{"name":"get_weather","args":{"city":"Oslo"}}},
           {"functionCall":{"name":"get_time","args":{"tz":"CET"}}}]},
          "finishReason":"STOP"}],
         "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}
        """
        # 3-arg ctor (cf. test/gemini.jl): a String body becomes BytesBody in HTTP 2.x,
        # which decode's JSON.parse can't consume; a Vector{UInt8} body works on HTTP 1.x + 2.x.
        dec = UniLM.decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(resp_json)))
        tcs = dec.message.tool_calls
        ids_ok = !isnothing(tcs) && length(tcs) == 2 && allunique([tc.id for tc in tcs])
        corr_ok = if ids_ok
            msgs = [Message(role=UniLM.RoleUser, content="hi"),
                    dec.message,
                    Message(role=UniLM.RoleTool, content="12C",   tool_call_id=tcs[1].id),
                    Message(role=UniLM.RoleTool, content="14:00", tool_call_id=tcs[2].id)]
            _, contents = UniLM._gemini_contents(msgs)
            frs = [p[:functionResponse] for p in contents[end][:parts]]
            length(frs) == 2 && frs[1][:name] == "get_weather" && frs[2][:name] == "get_time"
        else
            false
        end
        @test_broken ids_ok && corr_ok
    end

    @testset "P0-8 MCP stdio server survives non-object frames" begin
        # A JSON array (legacy batch — MUST be a single message per spec
        # 2025-11-25) currently raises MethodError at mcp_server.jl:456
        # (_dispatch_mcp only accepts Dict{String,Any}) and kills the serve loop
        # before the NEXT valid request is answered. FIXED contract: -32600 for
        # the bad frame, normal answer for the following ping.
        input = IOBuffer("""[{"jsonrpc":"2.0","id":1,"method":"ping"}]\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n""")
        output = IOBuffer()
        server = MCPServer("p0-test", "0.0.0")
        ok = try
            UniLM._serve_stdio(server; input=input, output=output)
            lines = split(String(take!(output)), '\n'; keepempty=false)
            length(lines) == 2 &&
                occursin("-32600", lines[1]) &&
                occursin("\"id\":2", lines[2]) && occursin("result", lines[2])
        catch
            false
        end
        @test_broken ok
    end

    @testset "P0-9 MCP client skips interleaved notifications" begin
        # Server-initiated notifications (tools/list_changed, logging — legal at
        # any time) currently get consumed as "the response": id mismatch is only
        # @warn'ed and `{}` is returned as the result (mcp_client.jl:186-194,337),
        # desyncing every subsequent call. FIXED contract: read until the
        # matching id, return the real result.
        t = UniLM.StdioTransport(`cat`)
        t.input = IOBuffer()
        t.output = IOBuffer("""{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n""")
        session = UniLM.MCPSession(t, UniLM.MCPServerCapabilities(), Dict{String,Any}(),
                                   UniLM.MCPToolInfo[], UniLM.MCPResourceInfo[],
                                   UniLM.MCPPromptInfo[], "2025-11-25", 0, :ready)
        res = UniLM._mcp_request!(session, "ping")
        @test_broken get(res, "ok", false) == true
    end
end
