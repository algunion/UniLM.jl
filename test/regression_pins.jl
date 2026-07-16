# ============================================================================
# Pinned regression contracts: each testset locks a behavior that a confirmed,
# reproduced bug once violated, so any regression re-breaks it here first.
# These are the drift gates for the fixes that shipped. Fully offline
# (zero-spend).
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
    # Port TOCTOU: the ephemeral port is discovered by binding a probe socket
    # that MUST be closed before HTTP.listen! can claim it — a window in which
    # another listener can steal the port (a once-observed, otherwise-unexplained
    # bind failure). Re-pick a fresh ephemeral port and retry a few times before
    # giving up, so a lost race doesn't fail the run.
    server = nothing
    port = 0
    for attempt in 1:3
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
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
            break
        catch
            attempt == 3 && rethrow()
        end
    end
    server, "http://127.0.0.1:$port"
end

# Test-only endpoint: routes chat streaming to a local mock server while
# delegating SSE semantics to the real Anthropic handler. This is the URL seam
# the production endpoint lacks (its base URL is a constant), so driver-level
# stream behavior gets exercised end to end through _chatrequeststream.
struct AnthropicWireMock <: UniLM.ServiceEndpoint   # Chat.service requires ServiceEndpointSpec
    base_url::String
end
UniLM.get_url(s::AnthropicWireMock, ::Chat) = s.base_url
UniLM.auth_header(::AnthropicWireMock) = ["Content-Type" => "application/json"]
UniLM.handle_sse_event!(::AnthropicWireMock, event::AbstractString, payload::AbstractString,
                        state::UniLM.StreamState) =
    UniLM.handle_sse_event!(ANTHROPICServiceEndpoint, event, payload, state)

@testset "pinned regression contracts" begin

    @testset "fork preserves every Chat field (drift gate)" begin
        kwargs = Dict{Symbol,Any}(
            :service => GenericOpenAIEndpoint("http://localhost:9999", ""),
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
        # FIXED contract: every remaining field survives a fork verbatim.
        # Extend this fixture with a non-default value whenever Chat gains a field — a field left at its default is invisible to this gate.
        @test all(name -> isequal(getfield(forked, name), getfield(chat, name)),
                  setdiff(fieldnames(Chat), skip))
    end

    @testset "chat SSE unit contracts (handle_sse_event! seam)" begin
        # (a) finish_reason=="stop" must NOT end the stream — only `data: [DONE]` may.
        state = UniLM.StreamState()
        carry = IOBuffer()
        finish_chunk = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n"
        st = UniLM._sse_dispatch!(OPENAIServiceEndpoint, carry, Ref(""), finish_chunk, state)
        @test st === :continue
        @test state.finish_reason == "stop"        # recorded, not terminal

        # (b) the stream_options.include_usage final chunk has `"choices": []`
        # (documented). It must yield usage AND leave the carry buffer empty.
        state2 = UniLM.StreamState()
        carry2 = IOBuffer()
        usage_chunk = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n"
        st2 = UniLM._sse_dispatch!(OPENAIServiceEndpoint, carry2, Ref(""), usage_chunk, state2)
        @test st2 === :continue
        @test state2.usage !== nothing && isempty(String(take!(carry2)))
    end

    @testset "streamed tool call end-to-end (fragmented)" begin
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
            # The callback fires exactly once for the completed call
            # (driver-owned completion detection + fired_tool_calls guard).
            @test fired[] == 1
            # The fragmented usage chunk must not turn success into failure.
            @test res isa LLMSuccess
            # Usage from the final chunk must be captured.
            @test res isa LLMSuccess && res.usage !== nothing &&
                  res.usage.total_tokens == 16
        finally
            close(server)
        end
    end

    @testset "_build_stream_message keeps text alongside tool calls; zero-arg args parse" begin
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

    @testset "Anthropic thinking block round-trip" begin
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
        # FIXED contract: the assistant turn opens with the thinking block —
        # text AND signature echoed verbatim — and still ends with tool_use.
        # Tolerate Symbol- or String-keyed blocks (verbatim echo of decoded
        # JSON is String-keyed).
        _get(b, k) = b isa AbstractDict ? get(b, k, get(b, String(k), nothing)) : nothing
        @test asst isa AbstractVector && length(asst) >= 2 &&
              _get(asst[1], :type) == "thinking" &&
              _get(asst[1], :thinking) == "user wants weather" &&
              _get(asst[1], :signature) == "sig==" &&
              _get(asst[end], :type) == "tool_use"
    end

    @testset "Anthropic streamed thinking turn round-trips (driver-level)" begin
        # Chunk boundaries group whole events (fragmentation is covered by the
        # unit contracts; this witnesses the DRIVER: assembly → Message →
        # provider_content → verbatim re-encode).
        chunks = [
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n" *
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" *
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"user wants weather\"}}\n\n" *
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig==\"}}\n\n" *
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" *
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Checking.\"}}\n\n" *
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n" *
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n" *
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Oslo\\\"}\"}}\n\n" *
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":25}}\n\n" *
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
        server, base = sse_mock_server(chunks)
        try
            chat = Chat(service=AnthropicWireMock(base), model="mock", stream=true,
                        tools=[GPTTool(func=GPTFunctionSignature(name="get_weather"))])
            push!(chat, Message(Val(:system), "s"))
            push!(chat, Message(Val(:user), "u"))
            fired = Ref(0)
            task = chatrequest!(chat; on_tool_call = tc -> (fired[] += 1))
            res = fetch(task)
            @test res isa LLMSuccess
            m = res.message
            @test m.content == "Checking." && !isnothing(m.tool_calls) &&
                  length(m.tool_calls) == 1 && m.tool_calls[1].id == "toolu_1"
            @test fired[] == 1
            pc = m.provider_content
            @test pc isa ProviderContent && pc.provider === :anthropic && length(pc.blocks) == 3
            @test pc.blocks[1]["type"] == "thinking" &&
                  pc.blocks[1]["thinking"] == "user wants weather" &&
                  pc.blocks[1]["signature"] == "sig=="
            @test pc.blocks[3]["type"] == "tool_use" &&
                  pc.blocks[3]["input"] == Dict{String,Any}("city" => "Oslo")
            @test res isa LLMSuccess && res.usage !== nothing && res.usage.completion_tokens == 25
            # Re-encoding the streamed turn echoes the captured blocks verbatim.
            msgs = [Message(role=UniLM.RoleUser, content="w?"), m,
                    Message(role=UniLM.RoleTool, content="12C", tool_call_id="toolu_1")]
            _, wire = UniLM._anthropic_messages(msgs)
            @test wire[2][:content] === pc.blocks
        finally
            close(server)
        end
    end

    @testset "Anthropic in-band stream error is not success (unit)" begin
        state = UniLM.StreamState()
        err_chunk = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
        st = UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), err_chunk, state)
        # FIXED contract: the in-band error (documented 529-equivalent on an
        # HTTP-200 stream) is captured on the state and signalled terminally,
        # so the driver returns LLMFailure/LLMCallError instead of
        # LLMSuccess-with-truncated-content.
        @test st === :error
        @test state.error !== nothing
        @test state.error["error"]["type"] == "overloaded_error"
    end

    @testset "in-band stream error yields a typed failure (driver-level)" begin
        # An HTTP-200 SSE stream that dies with a documented in-band `error`
        # event (529-equivalent) must never surface as LLMSuccess with
        # truncated content.
        chunks = [
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n" *
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" *
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        ]
        server, base = sse_mock_server(chunks)
        try
            chat = Chat(service=AnthropicWireMock(base), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s"))
            push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat))
            @test !(res isa LLMSuccess)
            @test res isa LLMFailure && res.status == 529
            @test occursin("overloaded_error", res.response)
            # The truncated partial text must not have been pushed into history.
            @test all(m -> m.role != UniLM.RoleAssistant, chat.messages)
        finally
            close(server)
        end
    end

    @testset "Interactions streaming surfaces function calls" begin
        # Real wire shape (live docs, verified 2026-07-12): a function call is
        # a step.start with an empty arguments placeholder, arguments_delta
        # partial-JSON string deltas, and a step.stop; requires_action is a
        # STATUS on interaction.completed (there is no dedicated SSE event for
        # it). The decoder must assemble the call and surface it in the
        # terminal response's output.
        st = UniLM.AgenticStreamState()
        seq = "event: step.start\ndata: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"name\":\"get_weather\",\"arguments\":{}}}\n\n" *
              "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"{\\\"city\\\":\"}}\n\n" *
              "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"\\\"Oslo\\\"}\"}}\n\n" *
              "event: step.stop\ndata: {\"event_type\":\"step.stop\",\"index\":0}\n\n" *
              "event: interaction.completed\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"i_1\",\"status\":\"requires_action\",\"model\":\"gemini-3.5-flash\",\"usage\":{\"total_input_tokens\":5,\"total_output_tokens\":3,\"total_tokens\":8}}}\n\n" *
              "event: done\ndata: [DONE]\n\n"
        r = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, seq, st)
        # FIXED contract: the terminal result is the completed interaction
        # (status requires_action) whose rebuilt output carries the assembled
        # function_call with its full arguments string.
        out = r.data isa AbstractDict ? get(get(r.data, "response", Dict{String,Any}()), "output", Any[]) : Any[]
        fc = findfirst(item -> item isa AbstractDict && get(item, "type", "") == "function_call", out)
        ok = r.done == true && r.data !== nothing &&
             get(get(r.data, "response", Dict{String,Any}()), "status", "") == "requires_action" &&
             !isnothing(fc) &&
             get(out[something(fc, 1)], "name", "") == "get_weather" &&
             get(out[something(fc, 1)], "call_id", "") == "fc_1" &&
             (JSON.parse(get(out[something(fc, 1)], "arguments", "{}"); dicttype=Dict{String,Any})["city"] == "Oslo")
        @test ok
    end

    @testset "Gemini id-less parallel call correlation" begin
        # FunctionCall.id is Optional in the Gemini API. FIXED contract: id-less
        # parallel calls receive unique synthetic positional ids, so every
        # functionResponse carries the right name with no fabricated wire id.
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
        @test ids_ok && corr_ok
    end

    @testset "MCP stdio server survives non-object frames" begin
        # A JSON array (legacy batch — a message MUST be a single JSON object
        # per spec 2025-11-25) gets a -32600 Invalid Request response and the
        # serve loop continues: the following valid ping is answered normally.
        input = IOBuffer("""[{"jsonrpc":"2.0","id":1,"method":"ping"}]\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n""")
        output = IOBuffer()
        server = MCPServer("pins-test", "0.0.0")
        ok = try
            UniLM._serve_stdio(server; input=input, output=output)
            lines = split(String(take!(output)), '\n'; keepempty=false)
            length(lines) == 2 &&
                occursin("-32600", lines[1]) &&
                occursin("\"id\":2", lines[2]) && occursin("result", lines[2])
        catch
            false
        end
        @test ok
    end

    @testset "MCP client skips interleaved notifications" begin
        # Server-initiated notifications (tools/list_changed, logging) are legal
        # at any time and must never be consumed as "the response": the client
        # reads until the frame with the matching id and returns that result.
        t = UniLM.StdioTransport(`cat`)
        t.input = IOBuffer()
        t.output = IOBuffer("""{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n""")
        session = UniLM.MCPSession(t, UniLM.MCPServerCapabilities(), Dict{String,Any}(),
                                   UniLM.MCPToolInfo[], UniLM.MCPResourceInfo[],
                                   UniLM.MCPPromptInfo[], "2025-11-25", 0, :ready)
        res = UniLM._mcp_request!(session, "ping")
        @test get(res, "ok", false) == true
    end

    @testset "azure deployment env is read at call time" begin
        # The Azure deployment map used to be baked from ENV at precompile/load
        # time, so a deployment name exported AFTER the package loaded was
        # ignored. FIXED contract: the deployment env var is read when the
        # request URL is built, so a value set at runtime is honored.
        model = "gpt-5.2"
        # Keep the test hermetic: an explicit registration would shadow the env
        # path, so drop any that exists and restore it afterwards.
        had = haskey(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, model)
        saved = had ? UniLM._MODEL_ENDPOINTS_AZURE_OPENAI[model] : nothing
        had && delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, model)
        try
            withenv(
                "AZURE_OPENAI_DEPLOY_NAME_GPT_5_2" => "runtime-deploy-xyz",
                "AZURE_OPENAI_BASE_URL" => "https://rt.openai.azure.com",
                "AZURE_OPENAI_API_VERSION" => "2099-01-01",
            ) do
                url = UniLM.get_url(AZUREServiceEndpoint,
                                    Chat(service=AZUREServiceEndpoint, model=model))
                @test occursin("/openai/deployments/runtime-deploy-xyz/chat/completions", url)
            end
        finally
            had && (UniLM._MODEL_ENDPOINTS_AZURE_OPENAI[model] = saved)
        end
    end

    @testset "tool loop propagates user interrupt" begin
        # A bare `catch e` around tool dispatch must not convert a user Ctrl-C
        # into a swallowed tool-error outcome. FIXED contract: InterruptException
        # propagates so the loop aborts.
        interrupting = (name, args) -> throw(InterruptException())
        @test_throws InterruptException UniLM._dispatch_tool("boom", Dict{String,Any}(), interrupting)

        # Every other exception keeps the existing behavior: a failed outcome.
        failing = (name, args) -> error("kaboom")
        outcome = UniLM._dispatch_tool("boom", Dict{String,Any}(), failing)
        @test outcome.success == false && occursin("kaboom", outcome.error)
    end
end
