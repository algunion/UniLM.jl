# ============================================================================
# Shared SSE machine (src/sse.jl) — unit + driver tests. Decision 1 of
# docs/superpowers/specs/2026-07-10-wave1-p0-architecture.md.
# Fully offline (zero-spend). Self-contained: own imports + own mock server.
# ============================================================================
using UniLM
using UniLM: StreamState, _build_stream_message, TOOL_CALLS, STOP
using Test, HTTP, JSON, Sockets

# Fragmenting SSE mock (portable across HTTP 1.9/2.x — same intersection APIs
# as test/regression_p0.jl's: no listen!(stream=true), drain with read()).
function ws1_stream_server(chunks::Vector{String}; gap::Float64=0.4)
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2])
    close(tcp)
    server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
        read(http)
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        for c in chunks
            write(http, c)
            flush(http)
            sleep(gap)
        end
    end
    server, "http://127.0.0.1:$port"
end

# Test seam: an endpoint that speaks OpenAI-wire ROUTING to the local mock but
# parses SSE with the ANTHROPIC handler — the sanctioned way to drive Anthropic
# wire through the provider-agnostic driver without base-URL injection (a
# wave-2 item per the spec). Also witnesses that handle_sse_event! is an
# overridable seam, as decode_stream_chunk was.
struct AnthropicWireEndpoint <: UniLM.ServiceEndpoint
    base_url::String
end
UniLM.get_url(s::AnthropicWireEndpoint, ::Chat) = s.base_url * "/v1/messages"
UniLM.auth_header(::AnthropicWireEndpoint) = ["Content-Type" => "application/json"]
UniLM.handle_sse_event!(::AnthropicWireEndpoint, event::AbstractString,
                        payload::AbstractString, state::UniLM.StreamState) =
    UniLM.handle_sse_event!(UniLM.ANTHROPICServiceEndpoint, event, payload, state)

@testset "layer 1 — _sse_complete_lines!" begin
    @testset "splits complete lines, stashes tail VERBATIM (no strip)" begin
        carry = IOBuffer()
        lines = UniLM._sse_complete_lines!(carry, "data: a\r\ndata: b\ndata: {\"x\": \"hel")
        @test lines == ["data: a", "data: b"]                # trailing \r dropped per line
        @test String(take!(carry)) == "data: {\"x\": \"hel"  # verbatim — whitespace intact
    end

    @testset "whitespace inside a split JSON string survives (the old strip() bug)" begin
        carry = IOBuffer()
        @test isempty(UniLM._sse_complete_lines!(carry, "data: {\"t\":\"a "))
        lines = UniLM._sse_complete_lines!(carry, " b\"}\n")
        @test lines == ["data: {\"t\":\"a  b\"}"]            # both spaces preserved
    end

    @testset "no newline at all → everything carried" begin
        carry = IOBuffer()
        @test isempty(UniLM._sse_complete_lines!(carry, "data: x"))
        @test String(take!(carry)) == "data: x"
    end

    @testset "CRLF + empty lines dropped" begin
        carry = IOBuffer()
        @test UniLM._sse_complete_lines!(carry, "a\r\n\r\n\nb\n") == ["a", "b"]
        @test isempty(take!(carry))
    end
end

@testset "layer 2 — _sse_events! field framing" begin
    @testset "event/data recognized; comments, id:, retry:, unknown ignored" begin
        ev = Ref(""); carry = IOBuffer()
        pairs = UniLM._sse_events!(carry, ev,
            "event: ping\ndata: {\"a\":1}\ndata:{\"b\":2}\n: keep-alive\nid: 42\nretry: 100\ndata: [DONE]\n")
        @test pairs == [("ping", "{\"a\":1}"), ("ping", "{\"b\":2}"), ("ping", "[DONE]")]
        @test ev[] == "ping"                                  # sticky until the next event: line
    end

    @testset "payload space handling: at most ONE leading space removed" begin
        ev = Ref(""); carry = IOBuffer()
        pairs = UniLM._sse_events!(carry, ev, "data:no-space\ndata: one-space\ndata:  two-spaces\n")
        @test pairs == [("", "no-space"), ("", "one-space"), ("", " two-spaces")]
    end

    @testset "event name updates mid-batch" begin
        ev = Ref(""); carry = IOBuffer()
        pairs = UniLM._sse_events!(carry, ev, "data: a\nevent: e2\ndata: b\n")
        @test pairs == [("", "a"), ("e2", "b")]
        @test ev[] == "e2"
    end

    @testset "every byte split reassembles identically (multibyte-safe)" begin
        sse = "event: e1\ndata: {\"delta\":\"héllo, wörld\"}\r\n\ndata: [DONE]\n"
        whole = UniLM._sse_events!(IOBuffer(), Ref(""), sse)
        bytes = Vector{UInt8}(sse)
        ok = true
        for k in 1:length(bytes)-1
            ev = Ref(""); carry = IOBuffer()
            got = vcat(UniLM._sse_events!(carry, ev, String(bytes[1:k])),
                       UniLM._sse_events!(carry, ev, String(bytes[k+1:end])))
            ok &= (got == whole)
        end
        @test ok
    end
end

@testset "_parse_tool_arguments — zero-arg contract (P0-15b mechanism)" begin
    @test UniLM._parse_tool_arguments("") == Dict{String,Any}()
    @test UniLM._parse_tool_arguments("  ") == Dict{String,Any}()
    @test UniLM._parse_tool_arguments("{\"city\":\"Oslo\"}") == Dict{String,Any}("city" => "Oslo")
    @test_throws ArgumentError UniLM._parse_tool_arguments("[1,2]")   # non-object: loud, not silent
end

@testset "StreamState — WS1 additions" begin
    st = StreamState()
    @test st.error === nothing
    @test st.fired_tool_calls == Set{Int}()
    @test st.pending_delta isa IOBuffer
end

@testset "handle_sse_event! (OpenAI-wire default) — Decision 1 contracts" begin
    S = OPENAIServiceEndpoint

    @testset "[DONE] is the ONLY EOS; finish_reason only records" begin
        st = StreamState()
        @test UniLM.handle_sse_event!(S, "", "[DONE]", st) === :done
        st2 = StreamState()
        @test UniLM.handle_sse_event!(S, "",
            "{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}", st2) === :continue
        @test st2.finish_reason == "stop"
    end

    @testset "choices:[] tolerated; usage captured from any chunk" begin
        st = StreamState()
        @test UniLM.handle_sse_event!(S, "",
            "{\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}",
            st) === :continue
        @test st.usage !== nothing && st.usage.total_tokens == 3
    end

    @testset "content delta lands in content AND pending_delta" begin
        st = StreamState()
        UniLM.handle_sse_event!(S, "", "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}", st)
        @test String(take!(st.pending_delta)) == "hi"
        @test String(take!(st.content)) == "hi"
    end

    @testset "refusal delta accumulates" begin
        st = StreamState()
        UniLM.handle_sse_event!(S, "", "{\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"no\"}}]}", st)
        @test String(take!(st.refusal)) == "no"
    end

    @testset "tool-call deltas accumulate by index" begin
        st = StreamState()
        UniLM.handle_sse_event!(S, "",
            "{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}", st)
        UniLM.handle_sse_event!(S, "",
            "{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\":\"}}]}}]}", st)
        UniLM.handle_sse_event!(S, "",
            "{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"NYC\\\"}\"}}]}}]}", st)
        @test st.tool_calls[0]["id"] == "call_abc"
        @test st.tool_calls[0]["function"]["name"] == "get_weather"
        @test st.tool_calls[0]["function"]["arguments"] == "{\"location\":\"NYC\"}"
    end
end

@testset "_sse_dispatch! (OpenAI wire) — drop policy + adversarial wire" begin
    S = OPENAIServiceEndpoint

    @testset "malformed COMPLETE line: dropped + counted, carry stays EMPTY" begin
        before = UniLM._SSE_DROPPED_LINES[]
        st = StreamState(); carry = IOBuffer()
        @test UniLM._sse_dispatch!(S, carry, Ref(""), "data: {invalid json\n", st) === :continue
        @test UniLM._SSE_DROPPED_LINES[] == before + 1
        @test isempty(take!(carry))          # never re-queued — the Azure/proxy poison fix
    end

    @testset "keep-alive comments and Azure-style empty-choices preamble are harmless" begin
        st = StreamState(); carry = IOBuffer()
        chunk = ": keep-alive\n\ndata: {\"choices\":[],\"prompt_filter_results\":[]}\n\n" *
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n"
        @test UniLM._sse_dispatch!(S, carry, Ref(""), chunk, st) === :continue
        @test String(take!(st.content)) == "ok"
        @test isempty(take!(carry))
    end

    @testset "golden stream re-split at every byte → identical final state" begin
        golden = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Saint-Exupéry\"},\"finish_reason\":null}]}\n\n" *
                 "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" *
                 "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n" *
                 "data: [DONE]\n\n"
        bytes = Vector{UInt8}(golden)
        ok = true
        for k in 1:length(bytes)-1
            st = StreamState(); carry = IOBuffer(); ev = Ref("")
            s1 = UniLM._sse_dispatch!(S, carry, ev, String(bytes[1:k]), st)
            s2 = s1 === :continue ? UniLM._sse_dispatch!(S, carry, ev, String(bytes[k+1:end]), st) : s1
            ok &= (s2 === :done) & (String(take!(st.content)) == "Saint-Exupéry") &
                  (st.finish_reason == "stop") & (st.usage !== nothing && st.usage.total_tokens == 12)
        end
        @test ok
    end
end

@testset "handle_sse_event! (Anthropic) — message_stop EOS + error capture + block-stop flag" begin
    A = ANTHROPICServiceEndpoint

    @testset "message_stop → :done" begin
        st = StreamState()
        @test UniLM.handle_sse_event!(A, "message_stop", "{\"type\":\"message_stop\"}", st) === :done
    end

    @testset "error event → :error with payload stored (P0-4 mechanism)" begin
        st = StreamState()
        r = UniLM.handle_sse_event!(A, "error",
            "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}", st)
        @test r === :error
        @test st.error isa Dict{String,Any}
        @test st.error["error"]["type"] == "overloaded_error"
    end

    @testset "content_block_stop marks a streamed tool call complete" begin
        st = StreamState()
        UniLM.handle_sse_event!(A, "content_block_start",
            "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"ping\"}}", st)
        @test get(st.tool_calls[0], "complete", false) == false
        UniLM.handle_sse_event!(A, "content_block_stop",
            "{\"type\":\"content_block_stop\",\"index\":0}", st)
        @test st.tool_calls[0]["complete"] === true
    end

    @testset "text deltas land in content AND pending_delta" begin
        st = StreamState()
        UniLM.handle_sse_event!(A, "content_block_delta",
            "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hey\"}}", st)
        @test String(take!(st.pending_delta)) == "hey"
        @test String(take!(st.content)) == "hey"
    end
end

@testset "handle_sse_event! (Gemini) — no sentinel: NEVER :done" begin
    G = GEMINIServiceEndpoint

    @testset "finishReason records but does not terminate" begin
        st = StreamState()
        r = UniLM.handle_sse_event!(G, "",
            "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}]}", st)
        @test r === :continue
        @test st.finish_reason == STOP
        @test String(take!(st.pending_delta)) == "hi"
    end

    @testset "trailing usageMetadata-only chunk is consumed (the reason EOF-reads exist)" begin
        st = StreamState()
        r = UniLM.handle_sse_event!(G, "",
            "{\"usageMetadata\":{\"promptTokenCount\":8,\"candidatesTokenCount\":5,\"totalTokenCount\":13}}", st)
        @test r === :continue
        @test st.usage !== nothing && st.usage.total_tokens == 13
    end

    @testset "functionCall parts arrive whole → marked complete, signature kept" begin
        st = StreamState()
        UniLM.handle_sse_event!(G, "",
            "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":\"fc_1\",\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\"}},\"thoughtSignature\":\"SIG\"}]}}]}", st)
        @test st.tool_calls[0]["complete"] === true
        @test st.tool_calls[0]["thought_signature"] == "SIG"
        @test st.tool_calls[0]["function"]["arguments"] == "{\"city\":\"Oslo\"}"
    end
end

@testset "driver — delta forwarding + EOF rules + error mapping" begin
    @testset "deltas forwarded verbatim, then the final Message" begin
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        ]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            received = Any[]
            res = fetch(chatrequest!(chat; callback=(c, _) -> push!(received, c)))
            @test res isa LLMSuccess
            @test res.message.content == "Hello"
            strs = [x for x in received if x isa String]
            @test join(strs) == "Hello"                     # exact deltas, no re-diffing
            @test !isempty(strs)
            @test received[end] isa Message && received[end].content == "Hello"
        finally
            close(server)
        end
    end

    @testset "EOF with finish_reason recorded (Gemini shape) → LLMSuccess" begin
        # No [DONE] sentinel at all — the server just closes after the final chunk.
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        ]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat))
            @test res isa LLMSuccess && res.message.content == "Hi"
        finally
            close(server)
        end
    end

    @testset "EOF with NO terminal signal (truncated stream) → LLMFailure" begin
        chunks = ["data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n"]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat))
            @test res isa LLMFailure
            @test res.status == 200            # HTTP was fine; the STREAM was truncated
        finally
            close(server)
        end
    end

    @testset "in-band overloaded error → LLMFailure(529), never LLMSuccess (P0-4 mapping)" begin
        chunks = [
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        ]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=AnthropicWireEndpoint(base), model="claude-mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat))
            @test res isa LLMFailure
            @test res.status == 529
            @test occursin("overloaded_error", res.response)
        finally
            close(server)
        end
    end

    @testset "in-band non-overloaded error → LLMCallError" begin
        chunks = ["event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"bad\"}}\n\n"]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=AnthropicWireEndpoint(base), model="claude-mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat))
            @test res isa LLMCallError
            @test occursin("invalid_request_error", res.error)
        finally
            close(server)
        end
    end

    @testset "Anthropic wire end-to-end: message_stop → LLMSuccess" begin
        chunks = [
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n" *
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=AnthropicWireEndpoint(base), model="claude-mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat))
            @test res isa LLMSuccess && res.message.content == "hello"
        finally
            close(server)
        end
    end

    @testset "parallel tool calls fire once each; early fire on max-index rule" begin
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"f1\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"f2\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n",
        ]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true,
                        tools=[GPTTool(func=GPTFunctionSignature(name="f1"))])
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            fired = String[]
            res = fetch(chatrequest!(chat; on_tool_call=tc -> push!(fired, tc.id)))
            @test res isa LLMSuccess
            @test fired == ["call_1", "call_2"]              # each exactly once, in order
            @test length(res.message.tool_calls) == 2
        finally
            close(server)
        end
    end

    @testset "user close via callback → no message, legacy LLMFailure preserved" begin
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        ]
        server, base = ws1_stream_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat; callback=(c, close_ref) -> (close_ref[] = true)))
            @test res isa LLMFailure                          # pre-WS1 user-close contract, unchanged
        finally
            close(server)
        end
    end
end
