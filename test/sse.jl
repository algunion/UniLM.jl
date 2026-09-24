# ============================================================================
# Shared SSE machine (src/sse.jl) — unit + driver tests.
# Fully offline (zero-spend). Self-contained: own imports + own mock server.
# ============================================================================
using UniLM
using UniLM: StreamState, _build_stream_message, TOOL_CALLS, STOP
using Test, HTTP, JSON, Sockets, Logging

# Fragmenting SSE mock: a stream handler that drains the request with read() and
# writes each chunk as its own flush, `gap` seconds apart.
function fragmented_sse_server(chunks::Vector{String}; gap::Float64=0.4)
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

# Test seam: an endpoint that speaks the OpenAI wire (request encoding + routing)
# to the local mock but parses SSE with the ANTHROPIC handler — the sanctioned way
# to drive Anthropic wire through the provider-agnostic driver without base-URL
# injection (the Anthropic endpoint type does not currently support overriding its
# base URL). Subtypes OpenAIWireEndpoint to inherit encode_request; overriding
# handle_sse_event! witnesses that the SSE seam is overridable (as decode_stream_chunk was).
struct AnthropicWireEndpoint <: UniLM.OpenAIWireEndpoint
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

@testset "_parse_tool_arguments — zero-arg contract" begin
    @test UniLM._parse_tool_arguments("") == Dict{String,Any}()
    @test UniLM._parse_tool_arguments("  ") == Dict{String,Any}()
    @test UniLM._parse_tool_arguments("{\"city\":\"Oslo\"}") == Dict{String,Any}("city" => "Oslo")
    @test_throws ArgumentError UniLM._parse_tool_arguments("[1,2]")   # non-object: loud, not silent
end

@testset "StreamState — streaming-machine additions" begin
    st = StreamState()
    @test st.error === nothing
    @test st.fired_tool_calls == Set{Int}()
    @test st.pending_delta isa IOBuffer
end

@testset "handle_sse_event! (OpenAI-wire default) — wire contracts" begin
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

    @testset "an in-band error payload is terminal and recorded" begin
        # vLLM and OpenAI-compatible proxies report a mid-stream failure as
        # `data: {"error": …}` on the HTTP-200 stream, usually followed by [DONE].
        st = StreamState()
        @test UniLM.handle_sse_event!(S, "",
            "{\"error\":{\"object\":\"error\",\"message\":\"backend died\",\"type\":\"InternalServerError\",\"code\":500}}",
            st) === :error
        @test st.error["message"] == "backend died" && st.error["code"] == 500
        st2 = StreamState()
        @test UniLM.handle_sse_event!(S, "", "{\"error\":\"boom\"}", st2) === :error
        @test st2.error["error"] == "boom"
        @test UniLM.handle_sse_event!(S, "", "{\"error\":null,\"choices\":[]}", StreamState()) === :continue
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
        # Fragments are buffered and joined once, when the call is read.
        fn = UniLM._tool_function!(st, 0)
        @test fn["name"] == "get_weather"
        @test fn["arguments"] == "{\"location\":\"NYC\"}"
        @test UniLM._tool_function!(st, 0) == fn            # joining is idempotent
    end
end

@testset "SSE machine cost is linear in what the wire carries" begin
    # Minimum of three runs: @allocated counts every thread's allocations.
    min_alloc(f) = minimum(_ -> (f(); @allocated f()), 1:3)

    @testset "a long line in 16 KiB reads allocates linearly in its length" begin
        # The old carry re-joined and re-scanned the whole pending line on every read:
        # quadratic, so a 4 MiB line allocated about 1 GB. Quadrupling the line
        # quadruples a linear cost and multiplies a quadratic one by 16, so the ratio is
        # the test; the absolute bound only rules out a constant-factor blow-up, with
        # room for IOBuffer's growth policy (a 4 MiB line measured ~2.6x its size).
        function measure(n)
            value = repeat("x", n - 6)
            chunks = let bytes = codeunits("data: " * value * "\n")
                [String(bytes[i:min(i + 16383, end)]) for i in 1:16384:length(bytes)]
            end
            events = Ref{Any}(nothing)
            feed() = (carry = IOBuffer(); ev = Ref("");
                      events[] = reduce(vcat, [UniLM._sse_events!(carry, ev, c) for c in chunks]); nothing)
            (; bytes = min_alloc(feed), exact = length(events[]) == 1 && events[][1][2] == value)
        end
        mib = 1024 * 1024
        small, large = measure(mib), measure(4mib)
        @test small.exact && large.exact                 # byte-identical payload
        @test large.bytes / small.bytes < 6
        @test large.bytes < 8 * 4mib
    end

    @testset "tool-call arguments in 2 KiB fragments accumulate linearly" begin
        # `s = s * piece` per fragment was quadratic in the argument length.
        frag = repeat("a", 2048)
        payloads = [JSON.json(Dict("choices" => [Dict("index" => 0, "delta" => Dict("tool_calls" =>
            [Dict("index" => 0, "function" => Dict("arguments" => frag))]))])) for _ in 1:1024]
        st = Ref{Any}(nothing)
        feed() = (s = StreamState();
                  foreach(p -> UniLM.handle_sse_event!(OPENAIServiceEndpoint, "", p, s), payloads);
                  st[] = s; nothing)
        total = 1024 * 2048
        @test min_alloc(feed) < 16total      # quadratic accumulation is ~1000x here
        @test UniLM._tool_function!(st[], 0)["arguments"] == repeat(frag, 1024)
    end
end

struct _InterruptingWire <: UniLM.OpenAIWireEndpoint end
UniLM.handle_sse_event!(::_InterruptingWire, event::AbstractString, payload::AbstractString,
                        state::UniLM.StreamState) = throw(InterruptException())

@testset "an interrupt raised while parsing a payload is never swallowed" begin
    st = StreamState()
    @test_throws InterruptException UniLM._sse_dispatch!(_InterruptingWire(), IOBuffer(), Ref(""),
                                                         "data: {}\n", st)
    @test st.sse_dropped == 0          # not miscounted as an undecodable line
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

    @testset "error event → :error with payload stored" begin
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
        server, base = fragmented_sse_server(chunks)
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
        server, base = fragmented_sse_server(chunks)
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
        server, base = fragmented_sse_server(chunks)
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

    @testset "in-band overloaded error → LLMFailure(529), never LLMSuccess" begin
        chunks = [
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        ]
        server, base = fragmented_sse_server(chunks)
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
        server, base = fragmented_sse_server(chunks)
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

    @testset "OpenAI-wire in-band error then [DONE] → typed failure, never the partial text" begin
        # vLLM's mid-stream failure shape: partial text, an error payload, then [DONE].
        # A numeric code is the server's own status for the failure (as for Gemini).
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":null}]}\n\n",
            "data: {\"error\":{\"object\":\"error\",\"message\":\"backend died\",\"type\":\"InternalServerError\",\"code\":500}}\n\n" *
            "data: [DONE]\n\n",
        ]
        server, base = fragmented_sse_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            deltas = String[]
            res = fetch(chatrequest!(chat; callback=(c, _) -> c isa String && push!(deltas, c)))
            @test !(res isa LLMSuccess)
            @test res isa LLMFailure && res.status == 500
            @test res isa LLMFailure && occursin("backend died", res.response)
            @test deltas == ["par"]
            @test all(m -> m.role != UniLM.RoleAssistant, chat.messages)   # nothing committed
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
        server, base = fragmented_sse_server(chunks)
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
        server, base = fragmented_sse_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true,
                        tools=[Tool(func=FunctionSignature(name="f1"))])
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

    @testset "user close via callback → typed cancellation, no message, nothing committed" begin
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        ]
        server, base = fragmented_sse_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            seen = Any[]
            res = fetch(chatrequest!(chat; callback=(c, close_ref) -> (push!(seen, c); close_ref[] = true)))
            @test res isa LLMCallError && isnothing(res.status)
            @test res isa LLMCallError && res.cause isa UniLMCancelled && res.cause.source === :callback
            @test seen == ["a"]                               # no terminal Message callback
            @test length(chat.messages) == 2
        finally
            close(server)
        end
    end
end

@testset "driver — per-turn SSE drop count" begin
    # The drop policy keeps a poisoned line from killing the turn, but a truncated
    # stream was indistinguishable from a clean one: the only trace was a
    # process-global counter no caller can attribute to its own request. The count
    # now rides the result, and a drop — an undecodable provider payload, never
    # routine — is stated once per turn.
    _dropwarns(logs) = count(l -> l.level == Logging.Warn &&
        occursin("undecodable data payloads dropped", string(l.message)), logs)

    @testset "one undecodable payload → sse_dropped == 1 and exactly one warning" begin
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {invalid json\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        ]
        server, base = fragmented_sse_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            # The driver runs in a spawned task, which inherits the logger in force
            # AT SPAWN — so the whole call happens under the collector, not just the fetch.
            logs, res = Test.collect_test_logs(min_level=Logging.Warn) do
                fetch(chatrequest!(chat))
            end
            @test res isa LLMSuccess
            @test res.message.content == "Hello"   # the surviving lines still built the turn
            @test res.sse_dropped == 1
            @test _dropwarns(logs) == 1
        finally
            close(server)
        end
    end

    @testset "clean stream → sse_dropped == 0, no warning" begin
        chunks = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ]
        server, base = fragmented_sse_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            logs, res = Test.collect_test_logs(min_level=Logging.Warn) do
                fetch(chatrequest!(chat))
            end
            @test res isa LLMSuccess && res.sse_dropped == 0
            @test _dropwarns(logs) == 0
        finally
            close(server)
        end
    end

    @testset "a dropped line on a stream that never terminates still reports on the failure" begin
        # Truncated stream: no [DONE], no finish_reason → LLMFailure. The drop count
        # is exactly what tells the caller WHY the terminal never arrived.
        chunks = ["data: {invalid json\n\n"]
        server, base = fragmented_sse_server(chunks)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
            push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "u"))
            res = fetch(chatrequest!(chat))
            @test res isa LLMFailure
            @test res.sse_dropped == 1
        finally
            close(server)
        end
    end

    @testset "the non-streamed path reports no drops" begin
        # `sse_dropped` defaults to 0, so a path that never ran the SSE machine is
        # truthfully quiet rather than absent.
        st = StreamState()
        @test st.sse_dropped == 0
        @test LLMSuccess(message=Message(role=UniLM.RoleAssistant, content="x"),
                         self=Chat(model="m")).sse_dropped == 0
    end
end

# ─── Driver: cancellation, stops, time in user code, user-code failures ─────────

# SSE server on an OS-assigned port. After reading the request it answers `status`
# (with `headers`) and streams `chunks` `gap` seconds apart, then holds the connection
# `hold` seconds. `hits` counts the requests that reached it.
function paced_sse_server(chunks::Vector{String}; gap::Real=0.1, hold::Real=0.0,
                          status::Int=200, headers::Vector{Pair{String,String}}=Pair{String,String}[])
    hits = Threads.Atomic{Int}(0)
    server = HTTP.listen!("127.0.0.1", 0; verbose=false) do http::HTTP.Stream
        read(http)
        Threads.atomic_add!(hits, 1)
        HTTP.setstatus(http, status)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        foreach(h -> HTTP.setheader(http, h), headers)
        HTTP.startwrite(http)
        for c in chunks
            write(http, c); flush(http); sleep(gap)
        end
        sleep(hold)
    end
    (; server, url="http://127.0.0.1:$(HTTP.port(server))", hits)
end

# A server that reads the request and then sends nothing — not even the response
# headers — for `hold` seconds.
function mute_header_server(; hold::Real=10.0)
    hits = Threads.Atomic{Int}(0)
    server = HTTP.listen!("127.0.0.1", 0; verbose=false) do http::HTTP.Stream
        read(http); Threads.atomic_add!(hits, 1); sleep(hold)
    end
    (; server, url="http://127.0.0.1:$(HTTP.port(server))", hits)
end

sse_text(s; finish=nothing) = "data: " * JSON.json(Dict("choices" => [Dict("index" => 0,
    "delta" => Dict("content" => s), "finish_reason" => finish)])) * "\n\n"

stream_chat(url) = Chat(service=GenericOpenAIEndpoint(url, ""), model="mock", stream=true,
                        messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])

const _SLOW_CFG = RequestConfig(request_timeout=30.0, stream_idle_timeout=30.0,
                                total_deadline=120.0, max_attempts=1)

# Run `call()` (a streaming verb returning a Task) on a task that also records when
# its result arrived; `stop()` fires `after` seconds after `ready()` holds (the phase
# under test was reached). Returns the result and the seconds from the stop to it.
function stop_after(call, stop; ready::Function, after::Real=1.0)
    t = Threads.@spawn (r = fetch(call()); (r, time()))
    timedwait(ready, 25.0) === :ok || return (; finished=false, result=nothing, latency=Inf)
    sleep(after)
    stopped_at = time()
    stop()
    finished = timedwait(() -> istaskdone(t), 25.0) === :ok
    finished || return (; finished, result=nothing, latency=Inf)
    r, done_at = fetch(t)
    (; finished, result=r, latency=done_at - stopped_at)
end

cancelled_by(r, source) = r isa LLMCallError && isnothing(r.status) &&
                          r.cause isa UniLMCancelled && r.cause.source === source

@testset "driver — a cancel ends the stream promptly, typed, and commits nothing" begin
    @testset "mid-stream: chunks 5 s apart, cancelled 1 s in" begin
        srv = paced_sse_server([sse_text("a"), sse_text("b"), sse_text("c"; finish="stop"),
                                "data: [DONE]\n\n"]; gap=5.0)
        try
            chat = stream_chat(srv.url); tok = CancelToken(); seen = Any[]
            o = stop_after(() -> chatrequest!(chat; config=_SLOW_CFG, cancel=tok,
                                              callback=(c, _) -> push!(seen, c)),
                           () -> cancel!(tok); ready=() -> !isempty(seen))
            @test o.finished && cancelled_by(o.result, :token)
            @test o.latency < 0.5
            @test seen == ["a"]                     # no terminal callback
            @test length(chat.messages) == 2        # the partial turn is not committed
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "during a mute header wait" begin
        srv = mute_header_server(hold=10.0)
        try
            chat = stream_chat(srv.url); tok = CancelToken()
            o = stop_after(() -> chatrequest!(chat; config=_SLOW_CFG, cancel=tok), () -> cancel!(tok);
                           ready=() -> srv.hits[] == 1)
            @test o.finished && cancelled_by(o.result, :token)
            @test o.latency < 0.5
            @test length(chat.messages) == 2
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "during a Retry-After backoff, never retried" begin
        srv = paced_sse_server(["slow down"]; status=429, headers=["Retry-After" => "30"])
        try
            chat = stream_chat(srv.url); tok = CancelToken()
            cfg = RequestConfig(request_timeout=30.0, stream_idle_timeout=30.0,
                                total_deadline=120.0, max_attempts=3)
            t = Threads.@spawn (r = fetch(chatrequest!(chat; config=cfg, cancel=tok)); (r, time()))
            @test timedwait(() -> srv.hits[] == 1, 25.0) === :ok   # the 429 is in: backing off
            sleep(0.3)
            cancelled_at = time()
            cancel!(tok)
            @test timedwait(() -> istaskdone(t), 25.0) === :ok
            r, done_at = fetch(t)
            @test cancelled_by(r, :token)
            @test done_at - cancelled_at < 0.5
            @test srv.hits[] == 1
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "a pre-cancelled token sends nothing — explicit or ambient" begin
        srv = paced_sse_server([sse_text("a"; finish="stop"), "data: [DONE]\n\n"])
        try
            tok = cancel!(CancelToken())
            @test cancelled_by(fetch(chatrequest!(stream_chat(srv.url); cancel=tok)), :token)
            @test cancelled_by(with_cancel(() -> fetch(chatrequest!(stream_chat(srv.url))), tok), :token)
            @test srv.hits[] == 0
            @test isempty(tok.hooks)
        finally
            HTTP.forceclose(srv.server)
        end
    end
end

@testset "driver — the callback's close flag is the same typed stop" begin
    @testset "set from another task: prompt, source :callback" begin
        srv = paced_sse_server([sse_text("a"), sse_text("b"), "data: [DONE]\n\n"]; gap=5.0)
        try
            chat = stream_chat(srv.url); handle = Ref{Any}(nothing); seen = Any[]
            o = stop_after(() -> chatrequest!(chat; config=_SLOW_CFG,
                                              callback=(c, close) -> (push!(seen, c); handle[] = close)),
                           () -> (handle[][] = true); ready=() -> handle[] !== nothing)
            @test o.finished && cancelled_by(o.result, :callback)
            @test o.latency < 0.5
            @test seen == ["a"]
            @test length(chat.messages) == 2
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "a stop after the terminal was recorded leaves the turn standing" begin
        srv = paced_sse_server([sse_text("hi"), sse_text(""; finish="stop"), "data: [DONE]\n\n"])
        srv2 = paced_sse_server([sse_text("yo"; finish="stop"), "data: [DONE]\n\n"])
        try
            chat = stream_chat(srv.url)
            r = fetch(chatrequest!(chat; callback=(c, close) -> c isa Message && (close[] = true)))
            @test r isa LLMSuccess && r.message.content == "hi"
            @test length(chat.messages) == 3                        # committed
            # ... and on the delta that carries the finish reason.
            chat2 = stream_chat(srv2.url); seen = Any[]
            r2 = fetch(chatrequest!(chat2; callback=(c, close) -> (push!(seen, c); close[] = true)))
            @test r2 isa LLMSuccess && r2.message.content == "yo"
            @test length(chat2.messages) == 3
            @test count(x -> x isa Message, seen) == 1              # the terminal callback, once
        finally
            HTTP.forceclose(srv.server)
            HTTP.forceclose(srv2.server)
        end
    end
end

@testset "driver — time in user code is not wire idle time" begin
    @testset "a callback 3 s per delta under stream_idle_timeout=1 still succeeds" begin
        srv = paced_sse_server([sse_text("a"), sse_text("b"), sse_text("c"; finish="stop"),
                                "data: [DONE]\n\n"]; gap=0.2)
        try
            chat = stream_chat(srv.url); deltas = String[]
            cfg = RequestConfig(request_timeout=10.0, stream_idle_timeout=1.0, total_deadline=60.0,
                                max_attempts=1)
            t = chatrequest!(chat; config=cfg,
                             callback=(c, _) -> c isa String && (push!(deltas, c); sleep(3.0)))
            @test timedwait(() -> istaskdone(t), 40.0) === :ok
            r = fetch(t)
            @test r isa LLMSuccess && r.message.content == "abc"
            @test join(deltas) == "abc"
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "a mute peer after the first chunk still fails :stream_idle on time" begin
        limit = 2.0
        period = min(limit / 4, 5.0)
        srv = paced_sse_server([sse_text("a")]; hold=15.0)
        try
            chat = stream_chat(srv.url)
            cfg = RequestConfig(request_timeout=10.0, stream_idle_timeout=limit, total_deadline=60.0,
                                max_attempts=1)
            t = chatrequest!(chat; config=cfg)
            @test timedwait(() -> istaskdone(t), 25.0) === :ok
            r = fetch(t)
            @test r isa LLMCallError && r.cause isa UniLMTimeout && r.cause.phase === :stream_idle
            # The native read-idle timer is armed when the read starts, just before our
            # stamp, hence the 0.95 floor; the ceiling adds 0.5 s of timer scheduling.
            @test r isa LLMCallError && 0.95limit <= r.cause.elapsed <= limit + period + 0.5
        finally
            HTTP.forceclose(srv.server)
        end
    end
end

@testset "driver — user-code failures are the outcome, never teardown noise" begin
    @testset "an IOError from the callback on the finishing delta" begin
        srv = paced_sse_server([sse_text("hi"; finish="stop") * "data: [DONE]\n\n"])
        try
            chat = stream_chat(srv.url); calls = String[]
            cfg = RequestConfig(request_timeout=10.0, stream_idle_timeout=10.0, total_deadline=60.0,
                                max_attempts=3)
            r = fetch(chatrequest!(chat; config=cfg, callback=(c, _) -> c isa String ?
                (push!(calls, "delta"); throw(Base.IOError("user sink closed", 0))) :
                push!(calls, "message")))
            @test r isa LLMCallError && r.cause isa Base.IOError
            @test calls == ["delta"]                 # no callback after the one that threw
            @test length(chat.messages) == 2
            @test srv.hits[] == 1                    # never retried
        finally
            HTTP.forceclose(srv.server)
        end
    end

    tool_sse = "data: " * JSON.json(Dict("choices" => [Dict("index" => 0, "delta" => Dict("tool_calls" =>
        [Dict("index" => 0, "id" => "call_1", "type" => "function",
              "function" => Dict("name" => "f", "arguments" => "{}"))]), "finish_reason" => "tool_calls")])) *
        "\n\ndata: [DONE]\n\n"

    @testset "an exception from on_tool_call" begin
        srv = paced_sse_server([tool_sse])
        try
            chat = stream_chat(srv.url)
            cfg = RequestConfig(request_timeout=10.0, stream_idle_timeout=10.0, total_deadline=60.0,
                                max_attempts=3)
            r = fetch(chatrequest!(chat; config=cfg, on_tool_call=_ -> error("tool sink failed")))
            @test r isa LLMCallError && r.cause isa ErrorException && r.cause.msg == "tool sink failed"
            @test length(chat.messages) == 2
            @test srv.hits[] == 1
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "an interrupt from on_tool_call propagates" begin
        srv = paced_sse_server([tool_sse])
        try
            t = chatrequest!(stream_chat(srv.url); on_tool_call=_ -> throw(InterruptException()))
            @test timedwait(() -> istaskdone(t), 25.0) === :ok
            @test istaskfailed(t) && t.exception isa InterruptException
        finally
            HTTP.forceclose(srv.server)
        end
    end
end

@testset "driver — a tool call cut mid-arguments leaves a truncated success" begin
    # The token limit ends the turn inside the call's arguments; usage follows the
    # finish chunk, then [DONE]. The turn stands with its reason and usage; the
    # partial call is dropped, so neither on_tool_call nor a tool loop can run it.
    tc(fields::Pair...) = "data: " * JSON.json(Dict("choices" => [Dict("index" => 0,
        "delta" => Dict("tool_calls" => [Dict{String,Any}("index" => 0, fields...)]))])) * "\n\n"
    usage = "data: " * JSON.json(Dict("choices" => [], "usage" => Dict("prompt_tokens" => 10,
        "completion_tokens" => 50, "total_tokens" => 60))) * "\n\n"
    srv = paced_sse_server([
        tc("id" => "call_1", "type" => "function", "function" => Dict("name" => "get_weather", "arguments" => "")),
        tc("function" => Dict("arguments" => "{\"city\": \"Par")),
        sse_text(""; finish="length"), usage * "data: [DONE]\n\n"])
    try
        chat = stream_chat(srv.url); seen = Any[]; fired = ToolCall[]
        r = fetch(chatrequest!(chat; config=_SLOW_CFG, callback=(c, _) -> push!(seen, c),
                               on_tool_call=call -> push!(fired, call)))
        @test r isa LLMSuccess && r.message.finish_reason == "length"
        @test isnothing(r.message.tool_calls) && isempty(fired)
        @test r.usage.completion_tokens == 50
        @test count(x -> x isa Message, seen) == 1 && length(chat.messages) == 3
    finally
        HTTP.forceclose(srv.server)
    end
end

# SSE server that writes `chunks` at once, then holds the response open `hold` seconds.
# It records each request's client address: HTTP/1.1 carries one exchange at a time
# per connection, so requests from one address rode one reused connection.
function peer_sse_server(chunks::Vector{String}; hold::Real=0.0)
    peers, lk = String[], ReentrantLock()
    server = HTTP.listen!("127.0.0.1", 0; verbose=false) do http::HTTP.Stream
        read(http)
        @lock lk push!(peers, string(HTTP.peeraddr(http)))
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        foreach(c -> (write(http, c); flush(http)), chunks)
        sleep(hold)
    end
    (; server, url="http://127.0.0.1:$(HTTP.port(server))", peers)
end

@testset "driver — a Chat stream ends at [DONE], not at the end of its HTTP body" begin
    chunks = [sse_text("hi"), sse_text(""; finish="stop") * "data: [DONE]\n\n"]

    @testset "a body held open after [DONE] holds neither the final callback nor the result" begin
        srv = peer_sse_server(chunks; hold=10.0)
        try
            chat = stream_chat(srv.url); final_at = Ref(Inf)
            t0 = time()
            r = fetch(chatrequest!(chat; config=_SLOW_CFG,
                                   callback=(c, _) -> c isa Message && (final_at[] = time() - t0)))
            elapsed = time() - t0
            @test r isa LLMSuccess && r.message.content == "hi" && length(chat.messages) == 3
            @test final_at[] < 1.0
            @test elapsed < 1.5
            # Its connection was closed rather than pooled: the next call opens another.
            @test fetch(chatrequest!(stream_chat(srv.url); config=_SLOW_CFG)) isa LLMSuccess
            @test length(unique(srv.peers)) == 2
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "a body that ends right after [DONE] leaves its connection reusable" begin
        srv = peer_sse_server(chunks)
        try
            rs = [fetch(chatrequest!(stream_chat(srv.url); config=_SLOW_CFG)) for _ in 1:3]
            @test all(r -> r isa LLMSuccess && r.message.content == "hi", rs)
            @test length(srv.peers) == 3 && length(unique(srv.peers)) == 1
        finally
            HTTP.forceclose(srv.server)
        end
    end
end
