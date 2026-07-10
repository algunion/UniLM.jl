# ============================================================================
# Shared SSE machine (src/sse.jl) — unit + driver tests. Decision 1 of
# docs/superpowers/specs/2026-07-10-wave1-p0-architecture.md.
# Fully offline (zero-spend). Self-contained: own imports + own mock server.
# ============================================================================
using UniLM
using UniLM: StreamState, _build_stream_message, TOOL_CALLS, STOP
using Test, HTTP, JSON, Sockets

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
