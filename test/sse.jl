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
