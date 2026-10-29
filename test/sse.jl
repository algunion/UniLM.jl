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

# Mistral's reasoning stream (https://docs.mistral.ai/studio/conversations/reasoning, "Streaming"):
# "1. Thinking phase: `delta.content` is a list containing a `ThinkChunk` 2. Transition: a single list
# containing both a closing `ThinkChunk` and the first `TextChunk` 3. Answer phase: `delta.content` is a
# plain string". A ThinkChunk is `{"type": "thinking", "thinking": [TextChunk…]}`, a TextChunk
# `{"type": "text", "text": …}`.
_mi_think(t) = Dict("type" => "thinking", "thinking" => [Dict("type" => "text", "text" => t)])
_mi_chunk(content; finish=nothing) = JSON.json(Dict("id" => "cmpl-1", "model" => "mock",
    "choices" => [Dict("index" => 0, "delta" => Dict("content" => content), "finish_reason" => finish)]))
const _MI_REASONING_STREAM = [
    _mi_chunk([_mi_think("17 × 23: 17 × 20 = 340")]),                                    # thinking
    _mi_chunk([_mi_think(", plus 51."), Dict("type" => "text", "text" => "The answer")]),  # transition
    _mi_chunk(" is 391."; finish="stop"),                                                  # answer
]

@testset "handle_sse_event! (OpenAI wire) — a content list: Mistral's reasoning stream" begin
    st = StreamState()
    for payload in _MI_REASONING_STREAM
        @test UniLM.handle_sse_event!(GenericOpenAIEndpoint("https://api.mistral.ai", ""), "", payload, st) === :continue
    end
    @test String(take!(copy(st.pending_delta))) == "The answer is 391."   # the callback's text, in order
    m = _build_stream_message(st)
    @test m.content == "The answer is 391."                              # no text chunk lost, no thinking in it
    @test reasoning_text(m) == "17 × 23: 17 × 20 = 340, plus 51."        # the thinking chunks, as reasoning
    @test st.sse_dropped == 0
end

@testset "driver — Mistral's reasoning stream reaches the callback and the message whole" begin
    server, base = fragmented_sse_server(["data: " .* _MI_REASONING_STREAM .* "\n\n"; "data: [DONE]\n\n"]; gap=0.05)
    try
        chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true)
        push!(chat, Message(Val(:system), "s")); push!(chat, Message(Val(:user), "What is 17 * 23?"))
        received = Any[]
        res = fetch(chatrequest!(chat; callback=(c, _) -> push!(received, c)))
        @test res isa LLMSuccess && res.sse_dropped == 0
        @test res.message.content == "The answer is 391."
        @test [x for x in received if x isa String] == ["The answer", " is 391."]
        @test reasoning_text(res) == "17 × 23: 17 × 20 = 340, plus 51."
        @test last(chat).content == "The answer is 391."
    finally
        close(server)
    end
end

@testset "decode_response (generic) — a reply with reasoning keeps every other field the OpenAI wire decodes" begin
    lp = Dict("content" => [Dict("token" => "391", "logprob" => -0.2, "bytes" => [51, 57, 49], "top_logprobs" => [])])
    resp = HTTP.Response(200, [], Vector{UInt8}(JSON.json(Dict("id" => "cmpl-1", "model" => "mock",
        "moderation" => Dict("output" => Dict("flagged" => false)),
        "choices" => [Dict("index" => 0, "finish_reason" => "stop", "logprobs" => lp,
                           "message" => Dict("role" => "assistant", "content" => "391", "reasoning_content" => "17 × 23."))]))))
    d, wire = UniLM.decode_response(GenericOpenAIEndpoint("http://localhost:8080", ""), resp), UniLM.extract_message(resp)
    @test keys(d) == keys(wire)
    @test d.logprobs == lp && d.moderation == wire.moderation
    @test reasoning_text(d.message) == "17 × 23."
end

@testset "decode_response (generic) — an unstreamed reply's reasoning reaches reasoning_text, as a streamed one's does" begin
    reply(message) = HTTP.Response(200, [], Vector{UInt8}(JSON.json(Dict("id" => "cmpl-1", "model" => "mock",
        "choices" => [Dict("index" => 0, "finish_reason" => "stop", "message" => merge(Dict("role" => "assistant"), message))]))))
    decode(s, message) = UniLM.decode_response(s, reply(message)).message
    mistral = MistralEndpoint(api_key="k")
    # Mistral's reasoning reply: "the response `message.content` is a list of chunks", a ThinkChunk and the
    # answer's TextChunks (https://docs.mistral.ai/studio/conversations/reasoning).
    thinking = [_mi_think("17 × 23: 17 × 20 = 340"), _mi_think(", plus 51."), Dict("type" => "text", "text" => "The answer"),
                Dict("type" => "text", "text" => " is 391.")]
    m = decode(mistral, Dict("content" => thinking))
    @test m.content == "The answer is 391."
    @test reasoning_text(m) == "17 × 23: 17 × 20 = 340, plus 51."
    st = StreamState()
    foreach(p -> UniLM.handle_sse_event!(mistral, "", p, st), _MI_REASONING_STREAM)
    @test reasoning_text(_build_stream_message(st)) == reasoning_text(m)          # the two reads agree
    # llama-server returns the reasoning "via the `reasoning_content` field", vLLM in `reasoning`.
    local_server = GenericOpenAIEndpoint("http://localhost:8080", "")
    for key in ("reasoning_content", "reasoning")
        m = decode(local_server, Dict("content" => "391 = 17 × 23.", key => "Try 17: 17 × 23 = 391."))
        @test m.content == "391 = 17 × 23."
        @test reasoning_text(m) == "Try 17: 17 × 23 = 391."
        @test !occursin(key, JSON.json(m))                       # sent back with no next request
        chunk(delta; finish=nothing) = JSON.json(Dict("id" => "cmpl-1", "model" => "mock",
            "choices" => [Dict("index" => 0, "delta" => delta, "finish_reason" => finish)]))
        st = StreamState()
        for p in (chunk(Dict(key => "Try 17: ")), chunk(Dict(key => "17 × 23 = 391.")), chunk(Dict("content" => "391 = 17 × 23.")),
                  chunk(Dict(); finish="stop"), "[DONE]")
            UniLM.handle_sse_event!(local_server, "", p, st)
        end
        streamed = _build_stream_message(st)
        @test streamed.content == m.content
        @test reasoning_text(streamed) == reasoning_text(m)
        # A stream cut before its finish reason keeps no partial reasoning.
        cut = StreamState()
        UniLM.handle_sse_event!(local_server, "", chunk(Dict(key => "Try 17")), cut)
        @test reasoning_text(_build_stream_message(cut)) === nothing
    end
    # A reply without reasoning carries none, and other endpoints' replies are read as before.
    @test reasoning_text(decode(local_server, Dict("content" => "Hi"))) === nothing
    @test reasoning_text(decode(OPENAIServiceEndpoint, Dict("content" => "Hi", "reasoning_content" => "r"))) === nothing
end

# Reads one HTTP request off `sock` — its headers, then a chunked or Content-Length body —
# so a raw-socket server answers only once the client has sent all of it.
function _sse_drain_request!(sock)
    req = ""
    while (he = findfirst("\r\n\r\n", req)) === nothing
        bytes = readavailable(sock)
        isempty(bytes) && return
        req *= String(bytes)
    end
    headers = SubString(req, 1, last(he))
    length_match = match(r"(?i)content-length:\s*(\d+)", headers)
    complete() = occursin(r"(?i)transfer-encoding:\s*chunked", headers) ? occursin("0\r\n\r\n", req) :
                 isnothing(length_match) || sizeof(req) >= last(he) + parse(Int, length_match.captures[1])
    while !complete()
        bytes = readavailable(sock)
        isempty(bytes) && return
        req *= String(bytes)
    end
end

@testset "handle_sse_event! (OpenAI wire) — delta.audio assembles the audio reply" begin
    S = OPENAIServiceEndpoint
    wav = UniLM._wav_bytes(UInt8[0x00, 0x00, 0x10, 0x00])
    b64 = UniLM.base64encode(wav)
    spoken = AudioOutput(id="audio_abc123", data=wav, expires_at=1792669190, transcript="Hi there!")
    feed!(st, delta) = UniLM.handle_sse_event!(S, "", JSON.json(Dict("choices" => [Dict("index" => 0, "delta" => delta)])), st)
    chunk(delta; finish=nothing) = "data: " * JSON.json(Dict("id" => "chatcmpl-1", "model" => "mock",
        "choices" => [Dict("index" => 0, "delta" => delta, "finish_reason" => finish)])) * "\n\n"

    @testset "id, data and transcript accumulate; the transcript is the turn's text" begin
        st = StreamState()
        feed!(st, Dict("role" => "assistant", "content" => nothing, "audio" => Dict("id" => "audio_abc123", "transcript" => "Hi ")))
        feed!(st, Dict("audio" => Dict("data" => b64[1:15])))                  # a split inside a 4-character group
        feed!(st, Dict("audio" => Dict("data" => b64[16:end], "transcript" => "there!")))
        feed!(st, Dict("audio" => Dict("expires_at" => 1792669190)))           # the last update: its expiry alone
        @test String(take!(st.pending_delta)) == "Hi there!" && position(st.content) == 0
        st.finish_reason = "stop"
        m = _build_stream_message(st)
        @test m.audio_output == spoken && m.content == "Hi there!" && m.finish_reason == "stop"
    end

    @testset "chunks encoded one by one, padding mid-stream, assemble the same bytes" begin
        st = StreamState()
        # The 1-byte chunk is "AA==": its padding sits inside the joined base64, not at its end.
        for audio in (Dict("id" => "audio_1"), Dict("data" => UniLM.base64encode(UInt8[0x00])),
                      Dict("data" => UniLM.base64encode(UInt8[0x01, 0x00, 0x02])), Dict("expires_at" => 7, "transcript" => ""))
            feed!(st, Dict("audio" => audio))
        end
        @test _build_stream_message(st).audio_output ==
              AudioOutput(id="audio_1", data=UInt8[0x00, 0x01, 0x00, 0x02], expires_at=7, transcript="")
    end

    @testset "an audio reply without its id is no reply; one without its expiry is kept, expiry unknown" begin
        st = StreamState()
        feed!(st, Dict("audio" => Dict("data" => b64, "expires_at" => 7)))
        @test_throws ArgumentError _build_stream_message(st)
        st = StreamState()
        feed!(st, Dict("audio" => Dict("id" => "audio_abc123", "data" => b64, "transcript" => "Hi there!")))
        st.finish_reason = "stop"
        @test _build_stream_message(st).audio_output ==
              AudioOutput(id="audio_abc123", data=wav, expires_at=nothing, transcript="Hi there!")
        st = StreamState()
        print(st.content, "plain")
        @test isnothing(_build_stream_message(st).audio_output)              # no audio: none
        # An expiry that is no integer is a malformed update, as it is unstreamed: the handler
        # throws, so the dispatcher drops the chunk and counts it, and the expiry stays unknown.
        for bad in ("soon", true, 1.7926e9)
            st = StreamState()
            feed!(st, Dict("audio" => Dict("id" => "audio_abc123", "data" => b64, "transcript" => "Hi there!")))
            @test_throws ArgumentError feed!(st, Dict("audio" => Dict("expires_at" => bad)))
            payload = JSON.json(Dict("choices" => [Dict("index" => 0, "delta" => Dict("audio" => Dict("expires_at" => bad)))]))
            UniLM._sse_dispatch!(S, IOBuffer(), Ref(""), "data: $payload\n\n", st)
            @test st.sse_dropped == 1
            st.finish_reason = "stop"
            @test _build_stream_message(st).audio_output.expires_at === nothing
        end
    end

    @testset "driver: the callback hears the transcript; the result holds the audio" begin
        chunks = [chunk(Dict("role" => "assistant", "content" => nothing,
                             "audio" => Dict("id" => "audio_abc123", "transcript" => "Hi "))),
                  chunk(Dict("audio" => Dict("data" => b64[1:30]))),
                  chunk(Dict("audio" => Dict("data" => b64[31:end], "transcript" => "there!"))),
                  chunk(Dict(); finish="stop"),
                  chunk(Dict("audio" => Dict("expires_at" => 1792669190))),
                  "data: " * JSON.json(Dict("choices" => [], "usage" => Dict("prompt_tokens" => 9,
                      "completion_tokens" => 21, "total_tokens" => 30))) * "\n\ndata: [DONE]\n\n"]
        server, base = fragmented_sse_server(chunks; gap=0.05)
        try
            chat = Chat(service=GenericOpenAIEndpoint(base, ""), model="mock", stream=true,
                        modalities=["text", "audio"], audio=Dict("voice" => "alloy", "format" => "wav"))
            push!(chat, Message(Val(:system), "Be brief.")); push!(chat, Message(Val(:user), "Say hi."))
            received = Any[]
            res = fetch(chatrequest!(chat; callback=(c, _) -> push!(received, c)))
            @test res isa LLMSuccess
            @test res.message.audio_output == spoken && res.message.content == "Hi there!"
            @test join(x for x in received if x isa String) == "Hi there!"
            @test received[end] isa Message && received[end].audio_output == spoken
            @test last(res.self.messages).audio_output == spoken                # appended with its audio
            @test res.usage.completion_tokens == 21 && res.model == "mock"
        finally
            close(server)
        end
    end

    @testset "driver: a connection dropped after the finish, before the expiry, keeps the turn" begin
        # OpenAI sends the expires_at-only update after the finish_reason chunk. This body
        # ends at the finish, and its chunked framing is cut there (the terminating chunk
        # without its trailer section), so the read that would carry the expiry fails. The
        # turn arrived whole and is billed: it is kept, once, with its expiry unknown, and
        # nothing is sent again.
        body = chunk(Dict("role" => "assistant", "content" => nothing,
                          "audio" => Dict("id" => "audio_abc123", "transcript" => "Hi "))) *
               chunk(Dict("audio" => Dict("data" => b64, "transcript" => "there!"))) *
               chunk(Dict(); finish="stop")
        listener = Sockets.listen(Sockets.localhost, 0)
        url = "http://127.0.0.1:$(Int(Sockets.getsockname(listener)[2]))"
        posts = Threads.Atomic{Int}(0)
        Threads.@spawn try
            while true
                sock = Sockets.accept(listener)
                Threads.atomic_add!(posts, 1)
                try
                    _sse_drain_request!(sock)
                    write(sock, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                    write(sock, string(string(sizeof(body); base=16), "\r\n", body, "\r\n", "0\r\n"))
                finally
                    close(sock)
                end
            end
        catch                                            # the listener closed: the test is over
        end
        try
            chat = Chat(service=GenericOpenAIEndpoint(url, ""), model="mock", stream=true,
                        modalities=["text", "audio"], audio=Dict("voice" => "alloy", "format" => "pcm16"))
            push!(chat, Message(Val(:system), "Be brief.")); push!(chat, Message(Val(:user), "Say hi."))
            terminals = Message[]
            cfg = RequestConfig(request_timeout=5.0, total_deadline=30.0, stream_idle_timeout=5.0, max_attempts=3)
            task = chatrequest!(chat; config=cfg, callback=(x, _) -> x isa Message && push!(terminals, x))
            bounded = timedwait(() -> istaskdone(task), 20.0) === :ok
            @test bounded
            res = bounded ? fetch(task) : nothing
            @test res isa LLMSuccess
            @test res.message.audio_output == AudioOutput(id="audio_abc123", data=wav, expires_at=nothing, transcript="Hi there!")
            @test res.message.content == "Hi there!" && res.message.finish_reason == "stop"
            @test length(terminals) == 1 && terminals[1].audio_output == res.message.audio_output
            @test last(chat.messages).audio_output.expires_at === nothing && posts[] == 1
        finally
            close(listener)
        end
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
            chat = Chat(service=AnthropicWireEndpoint(base), model="claude-haiku-5-5", stream=true)
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
            chat = Chat(service=AnthropicWireEndpoint(base), model="claude-haiku-5-5", stream=true)
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
            chat = Chat(service=AnthropicWireEndpoint(base), model="claude-haiku-5-5", stream=true)
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

# An endpoint of the caller's own that implements its wire's encoders and decoders but no stream
# handler: the chat seam's `handle_sse_event!`, the agentic seam's `decode_agentic_stream`.
struct _NoStreamHandler <: UniLM.ServiceEndpoint end
UniLM.get_url(::_NoStreamHandler, ::Chat) = _url_probe_base[] * "/v1/chat/completions"
UniLM.get_url(::_NoStreamHandler, ::Respond) = _url_probe_base[] * "/v1/responses"
UniLM.auth_header(::_NoStreamHandler) = ["Content-Type" => "application/json"]
UniLM.encode_request(::_NoStreamHandler, chat::Chat) = JSON.json(chat)
UniLM.decode_response(::_NoStreamHandler, resp::HTTP.Response) = UniLM.extract_message(resp)
UniLM.encode_agentic(::_NoStreamHandler, r::Respond) = JSON.json(r)
UniLM.decode_agentic(::_NoStreamHandler, resp::HTTP.Response) = UniLM.parse_response(resp)

@testset "a missing stream handler is refused before anything is sent, on both seams" begin
    sse(lines...) = HTTP.Response(200, ["Content-Type" => "text/event-stream"],
                                  Vector{UInt8}(join(("data: " * l * "\n\n" for l in lines))))
    chat = Chat(service=_NoStreamHandler(), model="m", stream=true,
                messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])
    err, seen = _with_scripted((_, _) -> sse("{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}",
                                             "[DONE]")) do
        try chatrequest!(chat) catch e; e end
    end
    @test err isa MethodError && err.f === UniLM.handle_sse_event! && isempty(seen)
    @test occursin("handle_sse_event!(::$(_NoStreamHandler)", sprint(showerror, err))
    @test length(chat) == 2
    r = Respond(service=_NoStreamHandler(), model="m", input="hi", stream=true)
    err, seen = _with_scripted((_, _) -> sse("{\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}")) do
        try respond(r) catch e; e end
    end
    @test err isa MethodError && err.f === UniLM.decode_agentic_stream && isempty(seen)
    @test occursin("decode_agentic_stream(::$(_NoStreamHandler)", sprint(showerror, err))
    # Every built-in endpoint has its handlers: none is refused.
    for s in (OPENAIServiceEndpoint, AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint, GEMINIServiceEndpoint,
              ANTHROPICServiceEndpoint(), DeepSeekEndpoint(api_key="k"), MistralEndpoint(), OllamaEndpoint())
        @test isnothing(UniLM._require_stream_handler(s))
    end
    for s in (OPENAIServiceEndpoint, GEMINIServiceEndpoint, GenericOpenAIEndpoint("http://localhost:1", ""), OllamaEndpoint())
        @test isnothing(UniLM._require_agentic_stream_decoder(s))
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
    @testset "mid-stream: chunks 20 s apart, cancelled 1 s in" begin
        srv = paced_sse_server([sse_text("a"), sse_text("b"), sse_text("c"; finish="stop"),
                                "data: [DONE]\n\n"]; gap=20.0)
        try
            chat = stream_chat(srv.url); tok = CancelToken(); seen = Any[]
            o = stop_after(() -> chatrequest!(chat; config=_SLOW_CFG, cancel=tok,
                                              callback=(c, _) -> push!(seen, c)),
                           () -> cancel!(tok); ready=() -> !isempty(seen))
            @test o.finished && cancelled_by(o.result, :token)
            @test o.latency < 5.0                   # seen only at the next chunk, 19 s later
            @test seen == ["a"]                     # no terminal callback
            @test length(chat.messages) == 2        # the partial turn is not committed
        finally
            HTTP.forceclose(srv.server)
        end
    end

    @testset "during a mute header wait" begin
        srv = mute_header_server(hold=20.0)
        try
            chat = stream_chat(srv.url); tok = CancelToken()
            o = stop_after(() -> chatrequest!(chat; config=_SLOW_CFG, cancel=tok), () -> cancel!(tok);
                           ready=() -> srv.hits[] == 1)
            @test o.finished && cancelled_by(o.result, :token)
            @test o.latency < 5.0                   # not aborted, it waits out the 20 s hold
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
            @test done_at - cancelled_at < 5.0      # not woken, it sleeps out the 30 s backoff
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
        srv = paced_sse_server([sse_text("a"), sse_text("b"), "data: [DONE]\n\n"]; gap=20.0)
        try
            chat = stream_chat(srv.url); handle = Ref{Any}(nothing); seen = Any[]
            o = stop_after(() -> chatrequest!(chat; config=_SLOW_CFG,
                                              callback=(c, close) -> (push!(seen, c); handle[] = close)),
                           () -> (handle[][] = true); ready=() -> handle[] !== nothing)
            @test o.finished && cancelled_by(o.result, :callback)
            @test o.latency < 5.0                   # seen only at the next chunk, 19 s later
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

@testset "driver — a stream that ends uncommitted adds the usage its provider reported, once" begin
    # A server may report usage before the text (an OpenAI-compatible one on every chunk): the request
    # was billed at least that much, so a call that ends uncommitted after it adds it to cumulative_cost,
    # once — whether stopped, cancelled, timed out, cut off or ended by user code. 100,000 prompt tokens
    # on gpt-6-luna, at $0.10 per 1M, are $0.01.
    usage = Dict("prompt_tokens" => 100_000, "completion_tokens" => 0, "total_tokens" => 100_000)
    chunk(d) = "data: " * JSON.json(merge(Dict{String,Any}("id" => "c", "model" => "gpt-6-luna"), d)) * "\n\n"
    billed = chunk(Dict("choices" => Any[], "usage" => usage))
    text(s; finish=nothing) =
        chunk(Dict("choices" => [Dict("index" => 0, "delta" => Dict("content" => s), "finish_reason" => finish)]))
    luna(url) = Chat(service=GenericOpenAIEndpoint(url, ""), model="gpt-6-luna", stream=true,
                     messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])
    cost = 100_000 * 0.10 / 1_000_000
    cfg = RequestConfig(request_timeout=10.0, stream_idle_timeout=1.0, total_deadline=30.0, max_attempts=1)
    boom = ErrorException("user code failed")
    for how in (:close, :token, :throw, :idle, :eof)
        srv = paced_sse_server([billed, text("hel")]; hold=how === :eof ? 0.0 : 5.0)
        try
            chat, tok = luna(srv.url), CancelToken()
            act(close) = how === :close ? (close[] = true) : how === :token ? cancel!(tok) :
                         how === :throw ? throw(boom) : nothing
            r = fetch(chatrequest!(chat; config=cfg, cancel=tok, callback=(c, close) -> c == "hel" && act(close)))
            @test how === :eof ? r isa LLMFailure && r.status == 200 : r isa LLMCallError
            how === :idle && @test r.cause isa UniLMTimeout && r.cause.phase === :stream_idle
            how in (:close, :token) && @test r.cause isa UniLMCancelled
            how === :throw && @test r.cause === boom
            @test length(chat.messages) == 2                         # nothing committed
            @test cumulative_cost(chat) ≈ cost
        finally
            HTTP.forceclose(srv.server)
        end
    end
    # Run to its end: the turn is committed and its usage added once.
    srv = paced_sse_server([billed, text("hel"), text("lo"; finish="stop"), "data: [DONE]\n\n"])
    stopped = paced_sse_server([text("hel"), billed, text("lo"; finish="stop"), "data: [DONE]\n\n"]; gap=1.0)
    failed = paced_sse_server([billed, text("hel"), chunk(Dict("error" => Dict("message" => "backend died",
                                                                              "code" => 500))) * "data: [DONE]\n\n"])
    try
        chat = luna(srv.url)
        r = fetch(chatrequest!(chat; config=cfg))
        @test r isa LLMSuccess && r.usage.prompt_tokens == 100_000 && length(chat.messages) == 3
        @test cumulative_cost(chat) ≈ cost
        # Stopped before any usage arrived: nothing is added, though the provider bills the request.
        chat = luna(stopped.url)
        r = fetch(chatrequest!(chat; config=cfg, callback=(c, close) -> c == "hel" && (close[] = true)))
        @test r isa LLMCallError && r.cause isa UniLMCancelled && cumulative_cost(chat) == 0.0
        # A stream the provider ended with an error event adds nothing: the reply is the provider's failure.
        chat = luna(failed.url)
        r = fetch(chatrequest!(chat; config=cfg))
        @test r isa LLMFailure && r.status == 500 && cumulative_cost(chat) == 0.0
    finally
        foreach(s -> HTTP.forceclose(s.server), (srv, stopped, failed))
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
    # Both tests read pooling off the peers, so they start from an empty pool. The client these
    # calls share keeps at most 64 idle connections in all and drops a dead one only when its own
    # host is asked for again, so the closed servers of earlier tests can fill it; a full pool
    # closes every finished connection rather than keeping it.
    HTTP.close_idle_connections!()

    @testset "a body held open after [DONE] holds neither the final callback nor the result" begin
        srv = peer_sse_server(chunks; hold=20.0)
        try
            chat = stream_chat(srv.url); final_at = Ref(Inf)
            t0 = time()
            r = fetch(chatrequest!(chat; config=_SLOW_CFG,
                                   callback=(c, _) -> c isa Message && (final_at[] = time() - t0)))
            elapsed = time() - t0
            @test r isa LLMSuccess && r.message.content == "hi" && length(chat.messages) == 3
            # ≈0.5 s expected; the forbidden outcome is the 20 s hold. A 4 s bound leaves
            # a runner-stall budget and still discriminates.
            @test final_at[] < 4.0
            @test elapsed < 4.0
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
