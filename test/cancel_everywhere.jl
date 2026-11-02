# test/cancel_everywhere.jl
# The `cancel` keyword of every platform, lifecycle and audio verb: a token cancelled before the
# call sends nothing and the call returns its family's call error; an explicit token wins over the
# ambient one of `with_cancel`, and an explicit `nothing` is the ambient one; a cancel while the
# server holds the reply ends the call within 2 s on each seam; `realtime_connect` keeps its
# exception. Local servers only.

# The Realtime WebSocket that `speak` and `translate` open, at a local address set per test.
struct _CERealtime <: UniLM.ServiceEndpoint end
const _ce_ws_base = Ref("")
UniLM._realtime_ws_url(::Type{_CERealtime}) = _ce_ws_base[]
UniLM.auth_header(::Type{_CERealtime}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_CERealtime}) = Set([:realtime, :speech])

# Gemini TTS at the scripted endpoint.
struct _CEGeminiSpeech <: UniLM.ServiceEndpoint end
UniLM._speech_wire(::Type{_CEGeminiSpeech}) = UniLM._GeminiSpeechWire()
UniLM._gemini_speech_base(::Type{_CEGeminiSpeech}) = _url_probe_base[]
UniLM.auth_header(::Type{_CEGeminiSpeech}) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]

const _CE_WAV = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")   # 16-bit PCM

# Every verb that sends a request from the platform, lifecycle and audio files.
const _CE_VERBS = (
    upload_file, list_files, retrieve_file, delete_file, file_content,
    create_upload, add_upload_part, complete_upload, cancel_upload,
    create_vector_store, retrieve_vector_store, list_vector_stores, delete_vector_store,
    add_vector_store_file, create_file_batch, retrieve_file_batch,
    create_conversation, retrieve_conversation, update_conversation, delete_conversation,
    add_conversation_items, list_conversation_items, retrieve_conversation_item, delete_conversation_item,
    create_container, retrieve_container, list_containers, delete_container, add_container_file,
    create_batch, retrieve_batch, cancel_batch, list_batches,
    create_fine_tuning_job, retrieve_fine_tuning_job, cancel_fine_tuning_job,
    list_fine_tuning_jobs, list_fine_tuning_events, list_fine_tuning_checkpoints,
    moderate, speak, transcribe, translate, mint_realtime_secret,
    get_response, delete_response, list_input_items, cancel_response, compact_response, count_input_tokens)

# Each method of those verbs, and each wire a verb speaks, as (label, the family's call error,
# kw -> the call with keywords `kw`). Every call is valid, so only the token decides whether it is
# sent. The endpoints are built when called: the scripted endpoint's address is known once it listens.
function _ce_rows(file::String)
    s = URLProbe
    ep() = GenericOpenAIEndpoint(_url_probe_base[], "")
    ollama() = OllamaEndpoint(base_url=_url_probe_base[])
    item = [Dict("type" => "message", "role" => "user", "content" => "hi")]
    Tuple{String,Type,Function}[
        ("upload_file(::FileUpload)", FileCallError,
         kw -> upload_file(FileUpload(file=file, purpose="batch", service=s); kw...)),
        ("upload_file(path, purpose)", FileCallError, kw -> upload_file(file, "batch"; service=s, kw...)),
        ("list_files", FileCallError, kw -> list_files(; service=s, kw...)),
        ("retrieve_file", FileCallError, kw -> retrieve_file("file_1"; service=s, kw...)),
        ("delete_file", FileCallError, kw -> delete_file("file_1"; service=s, kw...)),
        ("file_content", FileCallError, kw -> file_content("file_1"; service=s, kw...)),
        ("create_upload", UploadCallError,
         kw -> create_upload(; filename="a.jsonl", purpose="batch", bytes=3, mime_type="application/jsonl",
                             service=s, kw...)),
        ("add_upload_part", UploadCallError, kw -> add_upload_part("upload_1", UInt8[1, 2]; service=s, kw...)),
        ("complete_upload", UploadCallError, kw -> complete_upload("upload_1", ["part_1"]; service=s, kw...)),
        ("cancel_upload", UploadCallError, kw -> cancel_upload("upload_1"; service=s, kw...)),
        ("create_vector_store", VectorStoreCallError, kw -> create_vector_store(; name="vs", service=s, kw...)),
        ("retrieve_vector_store", VectorStoreCallError, kw -> retrieve_vector_store("vs_1"; service=s, kw...)),
        ("list_vector_stores", VectorStoreCallError, kw -> list_vector_stores(; service=s, kw...)),
        ("delete_vector_store", VectorStoreCallError, kw -> delete_vector_store("vs_1"; service=s, kw...)),
        ("add_vector_store_file", VectorStoreCallError,
         kw -> add_vector_store_file("vs_1", "file_1"; service=s, kw...)),
        ("create_file_batch", VectorStoreCallError, kw -> create_file_batch("vs_1", ["file_1"]; service=s, kw...)),
        ("retrieve_file_batch", VectorStoreCallError,
         kw -> retrieve_file_batch("vs_1", "vsfb_1"; service=s, kw...)),
        ("create_conversation", ConversationCallError, kw -> create_conversation(; service=s, kw...)),
        ("retrieve_conversation", ConversationCallError, kw -> retrieve_conversation("conv_1"; service=s, kw...)),
        ("update_conversation", ConversationCallError,
         kw -> update_conversation("conv_1", Dict("k" => "v"); service=s, kw...)),
        ("delete_conversation", ConversationCallError, kw -> delete_conversation("conv_1"; service=s, kw...)),
        ("add_conversation_items", ConversationCallError,
         kw -> add_conversation_items("conv_1", item; service=s, kw...)),
        ("list_conversation_items", ConversationCallError,
         kw -> list_conversation_items("conv_1"; service=s, kw...)),
        ("retrieve_conversation_item", ConversationCallError,
         kw -> retrieve_conversation_item("conv_1", "item_1"; service=s, kw...)),
        ("delete_conversation_item", ConversationCallError,
         kw -> delete_conversation_item("conv_1", "item_1"; service=s, kw...)),
        ("create_container", ContainerCallError, kw -> create_container(; name="c", service=s, kw...)),
        ("retrieve_container", ContainerCallError, kw -> retrieve_container("cntr_1"; service=s, kw...)),
        ("list_containers", ContainerCallError, kw -> list_containers(; service=s, kw...)),
        ("delete_container", ContainerCallError, kw -> delete_container("cntr_1"; service=s, kw...)),
        ("add_container_file", ContainerCallError, kw -> add_container_file("cntr_1", file; service=s, kw...)),
        ("create_batch", BatchCallError, kw -> create_batch("file_1", "/v1/responses"; service=s, kw...)),
        ("retrieve_batch", BatchCallError, kw -> retrieve_batch("batch_1"; service=s, kw...)),
        ("cancel_batch", BatchCallError, kw -> cancel_batch("batch_1"; service=s, kw...)),
        ("list_batches", BatchCallError, kw -> list_batches(; service=s, kw...)),
        ("create_fine_tuning_job", FineTuningCallError,
         kw -> create_fine_tuning_job(; model="m", training_file="file_1", service=s, kw...)),
        ("retrieve_fine_tuning_job", FineTuningCallError,
         kw -> retrieve_fine_tuning_job("ftjob_1"; service=s, kw...)),
        ("cancel_fine_tuning_job", FineTuningCallError, kw -> cancel_fine_tuning_job("ftjob_1"; service=s, kw...)),
        ("list_fine_tuning_jobs", FineTuningCallError, kw -> list_fine_tuning_jobs(; service=s, kw...)),
        ("list_fine_tuning_events", FineTuningCallError,
         kw -> list_fine_tuning_events("ftjob_1"; service=s, kw...)),
        ("list_fine_tuning_checkpoints", FineTuningCallError,
         kw -> list_fine_tuning_checkpoints("ftjob_1"; service=s, kw...)),
        ("moderate", ModerationCallError, kw -> moderate("hi"; service=s, kw...)),
        ("mint_realtime_secret", RealtimeCallError, kw -> mint_realtime_secret(; service=s, kw...)),
        ("speak(::SpeechRequest), Realtime", AudioCallError,
         kw -> speak(SpeechRequest(input="Hi.", service=_CERealtime); kw...)),
        ("speak(input), Realtime", AudioCallError, kw -> speak("Hi."; service=_CERealtime, kw...)),
        ("speak(input), Gemini TTS", AudioCallError, kw -> speak("Hi."; service=_CEGeminiSpeech, kw...)),
        ("speak(input), Ollama", AudioCallError, kw -> speak("Hi."; service=ollama(), model="phonon", kw...)),
        ("transcribe(::TranscriptionRequest)", AudioCallError,
         kw -> transcribe(TranscriptionRequest(file=_CE_WAV, service=s); kw...)),
        ("transcribe(path)", AudioCallError, kw -> transcribe(_CE_WAV; service=s, kw...)),
        ("translate, Realtime", AudioCallError, kw -> translate(_CE_WAV; service=_CERealtime, kw...)),
        ("translate, Ollama", AudioCallError,
         kw -> translate(_CE_WAV; service=ollama(), model="gemma4:e4b", kw...)),
        ("get_response", ResponseCallError, kw -> get_response("resp_1"; service=ep(), kw...)),
        ("delete_response", ResponseCallError, kw -> delete_response("resp_1"; service=ep(), kw...)),
        ("list_input_items", ResponseCallError, kw -> list_input_items("resp_1"; service=ep(), kw...)),
        ("cancel_response", ResponseCallError, kw -> cancel_response("resp_1"; service=ep(), kw...)),
        ("compact_response", ResponseCallError, kw -> compact_response(; model="m", input="hi", service=ep(), kw...)),
        ("count_input_tokens", ResponseCallError,
         kw -> count_input_tokens(; model="m", input="hi", service=ep(), kw...)),
    ]
end

# A token's outcome, as the image verbs return it: the family's call error, no HTTP status, and the
# cancellation as its cause.
_ce_cancelled(r, ::Type{T}) where {T} =
    r isa T && isnothing(r.status) && r.cause isa UniLMCancelled && r.cause.source === :token

@testset "every method of every platform, lifecycle and audio verb takes `cancel` as its last keyword" begin
    @test length(_CE_VERBS) == 50
    for f in _CE_VERBS, m in methods(f)
        named = filter(k -> !endswith(String(k), "..."), Base.kwarg_decl(m))
        @test (f, last(named)) == (f, :cancel)
    end
end

@testset "cancel= on every platform, lifecycle and audio verb: a cancelled token sends nothing" begin
    file = tempname() * ".jsonl"
    write(file, "{}\n")
    hits = Threads.Atomic{Int}(0)
    fired = cancel!(CancelToken())
    # The result of `call()`, or what it threw, and the requests it sent.
    sent(call) = (before = hits[]; r = try call() catch e; e end; (r, hits[] - before))
    try
        results, _ = _with_scripted((_, _) -> (Threads.atomic_add!(hits, 1); _json(200, "{}"))) do
            _ce_ws_base[] = replace(_url_probe_base[], r"^http" => "ws")
            [(label, T, (
                 "a cancelled token" => sent(() -> call((; cancel=fired))),
                 # `cancel=nothing` is the ambient token, as no keyword is.
                 "cancel=nothing in a cancelled scope" => sent(() -> with_cancel(() -> call((; cancel=nothing)), fired)),
                 "no keyword in a cancelled scope" => sent(() -> with_cancel(() -> call((;)), fired)),
                 # A live token passed to the call wins over a cancelled ambient one.
                 "a live token in a cancelled scope" =>
                     sent(() -> with_cancel(() -> call((; cancel=CancelToken())), fired))))
             for (label, T, call) in _ce_rows(file)]
        end
        @test length(results) == 56
        for (label, T, modes) in results, (mode, (r, n)) in modes
            @testset "$label, $mode" begin
                if startswith(mode, "a live token")
                    @test !_ce_cancelled(r, T)
                    @test n >= 1        # the request goes out
                else
                    @test _ce_cancelled(r, T)
                    @test n == 0        # nothing sent
                end
            end
        end
    finally
        rm(file; force=true)
    end
end

# Seconds from `cancel!` to the call's result, per seam, for the harness that runs these tests under
# coverage and CPU contention to read.
const _CE_LATENCY = Dict{String,Float64}()

# Run `call(tok)` on its own task while the scripted endpoint holds each reply up to 20 s, and cancel
# `tok` 0.3 s after the request arrived. Returns the result (or what the call threw), the seconds from
# the cancel to the result, and the requests the endpoint saw.
function _ce_cancel_held(call::Function)
    arrived, release = Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false)
    hold = (_, _) -> (arrived[] = true; timedwait(() -> release[], 20.0); _json(200, "{}"))
    (r, after), seen = _with_scripted(hold) do
        tok = CancelToken()
        t = Threads.@spawn (try call(tok) catch e; e end, time_ns())
        timedwait(() -> arrived[] || istaskdone(t), 25.0)
        sleep(0.3)
        cancelled_at = time_ns()
        cancel!(tok)
        finished = timedwait(() -> istaskdone(t), 25.0) === :ok
        release[] = true
        finished || return (nothing, Inf)
        res, done_at = fetch(t)
        (res, (Int64(done_at) - Int64(cancelled_at)) / 1e9)
    end
    (; r, after, requests=length(seen))
end

# Run `call(tok)` against a local Realtime WebSocket that opens the session and then answers no client
# event, and cancel `tok` 0.3 s after the first client event arrived. Returns the result, the seconds
# from the cancel to it, and the connections the server took.
function _ce_cancel_in_session(call::Function)
    opened, heard = Threads.Atomic{Int}(0), Threads.Atomic{Bool}(false)
    srv = HTTP.WebSockets.listen!("127.0.0.1", 0) do ws
        Threads.atomic_add!(opened, 1)
        try
            HTTP.WebSockets.send(ws, JSON.json(Dict("type" => "session.created")))
            for _ in ws
                heard[] = true
            end
        catch e
            e isa HTTP.WebSockets.WebSocketError || e isa Base.IOError || rethrow()
        end
    end
    _ce_ws_base[] = "ws://" * HTTP.WebSockets.server_addr(srv)
    try
        tok = CancelToken()
        t = Threads.@spawn (try call(tok) catch e; e end, time_ns())
        timedwait(() -> heard[] || istaskdone(t), 25.0)
        sleep(0.3)
        cancelled_at = time_ns()
        cancel!(tok)
        finished = timedwait(() -> istaskdone(t), 25.0) === :ok
        finished || return (; r=nothing, after=Inf, connections=opened[])
        r, done_at = fetch(t)
        (; r, after=(Int64(done_at) - Int64(cancelled_at)) / 1e9, connections=opened[])
    finally
        close(srv)
    end
end

@testset "cancel=: a cancel while the server holds the reply ends the call within 2 s, on every seam" begin
    cfg = RequestConfig(max_attempts=3, total_deadline=60.0)
    file = tempname() * ".jsonl"
    write(file, "{}\n")
    try
        for (seam, T, call) in (
                ("_http", BatchCallError, tok -> cancel_batch("batch_1"; service=URLProbe, config=cfg, cancel=tok)),
                ("_http_with_retries", ModerationCallError,
                 tok -> moderate("hi"; service=URLProbe, config=cfg, cancel=tok)),
                ("multipart, _http", FileCallError,
                 tok -> upload_file(file, "batch"; service=URLProbe, config=cfg, cancel=tok)),
                ("multipart, _http_with_retries", AudioCallError,
                 tok -> transcribe(_CE_WAV; service=URLProbe, config=cfg, cancel=tok)),
                ("_lifecycle_request", ResponseCallError,
                 tok -> get_response("resp_1"; service=GenericOpenAIEndpoint(_url_probe_base[], ""), config=cfg,
                                     cancel=tok)))
            @testset "$seam" begin
                o = _ce_cancel_held(call)
                _CE_LATENCY[seam] = o.after
                @test _ce_cancelled(o.r, T)
                @test o.after <= 2.0
                @test o.requests == 1   # sent once, never retried
            end
        end
    finally
        rm(file; force=true)
    end
    @testset "Realtime WebSocket" begin
        session = RequestConfig(connect_timeout=10.0, stream_idle_timeout=Inf, total_deadline=Inf)
        o = _ce_cancel_in_session(tok -> speak("Hi."; service=_CERealtime, config=session, cancel=tok))
        _CE_LATENCY["Realtime WebSocket"] = o.after
        @test _ce_cancelled(o.r, AudioCallError)
        @test o.after <= 2.0
        @test o.connections == 1
    end
end

@testset "realtime_connect keeps its exception: no `cancel` keyword, and its session ignores a cancelled token" begin
    @test_throws MethodError realtime_connect(_ -> nothing; service=_CERealtime, cancel=CancelToken())
    srv = HTTP.WebSockets.listen!("127.0.0.1", 0) do ws
        try
            HTTP.WebSockets.send(ws, JSON.json(Dict("type" => "session.created")))
            for _ in ws end
        catch e
            e isa HTTP.WebSockets.WebSocketError || e isa Base.IOError || rethrow()
        end
    end
    _ce_ws_base[] = "ws://" * HTTP.WebSockets.server_addr(srv)
    try
        ev = with_cancel(() -> realtime_connect(realtime_receive; service=_CERealtime), cancel!(CancelToken()))
        @test ev["type"] == "session.created"
    finally
        close(srv)
    end
end
