# ─── Gemini Integration Tests (live) ─────────────────────────────────────────
# Requires UNILM_LIVE=1 and GEMINI_API_KEY (billing-enabled). Uses gemini-3.8-flash — a
# thinking model: max_tokens budgets include thought tokens, so limits carry reasoning
# headroom (a tight budget yields an empty-text turn). Run once when green; do not rerun.

if !haskey(ENV, "GEMINI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping Gemini integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY to run live)"
else

@testset "Gemini Chat — basic" begin
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=1024)
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test result.usage.completion_tokens > 0
    @test result.model == "gemini-3.8-flash"            # modelVersion
    @test cumulative_cost(chat) > 0.0
end

@testset "Gemini Chat — tool round-trip" begin
    sig = FunctionSignature(name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict("type" => "object",
            "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
            "required" => ["location"]))
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=2048,
                tools=[Tool(func=sig)], tool_choice="required")
    push!(chat, Message(Val(:system), "Use the weather tool when asked about weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris?"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    m = result.message
    @test m.finish_reason == UniLM.TOOL_CALLS
    if m.finish_reason == UniLM.TOOL_CALLS
        @test m.tool_calls[1].func.name == "get_current_weather"
        @test haskey(m.tool_calls[1].func.arguments, "location")
        # feed the tool result back — exercises thoughtSignature echo on the next turn
        push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="72F and sunny"))
        follow_chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
            reasoning_effort="low", max_tokens=1024, messages=chat.messages, tools=chat.tools)
        follow = chatrequest!(follow_chat)
        @test follow isa LLMSuccess
        @test !isempty(something(follow.message.content, ""))
    end
end

@testset "Gemini Chat — streaming" begin
    payloads = Any[]
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low",
                max_tokens=1024, stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Count from 1 to 10, one number per line."))
    task = chatrequest!(chat; callback=(c, _) -> push!(payloads, c))
    result = fetch(task)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)   # handle_sse_event! accumulated real SSE
    @test !isempty(payloads)                 # callback fired (deltas and/or final message)
    @test result.model == "gemini-3.8-flash" # every chunk's modelVersion
    @test result.message.provider_content isa ProviderContent
    push!(chat, Message(Val(:user), "Reply with exactly: done"))
    follow = fetch(chatrequest!(chat))
    @test follow isa LLMSuccess
    @test occursin("done", lowercase(something(follow.message.content, "")))
end

# The cost of a reply priced by hand from its own usage, at the rates of a price row. A live run happens on any
# day and a scheduled price change applies from its date, so each test passes the row in effect:
# `UniLM._price_row(DEFAULT_PRICING, model)`.
_gemini_cost(u, row::UniLM.PriceRow) =
    (u.prompt_tokens - u.cached_tokens) * row.input + u.cached_tokens * row.cached_input + u.completion_tokens * row.output

@testset "Gemini 4 Argon — low effort; the reply names its model and is priced at its row" begin
    # The pricier model, for its own rules: low is its lowest level (minimal is an HTTP 400).
    r = chatrequest!(Chat(service=GEMINIServiceEndpoint, model="gemini-4-argon", reasoning_effort="low",
        max_tokens=256, messages=[Message(role=UniLM.RoleUser, content="Reply with exactly: OK")]))
    @test r isa LLMSuccess
    @info "gemini-4-argon" usage = r.usage cost = estimated_cost(r)
    @test r.model == "gemini-4-argon"
    # Argon names its billed tier in the x-gemini-service-tier header only (its usageMetadata
    # carried none on 2026-10-22).
    @test r.usage.service_tier == "standard"
    @test estimated_cost(r) > 0
    @test estimated_cost(r) ≈ _gemini_cost(r.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-4-argon"))
end

@testset "Gemini — the default model is gemini-4-hydrogen; it takes xhigh, and the reply is priced at its row" begin
    # A Chat with no model, at the level only Hydrogen's row lists (observed HTTP 200 on 2026-11-16); a tiny
    # prompt, and a budget with room for the thinking (73 thought tokens on the witness). Refuted by another
    # served model, an error, or an estimate other than the usage billed at gemini-4-hydrogen's row.
    r = chatrequest!(Chat(service=GEMINIServiceEndpoint, reasoning_effort="xhigh", max_tokens=1024,
        messages=[Message(role=UniLM.RoleUser, content="Reply with exactly: OK")]))
    @test r isa LLMSuccess
    @info "gemini-4-hydrogen xhigh" served = r.model usage = r.usage cost = estimated_cost(r)
    @test r.self.model == r.model == "gemini-4-hydrogen"
    @test estimated_cost(r) > 0
    @test estimated_cost(r) ≈ _gemini_cost(r.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-4-hydrogen"))
end

@testset "Gemini 4 Carbon — low effort; the reply names its model and is priced at its row" begin
    # Carbon's row bills $2 / $0.10 / $10 per 1M tokens through 2026-11-30 and twice that from 2026-12-01.
    r = chatrequest!(Chat(service=GEMINIServiceEndpoint, model="gemini-4-carbon", reasoning_effort="low",
        max_tokens=256, messages=[Message(role=UniLM.RoleUser, content="Reply with exactly: OK")]))
    @test r isa LLMSuccess
    @info "gemini-4-carbon" usage = r.usage cost = estimated_cost(r)
    @test r.model == "gemini-4-carbon"
    @test estimated_cost(r) > 0
    @test estimated_cost(r) ≈ _gemini_cost(r.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-4-carbon"))
end

@testset "Gemini OpenAI-compatible endpoint — the default model gemini-4-hydrogen answers, streamed and not" begin
    # Refuted by another served model, an error, or a reply priced other than at gemini-4-hydrogen's row.
    for stream in (false, true)
        chat = Chat(service=GEMINIOpenAIServiceEndpoint, reasoning_effort="low", stream=(stream ? true : nothing),
                    messages=[Message(Val(:user), "Reply with the single word OK.")])
        out = chatrequest!(chat)
        r = stream ? fetch(out) : out
        @test r isa LLMSuccess
        @info "Gemini compat default" stream served = r.model usage = r.usage cost = estimated_cost(r)
        @test chat.model == r.model == "gemini-4-hydrogen"
        # The reply counts its thinking only in total_tokens; the result bills all of it as output.
        @test r.usage.completion_tokens == r.usage.total_tokens - r.usage.prompt_tokens
        @test estimated_cost(r) > 0
        @test estimated_cost(r) ≈ _gemini_cost(r.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-4-hydrogen"))
    end
end

@testset "Gemini routes gemini-3.7-flash to gemini-3.8-flash, and the call is priced as served" begin
    r = chatrequest!(Chat(service=GEMINIServiceEndpoint, model="gemini-3.7-flash", reasoning_effort="low",
        max_tokens=256, messages=[Message(role=UniLM.RoleUser, content="Reply with exactly: OK")]))
    @test r isa LLMSuccess
    @info "gemini-3.7-flash" served = r.model usage = r.usage cost = estimated_cost(r)
    @test r.model == "gemini-3.8-flash"
    @test estimated_cost(r) ≈ _gemini_cost(r.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-3.8-flash"))
    @test estimated_cost(r) == estimated_cost(r; model="gemini-3.8-flash")
end

const _GEMINI_LIVE_MEDIA = joinpath(@__DIR__, "fixtures")

# One user message with one attachment on gemini-3.8-flash, at its lowest thinking level.
_gemini_ask(prompt, attachment) = chatrequest!(Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
    reasoning_effort="low", max_tokens=256, messages=[Message(Val(:user), prompt, attachment)]))

@testset "Gemini Chat — an image, a PDF and a sound clip are read" begin
    # Refuted by: an answer without the circle's colour, the document's secret word or the clip's.
    circle = joinpath(_GEMINI_LIVE_MEDIA, "ollama", "red_circle.png")
    img = _gemini_ask("What colour is the circle? Answer with one word.", ImageAttachment(circle))
    @test img isa LLMSuccess && occursin(r"\bred\b"i, img.message.content)
    pdf = _gemini_ask("What is the secret word in this document? Answer with the word only.",
                      DocumentAttachment(joinpath(_GEMINI_LIVE_MEDIA, "documents", "secret_word.pdf")))
    @test pdf isa LLMSuccess && occursin("PELICAN", uppercase(pdf.message.content))
    # gemini-3.8-flash's one input price names no modality, so the clip bills at it, as the text does.
    wav = _gemini_ask("What is the secret word in this recording? Answer with the word only.",
                      AudioAttachment(joinpath(_GEMINI_LIVE_MEDIA, "ollama", "secret_word.wav")))
    @test wav isa LLMSuccess && occursin("banana", lowercase(wav.message.content))
    @info "Gemini audio" usage = wav.usage cost = estimated_cost(wav)
    @test wav.usage.prompt_audio_tokens > 0 && wav.usage.cached_tokens == 0
    @test estimated_cost(wav) ≈ _gemini_cost(wav.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-3.8-flash")) &&
          cumulative_cost(wav.self) == estimated_cost(wav)
    # detail="low" is the LOW media resolution: the image costs 256 prompt tokens, not ~1,089.
    low = _gemini_ask("What colour is the circle? Answer with one word.", ImageAttachment(circle; detail="low"))
    @info "Gemini media" image = img.usage low_detail = low.usage
    @test low isa LLMSuccess && low.usage.prompt_tokens < 400 < img.usage.prompt_tokens
end

@testset "Gemini Chat — service_tier priority is the billed tier, priced at 1.8 times" begin
    for stream in (false, true)
        chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=64,
                    service_tier="priority", stream=stream, messages=[Message(Val(:user), "Reply with the single word OK.")])
        out = chatrequest!(chat)
        r = stream ? fetch(out) : out
        @test r isa LLMSuccess
        @info "Gemini priority" stream usage = r.usage cost = estimated_cost(r)
        @test r.usage.service_tier == "priority"
        @test estimated_cost(r) ≈ 1.8 * _gemini_cost(r.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-3.8-flash"))
    end
end

@testset "Gemini Chat — a sound clip on 3.1 Flash-Lite bills at the page's audio rate, streamed or not" begin
    # gemini-3.1-flash-lite, per 1M tokens: $0.25 text, $0.50 audio input, $1.50 output (pricing page,
    # 2026-10-22). Refuted by: no audio tokens reported, or an estimate other than that bill of the usage.
    for stream in (false, true)
        chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite", max_tokens=64, stream=stream,
            messages=[Message(Val(:user), "What is the secret word in this recording? Answer with the word only.",
                              AudioAttachment(joinpath(_GEMINI_LIVE_MEDIA, "ollama", "secret_word.wav")))])
        out = chatrequest!(chat)
        r = stream ? fetch(out) : out
        @test r isa LLMSuccess
        u = r.usage
        @info "Gemini audio input" stream usage = u cost = estimated_cost(r) answer = r.message.content
        @test r.model == "gemini-3.1-flash-lite"
        @test u.prompt_audio_tokens > 0 && u.cached_tokens == 0 && u.completion_audio_tokens == 0
        bill = (u.prompt_audio_tokens * 0.50 + (u.prompt_tokens - u.prompt_audio_tokens) * 0.25 + u.completion_tokens * 1.50) / 1e6
        @test isapprox(estimated_cost(r), bill; rtol=1e-12)
        @test cumulative_cost(chat) == estimated_cost(r)
    end
end

@testset "Gemini OpenAI-compatible endpoint — a priority reply names no tier; the result records the requested one" begin
    # Observed 2026-10-22: replies to flex and priority requests on this endpoint name their tier in
    # neither the body (service_tier) nor the head (x-gemini-service-tier), streamed or not, so a Chat
    # result records the tier its request asked for, priced at Priority: 1.8 times the row in effect,
    # an upper bound (Google bills a downgraded Priority request at Standard).
    # Refuted by a reply that names a tier, or a result that records another tier or price.
    svc = GEMINIOpenAIServiceEndpoint
    for stream in (false, true)
        chat = Chat(service=svc, model="gemini-3.8-flash", reasoning_effort="low", max_completion_tokens=64,
                    stream=(stream ? true : nothing), stream_options=(stream ? Dict("include_usage" => true) : nothing),
                    service_tier="priority", messages=[Message(Val(:user), "Reply with the single word OK.")])
        resp = UniLM._http("POST", UniLM.get_url(chat), UniLM.auth_header(svc), UniLM.encode_request(svc, chat))
        @test resp.status == 200
        text = String(copy(resp.body))
        payloads = stream ? [strip(l[6:end]) for l in split(text, '\n') if startswith(l, "data:") && !occursin("[DONE]", l)] : [text]
        replies = [JSON.parse(p) for p in payloads]
        head = HTTP.header(resp, "x-gemini-service-tier", "")
        @info "Gemini compat priority" stream header = head body_tiers = [get(d, "service_tier", nothing) for d in replies]
        @test isempty(head) && !any(d -> haskey(d, "service_tier"), replies)
        out = chatrequest!(chat)
        r = stream ? fetch(out) : out
        @test r isa LLMSuccess
        @info "Gemini compat priority result" stream usage = r.usage cost = estimated_cost(r)
        @test r.usage.prompt_tokens > 0 && r.usage.service_tier == "priority"
        @test isapprox(estimated_cost(r), 1.8 * _gemini_cost(r.usage, UniLM._price_row(DEFAULT_PRICING, "gemini-3.8-flash"));
                       rtol=1e-12)
        @test cumulative_cost(chat) == estimated_cost(r)
    end
end

@testset "Gemini Chat — seed and store are accepted" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=64,
                seed=7, store=false, messages=[Message(Val(:user), "Reply with the single word OK.")])
    @test chatrequest!(chat) isa LLMSuccess
end

@testset "Gemini Files — resumable upload, poll, get, list, delete; the download refused, a registration answered" begin
    # Free: the Files API bills no tokens, and stores up to 20 GB a project. A 22-byte text file, deleted at the end.
    path = joinpath(mktempdir(), "unilm-live.txt")
    write(path, "UniLM live Files test\n")
    g = GEMINIServiceEndpoint
    up = upload_file(path; service=g)
    @test up isa FileSuccess
    if up isa FileSuccess
        f = up.response
        try
            @test (f.filename, f.bytes, f.purpose) == ("unilm-live.txt", 22, nothing)
            @test f.status in ("PROCESSING", "ACTIVE") && abs(f.created_at - time()) < 600
            @test f.raw["mimeType"] == "text/plain"
            @test f.raw["uri"] == "https://generativelanguage.googleapis.com/v1beta/files/$(f.id)"
            ready = poll_file(f.id; service=g, interval=1.0, timeout=120.0)
            @test ready isa FileSuccess && ready.response.status == "ACTIVE"
            got = retrieve_file("files/" * f.id; service=g)
            @test got isa FileSuccess && got.response.id == f.id
            seen, after = false, nothing
            for _ in 1:5   # at most 500 files
                page = list_files(; limit=100, after, service=g)
                @test page isa FileListSuccess
                page isa FileListSuccess || break
                seen |= any(x -> x.id == f.id, page.response.data)
                (seen || !page.response.has_more) && break
                after = page.response.raw["nextPageToken"]
            end
            @test seen
            @test_throws ArgumentError file_content(f.id; service=g)
        finally
            del = delete_file(f.id; service=g)
            @test del isa FileDeleteSuccess && del.id == f.id
        end
    end
    # A bucket that does not exist: Gemini refuses the registration (HTTP 404 on 2026-11-16), free.
    reg = register_files(["gs://unilm-no-such-bucket-20261116/w.txt"])
    @test reg isa FileFailure && reg.status in (400, 403, 404)
end

# countTokens and the Models API's lookup and listing bill no tokens.
@testset "Gemini count_tokens — a Chat's generateContent request, system instruction and tools counted" begin
    sig = FunctionSignature(name="get_weather", description="Weather", parameters=Dict("type" => "object",
                            "properties" => Dict("city" => Dict("type" => "string"))))
    plain = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", messages=[Message(Val(:user), "Hello")])
    full = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=16,
                tools=[Tool(func=sig)], messages=[Message(Val(:system), "Be brief."), Message(Val(:user), "Hello")])
    p, f = count_tokens(plain), count_tokens(full)
    @test p isa TokenCountSuccess && f isa TokenCountSuccess
    @test f.input_tokens > p.input_tokens > 0
    @test f.raw["promptTokensDetails"][1]["modality"] == "TEXT"
end

@testset "Gemini retrieve_model — one model; an unknown one is a 404; the listing is one page" begin
    r = retrieve_model("gemini-3.8-flash"; service=GEMINIServiceEndpoint)
    @test r isa ModelSuccess && r.model.id == "gemini-3.8-flash" && r.model.input_token_limit > 0
    @test "countTokens" in r.model.capabilities["supportedGenerationMethods"]
    r = retrieve_model("gemini-made-up-0"; service=GEMINIServiceEndpoint)
    @test r isa ModelListFailure && r.status == 404
    listed = list_models(; service=GEMINIServiceEndpoint)
    @test listed isa ModelListSuccess && length(listed.raw) == 1   # one page of up to 1000 models
end

@testset "Gemini batches — two requests: create, retrieve, list, update, cancel, results and delete" begin
    # Two gemini-3.8-flash requests at thinking "low", cancelled at once (a cancelled batch was CANCELLED on the
    # next read in the runs of 2026-11-16, and ended without results); the gate does not wait for a batch to run.
    q() = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=64,
               messages=[Message(Val(:user), "Reply with exactly: ok")])
    created = create_batch(["one" => q(), "two" => q()])
    @test created isa BatchSuccess
    if created isa BatchSuccess
        id = created.response.id
        @test created.response.status in ("BATCH_STATE_PENDING", "BATCH_STATE_RUNNING")
        @test created.response.request_counts["requestCount"] == "2"
        @test retrieve_batch(id; service=GEMINIServiceEndpoint) isa BatchSuccess
        page = list_batches(; limit=10, service=GEMINIServiceEndpoint)
        @test page isa BatchListSuccess && any(b -> b.id == id, page.response.data)
        up = update_batch(id; priority=5, service=GEMINIServiceEndpoint)
        @test up isa BatchSuccess && up.response.raw["priority"] == "5"
        stopped = cancel_batch(id; service=GEMINIServiceEndpoint)
        @test stopped isa BatchSuccess && stopped.response.status == "BATCH_STATE_CANCELLED"
        r = batch_results(id; service=GEMINIServiceEndpoint)
        @test r isa BatchCallError && occursin("ended without results", r.error)
        d = delete_batch(id; service=GEMINIServiceEndpoint)
        @test d isa BatchDeleteSuccess && d.id == id
        @test retrieve_batch(id; service=GEMINIServiceEndpoint) isa BatchFailure     # gone: HTTP 404
    end
end

@testset "Gemini batches — two requests run to their end: polled, batch-priced successes, each on its request's Chat" begin
    # Two gemini-3.8-flash requests at thinking "low", left to run: a batch of 2026-11-16 succeeded in under five
    # minutes, and the poll gives it ten. Each success bills at the Batch rates, half the standard ones; read with
    # the requests, each result's self is its request's Chat with the reply appended.
    q() = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=64,
               messages=[Message(Val(:user), "Reply with exactly: ok")])
    requests = ["one" => q(), "two" => q()]
    created = create_batch(requests)
    @test created isa BatchSuccess
    if created isa BatchSuccess
        id = created.response.id
        seconds = @elapsed (ended = poll_batch(id; service=GEMINIServiceEndpoint, interval=15.0, timeout=600.0))
        done = ended isa BatchSuccess && ended.response.status == "BATCH_STATE_SUCCEEDED"
        @test done
        if done
            r = batch_results(id; service=GEMINIServiceEndpoint, requests)
            @test r isa BatchResultsSuccess && first.(r.results) == ["one", "two"]   # inline: the requests' order
            for (cid, x) in (r isa BatchResultsSuccess ? r.results : [])
                @test x isa LLMSuccess
                x isa LLMSuccess || continue
                standard = LLMSuccess(message=x.message, self=x.self, model=x.model, usage=UniLM._standard_tier(x.usage))
                @test x.usage.service_tier == "batch" && 0 < estimated_cost(x) < estimated_cost(standard)
                chat = Dict(requests)[cid]
                @test x.self !== chat && length(x.self.messages) == length(chat.messages) + 1 && last(x.self.messages) === x.message
            end
            r isa BatchResultsSuccess && @info "Gemini batch run to its end" id seconds cost = sum(
                (estimated_cost(x) for (_, x) in r.results if x isa LLMSuccess); init=0.0)
            @test delete_batch(id; service=GEMINIServiceEndpoint) isa BatchDeleteSuccess
        else
            cancel_batch(id; service=GEMINIServiceEndpoint)   # the poll gave up: stop the batch
        end
    end
end

end  # if GEMINI_API_KEY
