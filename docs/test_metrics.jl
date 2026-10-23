using Test
isdefined(Main, :DocMetrics) || include(joinpath(@__DIR__, "metrics.jl"))
using .DocMetrics, .ProviderFacts

function fixture_page(text::String)::String
    path, io = mktemp()
    write(io, text); close(io)
    path
end
function fixture_dir(files::Pair{String,String}...)::String
    dir = mktempdir()
    for (name, text) in files
        mkpath(dirname(joinpath(dir, name)))
        write(joinpath(dir, name), text)
    end
    dir
end
const fixture_docsrc = fixture_dir
const LEDGER_HEADER = "atom_id\tsource\tdestination\twitness"
fixture_tsv(rows::Vector{String})::String = fixture_page(join([LEDGER_HEADER; rows], '\n') * "\n")
remake(e; changes...) = ProviderEntry(; (f => get(changes, f, getfield(e, f)) for f in fieldnames(ProviderEntry))...)
caught(f) = try f(); nothing catch err; err end

@testset "metrics — prose lines skip fences, indented ones too, and tables" begin
    text = "# T\n\nOpenAI here.\n\n```julia\nAnthropic\n```\n\n| Anthropic |\n|---|\n\n!!! tip\n    Gemini is fine.\n    ```julia\n    x\n    ```\nplain\n"
    @test [l for (_, l) in prose_lines(text)] == ["# T", "OpenAI here.", "!!! tip", "    Gemini is fine.", "plain"]
end
@testset "metrics — a provider's own entries and model ids are never other-provider" begin
    for (line, reader) in ["OPENAIServiceEndpoint answers with gpt-6-luna." => :openai_decisions,
                           "OllamaEndpoint() runs tev1:4b." => :ollama_decisions,
                           "`GenericOpenAIEndpoint(\"http://localhost:8000\", \"\")` works." => :local_servers,
                           "Pull embeddinggemma-2:740m-q8_0." => :ollama,
                           "GEMINIOpenAIServiceEndpoint serves embeddings." => :gemini,
                           "Any OpenAI-compatible server works." => :anthropic,
                           "Azure OpenAI deploys gpt-5.2." => :azure,
                           "Local decision models choose." => :jev,
                           "See [streaming](@ref anthropic_streaming)." => :openai]
        @test share([fixture_page(line)], reader) == 0
    end
    for (line, reader) in ["Claude answers." => :openai, "Use GEMINIServiceEndpoint here." => :anthropic,
                           "Set AZURE_OPENAI_API_KEY." => :openai, "model=\"gpt-5.4-mini\"" => :anthropic]
        @test share([fixture_page(line * "\nNeutral line.\n")], reader) ≈ 1 / 2
    end
end
@testset "metrics — a concept page names no endpoint and no model after its binding line" begin
    page = fixture_page("```julia\nservice, model = OPENAIServiceEndpoint, \"gpt-5.4-mini\"\n```\n" *
                        "```julia\nChat(service=GEMINIServiceEndpoint)\nChat(service=service, model=\"claude-haiku-5-5\")\n```\n")
    @test length(concept_code_violations(page)) == 2
end
@testset "metrics — the ledger: each expected atom once, its witness inside its destination section" begin
    docsrc = fixture_docsrc("x.md" => "# [X](@id x_start)\n\n## [S](@id x_s)\n\nthe witness\ntext\n")
    ok = fixture_tsv(["anthropic:1\told.md:1\tx.md#x_s\tthe witness text", "anthropic:2\told.md:9\tdeleted\tduplicate of anthropic:1"])
    @test check_ledger(ok, docsrc, ["anthropic:1", "anthropic:2"]) == String[]
    bad = fixture_tsv(["anthropic:1\told.md:1\tx.md#x_s\tmissing words", "anthropic:1\told.md:2\tx.md#x_s\tthe witness text"])
    @test length(check_ledger(bad, docsrc, ["anthropic:1", "anthropic:2"])) == 3   # absent witness, twice, one missing
end
@testset "metrics — tokens match case-sensitively at identifier boundaries, longest first, each span once" begin
    @test provider_tokens("Azure OpenAI deploys gpt-5.2.") == [(Set([:azure]), "Azure OpenAI"), (Set([:openai, :azure]), "gpt-")]
    @test provider_tokens("OpenAIWireEndpoint wraps OPENAIServiceEndpoint; o3-mini, io3 and gpt-4o-mini.") ==
          [(Set([:openai]), "OPENAIServiceEndpoint"), (Set([:openai, :azure]), "o3"), (Set([:openai, :azure]), "gpt-")]
    for line in ["Every result is typesafe under an azure sky.",                       # the names, not the words
                 "The Jevons paradox: cheaper calls, more calls.",                     # a word that starts with a name
                 "Read [the guide](https://ai.google.dev/gemini-api/docs/thinking)."]  # a link's destination
        @test share([fixture_page(line)], :openai) == 0
    end
    @test share([fixture_page("Claude and Gemini answer.\nNeutral line.\n")], :openai) ≈ 1 / 2   # a line counts once
    @test share([fixture_page("Claude answers.\nNeutral line.\n")], nothing) ≈ 1 / 2             # a concept page: any provider
    @test share([fixture_page("OpenAI answers.\n")], nothing) == 1
    @test caught(() -> share([fixture_page("```julia\nx\n```\n")], :openai)) isa ArgumentError    # no prose line, no share
end
@testset "metrics — GPT- names and CamelCase identifiers count for their provider; a lowercase run never does" begin
    page(line) = fixture_page(line * "\nNeutral line.\n")
    @test share([page("GPT-6 Sol answers.")], :anthropic) ≈ 1 / 2
    @test share([page("Azure OpenAI deploys GPT-5.2.")], :azure) == 0                # Azure's own model names
    @test share([page("Azure OpenAI deploys GPT-5.2.")], :anthropic) ≈ 1 / 2
    @test share([page("OllamaOptions sets num_ctx.")], :ollama) == 0
    @test share([page("OllamaOptions sets num_ctx.")], :openai) ≈ 1 / 2
    @test share([page("TypeSafeProtocol is the wire.")], :jev) == 0
    @test share([page("TypeSafeProtocol is the wire.")], :openai) ≈ 1 / 2
    @test all(r -> share([page("The Jevons paradox: cheaper calls, more calls.")], r) == 0, [nothing; [e.key for e in REGISTRY]])
    @test provider_tokens("OpenAIDecisionsProtocol, OllamaSystemOneProtocol and GenericOpenAIEndpoint; GPT-5.4 mini") ==
          [(Set([:openai]), "OpenAI"), (Set([:ollama]), "Ollama"), (Set([:openai, :azure]), "GPT-")]
end
@testset "metrics — fenced code, nested fences and table rows are never prose; an unclosed fence is refused" begin
    page = fixture_page("Neutral prose.\n\n````markdown\n```julia\nClaude\n```\nGemini\n````\n\n| Claude | yes |\n|---|---|\n\n" *
                        "!!! note\n    ```julia\n    Chat(service=ANTHROPICServiceEndpoint)\n    ```\n")
    @test share([page], :openai) == 0
    err = caught(() -> prose_lines("a\n\n```julia\nx\n"; file="p.md"))
    @test err isa PageError && (err.file, err.line) == ("p.md", 3) && startswith(sprint(showerror, err), "PageError: p.md:3: ")
end
@testset "metrics — a binding block may bind several names; code outside it names no endpoint and no model" begin
    page = fixture_page("```julia\nservice, model = OPENAIServiceEndpoint, \"gpt-5.4-mini\"\n" *
                        "decider, decision_model = TYPESAFEServiceEndpoint, \"jev-latest\"\n```\n\nText.\n\n" *
                        "```julia\nask(TYPESAFEServiceEndpoint, \"Is it late?\")\n```\n")
    @test concept_code_violations(page) == ["$page:9: TYPESAFEServiceEndpoint"]
    before = fixture_page("```@setup page\nusing UniLM; Chat(service=GEMINIServiceEndpoint)\n```\n\n" *
                          "```julia\nservice, model = OPENAIServiceEndpoint, \"gpt-5.4-mini\"\n```\n")
    @test concept_code_violations(before) == ["$before:2: GEMINIServiceEndpoint"]   # a block before it, @setup too
    unbound = fixture_page("Pass `OPENAIServiceEndpoint` as `service`.\n\n```julia\n" *
                           "Chat(service=MistralEndpoint(), model=\"mistral-small\")\n" *
                           "respond(\"hi\"; model=\"o3\"); println(\"gpt-5.2 row: \")\n```\n")   # a label is no model id
    @test concept_code_violations(unbound) ==                    # no binding block: all code counts, prose never
          ["$unbound:4: MistralEndpoint", "$unbound:4: \"mistral-small\"", "$unbound:5: \"o3\""]
end
@testset "metrics — a reader's path: the entry's pages, then the concept pages they link" begin
    docsrc = fixture_docsrc(
        "p/a.md" => "# [A](@id a_start)\n\nSee [streaming](@ref streaming_guide), [chat](@ref chat_api), [`Chat`](@ref),\n" *
                    "[a protocol](@ref UniLM.TypeSafeProtocol) and [B](@ref b_start).\n\n| [Tools](@ref tools_guide) |\n|---|\n\n" *
                    "```julia\n# [MCP](@ref mcp_guide)\n```\n",
        "p/b.md" => "# [B](@id b_start)\n\n[Streaming again](@ref streaming_guide), [time\nouts](@ref\n    timeouts_guide).\n",
        "guide/streaming.md" => "# [Streaming](@id streaming_guide)\n", "guide/tool_calling.md" => "# [Tools](@id tools_guide)\n",
        "guide/timeouts.md" => "# [Timeouts](@id timeouts_guide)\n", "guide/mcp.md" => "# [MCP](@id mcp_guide)\n",
        "api/chat.md" => "# [Chat](@id chat_api)\n")
    e = remake(entry(:anthropic); pages=["p/a.md", "p/b.md", "p/unwritten.md"])
    @test reader_path(e; docsrc) ==
          ["p/a.md", "p/b.md", "p/unwritten.md", "guide/streaming.md", "guide/tool_calling.md", "guide/timeouts.md"]
    for (text, line) in ["# C\n\n[gone](@ref no_such_id)\n" => 3,          # an id defined nowhere
                         "# C\n\n\n[Streaming](@ref)\n" => 4]             # a heading's text, not an id
        broken = fixture_docsrc("p/c.md" => text, "guide/streaming.md" => "# [Streaming](@id streaming_guide)\n")
        err = caught(() -> reader_path(remake(e; pages=["p/c.md"]); docsrc=broken))
        @test err isa PageError && (err.file, err.line) == (joinpath(broken, "p/c.md"), line)
    end
    twice = fixture_docsrc("guide/streaming.md" => "# [Streaming](@id streaming_guide)\n",
                           "p/d.md" => "# D\n\n## [Streaming here](@id streaming_guide)\n")
    err = caught(() -> reader_path(e; docsrc=twice))                      # an id defined on two pages
    @test err isa PageError && (err.file, err.line) == (joinpath(twice, "p/d.md"), 3)
end
@testset "metrics — every @ref of today's pages resolves to an @id or a binding" begin
    src = joinpath(@__DIR__, "src")
    pages = sort!([relpath(joinpath(r, f), src) for (r, _, fs) in walkdir(src) for f in fs if endswith(f, ".md")])
    @test reader_path(remake(entry(:openai); pages); docsrc=src) == pages
end
@testset "metrics — the report: a missing page never passes, thresholds include their bound, code and ledgers count" begin
    pages = unique!([[p for e in REGISTRY for p in e.pages]; [p for c in instances(ProviderFacts.Chapter) for p in CONCEPT_PAGES[c]]])
    clean = [p => "# Title\n\nNeutral prose.\n" for p in pages]
    passes(files::Pair{String,String}...; options...) = report(IOBuffer(); docsrc=fixture_docsrc(files...), options...)
    @test !passes() && passes(; baseline=true)                                     # no pages: never a 0% share
    out = IOBuffer(); report(out; docsrc=fixture_docsrc())
    @test occursin("no pages", String(take!(out)))
    @test passes(clean...) && !passes(clean[2:end]...)                              # every page, or one missing
    deepseek(n) = "providers/deepseek.md" => "# Title\n" * "Neutral prose.\n"^(99 - n) * "Claude answers.\n"^n
    @test passes(clean..., deepseek(1)) && !passes(clean..., deepseek(2))          # 1 of 100 lines passes, 2 do not
    @test !passes(clean..., "guide/streaming.md" => "# Title\n\n```julia\nChat(service=MistralEndpoint())\n```\n")
    x = "x.md" => "# [X](@id x_s)\n\nthe witness text\n"
    atoms = fixture_dir("anthropic.json" => """{"atoms": [{"topic": "a"}, {"topic": "b"}]}""", "ia.json" => """{"verdict": "v"}""")
    ledgers = fixture_dir("anthropic.tsv" => join([LEDGER_HEADER, "anthropic:1\tindex.md:16\tdeleted\tmerged into anthropic:2",
                                                   "anthropic:2\tindex.md:20\tx.md#x_s\tthe witness text"], '\n') * "\n")
    @test passes(clean..., x; ledgers, atoms)
    @test !passes(clean..., x; ledgers=fixture_dir(), atoms)                        # the unit's ledger is missing
    @test !passes(clean..., x; ledgers, atoms=fixture_dir("ia.json" => """{"verdict": "v"}"""))   # no atoms at all
    @test caught(() -> report(IOBuffer(); docsrc=fixture_docsrc(clean...), ledgers)) isa ArgumentError
end
@testset "metrics — a ledger row's section holds its witness; an empty witness proves nothing" begin
    docsrc = fixture_docsrc("x.md" => "# [X](@id x_start)\n\n## [S](@id x_s)\n\n```julia\n# a comment, not a heading\n```\n\n" *
                                      "the witness\n\n## [T](@id x_t)\n\nanother passage\n")
    problems(rows::String...) = check_ledger(fixture_tsv(collect(rows)), docsrc, ["openai:1"])
    @test problems("openai:1\ts\tx.md#x_s\tthe witness") == String[]                 # past a comment in code
    @test length(problems("openai:1\ts\tx.md#x_s\tanother passage")) == 1           # the next section's text
    @test problems("openai:1\ts\tx.md#x_start\tanother passage") == String[]        # a title's section is the page
    @test length(problems("openai:1\ts\tx.md#x_s\t ")) == 1                         # no witness
    @test length(problems("openai:1\ts\tdeleted\t")) == 1                           # deleted without a reason
    @test length(problems("openai:1\ts\ty.md#x_s\tthe witness")) == 1               # no such page
    @test length(problems("openai:1\ts\tx.md#x_u\tthe witness")) == 1               # no such section
    @test length(problems("openai:1\ts\tx.md\tthe witness")) == 1                   # not page.md#id
    @test length(problems("openai:1\ts\tx.md#x_s")) == 2                            # three columns, so openai:1 has no row
    @test length(problems("openai:1\ts\tdeleted\tgone", "openai:7\ts\tdeleted\tgone")) == 1   # no atom of this ledger
    @test length(check_ledger(fixture_page("openai:1\ts\tdeleted\tgone\nopenai:1\ts\tdeleted\tgone\n"), docsrc,
                              ["openai:1"])) == 1                         # no header: no row is read
end
@testset "metrics — the command line takes --ledgers and --atoms together or not at all" begin
    dir = mktempdir()
    usage(args::String...) = redirect_stderr(() -> DocMetrics.main(collect(args)), devnull)
    @test usage("--ledgers", dir) == usage("--atoms", dir) == usage("--atoms") == usage("--baseline", "--baseline") ==
          usage("--ledgers", joinpath(dir, "absent"), "--atoms", dir) == usage("--check") == 2
end
