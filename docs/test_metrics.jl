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
    text = "# T\n\nOpenAI here.\n\n```julia\nAnthropic\n```\n\n| Anthropic |\n|---|\n| yes |\n\n!!! tip\n    Gemini is fine.\n    ```julia\n    x\n    ```\nplain\n"
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
    docsrc = fixture_docsrc("x.md" => "# [X](@id x_start)\n\n## [S](@id x_s)\n\nthe witness\ntext that proves the placement\n")
    expected = ["anthropic:1" => "old.md:1", "anthropic:2" => "old.md:9"]
    ok = fixture_tsv(["anthropic:1\told.md:1\tx.md#x_s\tthe witness text that proves the placement",
                      "anthropic:2\told.md:9\tdeleted\tduplicate of anthropic:1"])
    @test check_ledger(ok, docsrc, expected) == String[]
    bad = fixture_tsv(["anthropic:1\told.md:1\tx.md#x_s\tmissing words that are nowhere here",
                       "anthropic:1\told.md:1\tx.md#x_s\tthe witness text that proves the placement"])
    @test length(check_ledger(bad, docsrc, expected)) == 3   # absent witness, twice, one missing
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
@testset "metrics — an entry's key variables are tokens of its provider" begin
    @test all(e -> all(v -> provider_tokens("Set $v.") == [(Set([e.provider]), v)], e.key_vars), REGISTRY)
    # every other variable its code reads too; a name with `<MODEL>` in it, by the part before
    @test all(e -> all(v -> provider_tokens("Set $v.") == [(Set([e.provider]), first(split(v, '<')))], e.env_vars), REGISTRY)
    @test provider_tokens("Set AZURE_OPENAI_DEPLOY_NAME_GPT_5_2.") == [(Set([:azure]), "AZURE_OPENAI_DEPLOY_NAME_")]
    @test provider_tokens("Set AZURE_OPENAI_API_KEY, not OPENAI_API_KEY.") ==
          [(Set([:azure]), "AZURE_OPENAI_API_KEY"), (Set([:openai]), "OPENAI_API_KEY")]
    @test share([fixture_page("Set `OPENAI_API_KEY` before running.")], nothing) == 1          # a key variable alone names its provider
    @test share([fixture_page("Point OLLAMA_HOST at the server.\nNeutral line.\n")], :openai) ≈ 1 / 2
    @test share([fixture_page("Point OLLAMA_HOST at the server.\nNeutral line.\n")], :ollama_decisions) == 0
end
@testset "metrics — an Evidence box is provenance, not prose; every other admonition is prose" begin
    box = "!!! details \"Evidence\"\n    Measured on Jev, jev-1.13.0.\n\n    Jev again, in a second paragraph.\n"
    @test share([fixture_page("Neutral prose.\n\n" * box * "\nNeutral again.\n")], nothing) == 0
    @test share([fixture_page(box * "Claude, after the box.\n")], nothing) ≈ 1 / 2              # a dedented line ends it
    @test share([fixture_page(box * "\n!!! note \"Measured\"\n    Measured on Jev.\n")], nothing) ≈ 1 / 3
    # Julia's admonition body takes only lines indented four spaces or a tab, and blank lines
    @test share([fixture_page(box * " Claude answers here.\n Gemini too.\n")], nothing) ≈ 2 / 3
    @test share([fixture_page(box * "\n   Claude answers here.\n")], nothing) ≈ 1 / 2
    for opener in ["!!! details  \"Evidence\"", "!!! details \"Evidence\" (measured)", "!!! details \"Evidence",
                   "!!!\tdetails \"Evidence\"", "  !!! details \"Evidence\""]         # no admonition to Julia: prose
        @test share([fixture_page(opener * "\n    Measured on Jev.\n")], nothing) ≈ 1 / 2
    end
end
@testset "metrics — fenced code, nested fences and table rows are never prose; an unclosed fence is refused" begin
    page = fixture_page("Neutral prose.\n\n````markdown\n```julia\nClaude\n```\nGemini\n````\n\n| Claude | yes |\n|---|---|\n| Gemini | no |\n\n" *
                        "!!! note\n    ```julia\n    Chat(service=ANTHROPICServiceEndpoint)\n    ```\n")
    @test share([page], :openai) == 0
    err = caught(() -> prose_lines("a\n\n```julia\nx\n"; file="p.md"))
    @test err isa PageError && (err.file, err.line) == ("p.md", 3) && startswith(sprint(showerror, err), "PageError: p.md:3: ")
end
@testset "metrics — a binding block may bind several names; code outside it names no endpoint and no model" begin
    page = joinpath(fixture_docsrc("guide/system_one.md" =>                 # a Decisions concept page binds two endpoints
        "```julia\nservice, model = OPENAIServiceEndpoint, \"gpt-5.4-mini\"\n" *
        "decider, decision_model = TYPESAFEServiceEndpoint, \"jev-latest\"\n```\n\nText.\n\n" *
        "```julia\nask(TYPESAFEServiceEndpoint, \"Is it late?\")\n```\n"), "guide/system_one.md")
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
@testset "metrics — a binding block exempts only its binding lines, within its chapter's endpoints and their models" begin
    code(page, text) = (dir = fixture_docsrc(page => text); (joinpath(dir, page), concept_code_violations(joinpath(dir, page))))
    bind(lines...) = "```julia\n" * join(lines, '\n') * "\n```\n"
    openai, gemini = "service, model = OPENAIServiceEndpoint, \"gpt-5.4-mini\"", "gemini, gemini_model = GEMINIServiceEndpoint, \"gemini-3.8-flash\""
    aliased = bind(openai, gemini) * "\n" * bind("respond(\"again\"; service=gemini, model=gemini_model)")   # a second provider
    page, found = code("guide/agentic.md", aliased)                               # a language-model page binds one endpoint
    @test length(found) == 1 && startswith(only(found), "$page:3: GEMINIServiceEndpoint")
    @test last(code("guide/system_one.md", aliased)) == String[]                 # a Decisions page binds two
    @test length(last(code("guide/system_one.md", bind(openai, gemini, "local_model = OllamaEndpoint(), \"qwen3:4b\"")))) == 1
    @test length(last(code("guide/agentic.md", bind("service, model = OPENAIServiceEndpoint, \"claude-haiku-5-5\"")))) == 1
    @test last(code("guide/system_one.md", bind("service, model = TYPESAFEServiceEndpoint, \"jev-latest\"", "pinned = \"jev-1.13.0\""))) ==
          String[]                                                              # a model of the bound provider
    @test length(last(code("guide/agentic.md", bind(openai, "chat = Chat(service=MistralEndpoint(), model=\"mistral-small\")")))) == 2
end
@testset "metrics — a reader's path: the entry's pages, then the concept pages they link" begin
    docsrc = fixture_docsrc(
        "p/a.md" => "# [A](@id a_start)\n\nSee [streaming](@ref streaming_guide), [chat](@ref chat_api), [`Chat`](@ref),\n" *
                    "[a protocol](@ref UniLM.TypeSafeProtocol) and [B](@ref b_start).\n\n| [Tools](@ref tools_guide) |\n|---|\n| yes |\n\n" *
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
    x = "x.md" => "# [X](@id x_s)\n\nthe witness text of this placement\n"
    atoms = fixture_dir("anthropic.json" => """{"atoms": [{"location": "index.md:16"}, {"location": "index.md:20"}]}""",
                        "ia.json" => """{"verdict": "v"}""")
    ledgers = fixture_dir("anthropic.tsv" => join([LEDGER_HEADER, "anthropic:1\tindex.md:16\tdeleted\tmerged into anthropic:2",
                                                   "anthropic:2\tindex.md:20\tx.md#x_s\tthe witness text of this placement"], '\n') * "\n")
    @test passes(clean..., x; ledgers, atoms)
    @test !passes(clean..., x; ledgers=fixture_dir(), atoms)                        # the unit's ledger is missing
    @test !passes(clean..., x; ledgers, atoms=fixture_dir("ia.json" => """{"verdict": "v"}"""))   # no atoms at all
    @test caught(() -> report(IOBuffer(); docsrc=fixture_docsrc(clean...), ledgers)) isa ArgumentError
end
@testset "metrics — a ledger row's section holds its witness; an empty witness proves nothing" begin
    W, V = "the witness of this section, long enough", "another passage, long enough to prove"
    docsrc = fixture_docsrc("x.md" => "# [X](@id x_start)\n\n## [S](@id x_s)\n\n```julia\n# a comment, not a heading\n```\n\n" *
                                      "$W\n\n## [T](@id x_t)\n\n$V\n")
    problems(rows::String...) = check_ledger(fixture_tsv(collect(rows)), docsrc, ["openai:1" => "s"])
    @test problems("openai:1\ts\tx.md#x_s\t$W") == String[]                         # past a comment in code
    @test length(problems("openai:1\ts\tx.md#x_s\t$V")) == 1                        # the next section's text
    @test problems("openai:1\ts\tx.md#x_start\t$V") == String[]                     # a title's section is the page
    @test length(problems("openai:1\ts\tx.md#x_s\t ")) == 1                         # no witness
    @test length(problems("openai:1\ts\tdeleted\t")) == 1                           # deleted without a reason
    @test length(problems("openai:1\ts\ty.md#x_s\t$W")) == 1                        # no such page
    @test length(problems("openai:1\ts\tx.md#x_u\t$W")) == 1                        # no such section
    @test length(problems("openai:1\ts\tx.md\t$W")) == 1                            # not page.md#id
    @test length(problems("openai:1\ts\tx.md#x_s")) == 2                            # three columns, so openai:1 has no row
    @test length(problems("openai:1\ts\tdeleted\tgone", "openai:7\ts\tdeleted\tgone")) == 1   # no atom of this ledger
    @test length(check_ledger(fixture_page("openai:1\ts\tdeleted\tgone\nopenai:1\ts\tdeleted\tgone\n"), docsrc,
                              ["openai:1" => "s"])) == 1                  # no header: no row is read
end
@testset "metrics — a ledger row keeps its atom's source; a witness has 24 characters and stands below the heading" begin
    docsrc = fixture_docsrc("x.md" => "# [X](@id x_start)\n\n## [Prompt caching on every provider](@id x_cache)\n\n" *
                                      "A cached prefix is billed at its cached rate.\n")
    body, deleted = "A cached prefix is billed at its cached rate.", "u:2\tdocs/src/index.md:20\tdeleted\tsaid again in u:1"
    ledger(rows::String...) = check_ledger(fixture_tsv(collect(rows)), docsrc, ["u:1" => "docs/src/index.md:16", "u:2" => "docs/src/index.md:20"])
    @test ledger("u:1\tdocs/src/index.md:16\tx.md#x_cache\t$body", deleted) == String[]
    @test length(ledger("u:1\tdocs/src/index.md:20\tx.md#x_cache\t$body", "u:2\tdocs/src/index.md:16\tdeleted\tsaid again in u:1")) == 2
    @test length(ledger("u:1\tdocs/src/index.md:16\tx.md#x_cache\tPrompt caching on every provider", deleted)) == 1   # the heading
    @test length(ledger("u:1\tdocs/src/index.md:16\tx.md#x_cache\t$(body[1:23])", deleted)) == 1                     # 23 characters
    @test ledger("u:1\tdocs/src/index.md:16\tx.md#x_cache\t$(body[1:24])", deleted) == String[]                      # 24
    @test length(ledger("u:1\tdocs/src/index.md:20\tx.md#x_cache\tPrompt caching",                                     # shifted, too short
                        "u:2\tdocs/src/index.md:16\tx.md#x_cache\tcaching")) == 4
end
@testset "metrics — the command line takes --ledgers and --atoms together or not at all" begin
    dir = mktempdir()
    usage(args::String...) = redirect_stderr(() -> DocMetrics.main(collect(args)), devnull)
    @test usage("--ledgers", dir) == usage("--atoms", dir) == usage("--atoms") == usage("--baseline", "--baseline") ==
          usage("--ledgers", joinpath(dir, "absent"), "--atoms", dir) == usage("--check") == 2
end
@testset "metrics — a fence is read as Julia's Markdown reads it: padding, run, info string, closing fence" begin
    # A closing fence may carry text after its run: Julia closes the block there, and the rest starts the next block.
    doc = "A.\n\n```julia\nx = 1\n```julia\nOpenAI prose line.\n\n```julia\ny = 2\n```\n"
    @test [l for (_, l) in prose_lines(doc)] == ["A.", "julia", "OpenAI prose line."]
    # Only spaces, tabs and carriage returns pad an opening fence; after a no-break space the line is text.
    @test share([fixture_page("Neutral.\n\n ```julia\nClaude answers.\n ```\n")], nothing) ≈ 1 / 4
    # An info string that holds the fence character opens no block.
    @test share([fixture_page("Neutral.\n\n```julia `x`\nClaude answers.\n")], nothing) ≈ 1 / 3
    # A fence in a list item, as in an admonition, closes at its own indentation.
    @test share([fixture_page("1. Step\n\n   ```julia\n   Claude\n   ```\n")], :openai) == 0
    @test fenced_blocks(split("```julia\nx\n```julia\ny\n\n```\nz\n```\n", '\n')) == [1:3, 6:8]
end
@testset "metrics — a fence or a table row Julia's Markdown reads otherwise is refused, naming the line" begin
    refused(text) = (err = caught(() -> prose_lines(text; file="p.md")); err isa PageError ? err.line : nothing)
    @test refused("Neutral.\n\n  ```julia\n  Claude\n  ```\n") == 3        # at the top level Julia closes at column 0
    @test refused("Text ends here\\\n```julia\nClaude\n```\n") == 2       # a hard line break keeps it in the paragraph
    @test refused("Neutral.\n\n> ```julia\n> Claude\n> ```\n") == 3       # in a block quote
    @test refused("Neutral.\n\n| Claude | yes |\n") == 3                  # no alignment row: no table
    @test refused("Neutral.\n\n| A | B |\n|---|---|\n") == 3               # no body row: no table
    @test refused("Text.\n| A |\n|---|\n| Claude |\n") == 2                # a table starts a block: this is the paragraph
    @test refused("Neutral.\n\n| A |\n|---|\n| Claude |\n") === nothing
    # A row Julia reads without a leading `|` is prose to the metric: a provider there counts.
    @test share([fixture_page("Neutral.\n\nA | B\n--- | ---\nClaude | yes\n")], nothing) ≈ 1 / 4
end
@testset "metrics — an Evidence box opens as Julia's admonition does, and its body ends at a line that is not blank" begin
    box = "!!! details \"Evidence\"\n    Measured on Jev, jev-1.13.0.\n\n    Jev again, in a second paragraph.\n"
    # Julia strips the opener with `strip`, which keeps U+2028: no admonition, so its lines are prose.
    @test share([fixture_page("!!!  details \"Evidence\"\n    Measured on Jev.\n")], nothing) ≈ 1 / 2
    @test share([fixture_page("!!! details \"Evidence\" \n    Measured on Jev.\n")], nothing) ≈ 1 / 2
    # After a line ending in a backslash, the opener is the paragraph's: no box.
    @test share([fixture_page("Text ends here\\\n!!! details \"Evidence\"\n    Measured on Jev.\n")], nothing) ≈ 1 / 3
    # Blank is " \t\r" alone: a no-break space ends the body, so what follows is prose.
    @test share([fixture_page(box * " \n    Claude answers here.\n")], nothing) ≈ 1 / 2
    # Only a `details` box titled "Evidence…" is provenance: another category, or another title, is prose.
    @test share([fixture_page("!!! details \"Measured\"\n    Measured on Jev.\n")], nothing) ≈ 1 / 2
    @test share([fixture_page("!!! note \"Evidence\"\n    Measured on Jev.\n")], nothing) ≈ 1 / 2
end
@testset "metrics — a binding line binds and does nothing else; a quoted value interpolates nothing" begin
    code(text) = (dir = fixture_docsrc("guide/system_one.md" => text); p = joinpath(dir, "guide/system_one.md");
                  (p, concept_code_violations(p)))
    bind(lines...) = "```julia\n" * join(lines, '\n') * "\n```\n"
    jev = "service, model = TYPESAFEServiceEndpoint, \"jev-latest\""
    page, found = code(bind(jev, "pinned = \"jev-1.13.0\"; respond(\"hi\")"))          # trailing code: no binding line
    @test found == ["$page:3: \"jev-1.13.0\""]
    page, found = code(bind(jev, "label = \"\$(respond(GEMINIServiceEndpoint))\""))
    @test found == ["$page:3: GEMINIServiceEndpoint"]
    page, found = code(bind(jev, "local_llm = OllamaEndpoint(base_url=\"\$(respond(GEMINIServiceEndpoint))\")"))
    @test found == ["$page:3: OllamaEndpoint", "$page:3: GEMINIServiceEndpoint"]
end
@testset "metrics — the OpenAI-compatible wire names no provider in any spelling; an Ollama tag is Ollama's" begin
    page(line) = fixture_page(line * "\nNeutral line.\n")
    for v in ["OpenAI-compatible", "OpenAI compatible", "OpenAI-Compatible", "OpenAI Compatible", "OpenAI-compatibility",
              "OpenAI compatibility", "OpenAI-Compatibility", "OpenAI Compatibility", "OpenAI-compat", "OpenAI compat",
              "OpenAI-Compat", "OpenAI Compat", "OpenAI-wire", "OpenAI wire", "OpenAI-Wire", "OpenAI Wire"]
        @test share([page("Any $v server works.")], :anthropic) == 0
        @test v in NEUTRAL_TOKENS
    end
    @test share([page("Pull gpt-oss:120b-cloud or deepseek-r1:8b.")], :ollama) == 0
    @test share([page("Pull gpt-oss:120b-cloud.")], :openai) ≈ 1 / 2
    @test provider_tokens("gpt-oss:20b, library/gpt-oss:20b, gpt-oss-20b and gpt-5.4-mini: the default") ==
          [(Set([:ollama]), "gpt-"), (Set([:ollama]), "gpt-"), (Set([:openai, :azure]), "gpt-"), (Set([:openai, :azure]), "gpt-")]
end
@testset "metrics — OpenAI's audio and image model ids, Codestral and TypeSafe's variables name their provider" begin
    page(line) = fixture_page(line * "\nNeutral line.\n")
    @test provider_tokens("whisper-1, tts-1-hd and dall-e-3") ==
          [(Set([:openai, :azure]), "whisper-"), (Set([:openai, :azure]), "tts-"), (Set([:openai, :azure]), "dall-e-")]
    @test share([page("Use codestral-latest.")], :anthropic) ≈ 1 / 2
    @test share([page("Use codestral-latest.")], :openai_compatible) == 0
    @test share([page("Set TYPESAFE_DEFAULT_MODEL or TYPESAFE_BASE_URL.")], :openai) ≈ 1 / 2
    @test provider_tokens("TYPESAFE_TIMEOUT") == [(Set([:typesafe]), "TYPESAFE_")]
end
@testset "metrics — the report says what a prose line is and counts, per page, the Evidence lines it leaves out" begin
    pages = unique!([[p for e in REGISTRY for p in e.pages]; [p for c in instances(ProviderFacts.Chapter) for p in CONCEPT_PAGES[c]]])
    clean = [p => "# Title\n\nNeutral prose.\n" for p in pages]
    boxed = "guide/system_one.md" => "# Title\n\nNeutral prose.\n\n!!! details \"Evidence\"\n    Jev, jev-1.13.0.\n\n    Twice.\n"
    out = IOBuffer()
    @test report(out; docsrc=fixture_docsrc(clean..., boxed))
    text = String(take!(out))
    @test occursin("Evidence", first(split(text, '\n')))
    @test occursin("\n  guide/system_one.md: 2\n", text) && !occursin("guide/streaming.md: ", text)
end
@testset "metrics — --entry measures one entry, --unit checks one audit unit's ledger" begin
    pages = unique!([[p for e in REGISTRY for p in e.pages]; [p for c in instances(ProviderFacts.Chapter) for p in CONCEPT_PAGES[c]]])
    clean = [p => "# Title\n\nNeutral prose.\n" for p in pages]
    docsrc = fixture_docsrc(clean..., "guide/streaming.md" => "# Title\n\nClaude streams.\n")   # a concept page fails
    out = IOBuffer()
    @test !report(IOBuffer(); docsrc) && report(out; docsrc, entry_key=:anthropic)
    rows = filter(l -> occursin(r"^(entry|path|concept) ", l), split(String(take!(out)), '\n'))
    @test length(rows) == 2 && all(r -> occursin(r"^(entry|path) +anthropic ", r), rows)
    @test caught(() -> report(IOBuffer(); docsrc, entry_key=:nope)) isa ArgumentError
    x = "x.md" => "# [X](@id x_s)\n\nthe witness text of this placement\n"
    atoms = fixture_dir("anthropic.json" => """{"atoms": [{"location": "index.md:16"}]}""",
                        "gemini.json" => """{"atoms": [{"location": "index.md:20"}]}""")
    ledgers = fixture_dir("anthropic.tsv" => join([LEDGER_HEADER, "anthropic:1\tindex.md:16\tx.md#x_s\tthe witness text of this placement"], '\n') * "\n")
    docsrc = fixture_docsrc(clean..., x)
    @test !report(IOBuffer(); docsrc, ledgers, atoms)                                  # gemini's ledger is missing
    @test report(IOBuffer(); docsrc, ledgers, atoms, unit="anthropic")
    @test caught(() -> report(IOBuffer(); docsrc, ledgers, atoms, unit="ia")) isa ArgumentError
    usage(args::String...) = redirect_stderr(() -> DocMetrics.main(collect(args)), devnull)
    @test usage("--entry", "nope") == usage("--entry") == usage("--unit", "anthropic") ==
          usage("--ledgers", ledgers, "--atoms", atoms, "--unit", "ia") == 2
    @test redirect_stdout(() -> usage("--entry", "anthropic", "--baseline"), devnull) == 0
end
