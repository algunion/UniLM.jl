using Test, Markdown, UniLM
using Documenter: MarkdownAST
isdefined(Main, :ProviderFacts) || include(joinpath(@__DIR__, "ProviderFacts.jl")); using .ProviderFacts
hasvalue(n) = n.element isa MarkdownAST.JuliaValue || any(hasvalue, n.children)
@testset "ProviderFacts — every entry's facts come from the code" begin
    @test entry(:anthropic).default_model == UniLM.default_model(ANTHROPICServiceEndpoint) == "claude-opus-5-5"
    @test entry(:openai).default_model == "gpt-6-sol" && entry(:jev).default_model == "jev-latest"
    @test entry(:openai_decisions).default_model == "gpt-6-luna"
    @test entry(:ollama).default_model === nothing && entry(:ollama_decisions).default_model === nothing
    @test all(e -> e.chapter == ProviderFacts.llm, filter(e -> e.key in (:openai, :anthropic, :gemini), REGISTRY))
    @test length(unique(e.start_id for e in REGISTRY)) == length(REGISTRY) == 11
end
@testset "ProviderFacts — tables have one row per entry and link each start page" begin
    t = provider_table(ProviderFacts.llm).content[1]
    @test t isa Markdown.Table && length(t.rows) == 1 + count(e -> e.chapter == ProviderFacts.llm, REGISTRY)
    @test occursin("(@ref anthropic_start)", Markdown.plain(provider_table(ProviderFacts.llm)))
    @test occursin("embeddings", Markdown.plain(at_a_glance(:anthropic)))          # listed as not available
end
@testset "ProviderFacts — every generated table converts to MarkdownAST without raw values" begin
    for md in (provider_table(ProviderFacts.llm), provider_table(ProviderFacts.decisions), at_a_glance(:anthropic),
               at_a_glance(:jev), capability_matrix(), on_your_provider(ProviderFacts.llm, :streaming), openai_model_rules(),
               claude_family_table(), gemini_thinking_table(), price_table(:anthropic))
        @test all(r -> all(c -> c isa AbstractVector, r), only(md.content).rows)
        @test !hasvalue(convert(MarkdownAST.Node, md))
    end
end
@testset "ProviderFacts — rule and price tables cover every row of the code's tables" begin
    @test count(==('\n'), Markdown.plain(claude_family_table())) >= length(UniLM._CLAUDE_FAMILIES) + 2
    @test all(k -> occursin(k, Markdown.plain(gemini_thinking_table())), keys(UniLM._GEMINI_THINKING_LEVELS))
    @test sort([m for p in (:openai, :anthropic, :gemini, :deepseek, :jev) for m in price_rows(p)]) ==
          sort(collect(keys(UniLM.DEFAULT_PRICING)))                               # every row in exactly one table
    @test "claude-haiku-5-5" in price_rows(:anthropic)
end
@testset "ProviderFacts — a chapter's tables name only that chapter's entries" begin
    linked(md) = [chopprefix(only(only(r)).url, "@ref ") for r in only(md.content).rows[2:end]]
    chapter_of(id) = only(e for e in REGISTRY if id in values(e.features)).chapter
    @test all(id -> chapter_of(id) == ProviderFacts.llm, linked(on_your_provider(ProviderFacts.llm, :images)))
    @test all(id -> chapter_of(id) == ProviderFacts.decisions, linked(on_your_provider(ProviderFacts.decisions, :images)))
    supports(key) = [Symbol(c.code) for c in only(at_a_glance(key).content).rows[2][4] if c isa Markdown.Code]
    @test all(e -> e.chapter == ProviderFacts.llm || (s = supports(e.key); !isempty(s) && s ⊆ ProviderFacts.DECISION_CAPABILITIES),
              REGISTRY)
end
@testset "ProviderFacts — the registry and price checks refuse a broken table, naming what is wrong" begin
    remake(e; changes...) = ProviderEntry(; (f => get(changes, f, getfield(e, f)) for f in fieldnames(ProviderEntry))...)
    function refusal(entries)                       # the entry and field the check names; `nothing` when it accepts
        try ProviderFacts._validated(entries); nothing
        catch err; err isa ProviderFacts.RegistryError || rethrow(); (err.key, err.field) end
    end
    openai, gemini, jev, azure = entry(:openai), entry(:gemini), entry(:jev), entry(:azure)
    @test refusal(copy(REGISTRY)) === nothing
    @test refusal([openai, openai]) == (:openai, :key)
    @test refusal([remake(azure; endpoints=ServiceEndpointSpec[])]) == (:azure, :endpoints)
    @test refusal([remake(azure; pages=String[])]) == (:azure, :pages)
    @test refusal([remake(azure; pages=["providers/azure.html"])]) == (:azure, :pages)
    @test refusal([remake(jev; default_model="jev-1.13.0")]) == (:jev, :default_model)
    @test refusal([remake(openai; default_model=nothing)]) == (:openai, :default_model)
    @test refusal([openai, remake(gemini; start_id="openai_start")]) == (:gemini, :start_id)
    @test refusal([openai, remake(gemini; features=merge(gemini.features, Dict(:setup => "openai_setup")))]) ==
          (:gemini, :features)
    @test refusal([remake(jev; features=merge(jev.features, Dict(:images => "jev_images__next")))]) == (:jev, :features)
    row = UniLM.DEFAULT_PRICING["gpt-6-sol"]
    function price_refusal(ids, unprefixed...)       # the id and tables the partition names; `nothing` when it accepts
        try ProviderFacts._price_partition(Dict(id => row for id in ids), unprefixed...); nothing
        catch err; err isa ProviderFacts.PriceTableError || rethrow(); (err.id, err.tables) end
    end
    @test price_refusal(collect(keys(UniLM.DEFAULT_PRICING))) === nothing
    @test price_refusal(["chat-latest", "mistral-large-latest"]) == ("mistral-large-latest", Symbol[])
    @test price_refusal(["chat-latest", "gpt-6-sol"], ("chat-latest" => :openai, "gpt-6-sol" => :jev)) ==
          ("gpt-6-sol", [:openai, :jev])
    @test_throws ArgumentError ProviderFacts._price_partition(Dict("gpt-6-sol" => row))  # "chat-latest" is unpriced
end
@testset "ProviderFacts — the Gemini table asks the generateContent encoder's check which levels it takes" begin
    rows = only(gemini_thinking_table().content).rows[2:end]
    model(r) = only(r[1]).code
    levels(cell) = [c.code for c in cell if c isa Markdown.Code]
    twofive = filter(r -> startswith(model(r), "gemini-2.5-"), rows)
    @test !isempty(twofive) && all(r -> isempty(levels(r[3])), twofive)        # thinkingBudget, not levels
    @test any(r -> startswith(model(r), "gemini-3") && !isempty(levels(r[3])), rows)
    @test_throws ArgumentError ProviderFacts._generate_content_cell("gemini-3.8-flash", ["minimal"])  # other refusals propagate
end
@testset "ProviderFacts — regions regenerate and staleness is detected, in both marker forms" begin
    html = "a\n\n<!-- generated:providers -->\n\nold\n\n<!-- /generated:providers -->\n\nb\n"
    fence = "a\n\n```@setup generated\n# generated:provider-contracts (written by docs/regions.jl)\n```\n\nold\n\n```@setup generated\n# /generated:provider-contracts\n```\n\nb\n"
    for text in (html, fence)
        out = apply_regions(text)
        @test !occursin("\nold\n", out) && apply_regions(out) == out           # regenerated, idempotent
        @test startswith(out, "a\n\n") && endswith(out, "\n\nb\n")
        mktemp() do p, io; write(io, text); close(io); @test length(stale_regions(p)) == 1; end
    end
    @test occursin("Anthropic", apply_regions(html))
end
@testset "ProviderFacts — region markers are parsed strictly; each refusal names the file and the line" begin
    O, C = "<!-- generated:providers -->", "<!-- /generated:providers -->"
    FO = "```@setup generated\n# generated:provider-contracts (written by docs/regions.jl)\n```"
    FC = "```@setup generated\n# /generated:provider-contracts\n```"
    function refusal(text)                          # the file and line the parser names; `nothing` when it accepts
        try apply_regions(text; file="f.txt"); nothing
        catch err; err isa ProviderFacts.RegionError || rethrow(); (err.file, err.line) end
    end
    @test refusal("a\n\n$O\n\nold\n\n$C\n\n$FO\n\nold\n\n$FC\n\nb\n") === nothing         # both forms in one file
    @test refusal("a\n\n$O\n\nold\n") == ("f.txt", 3)                                    # not closed
    @test refusal("a\n\n$O\n\n$FO\n\n$FC\n\n$C\n\nb\n") == ("f.txt", 5)                   # opened inside another
    @test refusal("a\n\n$O\n\n$FC\n\n$C\n\nb\n") == ("f.txt", 5)                          # another region closes inside it
    @test refusal("a\n\n$C\n\nb\n") == ("f.txt", 3)                                      # closed, never opened
    @test refusal("a\n\n$O\n\n$C\n\n$O\n\n$C\n\nb\n") == ("f.txt", 7)                     # marked twice
    @test refusal("a\n\n<!-- generated:provider -->\n\n<!-- /generated:provider -->\n\nb\n") == ("f.txt", 3)  # unknown
    @test refusal("a\n\n<!-- generated:provider-contracts -->\n\n<!-- /generated:provider-contracts -->\n\nb\n") ==
          ("f.txt", 3)                                                                   # a region of the other form
    @test refusal("a\n\n$O\n\n<!-- /generated:providers --> -->\n\nb\n") == ("f.txt", 5)   # a closing marker's prefix
    @test refusal("a\n\n```@setup generated\nx = 1\n```\n\nb\n") == ("f.txt", 3)          # a setup block marking nothing
    @test refusal("a\n$O\n\n$C\n\nb\n") == ("f.txt", 2)                                  # no blank line above
    @test refusal("a\n\n$O\nold\n\n$C\n\nb\n") == ("f.txt", 3)                           # none below
    @test refusal("a\n\n$O\n\nold\n$C\n\nb\n") == ("f.txt", 6)                           # none above the closing one
    @test refusal("a\n\n$O\n\n$C\n") == ("f.txt", 5)                                     # the file ends after it
    mktemp() do p, io
        write(io, "a\n\n$O\n\n"); close(io)
        err = try stale_regions(p); nothing catch e; e end
        @test err isa ProviderFacts.RegionError && (err.file, err.line) == (p, 3) &&
              startswith(sprint(showerror, err), "RegionError: $p:3: ")
    end
    @test_throws ArgumentError region_text("provider")
end
@testset "ProviderFacts — a region holds its text between blank lines and is stale when one byte differs" begin
    O, C = "<!-- generated:providers -->", "<!-- /generated:providers -->"
    FO = "```@setup generated\n# generated:provider-contracts (written by docs/regions.jl)\n```"
    FC = "```@setup generated\n# /generated:provider-contracts\n```"
    two = "a\n\n$FO\n\nold\n\n$FC\n\n$O\n\nold\n\n$C\n\nb\n"
    current = apply_regions(two)
    @test occursin("$FO\n\n$(region_text("provider-contracts"))\n$FC\n\n$O\n\n$(region_text("providers"))\n$C\n\nb\n", current)
    stale(text) = mktemp() do p, io; write(io, text); close(io); stale_regions(p) end
    @test stale(two) == ["provider-contracts", "providers"]                             # in file order
    @test stale(current) == String[]
    @test stale(replace(current, "`gpt-6-luna`\n" => "`gpt-6-luna` \n")) == ["providers"]
end
@testset "ProviderFacts — each region lists its chapters' entries, linked the way its file is read" begin
    items(name) = [only(i).content for i in Markdown.parse(region_text(name)).content[2].items]
    in_order(chapters...) = [e for e in REGISTRY if e.chapter in chapters]
    for (name, entries) in ("provider-contracts" => in_order(ProviderFacts.llm),
                            "decision-provider-contracts" => in_order(ProviderFacts.decisions),
                            "providers" => in_order(ProviderFacts.llm, ProviderFacts.decisions))
        md = Markdown.parse(region_text(name))
        @test length(md.content) == 2 && md.content[1] isa Markdown.Header && !hasvalue(convert(MarkdownAST.Node, md))
        @test [only(p[1].text) for p in items(name)] == [e.name for e in entries]
        @test all(zip(items(name), entries)) do (p, e)                                  # usage, key variables, default model
            [c.code for c in p if c isa Markdown.Code] == ["service = $(e.usage)"; e.key_vars; something(e.default_model, String[])]
        end
    end
    @test [p[1].url for p in items("provider-contracts")] == ["@ref " * e.start_id for e in in_order(ProviderFacts.llm)]
    url(key) = only(p[1].url for p in items("providers") if only(p[1].text) == entry(key).name)
    @test url(:openai) == "https://algunion.github.io/UniLM.jl/dev/providers/openai/"          # an index.md page
    @test url(:local_servers) == "https://algunion.github.io/UniLM.jl/dev/providers/local_servers/"
    @test url(:openai_decisions) == "https://algunion.github.io/UniLM.jl/dev/decisions/openai/"
    @test endswith(region_text("provider-contracts"),
                   "`service = MistralEndpoint()`; reads `MISTRAL_API_KEY`; no default model (a call names one)\n")
    @test occursin("`; reads no environment variable; no default model", region_text("providers"))
end
@testset "ProviderFacts — llm.md and llms.txt hold their regions, well marked" begin
    regions(path) = first.(ProviderFacts._regions(split(read(path, String), '\n'), path))
    @test regions(joinpath(@__DIR__, "src", "llm.md")) == ["decision-provider-contracts", "provider-contracts"]
    @test regions(joinpath(dirname(@__DIR__), "llms.txt")) == ["providers"]
end
