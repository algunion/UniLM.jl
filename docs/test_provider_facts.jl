using Test, Markdown, UniLM
using Logging: with_logger, NullLogger
using Documenter: MarkdownAST
isdefined(Main, :ProviderFacts) || include(joinpath(@__DIR__, "ProviderFacts.jl")); using .ProviderFacts
isdefined(Main, :DocMetrics) || include(joinpath(@__DIR__, "metrics.jl")); using .DocMetrics: outside_fences, PageError
hasvalue(n) = n.element isa MarkdownAST.JuliaValue || any(hasvalue, n.children)
const SRC = joinpath(@__DIR__, "src")

# The explicit ids the headings of the page at `path` define, in order, each with its heading's level. Its fenced
# blocks are read by the metric's rule (docs/metrics.jl), so a page the metric cannot read fails here too.
heading_ids(path::String)::Vector{Tuple{Int,String}} =
    [(length(h[1]), String(h[2])) for l in outside_fences(readlines(path); file=path)
     for h in (match(r"^(#{1,6})\s.*\(@id\s+([^)\s]+)\s*\)", l),) if !isnothing(h)]
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
               claude_family_table(), gemini_thinking_table(), price_table(:anthropic), claude_price_rules(),
               openai_price_terms(), openai_audio_rates())
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
@testset "ProviderFacts — the Claude price rules list every long-prompt tier and every family model without a price row" begin
    rows = only(claude_price_rules().content).rows[2:end]
    ids = [only(r[1]).code for r in rows]
    tiers, priced = UniLM._CLAUDE_LONG_PROMPT_TIERS, price_rows(:anthropic)
    unpriced = [m for m in keys(UniLM._CLAUDE_FAMILIES) if m ∉ priced]
    @test !isempty(tiers) && !isempty(unpriced)                     # premise: today each kind has a row
    @test ids == sort!(unique!([collect(keys(tiers)); unpriced]))   # every tier and every unpriced family, once
    estimate(m) = Markdown.plain(Markdown.Paragraph(rows[findfirst(==(m), ids)][2]))
    for (m, t) in tiers
        m in priced || continue                                     # a model with no row is estimated at 0.0
        @test occursin("every rate × $(isinteger(t.multiplier) ? Int(t.multiplier) : t.multiplier) for a prompt over",
                       estimate(m))
        @test occursin("over $(t.threshold) tokens", replace(estimate(m), "," => ""))
    end
    @test all(m -> occursin("`0.0`, with a warning: it has no price row", estimate(m)), unpriced)
end
@testset "ProviderFacts — the OpenAI price terms give every multiple estimated_cost bills, and only those" begin
    rows = only(openai_price_terms().content).rows
    ids = [only(r[1]).code for r in rows[2:end]]
    terms = UniLM._OPENAI_PRICE_TERMS
    @test ids == sort!(collect(keys(terms)))                         # every model with terms, once
    plain(m, col) = String(strip(Markdown.plain(Markdown.Paragraph(rows[1 + findfirst(==(m), ids)][col]))))
    limit = UniLM._OPENAI_SHORT_CONTEXT
    # the estimate of `n` input and 100 output tokens at `tier`, its warnings for an unpriced tier muted
    cost(m, n, tier; cached=0) = with_logger(NullLogger()) do
        estimated_cost(LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=Chat(model=m),
                                  usage=UniLM.TokenUsage(prompt_tokens=n, completion_tokens=100, cached_tokens=cached,
                                                         service_tier=tier)))
    end
    # the multiple a cell names for prompts up to the limit (`long=false`) or over it, `nothing` for "—"
    function named(cell::String, long::Bool)::Union{Float64,Nothing}
        cell == "—" && return nothing
        parts = split(cell, ", ")
        part = something(findfirst(p -> occursin(long ? " over" : " up to", p), parts),
                         findfirst(p -> !occursin(" over", p) && !occursin(" up to", p), parts), Some(nothing))
        isnothing(part) ? nothing : parse(Float64, match(r"× ([0-9.]+)", parts[part])[1])
    end
    n_tiers = 0
    for m in ids, (col, tier) in ((4, "flex"), (5, "fast"), (6, "ultrafast")), long in (false, true)
        n = long ? limit + 1000 : 1000
        multiple = named(plain(m, col), long && terms[m].long_context)
        @test cost(m, n, tier) ≈ something(multiple, 1.0) * cost(m, n, nothing)   # "—": the standard rates
        n_tiers += !isnothing(multiple)
    end
    @test n_tiers > 0
    for m in ids                                                      # the long-context cell is the estimate's
        if terms[m].long_context
            input, cached, output = ProviderFacts._openai_long_multiples(m)
            @test plain(m, 3) == "input × $(ProviderFacts._factor(input)), cached input × " *
                                 "$(ProviderFacts._factor(cached)), output × $(ProviderFacts._factor(output))"
            @test cost(m, limit + 1, nothing; cached=limit + 1) ≈
                  cached * (limit + 1) * DEFAULT_PRICING[m].cached_input + output * 100 * DEFAULT_PRICING[m].output
        else
            @test plain(m, 3) == "—"
            @test cost(m, limit + 1000, nothing) ≈ (limit + 1000) * DEFAULT_PRICING[m].input + 100 * DEFAULT_PRICING[m].output
        end
        w30, w12 = UniLM._cache_write_multiples(terms[m])
        @test plain(m, 2) == (w30 == w12 == 1 ? "—" : "× $(ProviderFacts._factor(w30)) input for `\"30m\"`, " *
                                                      "× $(ProviderFacts._factor(w12)) for `\"12h\"`")
    end
    audio = only(openai_audio_rates().content).rows
    @test [only(r[1]).code for r in audio[2:end]] == sort!(collect(keys(UniLM._OPENAI_AUDIO_RATES)))
    for r in audio[2:end]
        rate = UniLM._OPENAI_AUDIO_RATES[only(r[1]).code]
        @test strip(Markdown.plain(Markdown.Paragraph(r[2]))) == ProviderFacts._per_million(rate.input)
        @test strip(Markdown.plain(Markdown.Paragraph(r[3]))) == ProviderFacts._per_million(rate.output)
    end
end
@testset "ProviderFacts — a header cell that names a request field is inline code" begin
    # a header cell holding only `field` as inline code; Markdown.Code compares by identity, so by type and text
    in_header(md, field) = any(c -> length(c) == 1 && only(c) isa Markdown.Code && only(c).code == field,
                               only(md.content).rows[1])
    @test in_header(claude_family_table(), "inference_geo")
    @test in_header(openai_model_rules(), "prompt_cache_retention")
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
@testset "ProviderFacts — a language-model entry's tables list language-model capabilities only" begin
    plain(md) = Markdown.plain(md)
    for k in (:openai, :anthropic, :ollama)         # system_one: supported by two of them, not available on one
        @test !occursin("system_one", plain(at_a_glance(k)))
    end
    @test occursin("system_one", plain(at_a_glance(:jev)))                       # a decision entry's row lists it
    supports = [c.code for c in only(at_a_glance(:anthropic).content).rows[2][4] if c isa Markdown.Code]
    @test "models" in supports                    # a capability decision entries share stays in an LLM row
    @test !occursin("system_one", plain(capability_matrix()))
    @test ProviderFacts.DECISION_CAPABILITIES == [:models, :system_one]
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
    taken = UniLM._GEMINI_THINKING_LEVELS            # model => its thinking levels on Interactions (`respond`)
    others = setdiff(union(values(taken)...), taken["gemini-3.8-flash"])           # other models' levels, not this one's
    @test !isempty(others) || error("premise: another model takes a thinking level gemini-3.8-flash does not")
    isempty(others) ||
        @test_throws ArgumentError ProviderFacts._generate_content_cell("gemini-3.8-flash", [first(others)])  # other refusals propagate
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
        @test all(zip(items(name), entries)) do (p, e)                         # usage, every variable it reads, default model
            [c.code for c in p if c isa Markdown.Code] ==
                ["service = $(e.usage)"; e.key_vars; e.env_vars; something(e.default_model, String[])]
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
@testset "ProviderFacts — key_vars are what a reader sets to make a call; env_vars every other variable read" begin
    @test entry(:jev).key_vars == [UniLM.TYPESAFE_API_KEY]
    @test entry(:jev).env_vars == [UniLM.TYPESAFE_BASE_URL_ENV, UniLM.TYPESAFE_DEFAULT_MODEL_ENV]
    @test entry(:azure).key_vars == [UniLM.AZURE_OPENAI_API_KEY, UniLM.AZURE_OPENAI_BASE_URL, UniLM.AZURE_OPENAI_API_VERSION]
    @test entry(:azure).env_vars == ["AZURE_OPENAI_DEPLOY_NAME_<MODEL>"]
    # OllamaEndpoint reads its cloud's key, OLLAMA_API_KEY, for a request to https://ollama.com.
    @test entry(:ollama).env_vars == entry(:ollama_decisions).env_vars == [UniLM.OLLAMA_API_KEY]
    @test occursin("reads `OLLAMA_HOST`, `OLLAMA_API_KEY`;", region_text("provider-contracts")) &&
          occursin("reads `OLLAMA_HOST`, `OLLAMA_API_KEY`;", region_text("decision-provider-contracts"))
    @test all(e -> isempty(e.env_vars), filter(e -> e.key ∉ (:jev, :azure, :ollama, :ollama_decisions), REGISTRY))
    @test occursin("reads `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`;",
                   region_text("decision-provider-contracts"))
    keys_column(chapter) = [[c.code for c in r[2] if c isa Markdown.Code] for r in only(provider_table(chapter).content).rows[2:end]]
    @test keys_column(ProviderFacts.decisions) == [e.key_vars for e in REGISTRY if e.chapter == ProviderFacts.decisions]
    @test keys_column(ProviderFacts.llm) == [e.key_vars for e in REGISTRY if e.chapter == ProviderFacts.llm]
end
@testset "ProviderFacts — the id reader reads fences as the metric does" begin
    page(text) = (p = tempname(); write(p, text); p)
    @test heading_ids(page("```julia\nx\n```julia\n# [T](@id t)\n\n## [U](@id u)\n")) == [(1, "t"), (2, "u")]
    @test heading_ids(page("```julia\n# [T](@id t)\n```\n")) == []
    @test_throws PageError heading_ids(page("# [T](@id t)\n\n```julia\nx\n"))   # never closed: refused, not skipped
end
@testset "ProviderFacts — every start and feature id is defined once, on one of its entry's pages" begin
    ids = Dict{String,Vector{String}}()             # an explicit id => the pages whose headings define it
    for (root, _, files) in walkdir(SRC), f in files
        endswith(f, ".md") || continue
        page = relpath(joinpath(root, f), SRC)
        foreach(((_, id),) -> push!(get!(ids, id, String[]), page), heading_ids(joinpath(SRC, page)))
    end
    @test [id => p for (id, p) in ids if length(p) > 1] == []                      # every explicit id once
    on(id) = get(ids, id, String[])
    # On one of the entry's pages; or, while its section has not moved, as `<id>__next` on one of
    # them, with `<id>` still on its old page.
    placed(e, id) = length(on(id)) == 1 && only(on(id)) in e.pages && isempty(on(id * "__next")) ||
                    length(on(id * "__next")) == 1 && only(on(id * "__next")) in e.pages &&
                    length(on(id)) == 1 && only(on(id)) ∉ e.pages
    @test [(e.key, id) for e in REGISTRY for id in [e.start_id; sort!(collect(values(e.features)))] if !placed(e, id)] == []
end
@testset "ProviderFacts — the sidebar lists each entry's pages and each chapter's concept pages, as the registry does" begin
    tree = include(joinpath(@__DIR__, "pages.jl"))          # the `pages` docs/make.jl builds
    groups(chapter) = [g => String[last(p) for p in pages] for (g, pages) in only(last(i) for i in tree if first(i) == chapter)]
    title_id(page) = (h = heading_ids(joinpath(SRC, page)); isempty(h) || first(first(h)) != 1 ? nothing : last(first(h)))
    all_pages = [relpath(joinpath(r, f), SRC) for (r, _, fs) in walkdir(SRC) for f in fs if endswith(f, ".md")]
    titles = Dict(title_id(p) => p for p in all_pages if !isnothing(title_id(p)))
    # A page being dissolved: the page that replaces it carries its title's id as `<id>__next`.
    heir(page) = (id = title_id(page); isnothing(id) ? nothing : get(titles, id * "__next", nothing))
    for (chapter, c) in ("Language models" => ProviderFacts.llm, "Decisions" => ProviderFacts.decisions)
        for (group, pages) in groups(chapter)
            kept = filter(p -> isnothing(heir(p)), pages)
            entries = [e for e in REGISTRY if e.chapter == c && first(e.pages) in kept]
            # a dissolving page sits right after the page that replaces it
            @test [p for (i, p) in pairs(pages) if !isnothing(heir(p)) && get(pages, i - 1, nothing) != heir(p)] == []
            startswith(group, "Concepts") && @test kept == CONCEPT_PAGES[c]
            isempty(entries) || @test kept == reduce(vcat, (e.pages for e in entries))
        end
        @test count(((g, _),) -> startswith(g, "Concepts"), groups(chapter)) == 1
        @test all(e -> e.chapter != c || any(((_, ps),) -> first(e.pages) in ps, groups(chapter)), REGISTRY)
    end
    # an entry's start id is the title of its first page, or that page's heir's while the old page stays
    @test [e.key => title_id(first(e.pages)) for e in REGISTRY if title_id(first(e.pages)) ∉ (e.start_id, e.start_id * "__next")] == []
end
@testset "ProviderFacts — every other page of an entry is titled <entry>_<stem>_page, or keeps an id of v0.26.0" begin
    old = Set(readlines(joinpath(@__DIR__, "anchors-v0.26.0.txt")))        # every explicit id of v0.26.0's pages
    title_id(page) = (h = heading_ids(joinpath(SRC, page)); isempty(h) || first(first(h)) != 1 ? nothing : last(first(h)))
    own(e, page) = string(e.key, '_', first(splitext(basename(page))), "_page")
    # a page that takes over a page of v0.26.0 keeps that page's id: `<id>__next` while the old page stays, then `<id>`
    kept(id) = chopsuffix(id, "__next") in old
    @test [(e.key, p) => id for e in REGISTRY for p in e.pages[2:end]
           for id in (title_id(p),) if isnothing(id) || id != own(e, p) && !kept(id)] == []
end
