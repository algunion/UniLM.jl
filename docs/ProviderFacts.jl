"""
    ProviderFacts

The manual's provider entries and the fact tables their pages embed. The registry holds what
only the manual knows — an entry's pages, section ids and the words that name it in prose;
every default model, capability, request rule and price comes from UniLM (the prices as
`DEFAULT_PRICING` holds them when this file is loaded), so no page restates one by hand.

Pages call the table functions from `@eval` blocks, which Documenter runs in a module of their
own: `Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)`. Each returns a `Markdown.MD`
holding one table and nothing else. A cell is a `Vector{Any}` of inline elements — text, inline
code, `@ref` links —, the form Documenter converts without raw values, and rows come in a fixed
order, so a table generated twice is the same bytes.

The same entries fill the generated regions of `docs/src/llm.md` and `llms.txt`, which
`docs/regions.jl` rewrites, so readers of those files as they are see the facts too.
"""
module ProviderFacts

using Markdown
using Printf: @sprintf
using UniLM

# The site writer's rule for the route a page is served at, which `llms.txt` links need.
isdefined(Main, :SiteWriter) || Base.include(Main, joinpath(@__DIR__, "site_writer", "SiteWriter.jl"))
using Main.SiteWriter: route

export Chapter, ProviderEntry, NEUTRAL_TOKENS, REGISTRY, CONCEPT_PAGES, entry, provider_table, at_a_glance,
       capability_matrix, on_your_provider, openai_model_rules, claude_family_table, gemini_thinking_table,
       price_rows, price_table, claude_price_rules, openai_price_terms, openai_audio_rates, region_text,
       apply_regions, stale_regions

# ── Table cells ──────────────────────────────────────────────────────────────

# What a cell holds: text, inline code and links.
const _Inline = Union{String,Markdown.Code,Markdown.Link}

# One table cell: its inline parts, at least one, as the `Vector{Any}` Documenter converts. A
# bare `String` cell would reach the site writer as raw values, which it refuses: every cell of
# this module is built here.
_cell(parts::Vector{<:_Inline})::Vector{Any} =
    isempty(parts) ? throw(ArgumentError("a table cell holds at least one inline element")) : Vector{Any}(parts)
_cell(parts::_Inline...)::Vector{Any} = _cell(collect(_Inline, parts))

_code(s::AbstractString)::Markdown.Code = Markdown.Code(String(s))
_ref(text::String, id::String)::Markdown.Link = Markdown.Link(Any[text], "@ref " * id)

_yes(b::Bool)::Vector{Any} = _cell(b ? "yes" : "—")
_model(::Nothing)::Vector{Any} = _cell("—")
_model(id::String)::Vector{Any} = _cell(_code(id))

# Each of `xs` as inline code, comma-separated; "—" when there are none.
function _codes(xs)::Vector{Any}
    isempty(xs) && return _cell("—")
    parts = _Inline[]
    for x in xs
        isempty(parts) || push!(parts, ", ")
        push!(parts, _code(string(x)))
    end
    _cell(parts)
end

# A `Markdown.MD` holding one left-aligned table: the `header` cells — a text, or inline code
# where a cell names a request field —, then `rows`, each a `Vector{Any}` of as many cells.
function _table(header::Vector{<:_Inline}, rows::Vector{Vector{Any}})::Markdown.MD
    n = length(header)
    all(r -> length(r) == n, rows) || throw(ArgumentError("every row of a $n-column table has $n cells"))
    Markdown.MD(Any[Markdown.Table(Vector{Any}[Any[_cell(h) for h in header], rows...], fill(:l, n))])
end

# ── The registry ─────────────────────────────────────────────────────────────

"""
    Chapter

A chapter of the manual with provider entries: `llm` (language models) or `decisions`
(decision models).
"""
@enum Chapter llm decisions

"""
    ProviderEntry

One provider entry of the manual.

- `key`: the entry, e.g. `:anthropic`; `chapter`: the chapter it belongs to.
- `provider`: the vendor; entries of one vendor in both chapters share it (`:openai` and
  `:openai_decisions` are `:openai`).
- `name`: what the manual calls the entry.
- `endpoints`: the endpoints the entry covers, primary first; the entry's capabilities are
  theirs together.
- `usage`: what code writes to select the provider — a type's name, or the constructor call
  of an instance.
- `key_vars`: the environment variables a reader sets to make a call — the credentials, or the
  host of a local server —, from UniLM's constants; the tables' key column.
- `env_vars`: every other environment variable the entry's code reads, from UniLM's constants;
  `<MODEL>` stands for the part of a name a model's id gives. A generated region line says the
  entry reads both lists, and the metric (docs/metrics.jl) counts both as its provider's words.
- `default_model`: the model a call that names none gets; `nothing` where it must name one.
- `pages`: the entry's pages under `docs/src`, start page first.
- `start_id`: the `@id` of the start page's title.
- `features`: each feature the entry has a section for, and that section's `@id`.
- `tokens`: case-sensitive words that name the provider in prose.
"""
@kwdef struct ProviderEntry
    key::Symbol
    chapter::Chapter
    provider::Symbol
    name::String
    endpoints::Vector{ServiceEndpointSpec}
    usage::String
    key_vars::Vector{String}
    env_vars::Vector{String}
    default_model::Union{String,Nothing}
    pages::Vector{String}
    start_id::String
    features::Dict{Symbol,String}
    tokens::Vector{String}
end

"""
    RegistryError(key, field, message)

Thrown when this file is loaded and the registry breaks one of its rules: `key` names the
entry, `field` the field at fault.
"""
struct RegistryError <: Exception
    key::Symbol
    field::Symbol
    message::String
end
Base.showerror(io::IO, e::RegistryError) =
    print(io, "RegistryError: entry :", e.key, ", field ", e.field, ": ", e.message)

# An entry's feature sections: `<key>_<feature>` for each of `own`, and for each inherited
# feature the id its section keeps from the page it moves from.
function _features(key::Symbol, own::Tuple{Vararg{Symbol}}, inherited::Pair{Symbol,String}...)::Dict{Symbol,String}
    sections = [Pair{Symbol,String}[f => string(key, '_', f) for f in own]; Pair{Symbol,String}[inherited...]]
    twice = unique!([f for (f, _) in sections if count(s -> first(s) === f, sections) > 1])
    isempty(twice) || throw(RegistryError(key, :features, "$(join(repr.(twice), ", ")) listed twice"))
    Dict{Symbol,String}(sections)
end

# The model a System One protocol's `ask` sends when a call names none. TypeSafe's is the
# package constant, not `default_typesafe_model()`, which reads TYPESAFE_DEFAULT_MODEL from the
# environment of the process that loads this file.
_decision_default(::UniLM.TypeSafeProtocol)::String = UniLM.TYPESAFE_DEFAULT_MODEL
_decision_default(::UniLM.OpenAIDecisionsProtocol)::String = UniLM._DECISIONS_MODEL
_decision_default(::UniLM.OllamaSystemOneProtocol)::Nothing = nothing     # a call names an installed model

# The code's default for an entry: its primary endpoint's chat default, or for a decision entry
# the default of the System One protocol that endpoint speaks.
_code_default(e::ProviderEntry)::Union{String,Nothing} =
    e.chapter == llm ? UniLM.default_model(first(e.endpoints)) :
                       _decision_default(UniLM.system_one_protocol(first(e.endpoints)))

# `entries` once every rule holds, else a `RegistryError` naming the entry and the field: keys
# are unique; an entry has an endpoint and its pages are `.md` paths under docs/src; its
# default model is the code's; and every start and feature id is final (no `__next`) and
# named once across the registry.
function _validated(entries::Vector{ProviderEntry})::Vector{ProviderEntry}
    seen = Set{Symbol}()
    owner = Dict{String,Symbol}()              # a section id => the entry that names it
    for e in entries
        e.key in seen && throw(RegistryError(e.key, :key, "another entry has the same key"))
        push!(seen, e.key)
        isempty(e.endpoints) && throw(RegistryError(e.key, :endpoints, "an entry covers at least one endpoint"))
        isempty(e.pages) && throw(RegistryError(e.key, :pages, "an entry has at least its start page"))
        for p in e.pages
            endswith(p, ".md") && !isabspath(p) ||
                throw(RegistryError(e.key, :pages, "$(repr(p)) is not a .md path under docs/src"))
        end
        code = _code_default(e)
        isequal(e.default_model, code) || throw(RegistryError(e.key, :default_model,
            "$(repr(e.default_model)) is not the code's default, $(repr(code))"))
        for (field, id) in [:start_id => e.start_id; [:features => i for i in sort!(collect(values(e.features)))]]
            endswith(id, "__next") && throw(RegistryError(e.key, field, "$(repr(id)) is not a final id"))
            other = get(owner, id, nothing)
            isnothing(other) || throw(RegistryError(e.key, field, "$(repr(id)) is already an id of :$other"))
            owner[id] = e.key
        end
    end
    entries
end

# The model-id prefixes of each provider with a price table: UniLM's shared table for the three
# it names; DeepSeek's and Jev's, which only the manual needs, here.
const _MODEL_PREFIXES = (openai=UniLM._MODEL_ID_PREFIXES.openai, anthropic=UniLM._MODEL_ID_PREFIXES.anthropic,
                         gemini=UniLM._MODEL_ID_PREFIXES.gemini, deepseek=("deepseek-",), jev=("jev-",))

# The words that name an OpenAI model in prose, for both entries that serve OpenAI's models: `GPT-`
# as names are written, the shared table's prefixes, and those of the audio and image models, which
# that table (prefixes a price row or a request rule needs) leaves out.
const _OPENAI_MODELS = ["GPT-", _MODEL_PREFIXES.openai..., "whisper-", "tts-", "dall-e-"]

# The server `OllamaEndpoint()` reaches when OLLAMA_HOST is unset, for both Ollama entries, named
# explicitly: `OllamaEndpoint()` itself would read OLLAMA_HOST from the environment that loads
# this file.
const _OLLAMA = OllamaEndpoint(base_url=UniLM._ollama_host_url(""))

"""
    REGISTRY

The manual's provider entries, language models first; every table lists entries in this order.
Checked when this file is loaded: a broken rule throws a `RegistryError` naming the entry and
the field.
"""
const REGISTRY = _validated([
    ProviderEntry(key=:openai, chapter=llm, provider=:openai, name="OpenAI",
        endpoints=[OPENAIServiceEndpoint], usage="OPENAIServiceEndpoint", key_vars=[UniLM.OPENAI_API_KEY],
        env_vars=String[], default_model=UniLM.default_model(OPENAIServiceEndpoint),
        pages=["providers/openai/index.md", "providers/openai/models.md", "providers/openai/chat_completions.md",
               "providers/openai/responses.md", "providers/openai/retrieval.md", "providers/openai/media.md",
               "providers/openai/platform.md"],
        start_id="openai_start",
        features=_features(:openai, (:setup, :first_call, :conversations, :streaming, :tools, :structured_output,
            :reasoning, :media, :embeddings, :images, :audio, :caching, :cost, :limits, :failures, :models)),
        tokens=["OpenAI", "ChatGPT", "OPENAIServiceEndpoint", _OPENAI_MODELS...]),
    ProviderEntry(key=:anthropic, chapter=llm, provider=:anthropic, name="Anthropic",
        endpoints=[ANTHROPICServiceEndpoint], usage="ANTHROPICServiceEndpoint", key_vars=[UniLM.ANTHROPIC_API_KEY],
        env_vars=String[], default_model=UniLM.default_model(ANTHROPICServiceEndpoint),
        pages=["providers/anthropic/index.md", "providers/anthropic/requests.md", "providers/anthropic/caching.md"],
        start_id="anthropic_start",
        features=_features(:anthropic, (:setup, :first_call, :conversations, :streaming, :tools, :structured_output,
            :reasoning, :media, :caching, :cost, :limits, :failures, :models)),
        tokens=["Anthropic", "Claude", "ANTHROPICServiceEndpoint", _MODEL_PREFIXES.anthropic...]),
    ProviderEntry(key=:gemini, chapter=llm, provider=:gemini, name="Google Gemini",
        endpoints=[GEMINIServiceEndpoint, GEMINIOpenAIServiceEndpoint], usage="GEMINIServiceEndpoint",
        key_vars=[UniLM.GEMINI_API_KEY], env_vars=String[], default_model=UniLM.default_model(GEMINIServiceEndpoint),
        pages=["providers/gemini/index.md", "providers/gemini/generate_content.md", "providers/gemini/interactions.md",
               "providers/gemini/media.md"],
        start_id="gemini_start",
        features=_features(:gemini, (:setup, :first_call, :conversations, :streaming, :tools, :structured_output,
            :reasoning, :media, :embeddings, :images, :audio, :caching, :cost, :limits, :failures, :models)),
        tokens=["Gemini", "Google", "GEMINIServiceEndpoint", "GEMINIOpenAIServiceEndpoint", _MODEL_PREFIXES.gemini...]),
    ProviderEntry(key=:ollama, chapter=llm, provider=:ollama, name="Ollama",
        endpoints=[_OLLAMA], usage="OllamaEndpoint()", key_vars=[UniLM.OLLAMA_HOST],
        env_vars=[UniLM.OLLAMA_API_KEY],   # read for a request to Ollama's cloud, https://ollama.com
        default_model=nothing, pages=["providers/ollama/index.md", "providers/ollama/embeddings.md", "providers/ollama/runtime.md"],
        start_id="ollama_guide",
        features=_features(:ollama, (:setup, :conversations, :streaming, :limits, :cost),
            :first_call => "ollama_first", :reasoning => "ollama_thinking", :tools => "ollama_tools",
            :structured_output => "ollama_json", :media => "ollama_media", :audio => "ollama_transcription",
            :embeddings => "ollama_embeddings", :models => "ollama_models", :failures => "ollama_failures",
            :completions => "ollama_respond"),
        tokens=["Ollama", "OllamaEndpoint"]),
    ProviderEntry(key=:local_servers, chapter=llm, provider=:local_servers, name="vLLM, LM Studio & llama.cpp",
        endpoints=[GenericOpenAIEndpoint("http://localhost:8000", "")],
        usage="GenericOpenAIEndpoint(\"http://localhost:8000\", \"\")", key_vars=String[], env_vars=String[],
        default_model=nothing,
        pages=["providers/local_servers.md"], start_id="local_servers_start",
        features=_features(:local_servers, (:setup, :first_call, :streaming, :tools, :structured_output, :limits,
            :failures, :embeddings, :completions)),
        tokens=["vLLM", "LM Studio", "llama.cpp", "llama-server"]),
    ProviderEntry(key=:azure, chapter=llm, provider=:azure, name="Azure OpenAI",
        endpoints=[AZUREServiceEndpoint], usage="AZUREServiceEndpoint",
        key_vars=[UniLM.AZURE_OPENAI_API_KEY, UniLM.AZURE_OPENAI_BASE_URL, UniLM.AZURE_OPENAI_API_VERSION],
        env_vars=[UniLM._azure_deploy_env_var("") * "<MODEL>"],     # read unless add_azure_deploy_name! registered one
        default_model=UniLM.default_model(AZUREServiceEndpoint),
        pages=["providers/azure.md"], start_id="azure_start",
        features=_features(:azure, (:setup, :first_call, :conversations, :streaming, :tools, :structured_output,
            :reasoning, :media, :caching, :cost, :limits, :failures, :models)),
        tokens=["Azure OpenAI", "Azure", "AZUREServiceEndpoint", "AZURE_OPENAI_", _OPENAI_MODELS...]),
    ProviderEntry(key=:deepseek, chapter=llm, provider=:deepseek, name="DeepSeek",
        endpoints=[DeepSeekEndpoint()], usage="DeepSeekEndpoint()", key_vars=[UniLM.DEEPSEEK_API_KEY],
        env_vars=String[], default_model=UniLM.default_model(DeepSeekEndpoint()),
        pages=["providers/deepseek.md"], start_id="deepseek_start",
        features=_features(:deepseek, (:setup, :first_call, :conversations, :streaming, :tools, :structured_output,
            :reasoning, :media, :completions, :caching, :cost, :limits, :failures, :models)),
        tokens=["DeepSeek", "DeepSeekEndpoint", _MODEL_PREFIXES.deepseek...]),
    ProviderEntry(key=:openai_compatible, chapter=llm, provider=:mistral, name="Mistral & OpenAI-compatible services",
        endpoints=[MistralEndpoint()], usage="MistralEndpoint()", key_vars=[UniLM.MISTRAL_API_KEY],
        env_vars=String[], default_model=nothing,
        pages=["providers/openai_compatible.md"], start_id="openai_compatible_start",
        features=_features(:openai_compatible, (:setup, :first_call, :conversations, :streaming, :tools,
            :structured_output, :media, :completions, :embeddings, :audio, :cost, :limits, :failures, :models)),
        tokens=["Mistral", "MistralEndpoint", "mistral-", "codestral-"]),
    ProviderEntry(key=:jev, chapter=decisions, provider=:typesafe, name="Jev (TypeSafe)",
        endpoints=[TYPESAFEServiceEndpoint], usage="TYPESAFEServiceEndpoint", key_vars=[UniLM.TYPESAFE_API_KEY],
        env_vars=[UniLM.TYPESAFE_BASE_URL_ENV, UniLM.TYPESAFE_DEFAULT_MODEL_ENV], default_model=UniLM.TYPESAFE_DEFAULT_MODEL,
        pages=["decisions/jev/index.md", "decisions/jev/abilities.md", "decisions/jev/operations.md"],
        start_id="jev_start",
        features=_features(:jev, (:first_decision, :questions, :confidence, :testing, :throughput),
            :setup => "jev_setup", :models => "jev_models", :limits => "jev_limits", :failures => "jev_errors",
            :cost => "jev_cost", :images => "jev_images"),
        tokens=["Jev", "TypeSafe", "TYPESAFEServiceEndpoint", "TYPESAFE_", _MODEL_PREFIXES.jev...]),
    ProviderEntry(key=:openai_decisions, chapter=decisions, provider=:openai, name="OpenAI Decisions API",
        endpoints=[OPENAIServiceEndpoint], usage="OPENAIServiceEndpoint", key_vars=[UniLM.OPENAI_API_KEY],
        env_vars=String[], default_model=UniLM._DECISIONS_MODEL,
        pages=["decisions/openai.md"], start_id="openai_decisions_start",
        features=_features(:openai_decisions, (:setup, :first_decision, :models, :questions, :limits, :confidence,
            :cost, :testing, :failures)),
        tokens=["Decisions API"]),
    ProviderEntry(key=:ollama_decisions, chapter=decisions, provider=:ollama, name="Local decision models",
        endpoints=[_OLLAMA], usage="OllamaEndpoint()", key_vars=[UniLM.OLLAMA_HOST],
        env_vars=[UniLM.OLLAMA_API_KEY],   # read for a request to Ollama's cloud, https://ollama.com
        default_model=nothing, pages=["decisions/ollama.md"], start_id="ollama_decisions_start",
        features=_features(:ollama_decisions, (:setup, :first_decision, :models, :questions, :limits, :confidence,
            :images, :testing, :throughput, :failures)),
        tokens=String[]),
])

"""
    NEUTRAL_TOKENS

Words that look like a provider's and name none: the endpoint types every provider shares, and
the OpenAI-compatible wire, which many providers speak — as "compatible", "compatibility",
"compat" or "wire", after a hyphen or a space, lower case or capitalised as in a title.
"""
const NEUTRAL_TOKENS = ["GenericOpenAIEndpoint", "OpenAIWireEndpoint", "ServiceEndpoint",
                        "OpenAI-compatible", "OpenAI compatible", "OpenAI-Compatible", "OpenAI Compatible",
                        "OpenAI-compatibility", "OpenAI compatibility", "OpenAI-Compatibility", "OpenAI Compatibility",
                        "OpenAI-compat", "OpenAI compat", "OpenAI-Compat", "OpenAI Compat",
                        "OpenAI-wire", "OpenAI wire", "OpenAI-Wire", "OpenAI Wire"]

"""
    CONCEPT_PAGES

Each chapter's concept pages under `docs/src`: the provider-free pages its entries link to.
"""
const CONCEPT_PAGES = Dict{Chapter,Vector{String}}(
    llm => ["guide/chat_completions.md", "guide/agentic.md", "guide/tool_calling.md", "guide/streaming.md",
            "guide/structured_output.md", "guide/attachments.md", "guide/embeddings.md", "guide/image_generation.md",
            "guide/completions.md", "guide/results.md", "guide/cost_tracking.md", "guide/timeouts.md",
            "guide/concurrency.md", "guide/testing.md", "guide/mcp.md"],
    decisions => ["guide/system_one.md", "guide/natural_language_dispatch.md", "guide/semantic_algorithms.md",
                  "guide/jev_with_llms.md", "guide/jev_testing.md"])

const _ENTRIES = Dict{Symbol,ProviderEntry}(e.key => e for e in REGISTRY)

"""
    entry(key::Symbol) -> ProviderEntry

The registry entry `key`; an `ArgumentError` naming the entries when there is none.
"""
entry(key::Symbol)::ProviderEntry = get(_ENTRIES, key) do
    throw(ArgumentError("no provider entry :$key; the entries are $(join((repr(e.key) for e in REGISTRY), ", "))"))
end

# ── Provider tables ──────────────────────────────────────────────────────────

# Every capability an endpoint of the registry declares, sorted.
const _CAPABILITIES = sort!(unique!([c for e in REGISTRY for ep in e.endpoints for c in provider_capabilities(ep)]))

# The capabilities only the Decisions chapter documents — `system_one`: serving decision models
# through `ask` —, which a language-model entry's tables leave out.
const _DECISIONS_ONLY = (:system_one,)

"""
    DECISION_CAPABILITIES

The capabilities a decision entry's at-a-glance table reports: all of `TYPESAFEServiceEndpoint`'s,
the endpoint that serves decisions and nothing else, sorted.
"""
const DECISION_CAPABILITIES = sort!(collect(provider_capabilities(TYPESAFEServiceEndpoint)))

# The capabilities of `_CAPABILITIES` that some endpoint of the entry has.
_capabilities(e::ProviderEntry)::Vector{Symbol} =
    filter(c -> any(ep -> has_capability(ep, c), e.endpoints), _CAPABILITIES)

# Whether the entry's capabilities are the code's permissive default, `GenericOpenAIEndpoint`'s: UniLM
# sends those requests to whatever server the endpoint names, which may serve them or not, so the tables
# say "sent unchecked" where they say "yes" or "Supports" for an endpoint whose set UniLM checks.
_unchecked(e::ProviderEntry)::Bool = any(ep -> ep isa GenericOpenAIEndpoint, e.endpoints)

# A capability cell: "sent unchecked: " where `unchecked` and `caps` is not empty, then each of `caps` as
# inline code; "—" when `caps` is empty.
function _capability_cell(caps::Vector{Symbol}; unchecked::Bool=false)::Vector{Any}
    parts = _Inline[]
    unchecked && !isempty(caps) && push!(parts, "sent unchecked: ")
    for (i, c) in pairs(caps)
        i > 1 && push!(parts, ", ")
        push!(parts, _code(string(c)))
    end
    isempty(parts) ? _cell("—") : _cell(parts)
end

# How code names each endpoint of an entry: the primary's usage, then the others' types.
_endpoints(e::ProviderEntry)::Vector{Any} =
    _codes([e.usage; [String(nameof(ep isa Type ? ep : typeof(ep))) for ep in e.endpoints[2:end]]])

"""
    provider_table(chapter::Chapter) -> Markdown.MD

One row per entry of `chapter`, in registry order — Provider | Key or host | Default model |
Start here: the entry's name, the variables a reader sets to make a call (`key_vars`; "—":
none), the model a call that names none gets ("—": a call names one), and the name again,
linking the start page.
"""
function provider_table(chapter::Chapter)::Markdown.MD
    rows = Vector{Any}[Any[_cell(e.name), _codes(e.key_vars), _model(e.default_model), _cell(_ref(e.name, e.start_id))]
                       for e in REGISTRY if e.chapter == chapter]
    _table(["Provider", "Key or host", "Default model", "Start here"], rows)
end

"""
    at_a_glance(key::Symbol) -> Markdown.MD

The entry's facts in one row — Endpoint | Key variables | Default model | Supports, and for a
language-model entry Not available: how code names each endpoint (primary first), the
variables a reader sets to make a call (`key_vars`), its default model ("—": a call names one), the
capabilities some endpoint of it has (for a decision entry, those of `DECISION_CAPABILITIES`),
and the capabilities other registry endpoints declare that none of its endpoints has. A
language-model entry's two capability cells leave out the capabilities only the Decisions
chapter documents (`system_one`: serving decision models); that chapter's entries list them.
The Supports cell of an entry whose capabilities are `GenericOpenAIEndpoint`'s permissive default
starts "sent unchecked: ": UniLM sends those requests without knowing whether the server serves them.
"""
function at_a_glance(key::Symbol)::Markdown.MD
    e = entry(key)
    has = _capabilities(e)
    header = ["Endpoint", "Key variables", "Default model", "Supports"]
    row = Any[_endpoints(e), _codes(e.key_vars), _model(e.default_model)]
    if e.chapter == llm
        push!(header, "Not available")
        push!(row, _capability_cell(filter(∉(_DECISIONS_ONLY), has); unchecked=_unchecked(e)),
              _capability_cell(setdiff(_CAPABILITIES, has, _DECISIONS_ONLY)))
    else
        push!(row, _codes(filter(in(DECISION_CAPABILITIES), has)))
    end
    _table(header, Vector{Any}[row])
end

"""
    capability_matrix() -> Markdown.MD

One row per capability a registry endpoint declares, sorted, and one column per
language-model entry, in registry order: "yes" where some endpoint of the entry has the
capability — "sent unchecked" for an entry whose capabilities are `GenericOpenAIEndpoint`'s
permissive default, whose requests UniLM sends without knowing whether the server serves them —,
else "—". It has no row for a capability only the Decisions chapter documents (`system_one`: serving
decision models), which no language-model entry lists.
"""
function capability_matrix()::Markdown.MD
    entries = filter(e -> e.chapter == llm, REGISTRY)
    has = [Set(_capabilities(e)) for e in entries]
    mark(e, yes) = yes ? _cell(_unchecked(e) ? "sent unchecked" : "yes") : _cell("—")
    rows = Vector{Any}[Any[_cell(_code(String(c))), (mark(e, c in h) for (e, h) in zip(entries, has))...]
                       for c in _CAPABILITIES if c ∉ _DECISIONS_ONLY]
    _table(["Capability"; [e.name for e in entries]], rows)
end

"""
    on_your_provider(chapter::Chapter, feature::Symbol) -> Markdown.MD

One row per entry of `chapter` with a section for `feature`, in registry order: the entry's
name, linking that section. An `ArgumentError` when no entry of `chapter` has one.
"""
function on_your_provider(chapter::Chapter, feature::Symbol)::Markdown.MD
    rows = Vector{Any}[Any[_cell(_ref(e.name, e.features[feature]))]
                       for e in REGISTRY if e.chapter == chapter && haskey(e.features, feature)]
    isempty(rows) && throw(ArgumentError("no $chapter entry has a section for the feature :$feature"))
    _table(["Provider"], rows)
end

# ── Model rule tables ────────────────────────────────────────────────────────

# Reasoning efforts: `nothing` is any effort, an empty list none.
_efforts(::Nothing)::Vector{Any} = _cell("any")
_efforts(efforts::Vector{String})::Vector{Any} = _codes(efforts)

"""
    openai_model_rules() -> Markdown.MD

One row per model with request rules on OpenAI's native endpoint, sorted by id: the reasoning
efforts the Responses API and Chat Completions take; the efforts at which the model takes
sampling controls and log probabilities, and function tools on Chat Completions ("any": every
effort, "—": none); whether it serves the Ultrafast tier and takes `prompt_cache_retention`;
and the effort of a request that sets none.
"""
function openai_model_rules()::Markdown.MD
    rows = Vector{Any}[Any[_cell(_code(id)), _efforts(r.efforts), _efforts(r.chat_efforts),
                           _efforts(r.sampling_efforts), _efforts(r.chat_tool_efforts), _yes(r.ultrafast),
                           _yes(r.prompt_cache_retention), _cell(_code(r.default_effort))]
                       for (id, r) in sort!(collect(UniLM._OPENAI_MODEL_RULES); by=first)]
    _table(_Inline["Model", "Efforts on Responses", "Efforts on Chat Completions", "Sampling and log probabilities at",
                   "Chat Completions tools at", "Ultrafast tier", _code("prompt_cache_retention"), "Default effort"], rows)
end

# A Claude row's thinking mode, as the page reads it.
function _thinking(mode::Symbol)::Vector{Any}
    mode === :always && return _cell("adaptive; cannot be disabled")
    mode === :on && return _cell("adaptive by default; ", _code("disabled"), " turns it off")
    mode === :between && return _cell("adaptive by default, cannot be disabled; ", _code("between_tools"),
                                      " turns up-front thinking off")
    mode === :off && return _cell("adaptive on request")
    mode === :manual && return _cell("extended thinking (", _code("budget_tokens"), ") only")
    throw(ArgumentError("no page text for the Claude thinking mode $(repr(mode))"))
end

# A Claude row's fast mode, as the page reads it.
function _fast(status::Symbol)::Vector{Any}
    status === :ga && return _cell("yes")
    status === :beta && return _cell("research preview, with the ", _code(UniLM._ANTHROPIC_FAST_BETA), " beta header")
    status === :none && return _cell("—")
    throw(ArgumentError("no page text for the Claude fast mode $(repr(status))"))
end

"""
    claude_family_table() -> Markdown.MD

One row per Claude model with request rules, sorted by id: its thinking mode, the effort
levels it takes ("—": no effort parameter), and whether it takes a forced tool choice, a
trailing assistant turn (prefill), system messages after the conversation starts, checks the
prefix before a thinking block, serves fast mode, and takes `inference_geo`.
"""
function claude_family_table()::Markdown.MD
    rows = Vector{Any}[Any[_cell(_code(id)), _thinking(f.thinking), _codes(f.efforts), _yes(f.forced), _yes(f.prefill),
                           _yes(f.system), _yes(f.prefix_checked), _fast(f.fast), _yes(f.inference_geo)]
                       for (id, f) in sort!(collect(UniLM._CLAUDE_FAMILIES); by=first)]
    _table(_Inline["Model", "Thinking", "Effort levels", "Forced tool choice", "Prefill",
                   "Mid-conversation system messages", "Thinking prefix check", "Fast mode", _code("inference_geo")], rows)
end

# The refusal the generateContent encoder's check gives every thinking level of Gemini 2.5, which
# takes a thinkingBudget there instead.
_budget_refusal(err)::Bool = err isa ArgumentError && occursin("generateContent uses thinkingBudget", err.msg)

# A model's thinking levels on generateContent: those the check its encoder runs
# (`_gemini_thinking_level(…; native=true)`) accepts, or "—" and the check's own reason where it
# refuses them all. Only that refusal is caught; any other error propagates.
function _generate_content_cell(model::String, levels::Vector{String})::Vector{Any}
    taken = String[]
    reason = nothing
    for level in levels
        try
            push!(taken, UniLM._gemini_thinking_level(model, level; native=true))
        catch err
            _budget_refusal(err) || rethrow()
            reason = err.msg
        end
    end
    isempty(taken) && !isnothing(reason) ? _cell("— ($reason)") : _codes(taken)
end

"""
    gemini_thinking_table() -> Markdown.MD

One row per Gemini model with thinking levels, sorted by id: the levels `respond` takes on the
Interactions API, and those `Chat` takes on generateContent, as the check that encoder runs
answers — where it refuses them all, "—" and the reason it gives.
"""
function gemini_thinking_table()::Markdown.MD
    rows = Vector{Any}[Any[_cell(_code(id)), _codes(levels), _generate_content_cell(id, levels)]
                       for (id, levels) in sort!(collect(UniLM._GEMINI_THINKING_LEVELS); by=first)]
    _table(["Model", "Levels with respond (Interactions)", "Levels with Chat (generateContent)"], rows)
end

# ── Price tables ─────────────────────────────────────────────────────────────

# Priced model ids that carry no provider's prefix, and the price table each belongs to.
const _UNPREFIXED = ("chat-latest" => :openai,)

# The package's price rows, read once: an example that adds or replaces a row while the manual
# builds leaves the documented prices as UniLM ships them.
const _PRICES = Dict{String,UniLM.PriceRow}(DEFAULT_PRICING)

"""
    PriceTableError(id, tables)

Thrown when this file is loaded and the priced model `id` is in no price table, or in more than
one: `tables` lists the tables it is in.
"""
struct PriceTableError <: Exception
    id::String
    tables::Vector{Symbol}
end
Base.showerror(io::IO, e::PriceTableError) =
    print(io, "PriceTableError: the DEFAULT_PRICING row ", repr(e.id), " is in ",
          isempty(e.tables) ? "no price table" : "the price tables " * join(e.tables, " and "),
          ": give it one table's prefix, or one unprefixed entry")

# Each price table's model ids in `pricing`, sorted. An id belongs to the table whose prefix it
# carries, or to the one `unprefixed` gives it; an id in no table, or in two, is a
# `PriceTableError`, and an unprefixed id that `pricing` lacks an `ArgumentError`.
function _price_partition(pricing::AbstractDict{String},
                          unprefixed::Tuple{Vararg{Pair{String,Symbol}}}=_UNPREFIXED)::Dict{Symbol,Vector{String}}
    ids = Dict{Symbol,Vector{String}}(t => String[] for t in keys(_MODEL_PREFIXES))
    for id in keys(pricing)
        tables = Symbol[t for (t, prefixes) in pairs(_MODEL_PREFIXES) if any(p -> startswith(id, p), prefixes)]
        append!(tables, (t for (u, t) in unprefixed if u == id))
        length(tables) == 1 || throw(PriceTableError(id, tables))
        push!(ids[only(tables)], id)
    end
    for (id, _) in unprefixed
        haskey(pricing, id) || throw(ArgumentError("the unprefixed price id $(repr(id)) has no DEFAULT_PRICING row"))
    end
    foreach(sort!, values(ids))
    ids
end

const _PRICE_IDS = _price_partition(_PRICES)

# A per-token USD rate in dollars per million tokens, to the cent at least: 7.5e-8 → "0.075",
# 1.0e-5 → "10.00".
function _per_million(rate::Float64)::String
    whole, frac = split(@sprintf("%.6f", rate * 1_000_000), '.')
    string(whole, '.', rpad(rstrip(frac, '0'), 2, '0'))
end

# A price table row: the model id, then its input, cached-input and output rates.
_price_row(id::String, r::UniLM.PriceRow)::Vector{Any} =
    Any[_cell(_code(id)), (_cell(_per_million(rate)) for rate in (r.input, r.cached_input, r.output))...]

"""
    price_rows(table::Symbol) -> Vector{String}

The `DEFAULT_PRICING` model ids of the price table `table` — `:openai`, `:anthropic`, `:gemini`,
`:deepseek` or `:jev` —, sorted, in a new vector. Every priced id is in exactly one table.
"""
function price_rows(table::Symbol)::Vector{String}
    haskey(_PRICE_IDS, table) ||
        throw(ArgumentError("no price table :$table; the tables are $(join(repr.(keys(_MODEL_PREFIXES)), ", "))"))
    copy(_PRICE_IDS[table])
end

"""
    price_table(table::Symbol) -> Markdown.MD

One row per model of `price_rows(table)`, sorted by id: its input, cached-input and output
rates in USD per million tokens, as `DEFAULT_PRICING` held them when this file was loaded.
"""
function price_table(table::Symbol)::Markdown.MD
    _table(["Model", "Input (USD per 1M tokens)", "Cached input (USD per 1M tokens)", "Output (USD per 1M tokens)"],
           Vector{Any}[_price_row(id, _PRICES[id]) for id in price_rows(table)])
end

# A whole number with a comma between each group of three digits: 100000 → "100,000".
_thousands(n::Integer)::String = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")
# A multiplier as written in prose: 5.0 → "5", 2.5 → "2.5".
_factor(x::Real)::String = isinteger(x) ? string(Int(x)) : string(x)

"""
    claude_price_rules() -> Markdown.MD

One row per Claude model that `estimated_cost` does not price at its row's rates alone, sorted
by id: a model of UniLM's family table (`UniLM._CLAUDE_FAMILIES`) with no row in the prices
this file loaded, estimated at `0.0` with a warning, and a model with a long-prompt multiplier
(`UniLM._CLAUDE_LONG_PROMPT_TIERS`), with the prompt length over which every rate is
multiplied, and by how much.
"""
function claude_price_rules()::Markdown.MD
    tiers = UniLM._CLAUDE_LONG_PROMPT_TIERS
    unpriced = [id for id in keys(UniLM._CLAUDE_FAMILIES) if !haskey(_PRICES, id)]
    estimate(id) = haskey(_PRICES, id) ?
        _cell("every rate × $(_factor(tiers[id].multiplier)) for a prompt over $(_thousands(tiers[id].threshold)) tokens") :
        _cell(_code("0.0"), ", with a warning: it has no price row")
    _table(["Model", "Estimated"],
           Vector{Any}[Any[_cell(_code(id)), estimate(id)] for id in sort!(unique!([collect(keys(tiers)); unpriced]))])
end

# A multiple as a cell writes it: 1.25 → "× 1.25", 2.0 → "× 2".
_times(x::Real)::String = "× " * _factor(x)

# The multiples OpenAI's bill applies to a prompt over the short-context limit, read from the
# estimate itself: of the input rate (a prompt one token over the limit, no output), of the
# cached-input rate (the same prompt, all of it read from the cache) and of the output rate
# (one output token more).
function _openai_long_multiples(model::String)::NTuple{3,Float64}
    rates, n = _PRICES[model], UniLM._OPENAI_SHORT_CONTEXT + 1
    bill(; cached=0, out=0) = UniLM._billed_cost(OPENAIServiceEndpoint, model,
        UniLM.TokenUsage(prompt_tokens=n, completion_tokens=out, cached_tokens=cached), rates)
    round.((bill() / (n * rates.input), bill(cached=n) / (n * rates.cached_input),
            (bill(out=1) - bill()) / rates.output); digits=6)
end

# A billed tier's cell: the multiple of every rate it costs, for prompts up to the short-context
# limit and over it where the model has a long-context rate and the two differ; "—" where
# OpenAI's pricing page gives the tier no price.
function _openai_tier_cell(terms, tier::String)::Vector{Any}
    short, long = get(terms.tiers, tier, nothing), get(terms.long_tiers, tier, nothing)
    limit = _thousands(UniLM._OPENAI_SHORT_CONTEXT) * " input tokens"
    isnothing(short) && isnothing(long) && return _cell("—")
    (!terms.long_context || isequal(short, long)) && return _cell(_times(something(short, long)))
    isnothing(long) && return _cell(_times(short), " up to ", limit)
    isnothing(short) && return _cell(_times(long), " over ", limit)
    _cell(_times(short), " up to ", limit, ", ", _times(long), " over")
end

"""
    openai_price_terms() -> Markdown.MD

One row per OpenAI model that `estimated_cost` bills beyond its price row's rates
(`UniLM._OPENAI_PRICE_TERMS`), sorted by id: what a cache write costs for each lifetime, as a
multiple of the input rate; the multiples of the input, cached-input and output rates for a
prompt over the short-context limit, read from the estimate; and the multiple of every rate
that each billed service tier costs. "—": OpenAI's pricing page gives no such price.
"""
function openai_price_terms()::Markdown.MD
    limit = _thousands(UniLM._OPENAI_SHORT_CONTEXT)
    function writes(terms)::Vector{Any}
        w30, w12 = UniLM._cache_write_multiples(terms)
        w30 == w12 == 1 ? _cell("—") : _cell(_times(w30), " input for ", _code("\"30m\""), ", ",
                                             _times(w12), " for ", _code("\"12h\""))
    end
    function long(model, terms)::Vector{Any}
        terms.long_context || return _cell("—")
        input, cached, output = _openai_long_multiples(model)
        _cell("input ", _times(input), ", cached input ", _times(cached), ", output ", _times(output))
    end
    rows = Vector{Any}[Any[_cell(_code(id)), writes(t), long(id, t),
                           (_openai_tier_cell(t, tier) for tier in ("flex", "fast", "ultrafast"))...]
                       for (id, t) in sort!(collect(UniLM._OPENAI_PRICE_TERMS); by=first)]
    _table(["Model", "Cache writes", "Over $limit input tokens", "Flex", "Fast", "Ultrafast"], rows)
end

"""
    openai_audio_rates() -> Markdown.MD

One row per OpenAI model whose audio tokens `estimated_cost` prices (`UniLM._OPENAI_AUDIO_RATES`),
sorted by id: its audio input and audio output rates in USD per million tokens.
"""
openai_audio_rates()::Markdown.MD =
    _table(["Model", "Audio input (USD per 1M tokens)", "Audio output (USD per 1M tokens)"],
           Vector{Any}[Any[_cell(_code(id)), _cell(_per_million(r.input)), _cell(_per_million(r.output))]
                       for (id, r) in sort!(collect(UniLM._OPENAI_AUDIO_RATES); by=first)])

# ── Generated regions ────────────────────────────────────────────────────────

"""
    MarkerForm

How a file marks the regions `docs/regions.jl` writes: the lines of the marker that opens a
region and of the one that closes it, `NAME` standing for the region's name, and whether the
region links with `@ref`s, which Documenter resolves, or with the site's URLs, for a file read
as it is.
"""
struct MarkerForm
    opening::Vector{String}
    closing::Vector{String}
    refs::Bool
end

# Documenter's Markdown has no HTML — a comment would show as a paragraph —, so a page marks a
# region with `@setup` blocks, which run a comment and render nothing.
const _SETUP_BLOCKS = MarkerForm(["```@setup generated", "# generated:NAME (written by docs/regions.jl)", "```"],
                                 ["```@setup generated", "# /generated:NAME", "```"], true)
# A plain-text file marks a region with HTML comments.
const _HTML_COMMENTS = MarkerForm(["<!-- generated:NAME -->"], ["<!-- /generated:NAME -->"], false)

# Each generated region: the form of its markers, the heading its text opens with, and the
# chapters whose entries it lists.
const _REGIONS = Dict{String,@NamedTuple{form::MarkerForm, heading::String, chapters::Vector{Chapter}}}(
    "provider-contracts" => (form=_SETUP_BLOCKS, heading="## Provider Contracts", chapters=[llm]),
    "decision-provider-contracts" => (form=_SETUP_BLOCKS, heading="### Decision provider contracts",
                                      chapters=[decisions]),
    "providers" => (form=_HTML_COMMENTS, heading="## Providers", chapters=[llm, decisions]))

_region_names()::String = join(sort!(collect(keys(_REGIONS))), ", ")

# The development build of the manual, where the site serves each page at its route.
const _SITE = "https://algunion.github.io/UniLM.jl/dev"

# An entry's line: its name linking its start page, the `service` that selects it, the
# environment variables it reads — `key_vars` and `env_vars` — and its default model.
function _contract(e::ProviderEntry, refs::Bool)::String
    link = refs ? "@ref " * e.start_id : _SITE * route(first(e.pages))
    read = [e.key_vars; e.env_vars]
    vars = isempty(read) ? "no environment variable" : join(("`$v`" for v in read), ", ")
    model = isnothing(e.default_model) ? "no default model (a call names one)" : "default model `$(e.default_model)`"
    "- [$(e.name)]($link): `service = $(e.usage)`; reads $vars; $model"
end

"""
    region_text(name::String) -> String

The text of the generated region `name`: its heading, a blank line, and one line per entry of
its chapters in registry order — the entry's name linking its start page (an `@ref` in
`docs/src/llm.md`, the page's URL on the site in `llms.txt`), the `service` that selects it,
the environment variables it reads and its default model —, every line ending in a newline.
An `ArgumentError` naming the regions when there is no region `name`.
"""
function region_text(name::String)::String
    r = get(_REGIONS, name) do
        throw(ArgumentError("no generated region $(repr(name)); the regions are $(_region_names())"))
    end
    string(r.heading, "\n\n", (_contract(e, r.form.refs) * "\n" for e in REGISTRY if e.chapter in r.chapters)...)
end

"""
    RegionError(file, line, message)

Thrown when the region markers of `file` break a rule; `line` is the first line of the marker
at fault.
"""
struct RegionError <: Exception
    file::String
    line::Int
    message::String
end
Base.showerror(io::IO, e::RegionError) = print(io, "RegionError: ", e.file, ":", e.line, ": ", e.message)

# A marker in a file: its region's name, whether it opens the region, and its lines.
const _Marker = @NamedTuple{name::String, opens::Bool, span::UnitRange{Int}}

# Every marker a file may hold, as lines — the opening and the closing one of each region, in
# the region's form —, and what the first line of a marker of either form starts with.
const _MARKERS = [(name=name, opens=opens, lines=replace.(opens ? r.form.opening : r.form.closing, "NAME" => name))
                  for (name, r) in _REGIONS for opens in (true, false)]
const _MARKER_STARTS = unique!([String(first(split(first(m), "NAME")))
                                for f in (_SETUP_BLOCKS, _HTML_COMMENTS) for m in (f.opening, f.closing)])

# Whether line `j` of a text split at '\n' is blank; the piece after a final newline is no line.
_blank(lines::AbstractVector{<:AbstractString}, j::Int)::Bool = 1 <= j < length(lines) && isempty(lines[j])

# The marker at line `i`, or `nothing` when the line does not start like one. A `RegionError`
# when it does and the lines from it are not exactly a region's marker, or when the marker does
# not stand between blank lines.
function _marker(lines::AbstractVector{<:AbstractString}, i::Int, file::String)::Union{_Marker,Nothing}
    any(s -> startswith(lines[i], s), _MARKER_STARTS) || return nothing
    k = findfirst(m -> view(lines, i:min(lastindex(lines), i + length(m.lines) - 1)) == m.lines, _MARKERS)
    isnothing(k) && throw(RegionError(file, i, "$(repr(String(lines[i]))) starts like a marker and begins no " *
                                               "marker of the regions $(_region_names())"))
    m = _MARKERS[k]
    span = i:(i + length(m.lines) - 1)
    _blank(lines, first(span) - 1) && _blank(lines, last(span) + 1) ||
        throw(RegionError(file, i, "the marker of region $(m.name) does not stand between blank lines"))
    (name=m.name, opens=m.opens, span=span)
end

# The regions `lines` marks, in order: each one's name and the lines between its markers. A
# region unclosed, opened or closed inside another, or marked twice is a `RegionError` naming
# `file` and the line, as are the refusals of `_marker`.
function _regions(lines::AbstractVector{<:AbstractString}, file::String)::Vector{Pair{String,UnitRange{Int}}}
    regions = Pair{String,UnitRange{Int}}[]
    open::Union{_Marker,Nothing} = nothing              # the opening marker of the region being read
    for i in eachindex(lines)
        m = _marker(lines, i, file)
        isnothing(m) && continue
        if m.opens
            isnothing(open) || throw(RegionError(file, i,
                "region $(m.name) opens inside region $(open.name), opened at line $(first(open.span))"))
            any(r -> first(r) == m.name, regions) && throw(RegionError(file, i, "region $(m.name) is marked twice"))
            open = m
        else
            isnothing(open) && throw(RegionError(file, i, "region $(m.name) closes and is not open"))
            open.name == m.name || throw(RegionError(file, i,
                "region $(m.name) closes inside region $(open.name), opened at line $(first(open.span))"))
            push!(regions, m.name => (last(open.span) + 1):(first(m.span) - 1))
            open = nothing
        end
    end
    isnothing(open) || throw(RegionError(file, first(open.span), "region $(open.name) is not closed"))
    regions
end

# The lines region `name` holds between its markers: a blank line, its text, a blank line.
_content(name::String)::Vector{SubString{String}} = split("\n" * region_text(name), '\n')

"""
    apply_regions(text::String; file::String = "string") -> String

`text` with every region it marks holding its current text — between the region's markers a
blank line, `region_text(name)` and a blank line — and every other byte unchanged. Markers
take either form: `@setup generated` blocks or HTML comments. A `RegionError` naming `file`
and the line when a region is unclosed, opened or closed inside another, marked twice or
unknown, when a marker does not stand between blank lines, or when a line starts like a
marker and is not one.
"""
function apply_regions(text::String; file::String="string")::String
    lines = split(text, '\n')
    out = SubString{String}[]
    from = 1                                            # the first line not yet copied
    for (name, inside) in _regions(lines, file)
        append!(out, view(lines, from:(first(inside) - 1)), _content(name))
        from = last(inside) + 1
    end
    join(append!(out, view(lines, from:lastindex(lines))), '\n')
end

"""
    stale_regions(path::String) -> Vector{String}

The regions of the file at `path`, in file order, whose lines between the markers differ from
those `apply_regions` writes, byte for byte; empty when every region is current. The refusals
are `apply_regions`'s, naming `path`.
"""
function stale_regions(path::String)::Vector{String}
    lines = split(read(path, String), '\n')
    [name for (name, inside) in _regions(lines, path) if view(lines, inside) != _content(name)]
end

end # module
