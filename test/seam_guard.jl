# test/seam_guard.jl
#
# Architectural regression guard for the bounded-request invariant: every
# outbound HTTP request in src/ must flow through the deadline+retry seam, so it
# inherits the connect/request/total bounds and the shared retry budget. Only the
# seam primitives may name an HTTP verb directly. A new HTTP.post/get/put/delete/
# request/open added anywhere else reintroduces an unbounded call — this test
# fails the moment that happens.
#
# Scan is file+function granular. Line numbers are never part of the contract
# (they move); each matching call is attributed to its enclosing top-level
# function and checked against the allow-list.

using Test

# Direct client HTTP verb calls. Same alternation as the release closeout grep.
# Server constructors (HTTP.serve!, HTTP.Response, HTTP.header) are deliberately
# NOT verbs here.
const _SEAM_HTTP_VERB = r"\bHTTP\.(?:post|get|put|delete|request|open)\("

# The ONLY (file, enclosing-function) homes allowed to issue a direct verb call.
# The deadline seam lives at the top of requests.jl; _http performs the single
# non-streaming attempt, _http_open the streaming attempt. mcp_server.jl is
# server-side and issues no client verbs, so it needs no entry — if a
# server-side outbound call is ever added, allow-list its function deliberately.
const _SEAM_ALLOW = Set{Tuple{String,String}}([
    ("requests.jl", "_http"),
    ("requests.jl", "_http_open"),
])

"Nearest enclosing top-level function name for line `i` (1-based) in `lines`."
function _enclosing_function(lines::Vector{String}, i::Int)::String
    longform  = r"^function\s+([A-Za-z_][A-Za-z0-9_!]*)\("
    shortform = r"^([A-Za-z_][A-Za-z0-9_!]*)\([^)]*\)\s*(?:::[^=]+)?\s*="
    for j in i:-1:1
        s = lines[j]
        m = match(longform, s)
        m === nothing && (m = match(shortform, s))
        m === nothing || return String(m.captures[1])
    end
    return "<top-level>"
end

"All (file, enclosing-function, lineno) direct-HTTP-verb call sites under `srcdir`."
function _seam_call_sites(srcdir::AbstractString)::Vector{Tuple{String,String,Int}}
    sites = Tuple{String,String,Int}[]
    for (root, _, files) in walkdir(srcdir)
        for f in files
            endswith(f, ".jl") || continue
            lines = readlines(joinpath(root, f))
            for (i, line) in enumerate(lines)
                occursin(_SEAM_HTTP_VERB, line) || continue
                push!(sites, (f, _enclosing_function(lines, i), i))
            end
        end
    end
    return sites
end

"Call sites whose (file, enclosing-function) is not on the seam allow-list."
_seam_violations(srcdir::AbstractString)::Vector{Tuple{String,String,Int}} =
    filter(s -> (s[1], s[2]) ∉ _SEAM_ALLOW, _seam_call_sites(srcdir))

@testset "seam guard detects a planted violation (non-vacuous)" begin
    mktempdir() do d
        write(joinpath(d, "planted.jl"),
              "function _rogue_call()\n    HTTP.get(\"http://x\")\nend\n")
        v = _seam_violations(d)
        @test !isempty(v)
        @test ("planted.jl", "_rogue_call") in ((f, fn) for (f, fn, _) in v)
    end
end

@testset "every outbound HTTP call routes through the deadline seam" begin
    srcdir = normpath(joinpath(@__DIR__, "..", "src"))
    violations = _seam_violations(srcdir)
    if !isempty(violations)
        @info "Direct HTTP verb calls outside the seam:\n" *
              join(("  $(f):$(ln) in `$(fn)`" for (f, fn, ln) in violations), "\n")
    end
    @test isempty(violations)

    # Guard against the guard going blind: the seam MUST still issue the verbs.
    pairs = Set((f, fn) for (f, fn, _) in _seam_call_sites(srcdir))
    @test ("requests.jl", "_http") in pairs
    @test ("requests.jl", "_http_open") in pairs
end
