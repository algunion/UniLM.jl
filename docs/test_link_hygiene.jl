using Test
const SRC = joinpath(@__DIR__, "src")
@testset "docs/src links pages by @id, never by relative path or heading text" begin
    md = [joinpath(r, f) for (r, _, fs) in walkdir(SRC) for f in fs if endswith(f, ".md")]
    rel = [(f, m.match) for f in md for m in eachmatch(r"\]\((?!https?://|mailto:|@ref)[^)]*\.md(#[^)]*)?\)", read(f, String))]
    bare = [(f, m.match) for f in md for m in eachmatch(r"\[(?!`[^`\]]+`\])[^\]]+\]\(@ref\)", read(f, String))]
    quoted = [(f, m.match) for f in md for m in eachmatch(r"\]\(@ref\s+\"[^\"]*\"\)", read(f, String))]
    @test isempty(rel)
    @test isempty(bare) && isempty(quoted)
end
