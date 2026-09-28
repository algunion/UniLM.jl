# Deploys the documentation site's static export, site/out, on the gh-pages branch
# with Documenter's versioned deployment, the scheme the manual's HTML used: a push
# to main replaces dev/; a version tag vX.Y.Z adds vX.Y.Z/ and points stable at the
# newest release; versions.js and the root redirect are regenerated; other folders
# are left as they are.
#
# The export's pages load their scripts and styles, and link each other, under the
# path the site was built for, so the export must land in the folder that path
# names: it is built with DOCS_BASE_PATH=/UniLM.jl/<folder>. A tag's folder is its
# version, never stable, so its pages keep working after stable moves on.
#
#   julia --project=docs docs/deploy.jl plan     the folder to build for, printed as
#                                                `version=…` and `base_path=…` lines
#   julia --project=docs docs/deploy.jl dry-run  Documenter's decision; deploys nothing
#   julia --project=docs docs/deploy.jl          deploys (the Documentation workflow)
using Documenter

const ROOT = dirname(@__DIR__)
const TARGET = joinpath("site", "out")  # deploydocs resolves `target` inside `root`
const SITE_PATH = "/UniLM.jl"           # GitHub Pages serves gh-pages under the repository's name
const REPO = "github.com/algunion/UniLM.jl"
const DEVBRANCH = "main"

"The folder the site is built for: the version tag being built, else dev."
function folder(ref::String)
    tag = match(r"^refs/tags/(v\d+\.\d+\.\d+)$", ref)
    tag === nothing ? "dev" : String(tag[1])
end

"Refuses an export that is missing or was built for another folder than `subfolder`."
function built_for(subfolder::String)
    out, base = joinpath(ROOT, TARGET), "$SITE_PATH/$subfolder"
    build = "build it with DOCS_BASE_PATH=$base npm run build in site/"
    isfile(joinpath(out, "index.html")) || error("no site export in $out: $build")
    occursin("\"$base/_next/", read(joinpath(out, "index.html"), String)) ||
        error("the site export in $out was not built for $subfolder/: $build")
end

"""
    SiteDeploy(ci)

Deploys where the GitHub Actions deployment `ci` decides, and only an export built
for that folder.
"""
struct SiteDeploy <: Documenter.DeployConfig
    ci::Documenter.GitHubActions
end
function Documenter.deploy_folder(c::SiteDeploy; kwargs...)
    d = Documenter.deploy_folder(c.ci; kwargs...)
    d.all_ok && built_for(d.subfolder)
    d
end
Documenter.authentication_method(c::SiteDeploy) = Documenter.authentication_method(c.ci)
Documenter.authenticated_repo_url(c::SiteDeploy) = Documenter.authenticated_repo_url(c.ci)
Documenter.post_status(c::SiteDeploy; kwargs...) = Documenter.post_status(c.ci; kwargs...)

config() = SiteDeploy(Documenter.GitHubActions())
deploy(cfg::Documenter.DeployConfig; kwargs...) =
    deploydocs(; root = ROOT, target = TARGET, repo = REPO, devbranch = DEVBRANCH,
               versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"], deploy_config = cfg, kwargs...)

if ARGS == ["plan"]
    f = folder(get(ENV, "GITHUB_REF", ""))
    println("version=$f\nbase_path=$SITE_PATH/$f")
elseif ARGS == ["dry-run"]
    # the question deploydocs asks, with its defaults for devurl and push_preview
    d = Documenter.deploy_folder(config(); repo = REPO, devbranch = DEVBRANCH, devurl = "dev", push_preview = false)
    println("target=", joinpath(ROOT, TARGET))
    foreach(f -> println(f, "=", getfield(d, f)), fieldnames(Documenter.DeployDecision))
elseif isempty(ARGS)
    deploy(config())
else
    error("usage: julia --project=docs docs/deploy.jl [plan | dry-run]; got $(repr(ARGS))")
end
