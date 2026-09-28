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
#                                                `folder=…`, `base_path=…` and `version=…` lines
#   julia --project=docs docs/deploy.jl dry-run  Documenter's decision; deploys nothing
#   julia --project=docs docs/deploy.jl          deploys (the Documentation workflow)
#
# REDEPLOY_VERSION=vX.Y.Z deploys this checkout's manual into a released version's
# folder instead, leaving stable and versions.js pointing where they did. It is
# refused, by `plan` before anything is built, unless it runs on main and the tag
# exists and src/ and Project.toml are unchanged since it: a release's manual must
# not describe unreleased code.
using Documenter

const ROOT = dirname(@__DIR__)
const TARGET = joinpath("site", "out")  # deploydocs resolves `target` inside `root`
const SITE_PATH = "/UniLM.jl"           # GitHub Pages serves gh-pages under the repository's name
const REPO = "github.com/algunion/UniLM.jl"
const DEVBRANCH = "main"
const REDEPLOY = let v = get(ENV, "REDEPLOY_VERSION", "")
    isempty(v) ? nothing : v
end

"""
`version`, if this checkout's manual may be published as its docs: a `vX.Y.Z` tag whose
`src/` and `Project.toml` are `HEAD`'s, redeployed by a run on the development branch
(`ref` is the run's `GITHUB_REF`).
"""
function release(version::String, ref::String)
    occursin(r"^v\d+\.\d+\.\d+\z", version) ||
        error("REDEPLOY_VERSION must be a release version such as v0.22.0; got $(repr(version))")
    ref == "refs/heads/$DEVBRANCH" ||
        error("cannot redeploy $version from $(repr(ref)): run the workflow on $DEVBRANCH")
    tag = "refs/tags/$version"
    success(pipeline(`git -C $ROOT rev-parse --verify --quiet "$tag^{commit}"`; stdout = devnull)) ||
        error("cannot redeploy $version: there is no tag $version")
    changed = readchomp(`git -C $ROOT diff --name-only $tag HEAD -- src Project.toml`)
    isempty(changed) ||
        error("cannot redeploy $version: its manual would describe unreleased code; changed since the tag:\n$changed")
    version
end

"The folder the site is built for: the release being redeployed, the version tag being built, else dev."
function folder(redeploy::Union{Nothing,String}, ref::String)
    redeploy === nothing || return release(redeploy, ref)
    tag = match(r"^refs/tags/(v\d+\.\d+\.\d+)\z", ref)
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
    SiteDeploy(ci, version)

Deploys where the GitHub Actions deployment `ci` decides — into `version`'s folder
instead when `version` is a release being redeployed — and only an export built for
that folder. A redeploy that `ci` would not deploy is an error, not a skipped step.
"""
struct SiteDeploy <: Documenter.DeployConfig
    ci::Documenter.GitHubActions
    version::Union{Nothing,String}
end
function Documenter.deploy_folder(c::SiteDeploy; kwargs...)
    d = Documenter.deploy_folder(c.ci; kwargs...)
    if !d.all_ok
        c.version === nothing || error("cannot redeploy $(c.version): the deployment criteria above do not hold")
        return d
    end
    subfolder = something(c.version, d.subfolder)
    built_for(subfolder)
    Documenter.DeployDecision(; all_ok = true, d.branch, d.is_preview, d.repo, subfolder)
end
Documenter.authentication_method(c::SiteDeploy) = Documenter.authentication_method(c.ci)
Documenter.authenticated_repo_url(c::SiteDeploy) = Documenter.authenticated_repo_url(c.ci)
Documenter.post_status(c::SiteDeploy; kwargs...) = Documenter.post_status(c.ci; kwargs...)

config() = SiteDeploy(Documenter.GitHubActions(),
                      REDEPLOY === nothing ? nothing : release(REDEPLOY, get(ENV, "GITHUB_REF", "")))
deploy(cfg::Documenter.DeployConfig; kwargs...) =
    deploydocs(; root = ROOT, target = TARGET, repo = REPO, devbranch = DEVBRANCH,
               versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"], deploy_config = cfg, kwargs...)

if ARGS == ["plan"]
    f = folder(REDEPLOY, get(ENV, "GITHUB_REF", ""))
    println("folder=$f\nbase_path=$SITE_PATH/$f\nversion=$f")
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
