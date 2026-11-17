# ============================================================================
# Gemini context caching — explicit caches (cachedContents) on the Gemini API
# (https://ai.google.dev/gemini-api/docs/caching, https://ai.google.dev/api/caching). A cache holds a
# prompt's start — system instruction, contents, tools and tool config — which a generateContent request
# names in `cachedContent` (a Chat's `cached_content`) and reads at the cache-hit rate.
# ============================================================================

# A Gemini resource as the API names it, `<collection>/<id>`, from that name or its id: the id is one path
# segment, neither empty nor a dot segment, which a URL would resolve away.
_gemini_segment(s::AbstractString)::Bool = !isempty(s) && !occursin('/', s) && !(s in (".", ".."))

function _gemini_resource_name(collection::String, name::AbstractString, what::String)::String
    id = chopprefix(name, collection * "/")
    _gemini_segment(id) || throw(ArgumentError(
        "$what names a resource as \"$collection/<id>\" or by its id (got $(repr(name)))"))
    collection * "/" * id
end

# The cache a Chat names (api.jl), stored as its name.
_cache_reference(::Nothing) = nothing
_cache_reference(name::AbstractString)::String = _gemini_resource_name("cachedContents", name, "cached_content")

# Explicit caching "can be manually enabled on Gemini 3 and 2.5 models ...; not supported for Gemini 4 models"
# (https://ai.google.dev/gemini-api/docs/caching); the API answered a cache for gemini-4-hydrogen with HTTP 400
# "Explicit caching is not supported for models/gemini-4-hydrogen." (observed 2026-11-17).
function _gemini_explicit_cache_model(model::AbstractString, what::String)::Nothing
    startswith(model, "gemini-4-") && throw(ArgumentError("$what: $model takes no explicit cache: Google " *
        "caches explicitly on Gemini 3 and 2.5 models, \"not supported for Gemini 4 models\", which cache " *
        "implicitly (https://ai.google.dev/gemini-api/docs/caching); use a Gemini 3 or 2.5 model"))
    nothing
end
