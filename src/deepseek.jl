# ============================================================================
# DeepSeek thinking mode on the OpenAI-wire seam
# DeepSeek models think by default and return the chain of thought as
# `reasoning_content` beside `content` (`delta.reasoning_content` when streaming).
# A request that carries `tools` must pass back the reasoning_content of every
# earlier assistant turn, turns without tool calls included, or the API answers
# 400; without tools it is not needed and the API ignores it
# (https://api-docs.deepseek.com/guides/thinking_mode). The OpenAI-wire defaults
# drop the field, so these more specific methods delegate to them and carry the
# reasoning on the Message as ProviderContent(:deepseek, [{"reasoning_content"}]).
# They also read DeepSeek's context-cache hit count into the usage.
# ============================================================================

# DeepSeek takes one effort more than OpenAI's, which it maps to its highest: "| ultra | max"
# (https://api-docs.deepseek.com/guides/thinking_mode, the requested-to-actual effort table).
_reasoning_efforts(::DeepSeekEndpoint)::Tuple{Vararg{String}} = (_OPENAI_REASONING_EFFORTS..., "ultra")

# An explicit-mode Chat's remedy (api.jl): DeepSeek's context caching "is enabled by default for all
# users" (https://api-docs.deepseek.com/guides/kv_cache), and its reference lists no prompt_cache_options.
_explicit_cache_remedy(::DeepSeekEndpoint)::String =
    "DeepSeek caches prompts on its own, and its reference lists no prompt_cache_options: unset prompt_cache_options"

# Copy of `m` with one field replaced; field-generic, so fields added to Message later
# are carried over.
_message_with(m::Message, field::Symbol, value) =
    Message((f === field ? value : getfield(m, f) for f in fieldnames(Message))...)

# The reasoning captured on an assistant turn DeepSeek produced, else `nothing`.
function _deepseek_reasoning(m::Message)::Union{String,Nothing}
    pc = m.provider_content
    m.role == RoleAssistant && pc isa ProviderContent && pc.provider === :deepseek || return nothing
    parts = String[b["reasoning_content"] for b in pc.blocks
                   if b isa AbstractDict && get(b, "reasoning_content", nothing) isa AbstractString]
    isempty(parts) ? nothing : join(parts)
end

# DeepSeek reports the prompt's context-cache hits as `usage.prompt_cache_hit_tokens`
# (the rest of `prompt_tokens` as `prompt_cache_miss_tokens`,
# https://api-docs.deepseek.com/guides/kv_cache). The OpenAI-wire builder reads only
# `prompt_tokens_details.cached_tokens`; without this, a usage carrying just the
# native count prices every cached token at the miss rate. `t` is that builder's usage
# of the raw usage `u`; only its cached count is replaced, field by field, so every other
# field, one added to TokenUsage later included, is carried over.
function _deepseek_usage(t::TokenUsage, u::AbstractDict)::TokenUsage
    hit = get(u, "prompt_cache_hit_tokens", nothing)
    hit isa Integer || return t
    TokenUsage((f === :cached_tokens ? Int(hit) : getfield(t, f) for f in fieldnames(TokenUsage))...)
end
_deepseek_usage(t, _) = t

function decode_response(service::DeepSeekEndpoint, resp::HTTP.Response)
    decoded = invoke(decode_response, Tuple{OpenAIWireEndpointSpec,HTTP.Response}, service, resp)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    usage = _deepseek_usage(decoded.usage, get(data, "usage", nothing))
    # The OpenAI decoder read choices[1] already, so it is there.
    message = get(data["choices"][1], "message", nothing)
    rc = message isa AbstractDict ? get(message, "reasoning_content", nothing) : nothing
    rc isa AbstractString && !isempty(rc) ||
        return (; decoded.message, usage, decoded.model, decoded.logprobs, decoded.moderation)
    pc = ProviderContent(:deepseek, Any[Dict{String,Any}("reasoning_content" => rc)])
    (; message=_message_with(decoded.message, :provider_content, pc), usage, decoded.model, decoded.logprobs,
       decoded.moderation)
end

# DeepSeek's Chat Completions reference lists the token cap as `max_tokens` and no
# `max_completion_tokens` (https://api-docs.deepseek.com/api/create-chat-completion): a Chat's
# `max_completion_tokens` goes out under that name, taken over a `max_tokens` also set.
function encode_request(service::DeepSeekEndpoint, chat::Chat)::String
    body = _renamed_json(chat, :max_completion_tokens => :max_tokens)
    isnothing(chat.tools) && return body
    echo = Pair{Int,String}[]
    for (i, m) in pairs(chat.messages)
        rc = _deepseek_reasoning(m)
        isnothing(rc) || push!(echo, i => rc)
    end
    isempty(echo) && return body
    # The OpenAI wire lowers `chat.messages` one to one, in order.
    wire = JSON.parse(body)
    foreach(((i, rc),) -> wire["messages"][i]["reasoning_content"] = rc, echo)
    JSON.json(wire)
end

# Streamed reasoning accumulates in one pending block, finalized when the choice
# reports its finish_reason or the stream ends; a block still pending marks the
# capture incomplete, so a truncated stream carries no partial chain of thought.
function _deepseek_reasoning_buffer!(state::StreamState)::IOBuffer
    blk = get!(state.raw_pending, 0) do
        state.raw_provider = :deepseek
        Dict{String,Any}("reasoning_content" => IOBuffer())
    end
    blk["reasoning_content"]::IOBuffer
end

function _deepseek_finalize_reasoning!(state::StreamState)
    blk = pop!(state.raw_pending, 0, nothing)
    isnothing(blk) ||
        push!(state.raw_blocks, Dict{String,Any}("reasoning_content" => takestring!(blk["reasoning_content"])))
    nothing
end

function handle_sse_event!(service::DeepSeekEndpoint, event::AbstractString,
                           payload::AbstractString, state::StreamState)::Symbol
    status = invoke(handle_sse_event!, Tuple{OpenAIWireEndpointSpec,AbstractString,AbstractString,StreamState},
                    service, event, payload, state)
    if status === :continue
        parsed = JSON.parse(payload; dicttype=Dict{String,Any})   # the default handler parsed it too
        u = parsed isa AbstractDict ? get(parsed, "usage", nothing) : nothing
        u isa AbstractDict && (state.usage = _deepseek_usage(state.usage, u))   # the default's usage of `u`
        choices = parsed isa AbstractDict ? get(parsed, "choices", nothing) : nothing
        for cho in (choices isa AbstractVector ? choices : Any[])
            (cho isa AbstractDict && _reply_choice(cho)) || continue
            delta = get(cho, "delta", nothing)
            rc = delta isa AbstractDict ? get(delta, "reasoning_content", nothing) : nothing
            rc isa AbstractString && !isempty(rc) && print(_deepseek_reasoning_buffer!(state), rc)
            get(cho, "finish_reason", nothing) isa AbstractString && _deepseek_finalize_reasoning!(state)
        end
    end
    status === :done && _deepseek_finalize_reasoning!(state)
    status
end
