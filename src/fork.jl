"""
    fork(chat::Chat) -> Chat

Create an independent copy of a `Chat`: `messages` is deep-copied, the
cumulative-cost `Ref` is fresh (copied by value), and EVERY other field is
copied verbatim by construction (a `fieldnames` loop), so new `Chat` fields
survive forking automatically. `fork` itself applies no normalization or
rewrite — a fork is configuration-identical to its source.
"""
function fork(chat::Chat)::Chat
    kwargs = Dict{Symbol,Any}()
    for field in fieldnames(Chat)
        field in (:messages, :_cumulative_cost) && continue
        kwargs[field] = getfield(chat, field)
    end
    Chat(; messages=deepcopy(chat.messages),
           _cumulative_cost=Ref(chat._cumulative_cost[]), kwargs...)
end

"""
    fork(chat::Chat, n::Int) -> Vector{Chat}

Create `n` independent forks of a `Chat`.
"""
fork(chat::Chat, n::Int)::Vector{Chat} = [fork(chat) for _ in 1:n]
