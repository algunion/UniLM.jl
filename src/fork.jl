"""
    fork(chat::Chat) -> Chat

Create an independent copy of a `Chat`: every field except `service` is deep-copied
in one pass, so mutating a fork's `messages`, `tools`, `metadata`, `stop` or any other
field never reaches its source or its siblings; `service` is shared (the endpoint a
chat talks to, not conversation state); the cumulative-cost `Ref` is fresh (copied by
value). The copy runs over `fieldnames(Chat)`, so new `Chat` fields fork
automatically. `fork` itself applies no normalization or rewrite — a fork is
configuration-identical to its source.
"""
function fork(chat::Chat)::Chat
    fields = Dict(f => getfield(chat, f) for f in fieldnames(Chat)
                  if f ∉ (:service, :_cumulative_cost))
    Chat(; service=chat.service, _cumulative_cost=Ref(chat._cumulative_cost[]),
           deepcopy(fields)...)
end

"""
    fork(chat::Chat, n::Int) -> Vector{Chat}

Create `n` independent forks of a `Chat`.
"""
fork(chat::Chat, n::Int)::Vector{Chat} = [fork(chat) for _ in 1:n]
