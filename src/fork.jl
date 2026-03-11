"""
    fork(chat::Chat) -> Chat

Create an independent copy of a `Chat`. Deep-copies `messages`, shallow-copies config.
The `_cumulative_cost` is copied by value (new `Ref`), so mutations are independent.
"""
function fork(chat::Chat)::Chat
    Chat(
        service=chat.service,
        model=chat.model,
        messages=deepcopy(chat.messages),
        history=chat.history,
        tools=chat.tools,
        tool_choice=chat.tool_choice,
        parallel_tool_calls=isnothing(chat.tools) ? false : something(chat.parallel_tool_calls, false),
        temperature=chat.temperature,
        top_p=chat.top_p,
        n=chat.n,
        stream=chat.stream,
        stop=chat.stop,
        max_tokens=chat.max_tokens,
        presence_penalty=chat.presence_penalty,
        response_format=chat.response_format,
        frequency_penalty=chat.frequency_penalty,
        logit_bias=chat.logit_bias,
        user=chat.user,
        seed=chat.seed,
        _cumulative_cost=Ref(chat._cumulative_cost[])
    )
end

"""
    fork(chat::Chat, n::Int) -> Vector{Chat}

Create `n` independent forks of a `Chat`.
"""
fork(chat::Chat, n::Int)::Vector{Chat} = [fork(chat) for _ in 1:n]
