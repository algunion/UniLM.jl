makecall(::Nothing) = nothing

function makecall(m::Dict)
    # temporary JSON3.read workaround - might not be an issue since only one makecall is issued per message
    args = m["arguments"] isa String ? JSON3.read(m["arguments"], Dict) : m["arguments"] 
    Expr(:call, Symbol(m["name"]), (Expr(:kw, Symbol(k), v) for (k, v) in args)...)
end

"""
    makecall(m::Message)

Generates an `Expr` from a `Message` containing a `function_call`.
"""
makecall(m::Message) = makecall(m.function_call)

evalcall(m::Message) = eval(makecall(m))

"""
    evalcall!(chat::Chat)::GPTFunctionCallResult

    Evaluates the `function_call` of the last `Message` in `chat` and updates the `chat` with the result.
"""
function evalcall!(chat::Chat)::GPTFunctionCallResult
    m = last(chat)
    result = evalcall(m)
    update!(chat, Message(role=GPTFunction, name=something(m.function_call)["name"], content=JSON3.write(result)))
    GPTFunctionCallResult(Symbol(something(m.function_call)["name"]), something(m.function_call), result)
end
