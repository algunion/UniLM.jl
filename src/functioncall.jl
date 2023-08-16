makecall(::Nothing) = nothing

function makecall(m::Union{Dict,String})
    # temporary JSON3.read workaround - might not be an issue since only one makecall is issued per message   
    # this is cause by the way OpenAI API returns the function_call 
    try
        args = m["arguments"] isa String ? JSON3.read(m["arguments"], Dict) : m["arguments"]
        #Expr(:call, Symbol(m["name"]), (Expr(:kw, Symbol(k), v) for (k, v) in args)...)
        f = eval(Symbol(m["name"]))
        ans = f(; ((Symbol(k), v) for (k, v) in args)...)
        return ans
    catch e
        @error "makecall failed with error: $e"
        return nothing
    end
end

"""
    makecall(m::Message)

Generates an `Expr` from a `Message` containing a `function_call`. 
"""
makecall(m::Message) = makecall(m.function_call)

"""
    evalcall!(chat::Chat)::GPTFunctionCallResult

    Evaluates the `function_call` of the last `Message` in `chat` and updates the `chat` with the result.
"""
function evalcall!(chat::Chat)::GPTFunctionCallResult
    m = last(chat)
    result = makecall(m)
    #result = evalcall!(e)
    @info "result: " result
    update!(chat, Message(role=GPTFunction, name=something(m.function_call)["name"], content=JSON3.write(result)))
    #@info length(chat)

    GPTFunctionCallResult(Symbol(something(m.function_call)["name"]), something(m.function_call), result)
end