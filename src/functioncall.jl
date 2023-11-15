function makecall(m::Dict)
    # temporary JSON3.read workaround - might not be an issue since only one makecall is issued per message   
    # this is cause by the way OpenAI API returns the function_call 
    try
        args = JSON3.read(m["arguments"], Dict) # m["arguments"] isa String ? JSON3.read(m["arguments"], Dict) : m["arguments"]
        #Expr(:call, Symbol(m["name"]), (Expr(:kw, Symbol(k), v) for (k, v) in args)...)
        f = eval(Symbol(m["name"]))
        ans = f(; ((Symbol(k), v) for (k, v) in args)...)
        return ans
    catch e
        @error "makecall failed with error: $e"
        throw(e)
    end
end

"""
    makecall(m::Message)

Generates an `result`` from a `Message` containing a `tool_call`. 
"""
makecall(m::Message) = makecall(something(m.tool_calls)[1].func)

"""
    evalcall!(chat::Chat)::GPTFunctionCallResult

    Evaluates the `function_call` of the last `Message` in `chat` and updates the `chat` with the result.
"""
function evalcall!(chat::Chat)::Union{GPTFunctionCallResult,Nothing}
    m = last(chat)
    if !isnothing(m.tool_calls)
        result = makecall(m)
        #result = evalcall!(e)
        @info "result: " result

        update!(chat, Message(role=RoleTool, tool_call_id=something(m.tool_calls)[1].id, content=JSON3.write(result)))
        #@info length(chat)

        return GPTFunctionCallResult(Symbol(something(m.tool_calls)[1].id), something(m.tool_calls)[1].func, result)
    end
end