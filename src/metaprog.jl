makecall(::Nothing) = nothing

function makecall(m::Dict)
    Expr(:call, Symbol(m["name"]), (Expr(:kw, Symbol(k), v) for (k, v) in m["arguments"])...)
end

"""
    makecall(m::Message)

Generates an `Expr` from a `Message` containing a `function_call`.
"""
makecall(m::Message) = makecall(m.function_call)
