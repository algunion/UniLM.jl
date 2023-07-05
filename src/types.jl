struct Model
    name::Symbol
    endpoint::Symbol
end

model(name::Symbol, endpoint::Symbol) = Model(name, endpoint)

Base.propertynames(::typeof(model)) = _endpoints_syn
Base.getproperty(::typeof(model), x::Symbol) = model(:gpt35turbo, x)

function Base.propertynames(m::Model)
    m.endpoint == :chat && return _chat_completions_syn    
    return [:test1, :test2]
end
function Base.getproperty(m::Model, x::Symbol)
    if x == :name || x == :endpoint
        return getfield(m, x)
    end
    model(x, m.endpoint)
end