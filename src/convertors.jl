function Base.convert(::Type{GPTFunctionSignature}, ms::MethodSchemas.MethodSignature)
    req = [string(getfield(arg, :name)) for arg in ms.args if getfield(arg, :required)]
    name = MethodSchemas.nobang(ms.name)
    GPTFunctionSignature(name=string(name), description=ms.description,
        parameters=MethodSchemas.JsonObject(properties=Dict(string(getfield(arg, :name)) => convert(MethodSchemas.JsonSchema,
                arg)
                                                            for arg in ms.args
                                                            if MethodSchemas.isincluded(arg)),
            required=req))
end