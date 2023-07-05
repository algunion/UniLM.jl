"""
    extractanswer(s::String; include=false, first="```", last="```") :: String

    Extract the string between `first` and `last` in `s`. If `include` is true, then the `first` and `last` are included in the output. If `first` and `last` are not found, then the original input is returned.
"""
function extractanswer(s::String; include=false, first="```", last="```") :: String    
    fi = findfirst(first , s)
    li = findlast(last, s)
    (fi === nothing || li === nothing || fi == li) && return s
    include && return s[fi[1]:li[end]]
    return s[fi[end]+1:li[1]-1]    
end