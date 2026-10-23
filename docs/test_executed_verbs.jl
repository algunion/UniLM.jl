# julia --startup-file=no --project=docs docs/test_executed_verbs.jl
#
# The manual's executed blocks run in a build without keys, so each exported function or macro they call has its
# answer replayed by `with_recorded_answers` (RECORDABLE) or sends nothing (LOCAL); UNRECORDABLE holds the rest.
# Every export is in exactly one class, so a new export fails here until it is classified.
import UniLM
using Test

# The verbs a recording scope replays; `list_models` only TypeSafe's listing, which `unrecordable_calls` checks.
const RECORDABLE = Set{String}([
    "chatrequest!", "respond", "embeddingrequest!", "tool_loop", "tool_loop!",
    "get_response", "delete_response", "list_input_items", "cancel_response", "compact_response",
    "count_input_tokens", "fim_complete", "prefix_complete",
    "ask", "ask_each", "nl_dispatch", "nl_classify", "@branch", "list_models",
])

const LOCAL = Set{String}([
    # constructors and builders
    "MistralEndpoint", "choice", "score", "noul", "box", "@nl_str", "fork",
    "function_tool", "custom_tool", "web_search", "file_search", "code_interpreter", "computer_tool", "shell",
    "apply_patch_tool", "image_generation_tool", "mcp_tool", "gemini_code_execution", "gemini_google_search",
    "gemini_url_context", "input_text", "input_image", "input_file", "configuration_update", "tool_result",
    "mcp_approval_response", "json_object_format", "json_schema_format", "text_format", "tool_choice_allowed",
    "tool_choice_custom", "tool_choice_function", "tool_choice_hosted", "tool_choice_mcp", "to_tool",
    "realtime_event", "session_update", "input_audio_append", "response_create",
    # accessors
    "answer", "answers", "text", "output_text", "fim_text", "transcript_text", "embedding_vectors", "image_data",
    "function_calls", "reasoning_text", "reasoning_items", "reasoning_summaries", "refusals", "url_citations",
    "web_search_results", "file_search_results", "code_interpreter_outputs", "image_generation_results",
    "mcp_approval_requests", "mcp_call_outputs", "incomplete_details", "response_status", "usage_details",
    "token_usage", "estimated_cost", "cumulative_cost", "conversation_id", "vector_store_id", "is_flagged",
    "issuccess", "isfailure", "issendvalid", "iscancelled", "default_model", "has_capability",
    "provider_capabilities", "meanings", "meaning_gaps", "current_config", "get_url", "auth_header",
    # scopes, local writes, webhooks, the provider protocol's methods and the MCP server's definitions
    "with_recorded_answers", "with_system_one", "with_request_config", "with_cancel", "set_default_config!",
    "cancel!", "add_azure_deploy_name!", "update!", "save_image", "save_audio", "save_file_content",
    "parse_webhook", "verify_webhook", "encode_request", "decode_response", "encode_agentic", "decode_agentic",
    "decode_agentic_stream", "encode_system_one", "decode_system_one", "system_one_protocol", "handle_sse_event!",
    "register_tool!", "register_resource!", "register_resource_template!", "register_prompt!",
    "@mcp_tool", "@mcp_resource", "@mcp_prompt",
])

const UNRECORDABLE = Set{String}([
    "generate_image", "edit_image", "speak", "transcribe", "translate", "moderate",
    "upload_file", "list_files", "retrieve_file", "delete_file", "file_content",
    "create_upload", "add_upload_part", "complete_upload", "cancel_upload",
    "create_vector_store", "retrieve_vector_store", "list_vector_stores", "delete_vector_store",
    "add_vector_store_file", "create_file_batch", "retrieve_file_batch", "poll_file_batch",
    "create_batch", "retrieve_batch", "list_batches", "cancel_batch", "poll_batch",
    "create_fine_tuning_job", "retrieve_fine_tuning_job", "list_fine_tuning_jobs", "cancel_fine_tuning_job",
    "list_fine_tuning_events", "list_fine_tuning_checkpoints",
    "create_container", "retrieve_container", "list_containers", "delete_container", "add_container_file",
    "create_conversation", "retrieve_conversation", "update_conversation", "delete_conversation",
    "add_conversation_items", "list_conversation_items", "delete_conversation_item",
    "realtime_connect", "realtime_send", "realtime_receive", "mint_realtime_secret",
    # the MCP client and the tools it makes, which call its server; an MCP server's transport
    "mcp_connect", "mcp_disconnect!", "mcp_tools", "mcp_tools_respond", "call_tool", "list_tools!",
    "list_resources!", "list_prompts!", "read_resource", "get_prompt", "ping", "serve",
    "model_info", "pull_model", "running_models", "load_model", "unload_model", "delete_model",
])

# A fenced block: the block quote, list item and indentation markers before its fence (Documenter runs an executed
# block inside an admonition, a block quote or a list item too), the fence, its info string, and the code up to the
# line that closes it.
const BLOCK = r"^(?<prefix>(?:[ >]|[-*+] |\d+[.)] )*)(?<fence>`{3,}|~{3,})(?<info>[^`\n]*)\n(?<code>(?s:.*?))^[ >]*\k<fence>[ \t]*$"m
const EXECUTED = r"^\s*@(?:example|setup|eval|repl)\b"
const TYPESAFE = (:TYPESAFEServiceEndpoint, :(TYPESAFEServiceEndpoint()))

# The export a callee names: `f` and `UniLM.f` (macros included) name "f"; any other callee names none.
callee(f::Symbol)::Union{Nothing,String} = string(f)
callee(f::Expr)::Union{Nothing,String} =
    Meta.isexpr(f, :., 2) && f.args[1] === :UniLM && f.args[2] isa QuoteNode ? callee(f.args[2].value) : nothing
callee(_)::Union{Nothing,String} = nothing

# What `ex` calls: `f(x)` (a `do` block's call too), `f.(x)` or `@f x`.
called(ex::Expr)::Union{Nothing,String} =
    ex.head in (:call, :macrocall) || Meta.isexpr(ex, :., 2) && Meta.isexpr(ex.args[2], :tuple) ?
        callee(ex.args[1]) : nothing

# A recording scope replays TypeSafe's listing alone: each argument is a keyword, and `service` names TypeSafe.
typesafe_listing(call::Expr)::Bool =
    all(a -> Meta.isexpr(a, :kw) && (a.args[1] !== :service || a.args[2] in TYPESAFE),
        Iterators.flatten(Meta.isexpr(a, :parameters) ? a.args : (a,) for a in call.args[2:end]))

# Pushes onto `found` each call in `ex` that reaches a service unrecorded, at `line`, the line of the statement being
# walked (a `LineNumberNode` precedes each statement); returns the line reached. Code that does not parse is an error.
function calls!(found::Vector{Tuple{Int,String}}, ex, line::Int)::Int
    ex isa LineNumberNode && return ex.line
    ex isa Expr || return line
    Meta.isexpr(ex, (:error, :incomplete)) && throw(only(ex.args))
    name = called(ex)
    (name in UNRECORDABLE || name == "list_models" && !typesafe_listing(ex)) && push!(found, (line, name))
    foldl((l, a) -> calls!(found, a, l), ex.args; init = line)
end

"""
    unrecordable_calls(text; file = "page") -> Vector{Tuple{Int,String}}

The calls in the `@example`, `@setup`, `@eval` and `@repl` blocks of the Markdown `text` that reach a service
unrecorded, as (line, name): an `UNRECORDABLE` function or macro, or a `list_models` that may not ask TypeSafe. A
block that does not parse throws its `Meta.ParseError`, located at `file` and the line.
"""
function unrecordable_calls(text::String; file::String = "page")::Vector{Tuple{Int,String}}
    found = Tuple{Int,String}[]
    for m in eachmatch(BLOCK, text)
        occursin(EXECUTED, m[:info]) || continue
        line = count(==('\n'), SubString(text, 1, m.offset - 1)) + 2
        code = replace(m[:code], Regex("^[ >]{0,$(length(m[:prefix]))}", "m") => "")
        calls!(found, Meta.parseall(code; filename = file, lineno = line), line)
    end
    found
end

"`page:line name` for each call `unrecordable_calls` reports in the Markdown files under `dir`."
function scan_docs(dir::String)::Vector{String}
    pages = [relpath(joinpath(root, f), dir) for (root, _, files) in walkdir(dir) for f in files if endswith(f, ".md")]
    [string(page, ':', line, ' ', name)
     for page in pages for (line, name) in unrecordable_calls(read(joinpath(dir, page), String); file = page)]
end

@testset "executed verbs — unrecordable calls in executed blocks are reported" begin
    @test unrecordable_calls("```@example a\nr = generate_image(\"x\")\n```\n") == [(2, "generate_image")]
    @test unrecordable_calls("```@setup a\nupload_file(\"x\")\n```\n") == [(2, "upload_file")]
    @test isempty(unrecordable_calls("```julia\nr = generate_image(\"x\")\n```\n"))
    @test isempty(unrecordable_calls("```@example a\nlist_models()\n```\n"))
    @test unrecordable_calls("```@example a\nlist_models(; service=OllamaEndpoint())\n```\n") == [(2, "list_models")]
    @test isempty(unrecordable_calls("```@example a\nlist_models(service=TYPESAFEServiceEndpoint)\n```\n"))
    # Documenter runs these blocks inside admonitions, block quotes and list items too.
    @test unrecordable_calls("!!! note\n    ```@example a\n    speak(\"x\")\n    ```\n") == [(3, "speak")]
    @test unrecordable_calls("> ```@repl\n> UniLM.moderate(\"x\")\n> ```\n") == [(2, "moderate")]
    @test unrecordable_calls("1. ```@eval\n   transcribe.(clips)\n   ```\n") == [(2, "transcribe")]
    @test isempty(unrecordable_calls("````markdown\n```@example a\ngenerate_image(\"x\")\n```\n````\n"))
    @test unrecordable_calls("```@example a\nx = 1\nwith_cancel(tok) do\n    edit_image(x)\nend\n```\n") ==
          [(4, "edit_image")]
    @test_throws Meta.ParseError unrecordable_calls("```@example a\nf(\n```\n")
end

@testset "executed verbs — every export is classified, and docs/src is clean" begin
    exported = Set(string(n) for n in names(UniLM) if getfield(UniLM, n) isa Function)
    @test isempty(setdiff(exported, RECORDABLE ∪ LOCAL ∪ UNRECORDABLE))
    @test isempty(setdiff(RECORDABLE ∪ LOCAL ∪ UNRECORDABLE, exported))
    @test isempty((RECORDABLE ∩ LOCAL) ∪ (RECORDABLE ∩ UNRECORDABLE) ∪ (LOCAL ∩ UNRECORDABLE))
    @test isempty(scan_docs(joinpath(@__DIR__, "src")))
end
