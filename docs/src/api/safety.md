# [Safety & Provenance API](@id safety_api)

Read the safety case behind a `safety.warning_issued` or `safety.deactivation_issued` webhook and
the project safety alert behind a `safety.alert.created` one (the `:safety` capability), and check
an image, audio or PDF file for OpenAI's provenance signals (`:content_provenance`): "Images: PNG,
JPEG, WebP, and GIF. Audio: MP3, Opus, AAC, FLAC, WAV, and PCM. Documents: PDF."
([content provenance](https://developers.openai.com/api/docs/guides/content-provenance)). OpenAI only.

## Objects

```@docs
SafetyCase
SafetyAlert
ProvenanceCheck
ProvenanceResult
```

## Result Types

```@docs
SafetyCaseSuccess
SafetyAlertSuccess
ProvenanceCheckSuccess
SafetyFailure
SafetyCallError
```

## Request Functions

```@docs
retrieve_safety_case
retrieve_safety_alert
check_content_provenance
```

## Usage

```julia
# The record behind a verified safety webhook
event = parse_webhook(payload)
if event.type == "safety.alert.created"
    alert = retrieve_safety_alert(event.data["id"])
    alert isa SafetyAlertSuccess && println(alert.response.error_type, ": ", alert.response.response_id)
elseif event.type in ("safety.warning_issued", "safety.deactivation_issued")
    case = retrieve_safety_case(event.data["id"])
    case isa SafetyCaseSuccess && println(case.response.notice, " for ", case.response.entity_identifier)
end

# Provenance signals of an image
check = check_content_provenance("generated.png")
check isa ProvenanceCheckSuccess && [(r.type, r.outcome) for r in check.response.results]
```
