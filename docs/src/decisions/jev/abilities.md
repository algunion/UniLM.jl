# [Images, Box, long context, structured guidance](@id jev_abilities_page)

```@setup jev
using UniLM
```

What Jev takes beyond a question about a text, a section each:

| Section | What it holds |
| :--- | :--- |
| [Images](@ref jev_images__next) | a picture wherever a string can stand: how it is sent, checked and billed |
| [Locate with Box](@ref jev_box__next) | where something is in a picture, as fractions of its size |
| [Long context](@ref jev_long_context__next) | 262k tokens of context, in place of 64k and 32k |
| [Structured guidance](@ref jev_structured) | instructions, options, levels and yes/no written as objects and arrays |

## [Images](@id jev_images__next)

**The job:** judge what a picture shows — the photo a customer attached, a
screenshot of an error — with the questions you ask about a text. **Without
Jev:** a vision LLM whose description you parse. **With Jev:** put an
[`ImageAttachment`](@ref) in the request, and a Choice, Score or Noul is
answered about it as about a text.

The picture here is a test image of 512 × 512 pixels: a red rectangle and a
blue disc on white.

![A red rectangle and a blue disc on a white background](../../assets/scene.png)

```@example jev
scene = ImageAttachment("../../assets/scene.png")      # a PNG or JPEG file, or its bytes

r = ask((message = "I ordered the round blue sticker. Is that what the photo shows?", photo = scene),
        "shows_it"  => noul("Does `photo` show what `message` asks about?"),
        "red_shape" => choice("What is the red shape in `photo`?",
                              ["a rectangle", "a disc", "a triangle", "there is no red shape"]))

println("P(the photo shows it) = ", r["shows_it"].noul)
println("red shape: ", r["red_shape"].choice, " (confidence ", r["red_shape"].confidence, ")")
println(token_usage(r).prompt_tokens, " input tokens")
```

The message and the picture went out in one state, and each question names its
part in backticks. Jev puts 0.82 on the photo showing what the customer asks
about, and names the red shape a rectangle at confidence 1.0. The picture is
most of the 1,114 input tokens: about 720 of them, by the measurement below.

An image may stand anywhere a string can: in the state, alone or among text in a
`NamedTuple`, `Dict`, vector or tuple; in a question's instructions; in its
criteria — a Choice whose options are example pictures, a Score whose levels
are. A Score level that is an image comes back in the answer's `legend` as the
same `ImageAttachment`. UniLM looks for images in those containers, not in the
fields of a struct of your own: a request whose image only a struct holds is
refused with an `ArgumentError` before anything is sent, rather than writing the
image's bytes into the JSON as numbers ([Limits and refusals](@ref jev_limits__next)).

- **Sent once, billed per use.** Each distinct image goes out once, as a part of
  a multipart request, and every place it stood holds a reference to it. Every
  use is billed, so an image several questions need belongs in the state, once.
- **The opt-in.** TypeSafe calls images dreamer-ready, not yet ready for
  production — their request format, answers and limits can change between
  releases ([Images](https://docs.typesafe.ai/images)) — and reads them only from
  a request that carries the header `TypeSafe-Danger-Zone-Images: true`. UniLM
  adds it whenever a request holds an image: passing one is the opt-in.
- **Checked before sending.** PNG or JPEG, a still image (no animated PNG), at
  most 16 MiB, 280 to 1568 pixels on each side, a JPEG ending with its
  end-of-image marker (one cut short is refused); the whole request at most
  32 MiB ([Limits and refusals](@ref jev_limits__next)). TypeSafe applies no EXIF
  rotation, so turn a photo's pixels upright before you send it.
- **Other providers.** Where a picture may stand differs per provider
  ([Choosing a decision provider](@ref system_one_backends__next)).

!!! details "Evidence"
    Observed on 19 October 2026 (`jev-1.13.0`): an image of W × H pixels was
    billed 2·⌈W/28⌉·⌈H/28⌉ input tokens, plus about 4 for each use — 722 for a
    512 × 512 picture, 200 at 280 × 280, 6,272 at 1568 × 1568 — and the same
    image used twice in one request was billed twice. TypeSafe does not document
    this formula: it is our fit to six sizes, not a contract.

**Tune it**

- Crop to what the question is about: a smaller image costs fewer tokens, as
  long as each side stays between 280 and 1568 pixels.
- Name the image's key in the question, `photo` above, as you would name any
  part of the state.

## [Locate with Box](@id jev_box__next)

**The job:** find where something is in a picture: the dent on a parcel, the
signature on a form. **Without Jev:** an object detector trained on your
classes, or a vision LLM asked for coordinates as text. **With Jev:** a
[`box`](@ref) question names what to locate, and the answer is a
[`BoxAnswer`](@ref): the box's edges as fractions of the image, and a
confidence.

A Box needs Jev 1.14 or later ([Images](https://docs.typesafe.ai/images)) —
`model = "jev-1.14.0"`, or an alias that points to it: on 25 October 2026
TypeSafe points `jev-latest` and `jev-preview` at `jev-1.14.0`
([Models](@ref jev_models__next)). Asked of Jev 1.13 in October 2026, a Box
question drew HTTP 400 and `"Invalid request."`; [`ask`](@ref) returns that
[`SystemOneFailure`](@ref) with a hint naming the models that answer one.

```@example jev
r = ask(scene, "rectangle" => box("the red rectangle"), "disc" => box("the blue disc"); model = "jev-1.14.0")

for name in ("rectangle", "disc")
    b = r[name]
    println(rpad(name, 10), b.box, "  confidence ", b.confidence)
end
# Edges are fractions: of the height for top and bottom, of the width for left and right.
pixels(f) = round(Int, 512f)
b = r["rectangle"].box
println("rectangle: rows ", pixels(b.top), "–", pixels(b.bottom), ", columns ", pixels(b.left), "–", pixels(b.right))
```

The rectangle is drawn at 0.30–0.70 of the height and 0.20–0.55 of the width —
rows 153–358 and columns 102–281 of the 512 — and the disc at 0.12–0.32 and
0.68–0.88: every edge of both answers is within 0.02 of the drawing.

A [`BoxAnswer`](@ref) holds `box`, a [`BoxCoordinates`](@ref) — `top` and
`bottom` as fractions of the image's height from its top edge, `left` and
`right` as fractions of its width from its left edge, so `(0, 1, 0, 1)` is the
whole image — then `confidence`, `nothing` when the model reports none, and
`raw`. A Box question needs an image in the request, and [`ask`](@ref) refuses
one without before anything is sent ([Limits and refusals](@ref jev_limits__next)):

```@example jev
try
    ask("The parcel arrived with a dent on its left side.", "dent" => box("the dent"); model = "jev-1.14.0")
catch err
    println(sprint(showerror, err))
end
```

Sent anyway, such a question is still answered — HTTP 200 and a box at a low
confidence, 0.05 to 0.10 in our probes on 19 and 20 October 2026 — and no box
drawn on a picture that is not there means anything. An answer whose box is
turned inside out, its top below its bottom or its left right of its right, is
a [`SystemOneCallError`](@ref). Only TypeSafe answers Box questions; the other
backends' wrappers refuse them ([Backends](@ref backends_refusals)).

!!! details "Evidence"
    Measured on 19 October 2026 (`jev-1.14.0`) on this picture, in a run apart
    from the answer above: every edge of both boxes came back within 0.02 of the
    drawing then too. In probes on 19 and 20 October 2026, a Box question about
    a request with no image came back with HTTP 200 and a box at a confidence
    of 0.05 to 0.10. Limits: one picture, two shapes on a plain background.

**Tune it**

- Name the target as a person would point at it — "the red rectangle", not
  "object 1".
- Pin `jev-1.14.0` rather than `jev-preview`, which moves to the next preview
  build when one ships.

## [Long context](@id jev_long_context__next)

TypeSafe reads at most 64k tokens per request, and 32k for the state with any
one question. `long_context = true` raises both to 262k tokens, at the same
price per token, by sending the header `TypeSafe-Danger-Zone-Long-Context: true`.
The docs build does not run this block: a real example needs more than 32k
tokens of text, and its recording would hold all of it.

```julia
contract = read("supplier_agreement.txt", String)     # a long document
r = ask((contract = contract,),
        "auto_renews" => noul("Does `contract` renew automatically unless it is cancelled?");
        long_context = true)

r isa SystemOneFailure && r.error_type == "max_tokens_exceeded" &&
    println("too long even for long context: ", r.message)
```

TypeSafe documents long context as slower and less accurate than the default
limits ([Models](https://docs.typesafe.ai/models)), and an answer does not say
which ran. A request over its context comes back as a 400
[`SystemOneFailure`](@ref) whose `error_type` is `"max_tokens_exceeded"`; its
`message` names the limits, and `long_context` when the request did not set it. `long_context` is a TypeSafe
option: any other backend refuses it before sending, as it refuses every option
it does not know. The natural-language verbs take it as
`options = (long_context = true,)`, and
`with_system_one(TYPESAFEServiceEndpoint; long_context = true) do … end` sets it
for every call inside that names no backend and no options of its own.

!!! details "Evidence"
    Measured on 19 October 2026 (`jev-1.13.0`): a request of about 100k tokens
    came back 400 without the option and was answered in 1.5 s with it; one of
    257k to 262k tokens in a single state took 4.7 to 5.3 s, against 0.25 to
    0.42 s for a small request. The same request was billed the same number of
    tokens with and without the option.

**Tune it**

- Send less before you send more: accuracy falls as unrelated text grows around
  the decision ([Designing good questions](@ref jev_designing_questions)).
  Retrieve the part the question needs, and keep `long_context` for a state
  that is long all the way through.
- Many questions about a long state belong in one request: the state is read,
  and billed, once.

## [Structured guidance](@id jev_structured)

Wherever guidance appears — a question's `instructions`, a Choice option's description, a Score level, a Noul's
`yes` and `no` — Jev takes a string, an object (a `NamedTuple` or a `Dict`), an array (a vector; a tuple only
inside an object or an array, where it is written as an array), or an image, nested up to the limit ([Limits and refusals](@ref jev_limits__next);
[Advanced: structure](https://docs.typesafe.ai/primitives/advanced)). Start with strings. When two options keep
getting confused, give each what it covers, what it does not, and a few examples:

```@example jev
return_topic = choice("Which returns topic is the customer asking about?", (
    return_policy = (what     = "Whether and how an item can be returned",
                     not_for  = "Progress of a return already sent",
                     examples = ["Can I return shoes I've worn once?",
                                 "How long do I have to return an order?"]),
    return_status = (what     = "Progress of a return already sent",
                     not_for  = "Whether and how an item can be returned",
                     examples = ["Has my return arrived yet?",
                                 "When will my refund be paid?"])))

a = ask("I sent the shoes back two weeks ago and have heard nothing since.", "topic" => return_topic)["topic"]
a.choice, a.confidence
```

The field names `what`, `not_for` and `examples` are yours, not the API's. The model reads them with the values,
so keep them short and the same on every option ([Choice](https://docs.typesafe.ai/primitives/choice)). Not every
decision provider takes structure everywhere ([Choosing a decision provider](@ref system_one_backends__next)).

Recorded on 19 October 2026 on `jev-1.13.0` and `jev-1.14.0` with UniLM's development branch after 0.23.0, and on
25 October 2026 on `jev-1.14.0` with UniLM's development branch after 0.26.0, through TypeSafe's API at `/v1`.
