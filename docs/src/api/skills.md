# [Skills API](@id skills_api)

Upload a skill — a folder whose root holds a SKILL.md, or a `.zip` of that folder — version it, read
it back, download it and delete it, on each provider's own Skills API: OpenAI's
(`OPENAIServiceEndpoint`, the default) and Anthropic's (`ANTHROPICServiceEndpoint`), each declaring
`:skills`. The verbs and their results are the same on both; what differs is below. An OpenAI skill
loads into a container through a [`SkillReference`](@ref), and a zip goes into one directly as an
[`InlineSkill`](@ref) ([Containers API](@ref containers_api)). An Anthropic skill loads into the
code-execution container of a Messages request (`container.skills`), which a `Chat` does not send yet.

A skill's SKILL.md starts with front matter naming it and saying what it does — the model reads the
`description` to decide when to use it — and goes on with its instructions:

```markdown
---
name: csv-sum
description: Sums a column of a CSV file. Use it when asked for a column's total.
---
Read the file with pandas and print the column's sum.
```

A folder goes as one multipart part per file — every file in it, its subfolders' too — each named by
its path under the folder's name (`csv-sum/SKILL.md`), the layout both providers document: "All files
must be in the same top-level directory and must include a SKILL.md file at the root of that
directory" ([Anthropic](https://platform.claude.com/docs/en/api/skills/create)); "Each part's filename
includes the path inside a single top-level folder"
([OpenAI](https://developers.openai.com/api/docs/guides/tools-skills)). A `.zip` goes as one part.

| | OpenAI | Anthropic |
| :--- | :--- | :--- |
| `create_skill` | a folder as `files[]` parts, a zip as `files`; at most 500 files | a folder or a zip as `files[]` parts; `display_name`, one line of at most 255 characters |
| `SkillObject` | `name`, `description`, `default_version`, `latest_version` (version numbers) | `name` is the `display_name`; `latest_version` the `latest_version_id`; no description or default version; `source` and `updated_at` in `raw` |
| a version's `version` | its number: `"1"`, `"2"`, … | its id: `"skver_…"` |
| `list_skills`, `list_skill_versions` | `limit` (0 to 100), `after` (the `last_id`), `order` (`"asc"`, `"desc"`) | `limit` (1 to 1000), `after` (the `next_page` cursor), `source` (skills only); versions newest first |
| `update_skill(id; default_version)` | sets the default version | refused: an Anthropic skill has no default version |
| `create_skill_version(id, path; default)` | `default=true` makes it the default | `default` refused: the newest version is the `latest_version` |
| `skill_content(id; version)` | the skill's default version without `version` | a version only: without `version`, refused |
| `delete_skill` | `object` `"skill.deleted"`, `"deleted": true` | `type` `"skill_deleted"`; deletes every version too |
| `delete_skill_version` | not the default while other versions remain; the last one deletes the skill | not a skill's only version |

Each refusal in the table is an `ArgumentError` before anything is sent, and so are a `path` that is no
folder or `.zip` file, a folder without a SKILL.md at its root, and an empty skill id or version. An
Anthropic Skills request sends the key, `anthropic-version` and the endpoint's `workspace_id`, and
neither its `betas` nor its `user_profile_id`: the API "is out of beta and needs no beta header", and
`skills-2025-10-02` would select the beta's response shapes, whose versions are timestamps
([Skills guide](https://platform.claude.com/docs/en/build-with-claude/skills-guide)). A reply without a
field the provider's reference requires, or naming another kind of record, is a
[`SkillCallError`](@ref), never a result with the field made up.

## Objects

```@docs
SkillObject
SkillVersionObject
SkillList
SkillVersionList
```

## Result Types

```@docs
SkillSuccess
SkillListSuccess
SkillVersionSuccess
SkillVersionListSuccess
SkillDeleteSuccess
SkillFailure
SkillCallError
```

`skill_content` returns the zip's bytes as a [`FileContentSuccess`](@ref), which
[`save_file_content`](@ref) writes to disk.

## Request Functions

```@docs
create_skill
list_skills
retrieve_skill
update_skill
delete_skill
skill_content
create_skill_version
list_skill_versions
retrieve_skill_version
delete_skill_version
```

## Usage

```julia
# OpenAI: upload the folder, add a version, make it the default
s = create_skill("skills/csv-sum")
s isa SkillSuccess || error("upload failed: $s")
id = s.response.id
v = create_skill_version(id, "skills/csv-sum"; default=true)
retrieve_skill(id).response.default_version          # "2"

# Load it into a container by reference, and its version 1 inline, from its zip: an inline skill is
# named and described as its SKILL.md names and describes it
zip = skill_content(id; version="1")
zip isa FileContentSuccess && save_file_content(zip, "csv-sum-1.zip")
c = create_container(name="sandbox", skills=[SkillReference(s.response),
                     InlineSkill("csv-sum-1.zip"; name=s.response.name, description=s.response.description)])

# Anthropic: a label, versions by id, the zip of one version
a = create_skill("skills/csv-sum"; display_name="CSV sum", service=ANTHROPICServiceEndpoint)
versions = list_skill_versions(a.response.id; service=ANTHROPICServiceEndpoint)
skill_content(a.response.id; version=a.response.latest_version, service=ANTHROPICServiceEndpoint)
delete_skill(a.response.id; service=ANTHROPICServiceEndpoint)   # its versions with it
```
