# [Fine-tuning API](@id fine_tuning_api)

OpenAI restricts new jobs to eligible existing customers. Job creation ends
January 6, 2027; inference remains available until the base model retires.
See [OpenAI's availability schedule](https://developers.openai.com/api/docs/deprecations).

Create, retrieve, cancel, pause, resume and list fine-tuning jobs, list a job's events and
checkpoints, and share a checkpoint with other projects. Training and validation data is uploaded
via the [Files API](@ref files_api) with `purpose="fine-tune"`. OpenAI only.

## Objects

```@docs
FineTuningJob
FineTuningList
WandbIntegration
```

## Result Types

```@docs
FineTuningSuccess
FineTuningListSuccess
FineTuningDeleteSuccess
FineTuningFailure
FineTuningCallError
```

## Request Functions

```@docs
create_fine_tuning_job
retrieve_fine_tuning_job
cancel_fine_tuning_job
pause_fine_tuning_job
resume_fine_tuning_job
list_fine_tuning_jobs
list_fine_tuning_events
list_fine_tuning_checkpoints
create_fine_tuning_checkpoint_permission
list_fine_tuning_checkpoint_permissions
delete_fine_tuning_checkpoint_permission
```

## Usage

```julia
# Upload training data (purpose="fine-tune"), then create a job on a base model your
# organization may still fine-tune: OpenAI's GPT-6 and GPT-5.6 models take no fine-tuning
base_model = "…"
train = upload_file("train.jsonl", "fine-tune")
job = create_fine_tuning_job(model=base_model, training_file="file-abc123", seed=7,
                             integrations=[WandbIntegration(project="my-runs")],
                             metadata=Dict("team" => "search"))
job isa FineTuningSuccess && println("Job: ", job.response.id)

# Retrieve status, list events / checkpoints, list jobs by metadata, pause / resume, cancel
retrieve_fine_tuning_job("ftjob-abc123")
list_fine_tuning_events("ftjob-abc123")
list_fine_tuning_checkpoints("ftjob-abc123")
list_fine_tuning_jobs(limit=10, metadata=Dict("team" => "search"))
pause_fine_tuning_job("ftjob-abc123")
resume_fine_tuning_job("ftjob-abc123")
cancel_fine_tuning_job("ftjob-abc123")
```
