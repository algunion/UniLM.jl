# [Fine-tuning API](@id fine_tuning_api)

Create, retrieve, cancel, and list fine-tuning jobs, and list a job's events and
checkpoints. Training and validation data is uploaded via the [Files API](@ref files_api)
with `purpose="fine-tune"`. OpenAI only.

## Objects

```@docs
FineTuningJob
FineTuningList
```

## Result Types

```@docs
FineTuningSuccess
FineTuningListSuccess
FineTuningFailure
FineTuningCallError
```

## Request Functions

```@docs
create_fine_tuning_job
retrieve_fine_tuning_job
cancel_fine_tuning_job
list_fine_tuning_jobs
list_fine_tuning_events
list_fine_tuning_checkpoints
```

## Usage

```julia
# Upload training data (purpose="fine-tune"), then create a job
train = upload_file("train.jsonl", "fine-tune")
job = create_fine_tuning_job(model="gpt-4o-mini", training_file="file-abc123")
job isa FineTuningSuccess && println("Job: ", job.response.id)

# Retrieve status, list events / checkpoints, list all jobs, cancel
retrieve_fine_tuning_job("ftjob-abc123")
list_fine_tuning_events("ftjob-abc123")
list_fine_tuning_checkpoints("ftjob-abc123")
list_fine_tuning_jobs(limit=10)
cancel_fine_tuning_job("ftjob-abc123")
```
