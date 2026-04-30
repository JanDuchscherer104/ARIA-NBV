# Service Desk Templates

Use these as editable drafts. Replace placeholders before sending. Do not commit filled project IDs, user IDs, or DSS paths.

## AI Systems DSS Quota Request

Subject: Request AI Systems DSS allocation for ARIA-NBV workloads

Hello LRZ Support,

please allocate an AI Systems DSS container for project `<PROJECT_ID>` for ARIA-NBV dataset creation, oracle/VIN cache generation, and training workloads.

Initial request:

- Capacity: 1 TB, scalable up to 4 TB if needed.
- File count: sufficient for sharded/chunked ML datasets and checkpoint/log outputs.
- Users/groups: `<PROJECT_GROUP_OR_USERS>`.
- Workload: GPU/CPU Slurm jobs on LRZ AI Systems using containerized PyTorch workloads.

The storage will hold datasets, generated caches, immutable VIN offline stores, checkpoints, Slurm logs, W&B logs, containers, package/model caches, and temporary files. `$HOME` will be used only for code and small configuration.

Thank you.

## AI Systems Access Check

Subject: Check AI Systems access for project `<PROJECT_ID>`

Hello LRZ Support,

please confirm whether user `<LRZ_USER>` is enabled for LRZ AI Systems access through project `<PROJECT_ID>` and the relevant AI compute group. If additional group membership or QOS configuration is required, please advise the Project Master User steps.

Thank you.

## MCML Access Check

Subject: Check MCML partition/QOS access for project `<PROJECT_ID>`

Hello LRZ Support,

please confirm whether project `<PROJECT_ID>` and user `<LRZ_USER>` can use MCML AI Systems partitions and which QOS or Slurm account settings are required.

Thank you.
