# Published result records

`audits.json` preserves the nine resolution audits and completion record printed by the successful full protocol-v6 execution. It contains all 36 method summaries, timing summaries, duration-bin errors, and matrix precomputation costs. `metrics.csv` is a flat export of the same method summaries.

These were exported from the completed run’s console. Raw per-trip predictions, exact map extracts, and trained checkpoints are not included here. Running the experiment produces those artifacts. `source_manifest.json` identifies the unchanged core modules.

Timings were measured on one CPU, 2 GiB RAM, without a GPU. Matrix storage is payload only. See the article and protocol for sampling and measurement scope.
