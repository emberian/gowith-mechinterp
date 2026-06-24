# `infra/` — the ephemeral GPU box for the white-box run

This directory rents **one** GPU instance, runs the Gemma-3-12B + Gemma Scope 2
white-box stage on it, pulls the results back, and tears the box down. Everything
that costs money is gated behind explicit confirmation and **two** independent
auto-shutdown guards.

All knobs (region, instance type, auto-shutdown, USD cap) come from
[`config/experiment.yaml`](../config/experiment.yaml) `budget:` block — the
scripts read it; they do not hardcode it.

## The flow

```
bash infra/provision.sh      # 1. rent the g6e.xlarge (asks you to type 'launch')
bash infra/setup_remote.sh   # 2. sync + uv venv + pinned CUDA torch + [gpu] + preflight
bash infra/run_remote.sh     # 3. render -> run_white -> steer  (on the box)
bash infra/fetch.sh          # 4. pull data/runs/ back to local
bash infra/teardown.sh       # 5. TERMINATE the box (asks you to type 'terminate')
```

`provision`, `fetch`, and `teardown` are also wired into the `Justfile`
(`just provision` / `just fetch` / `just teardown`). `setup_remote` and
`run_remote` are run directly with `bash`.

Each step writes/reads `infra/instance.env` (the live box's id + DNS), so steps
2–5 just work after step 1. `setup_remote.sh` internally calls `sync.sh`; you can
also re-run `bash infra/sync.sh` any time to push code changes up.

## What each script does

| script | runs where | does |
|---|---|---|
| `provision.sh` | local | Resolves a Deep Learning AMI via SSM, creates `gowexp-key`/`gowexp-sg` (SSH from your IP only), launches the instance with a resized root + auto-shutdown user-data, waits for status-ok, writes `instance.env`. |
| `sync.sh` | local | `rsync` the repo up to `ubuntu@<box>:~/gowexp` (excludes `.venv .git data/runs report/figs *.pem __pycache__`). |
| `setup_remote.sh` | local→box | sync, install `uv`, `uv venv`, **pinned CUDA torch first**, then `uv pip install -e '.[gpu]'`, copy the HF token up, run `preflight.py`. Stops on any failure. |
| `preflight.py` | box | Asserts CUDA; verifies HF access to the gated model + the SAE repo; **discovers the real SAE ids**; records the model commit; writes `preflight.json`. The gate before GPU spend. |
| `run_remote.sh` | local→box | `cd ~/gowexp`, activate venv, `PYTHONPATH=src`, run `gowexp.render`, `gowexp.run_white`, `gowexp.steer`. |
| `fetch.sh` | box→local | `rsync` `~/gowexp/data/runs/` back to local `data/runs/` (and `preflight.json`). |
| `teardown.sh` | local | Confirm, `terminate-instances`, wait `terminated`, remove `instance.env`. |

## Cost guardrails (read this)

Renting a GPU is the only part of this repo that spends money. Guards, in layers:

1. **On-boot hard stop.** `provision.sh` injects user-data that runs
   `shutdown -h +<budget.auto_shutdown_minutes>` (currently **90 min**) at first
   boot. Even if you walk away mid-run, the box halts on schedule.
2. **Halt ⇒ terminate.** The instance is launched with
   `--instance-initiated-shutdown-behavior terminate`, so that scheduled halt
   **terminates** the instance (not just stops it). Terminating releases the EBS
   root volume too, so storage stops billing as well.
3. **Locked-down ingress.** The `gowexp-sg` security group allows inbound TCP 22
   from **only your current public IP** (`curl https://checkip.amazonaws.com`),
   egress all. Re-running `provision.sh` from a new network adds your new IP.
4. **Explicit human gates.** `provision.sh` prints a COST WARNING and requires you
   to type `launch`; `teardown.sh` requires you to type `terminate`. Set
   `GOWEXP_ASSUME_YES=1` to skip the prompts in automation.
5. **Estimate vs cap.** `g6e.xlarge` (L40S 48 GB) is ~**$1.86/hr** on-demand; the
   run is a few GPU-hours (≈$10). The config `budget.usd_cap` is **$30**.

> Always finish with `bash infra/teardown.sh`. It is the thing that stops the
> bill. If you are unsure whether a box is still up:
> ```
> aws ec2 describe-instances --region us-east-1 \
>   --filters Name=tag:Name,Values=gowexp Name=instance-state-name,Values=running,pending \
>   --query 'Reservations[].Instances[].InstanceId' --output text
> ```

## Environment overrides

| var | used by | effect |
|---|---|---|
| `SPOT=1` | `provision.sh` | request a one-time **spot** instance (cheaper, may be reclaimed). |
| `AMI_ID=ami-…` | `provision.sh` | skip SSM resolution, use this AMI. |
| `PYTORCH_VER=2.7` | `provision.sh` | which DLAMI PyTorch line to resolve (default `2.6`). |
| `ROOT_GB=200` | `provision.sh` | root volume size in GiB (default `200`). |
| `TORCH_SPEC` / `TORCH_INDEX` | `setup_remote.sh` | override the pinned torch wheel + index (default `torch==2.6.0` @ `cu124`). |
| `GOWEXP_ASSUME_YES=1` | `provision.sh`, `teardown.sh` | skip the confirmation prompt. |

## Resolved facts (verified read-only at authoring time, us-east-1)

- **AMI** (SSM `…/oss-nvidia-driver-gpu-pytorch-2.6-ubuntu-22.04/latest/ami-id`):
  `ami-0c702567ccf8b120a` — *Deep Learning OSS Nvidia Driver AMI GPU PyTorch
  2.6.0 (Ubuntu 22.04)*, x86_64, root device `/dev/sda1`. (Resolved live at
  provision time; this is just the value seen while authoring.)
- **Model** `google/gemma-3-12b-it` — **gated** (you must accept its license),
  commit `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- **SAE repo** `google/gemma-scope-2-12b-it` — public. The config's friendly
  width **`64k` maps to the repo's `width_65k`** (2¹⁶ = 65536). Resolved
  `resid_post` ids at width 65k / l0 medium:
  - layer 12 → `layer_12_width_65k_l0_medium`
  - layer 24 → `layer_24_width_65k_l0_medium` *(primary)*
  - layer 31 → `layer_31_width_65k_l0_medium`
  - layer 41 → `layer_41_width_65k_l0_medium`

  `preflight.py` re-discovers these live and writes them to `infra/preflight.json`.

## Files written at runtime (git-ignored)

- `infra/instance.env` — the live box's `INSTANCE_ID` / `PUBLIC_DNS` / `PUBLIC_IP`.
- `infra/gowexp-key.pem` — the SSH private key (mode 600; `*.pem` is git-ignored).
- `infra/preflight.json` — resolved SAE ids + model commit + GPU info.

> The keypair and security group are **left in place** by `teardown.sh` (free,
> reusable). The teardown output prints the exact commands to delete them for a
> full cleanup.
