#!/usr/bin/env bash
# =============================================================================
# infra/provision.sh — rent ONE GPU box for the gowexp white-box run.
#
#   Rents a single g6e.xlarge (NVIDIA L40S, 48 GB) on-demand instance in
#   us-east-1, on a Deep Learning AMI that already ships PyTorch + NVIDIA
#   drivers, locked down to SSH-from-your-IP-only, with TWO independent
#   cost guards:
#     1. user-data runs `shutdown -h +<auto_shutdown_minutes>` on boot, and
#     2. --instance-initiated-shutdown-behavior terminate, so that shutdown
#        TERMINATES (not just stops) the box -> the EBS volume stops billing too.
#
#   Source of truth for region / instance_type / auto_shutdown is
#   config/experiment.yaml (budget: block). Nothing here is hardcoded silently.
#
# Usage:
#     bash infra/provision.sh            # on-demand g6e.xlarge
#     SPOT=1 bash infra/provision.sh     # request a spot instance instead
#     AMI_ID=ami-xxxx bash infra/provision.sh   # override the resolved AMI
#
# Env overrides (all optional):
#     SPOT=1           request a spot instance (cheaper, can be reclaimed)
#     AMI_ID=ami-...   skip SSM resolution, use this AMI
#     PYTORCH_VER=2.6  which DLAMI PyTorch line to resolve (default 2.6)
#     ROOT_GB=200      root volume size in GiB (default 200)
#
# Writes: infra/instance.env  (INSTANCE_ID / PUBLIC_DNS / PUBLIC_IP)
#         infra/gowexp-key.pem (private key, chmod 600, if newly created)
#
# Read-only AWS calls only happen here for resolution; the single billable
# call is `aws ec2 run-instances`, guarded behind the cost warning below.
# =============================================================================
set -euo pipefail

# --- locate repo root regardless of where we're invoked from -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/config/experiment.yaml"
KEY_NAME="gowexp-key"
KEY_FILE="${SCRIPT_DIR}/gowexp-key.pem"
SG_NAME="gowexp-sg"
INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
NAME_TAG="gowexp"

# Defaults that may be overridden by env.
PYTORCH_VER="${PYTORCH_VER:-2.6}"   # DLAMI PyTorch line; 2.6 pairs with the cu124 wheel in setup_remote.sh
ROOT_GB="${ROOT_GB:-200}"
SPOT="${SPOT:-0}"

# Rough on-demand list price for g6e.xlarge in us-east-1 (USD/hr). Display-only;
# AWS billing is authoritative. Used purely to print a cost estimate.
EST_USD_PER_HR="1.86"

err()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

command -v aws >/dev/null 2>&1 || err "aws CLI not found on PATH."
command -v python3 >/dev/null 2>&1 || err "python3 not found on PATH."
command -v curl >/dev/null 2>&1 || err "curl not found on PATH."
[[ -f "${CONFIG}" ]] || err "config not found: ${CONFIG}"

# --- read the budget block from config/experiment.yaml (source of truth) -----
# A tiny pyyaml reader. We deliberately do NOT hardcode these values; the YAML
# is the frozen contract. Fail loudly if pyyaml is missing or keys are absent.
read_cfg() {
  python3 - "$CONFIG" "$1" <<'PY'
import sys, yaml
cfg_path, dotted = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    data = yaml.safe_load(f)
cur = data
for part in dotted.split("."):
    if not isinstance(cur, dict) or part not in cur:
        sys.stderr.write(f"missing key in config: {dotted}\n")
        sys.exit(3)
    cur = cur[part]
print(cur)
PY
}

REGION="${REGION:-$(read_cfg budget.region)}"            || err "could not read budget.region from config"
INSTANCE_TYPE="${INSTANCE_TYPE:-$(read_cfg budget.instance_type)}" || err "could not read budget.instance_type from config"
AUTO_SHUTDOWN_MIN="$(read_cfg budget.auto_shutdown_minutes)" || err "could not read budget.auto_shutdown_minutes from config"
USD_CAP="$(read_cfg budget.usd_cap)"          || err "could not read budget.usd_cap from config"

# Sanity: auto_shutdown must be a positive integer (it becomes `shutdown -h +N`).
[[ "${AUTO_SHUTDOWN_MIN}" =~ ^[0-9]+$ && "${AUTO_SHUTDOWN_MIN}" -gt 0 ]] \
  || err "budget.auto_shutdown_minutes must be a positive integer, got '${AUTO_SHUTDOWN_MIN}'"

note "config: region=${REGION} instance_type=${INSTANCE_TYPE} auto_shutdown=${AUTO_SHUTDOWN_MIN}min usd_cap=\$${USD_CAP}"

# --- resolve the Deep Learning AMI via SSM public parameters ------------------
# AWS publishes the latest DLAMI ids as public SSM parameters. Resolving by the
# parameter NAME (not a hardcoded ami-id) means we always get the current,
# patched image and it works across accounts. The PyTorch-2.6 / Ubuntu-22.04
# line ships NVIDIA drivers + a CUDA 12.4-compatible stack, matching the torch
# wheel we pin in setup_remote.sh.
#
# Verified working (read-only) at authoring time:
#   /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.6-ubuntu-22.04/latest/ami-id
#   -> ami-0c702567ccf8b120a  ("Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04)")
SSM_AMI_PARAM="/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-${PYTORCH_VER}-ubuntu-22.04/latest/ami-id"

if [[ -n "${AMI_ID:-}" ]]; then
  note "using AMI_ID override: ${AMI_ID}"
else
  note "resolving DLAMI from SSM: ${SSM_AMI_PARAM}"
  AMI_ID="$(aws ssm get-parameters \
              --names "${SSM_AMI_PARAM}" \
              --region "${REGION}" \
              --query 'Parameters[0].Value' \
              --output text 2>/dev/null || true)"
  if [[ -z "${AMI_ID}" || "${AMI_ID}" == "None" ]]; then
    # Surface invalid parameters so the failure is diagnosable, then bail.
    aws ssm get-parameters --names "${SSM_AMI_PARAM}" --region "${REGION}" \
      --query 'InvalidParameters' --output text 2>/dev/null || true
    err "SSM did not resolve an AMI for '${SSM_AMI_PARAM}'. Try a different PYTORCH_VER (e.g. 2.7) or pass AMI_ID=ami-..."
  fi
  note "resolved AMI: ${AMI_ID}"
fi

# --- ensure the SSH keypair exists -------------------------------------------
# If AWS already has the keypair but we lack the .pem locally, we cannot recover
# the private key (AWS never re-exposes it) -> fail loudly with the fix.
if aws ec2 describe-key-pairs --key-names "${KEY_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  note "keypair '${KEY_NAME}' already exists in AWS."
  if [[ ! -f "${KEY_FILE}" ]]; then
    err "keypair '${KEY_NAME}' exists in AWS but ${KEY_FILE} is missing locally.
     AWS cannot re-export a private key. Either restore ${KEY_FILE} from backup,
     or delete the keypair and re-run:
         aws ec2 delete-key-pair --key-name ${KEY_NAME} --region ${REGION}"
  fi
else
  note "creating keypair '${KEY_NAME}' -> ${KEY_FILE}"
  aws ec2 create-key-pair \
    --key-name "${KEY_NAME}" \
    --region "${REGION}" \
    --query 'KeyMaterial' \
    --output text > "${KEY_FILE}"
  chmod 600 "${KEY_FILE}"
  note "saved private key (chmod 600)."
fi
# Always re-assert perms; ssh refuses world-readable keys.
chmod 600 "${KEY_FILE}" 2>/dev/null || true

# --- resolve our current public IP for the SG ingress rule -------------------
MY_IP="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')" \
  || err "could not determine your public IP via checkip.amazonaws.com"
[[ "${MY_IP}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || err "got a non-IPv4 public IP: '${MY_IP}'"
MY_CIDR="${MY_IP}/32"
note "your public IP: ${MY_IP} (SSH will be locked to ${MY_CIDR})"

# --- ensure the security group exists, allow 22 from our IP only -------------
VPC_ID="$(aws ec2 describe-vpcs \
            --filters Name=isDefault,Values=true \
            --region "${REGION}" \
            --query 'Vpcs[0].VpcId' --output text 2>/dev/null || true)"
[[ -n "${VPC_ID}" && "${VPC_ID}" != "None" ]] || err "no default VPC found in ${REGION}; create one or set a VPC manually."

SG_ID="$(aws ec2 describe-security-groups \
           --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
           --region "${REGION}" \
           --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"

if [[ -z "${SG_ID}" || "${SG_ID}" == "None" ]]; then
  note "creating security group '${SG_NAME}' in ${VPC_ID}"
  SG_ID="$(aws ec2 create-security-group \
             --group-name "${SG_NAME}" \
             --description "gowexp ephemeral GPU box: SSH from caller IP only" \
             --vpc-id "${VPC_ID}" \
             --region "${REGION}" \
             --query 'GroupId' --output text)"
  note "created SG: ${SG_ID}"
else
  note "reusing security group '${SG_NAME}' (${SG_ID})"
fi

# Add the ingress rule for our /32 if it's not already present. Re-running with a
# new IP simply adds another rule; AWS rejects exact duplicates, which we ignore.
if aws ec2 describe-security-groups --group-ids "${SG_ID}" --region "${REGION}" \
     --query "SecurityGroups[0].IpPermissions[?ToPort==\`22\`].IpRanges[].CidrIp" \
     --output text 2>/dev/null | tr '\t' '\n' | grep -qx "${MY_CIDR}"; then
  note "SG already allows SSH from ${MY_CIDR}"
else
  note "authorizing SSH (tcp/22) from ${MY_CIDR}"
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_CIDR},Description=gowexp-ssh}]" \
    --region "${REGION}" >/dev/null 2>&1 \
    || note "(ingress rule already present or partially applied — continuing)"
fi
# Egress all is the default-VPC SG behavior; we leave it as-is (egress all).

# --- build user-data: the on-box hard cost guard -----------------------------
# This runs as root at first boot. `shutdown -h +N` schedules a halt N minutes
# out; combined with --instance-initiated-shutdown-behavior terminate below,
# that halt TERMINATES the instance and releases the EBS volume. Even if the
# experiment hangs or we walk away, the box self-destructs.
#
# IMPORTANT: the AWS CLI v2 base64-encodes --user-data FOR US. So we pass the
# RAW script via file:// (never pre-encode, or it double-encodes and the guard
# silently never runs). A temp file also sidesteps shell-quoting the script.
USER_DATA_FILE="$(mktemp "${TMPDIR:-/tmp}/gowexp-userdata.XXXXXX")"
# Clean up the temp user-data file on any exit path.
trap 'rm -f "${USER_DATA_FILE}"' EXIT
cat > "${USER_DATA_FILE}" <<EOF
#!/bin/bash
# gowexp auto-shutdown guard — terminates the box ${AUTO_SHUTDOWN_MIN} min after boot.
echo "gowexp: scheduling shutdown -h +${AUTO_SHUTDOWN_MIN} (hard cost guard)" | logger -t gowexp
shutdown -h +${AUTO_SHUTDOWN_MIN} "gowexp auto-shutdown cost guard (${AUTO_SHUTDOWN_MIN} min)"
EOF

# Root device name for this AMI family is /dev/sda1 (verified via describe-images).
# We resize the root to ROOT_GB on gp3 and force DeleteOnTermination=true so the
# disk dies with the instance.
BLOCK_DEV_MAPPING="$(cat <<EOF
[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":${ROOT_GB},"VolumeType":"gp3","DeleteOnTermination":true}}]
EOF
)"

TAG_SPEC="ResourceType=instance,Tags=[{Key=Name,Value=${NAME_TAG}},{Key=project,Value=gowexp},{Key=auto-shutdown-min,Value=${AUTO_SHUTDOWN_MIN}}]"

# Optional explicit subnet (targets a specific AZ — used to sweep around
# InsufficientInstanceCapacity in the default AZ).
SUBNET_ARGS=()
if [[ -n "${SUBNET_ID:-}" ]]; then
  SUBNET_ARGS+=(--subnet-id "${SUBNET_ID}")
  note "targeting subnet ${SUBNET_ID}"
fi

# Optional spot market options.
SPOT_ARGS=()
if [[ "${SPOT}" == "1" ]]; then
  # one-time spot request; if reclaimed, the instance terminates (we don't persist).
  SPOT_ARGS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  note "SPOT=1 -> requesting a one-time SPOT instance (cheaper, may be reclaimed)."
fi

# ============================== COST WARNING =================================
cat <<WARN

  ##############################################################################
  #                            >>> COST WARNING <<<                            #
  #                                                                            #
  #  About to launch a billable GPU instance:                                 #
  #      type   : ${INSTANCE_TYPE}  (NVIDIA L40S 48GB)
  #      region : ${REGION}
  #      market : $( [[ "${SPOT}" == "1" ]] && echo "SPOT (variable price)" || echo "ON-DEMAND ~\$${EST_USD_PER_HR}/hr" )
  #      disk   : ${ROOT_GB} GiB gp3 root (deleted on terminate)
  #                                                                            #
  #  GUARDS: shutdown -h +${AUTO_SHUTDOWN_MIN}min on boot AND shutdown=>TERMINATE.
  #  Experiment USD cap (config): \$${USD_CAP}.  Run infra/teardown.sh when done.
  #  THIS COSTS REAL MONEY UNTIL THE BOX TERMINATES.                           #
  ##############################################################################

WARN

# Honor an explicit opt-out gate so this is safe to source/dry-run in CI.
if [[ "${GOWEXP_ASSUME_YES:-0}" != "1" ]]; then
  read -r -p "Type 'launch' to provision (anything else aborts): " CONFIRM
  [[ "${CONFIRM}" == "launch" ]] || err "aborted by user (no instance launched)."
fi

# --- launch -------------------------------------------------------------------
note "launching ${INSTANCE_TYPE} from ${AMI_ID} ..."
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "${AMI_ID}" \
  --instance-type "${INSTANCE_TYPE}" \
  --key-name "${KEY_NAME}" \
  --security-group-ids "${SG_ID}" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "${BLOCK_DEV_MAPPING}" \
  --user-data "file://${USER_DATA_FILE}" \
  --tag-specifications "${TAG_SPEC}" \
  ${SUBNET_ARGS[@]+"${SUBNET_ARGS[@]}"} \
  ${SPOT_ARGS[@]+"${SPOT_ARGS[@]}"} \
  --region "${REGION}" \
  --query 'Instances[0].InstanceId' \
  --output text)"

[[ -n "${INSTANCE_ID}" && "${INSTANCE_ID}" != "None" ]] || err "run-instances did not return an instance id."
note "launched instance: ${INSTANCE_ID}"

# --- wait for running, then status-ok ----------------------------------------
note "waiting for instance to enter 'running' ..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${REGION}"

note "waiting for status checks (instance + system) to pass — this can take a few minutes ..."
aws ec2 wait instance-status-ok --instance-ids "${INSTANCE_ID}" --region "${REGION}"

# --- fetch connection details ------------------------------------------------
read -r PUBLIC_DNS PUBLIC_IP < <(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --region "${REGION}" \
  --query 'Reservations[0].Instances[0].[PublicDnsName,PublicIpAddress]' \
  --output text)

[[ -n "${PUBLIC_DNS}" && "${PUBLIC_DNS}" != "None" ]] || err "instance has no public DNS; cannot SSH."

# --- persist instance.env (consumed by sync/setup/run/fetch/teardown) --------
cat > "${INSTANCE_ENV}" <<EOF
# gowexp instance — written by infra/provision.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Source of truth for the live box. Deleted/overwritten on teardown/re-provision.
INSTANCE_ID=${INSTANCE_ID}
PUBLIC_DNS=${PUBLIC_DNS}
PUBLIC_IP=${PUBLIC_IP}
REGION=${REGION}
INSTANCE_TYPE=${INSTANCE_TYPE}
AMI_ID=${AMI_ID}
AUTO_SHUTDOWN_MIN=${AUTO_SHUTDOWN_MIN}
EOF
note "wrote ${INSTANCE_ENV}"

# --- summary -----------------------------------------------------------------
cat <<DONE

  ----------------------------------------------------------------------------
  gowexp box is UP.
      instance : ${INSTANCE_ID}  (${INSTANCE_TYPE}, ${REGION})
      dns      : ${PUBLIC_DNS}
      ip       : ${PUBLIC_IP}
      est cost : ~\$${EST_USD_PER_HR}/hr on-demand  (self-terminates in ${AUTO_SHUTDOWN_MIN} min)

  SSH:
      ssh -i ${KEY_FILE} -o StrictHostKeyChecking=accept-new ubuntu@${PUBLIC_DNS}

  Next:
      bash infra/setup_remote.sh     # sync + venv + torch + preflight
      bash infra/run_remote.sh       # render + run-white + steer
      bash infra/fetch.sh            # pull data/runs back
      bash infra/teardown.sh         # TERMINATE the box (do this when done!)
  ----------------------------------------------------------------------------

DONE
