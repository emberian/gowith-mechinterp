#!/usr/bin/env bash
# =============================================================================
# infra/teardown.sh — TERMINATE the GPU box. The thing that stops the bill.
#
#   Reads infra/instance.env, confirms, terminates the instance, waits for the
#   terminated state, and removes instance.env. Leaves the keypair and security
#   group in place (cheap/free, reusable) — deletion commands are printed for
#   when you want a full cleanup.
#
# Usage:  bash infra/teardown.sh
#         GOWEXP_ASSUME_YES=1 bash infra/teardown.sh   # skip the prompt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
KEY_NAME="gowexp-key"
KEY_FILE="${SCRIPT_DIR}/gowexp-key.pem"
SG_NAME="gowexp-sg"

err()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

command -v aws >/dev/null 2>&1 || err "aws CLI not found on PATH."
[[ -f "${INSTANCE_ENV}" ]] || err "no ${INSTANCE_ENV} — nothing recorded to tear down.
     (If a box is running, find it with: aws ec2 describe-instances \\
        --filters Name=tag:Name,Values=gowexp Name=instance-state-name,Values=running \\
        --query 'Reservations[].Instances[].InstanceId' --output text)"

# shellcheck source=/dev/null
source "${INSTANCE_ENV}"
[[ -n "${INSTANCE_ID:-}" ]] || err "INSTANCE_ID not set in ${INSTANCE_ENV}."
REGION="${REGION:-us-east-1}"

# Show current state before we act.
STATE="$(aws ec2 describe-instances \
           --instance-ids "${INSTANCE_ID}" \
           --region "${REGION}" \
           --query 'Reservations[0].Instances[0].State.Name' \
           --output text 2>/dev/null || echo "unknown")"

note "instance ${INSTANCE_ID} (${REGION}) current state: ${STATE}"

if [[ "${STATE}" == "terminated" ]]; then
  note "already terminated. cleaning up local instance.env."
  rm -f "${INSTANCE_ENV}"
  exit 0
fi
if [[ "${STATE}" == "unknown" ]]; then
  note "could not read instance state (already gone, or wrong region?). Removing ${INSTANCE_ENV}."
  rm -f "${INSTANCE_ENV}"
  exit 0
fi

# Confirm before destroying.
if [[ "${GOWEXP_ASSUME_YES:-0}" != "1" ]]; then
  echo
  echo "  This will TERMINATE ${INSTANCE_ID} (${INSTANCE_TYPE:-?}) and DELETE its root volume."
  read -r -p "  Type 'terminate' to proceed (anything else aborts): " CONFIRM
  [[ "${CONFIRM}" == "terminate" ]] || err "aborted by user (instance left running)."
fi

note "terminating ${INSTANCE_ID} ..."
aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" \
  --query 'TerminatingInstances[0].{Id:InstanceId,Prev:PreviousState.Name,Now:CurrentState.Name}' \
  --output table

note "waiting for 'terminated' ..."
aws ec2 wait instance-terminated --instance-ids "${INSTANCE_ID}" --region "${REGION}"
note "instance ${INSTANCE_ID} is TERMINATED — billing for it has stopped."

# Drop the local pointer so subsequent sync/run/fetch fail loud instead of
# targeting a dead box.
rm -f "${INSTANCE_ENV}"
note "removed ${INSTANCE_ENV}"

cat <<DONE

  ----------------------------------------------------------------------------
  Teardown complete. The instance and its EBS root volume are gone.

  Left in place (free / reusable for the next provision):
      keypair:  ${KEY_NAME}   (local key: ${KEY_FILE})
      sec grp:  ${SG_NAME}

  To delete those too (full cleanup):
      aws ec2 delete-key-pair --key-name ${KEY_NAME} --region ${REGION}
      rm -f ${KEY_FILE}
      SG_ID=\$(aws ec2 describe-security-groups --filters Name=group-name,Values=${SG_NAME} \\
                --region ${REGION} --query 'SecurityGroups[0].GroupId' --output text)
      aws ec2 delete-security-group --group-id "\$SG_ID" --region ${REGION}

  REMINDER: confirm nothing lingers ->
      aws ec2 describe-instances --region ${REGION} \\
        --filters Name=tag:Name,Values=gowexp Name=instance-state-name,Values=running,pending \\
        --query 'Reservations[].Instances[].InstanceId' --output text
  ----------------------------------------------------------------------------

DONE
