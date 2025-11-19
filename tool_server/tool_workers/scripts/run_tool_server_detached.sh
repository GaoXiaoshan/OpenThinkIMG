#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/tool_server/tool_workers/scripts/launch_scripts/config/service_apptainer.yaml}"
SERVER_SCRIPT="${PROJECT_ROOT}/tool_server/tool_workers/scripts/launch_scripts/start_server_local.py"
LOG_DIR="${PROJECT_ROOT}/tool_server/logs"
PID_FILE="${LOG_DIR}/tool_server.pid"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/tool_server_${TIMESTAMP}.log"

if [[ -f "${PID_FILE}" ]] && ps -p "$(cat "${PID_FILE}")" > /dev/null 2>&1; then
  echo "Tool server already running with PID $(cat "${PID_FILE}")."
  exit 0
fi

echo "Starting tool server..."
nohup "${PYTHON_BIN}" "${SERVER_SCRIPT}" --config "${CONFIG_PATH}" \
  >> "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

echo "${SERVER_PID}" > "${PID_FILE}"
echo "Tool server started (PID ${SERVER_PID}). Logs: ${LOG_FILE}"
