set -x
pkill -9 VLLM::Worker_DP
pkill -9 VLLM::DPCoordin
pkill -9 vllm
pkill -9 VLLM::EngineCor
