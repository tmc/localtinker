// Package tinkermgccl adapts localtinker node state to Magical planning data.
//
// The adapter builds offline planner snapshots only. It does not start
// providers, probe hardware, or claim JACCL, RDMA, NCCL, or transport
// readiness.
package tinkermgccl
