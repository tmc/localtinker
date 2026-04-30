//go:build unix

package tinkernodecap

import "syscall"

func probeDisk(root string) (DiskInfo, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(rootStatPath(root), &st); err != nil {
		return DiskInfo{}, err
	}
	return DiskInfo{
		RootBytes:      uint64(st.Blocks) * uint64(st.Bsize),
		AvailableBytes: uint64(st.Bavail) * uint64(st.Bsize),
	}, nil
}

func probeMemory() (MemoryInfo, bool) {
	return MemoryInfo{}, false
}
