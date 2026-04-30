//go:build !unix

package tinkernodecap

func probeDisk(root string) (DiskInfo, error) {
	return DiskInfo{}, nil
}

func probeMemory() (MemoryInfo, bool) {
	return MemoryInfo{}, false
}
