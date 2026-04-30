package tinkerartifact

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// FileLock is a narrow placeholder for Hugging Face compatible lock files.
type FileLock struct {
	path string
	file *os.File
	once sync.Once
}

// LockFile opens path and takes an exclusive advisory lock when supported.
func LockFile(path string) (*FileLock, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create lock directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o666)
	if err != nil {
		return nil, fmt.Errorf("open lock file: %w", err)
	}
	l := &FileLock{path: path, file: f}
	if err := lockFile(f); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("lock %s: %w", path, err)
	}
	return l, nil
}

// Path returns the lock file path.
func (l *FileLock) Path() string {
	if l == nil {
		return ""
	}
	return l.path
}

// Unlock releases the advisory lock and closes the file.
func (l *FileLock) Unlock() error {
	if l == nil || l.file == nil {
		return nil
	}
	var err error
	l.once.Do(func() {
		if e := unlockFile(l.file); e != nil {
			err = e
		}
		if e := l.file.Close(); err == nil && e != nil {
			err = e
		}
	})
	return err
}
