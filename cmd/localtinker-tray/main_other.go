//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("localtinker-tray is only supported on macOS")
}
