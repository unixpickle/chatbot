package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) != 4 {
		dieUsage()
	}
	switch os.Args[1] {
	case "train":
		Train(os.Args[2], os.Args[3])
	case "serve":
		port, err := strconv.Atoi(os.Args[2])
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid port:", err)
			os.Exit(1)
		}
		Serve(port, os.Args[3])
	default:
		dieUsage()
	}
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage: dist_train train <param_url> <samples>")
	fmt.Fprintln(os.Stderr, "       dist_train serve <port> <net_file>")
	os.Exit(1)
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
