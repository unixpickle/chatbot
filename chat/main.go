package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/chatbot"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: chat <bot_file>")
		os.Exit(1)
	}
	bot, err := chatbot.LoadBot(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load bot:", err)
		os.Exit(1)
	}
	chat := chatbot.NewChat(bot)

	for {
		msg := readMessage()
		chat.Send(msg)
		for {
			resp, more := chat.Receive()
			fmt.Println("Bot>", resp)
			if !more {
				break
			}
		}
	}
}

func readMessage() string {
	fmt.Print("You> ")
	var res []byte
	for {
		ch := make([]byte, 1)
		if n, err := os.Stdin.Read(ch); err != nil {
			panic(err)
		} else if n == 0 {
			continue
		}
		if ch[0] == '\n' {
			break
		}
		res = append(res, ch[0])
	}
	return string(res)
}
