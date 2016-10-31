// Command facebook runs a chatbot on a Facebook account,
// automatically responding to messages as they come in
// from various sources.
package main

import (
	"fmt"
	"os"
	"time"

	"github.com/howeyc/gopass"
	"github.com/unixpickle/chatbot"
	"github.com/unixpickle/fbmsgr"
)

type State int

const (
	WaitingForHuman State = iota
	HumanTyping
	BotTyping
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: facebook <bot_file> <fb_username>")
		os.Exit(1)
	}
	bot, err := chatbot.LoadBot(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load bot:", err)
		os.Exit(1)
	}

	fmt.Print("FB password: ")
	passwd, err := gopass.GetPasswd()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read password:", err)
		os.Exit(1)
	}

	sess, err := fbmsgr.Auth(os.Args[2], string(passwd))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not authenticate:", err)
		os.Exit(1)
	}

	messageLoop(sess, bot)
}

func messageLoop(sess *fbmsgr.Session, bot *chatbot.Bot) {
	chats := map[string]chan<- fbmsgr.Event{}
	for {
		event, err := sess.ReadEvent()
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error reading event:", err)
			os.Exit(1)
		}
		var threadID string
		var group bool
		switch event := event.(type) {
		case fbmsgr.MessageEvent:
			if event.SenderFBID != sess.FBID() {
				if event.GroupThread != "" {
					group = true
					threadID = event.GroupThread
				} else {
					threadID = event.OtherUser
				}
			}
		case fbmsgr.TypingEvent:
			if event.SenderFBID != sess.FBID() {
				if event.GroupThread != "" {
					group = true
					threadID = event.GroupThread
				} else {
					threadID = event.SenderFBID
				}
			}
		}
		if threadID != "" {
			ch := chats[threadID]
			if ch == nil {
				newChan := make(chan fbmsgr.Event, 10)
				chats[threadID] = newChan
				go handleThread(threadID, group, newChan, sess, bot)
				ch = newChan
			}
			ch <- event
		}
	}
}

func handleThread(thread string, group bool, events <-chan fbmsgr.Event, sess *fbmsgr.Session,
	bot *chatbot.Bot) {
	chat := chatbot.NewChat(bot)

	state := WaitingForHuman
	readHistory(thread, sess, chat)

	var noResponseCount int
	for {
		switch state {
		case WaitingForHuman:
			select {
			case <-time.After(time.Second * 20):
				if noResponseCount < 3 {
					state = BotTyping
					sendTyping(sess, thread, group, true)
				}
			case e := <-events:
				if startedTyping(e) {
					state = HumanTyping
				} else if msg := eventMessage(e); msg != "" {
					noResponseCount = 0
					markMessageRead(sess, e)
					if chat.Send(msg) {
						state = WaitingForHuman
					} else {
						state = BotTyping
						sendTyping(sess, thread, group, true)
					}
				}
			}
		case BotTyping:
			select {
			case <-time.After(time.Second * 10):
				msg, more := chat.Receive()
				sendMessage(sess, thread, group, msg)
				noResponseCount++
				if more {
					state = BotTyping
					sendTyping(sess, thread, group, true)
				} else {
					sendTyping(sess, thread, group, false)
					state = WaitingForHuman
				}
			case e := <-events:
				if startedTyping(e) {
					sendTyping(sess, thread, group, false)
					state = HumanTyping
				} else if msg := eventMessage(e); msg != "" {
					if chat.Send(msg) {
						state = WaitingForHuman
						sendTyping(sess, thread, group, false)
					}
				}
			}
		case HumanTyping:
			e := <-events
			if stoppedTyping(e) {
				state = WaitingForHuman
			} else if msg := eventMessage(e); msg != "" {
				markMessageRead(sess, e)
				noResponseCount = 0
				if chat.Send(msg) {
					state = WaitingForHuman
				} else {
					state = BotTyping
					sendTyping(sess, thread, group, true)
				}
			}
		}
	}
}

func readHistory(threadID string, sess *fbmsgr.Session, chat *chatbot.Chat) (shouldSend bool) {
	history, err := sess.ActionLog(threadID, time.Time{}, 0, 20)
	if err != nil {
		fmt.Fprintln(os.Stderr, "List actions:", err)
		os.Exit(1)
	}
	for _, action := range history {
		if msg, ok := action.(*fbmsgr.MessageAction); ok {
			if msg.Body != "" {
				if msg.AuthorFBID() == sess.FBID() {
					shouldSend = chat.ReceiveMessage(msg.Body)
				} else {
					shouldSend = !chat.Send(msg.Body)
				}
			}
		}
	}
	return
}

func stoppedTyping(e fbmsgr.Event) bool {
	if evt, ok := e.(fbmsgr.TypingEvent); ok {
		return !evt.Typing
	}
	return false
}

func startedTyping(e fbmsgr.Event) bool {
	if evt, ok := e.(fbmsgr.TypingEvent); ok {
		return evt.Typing
	}
	return false
}

func eventMessage(e fbmsgr.Event) string {
	if evt, ok := e.(fbmsgr.MessageEvent); ok {
		if evt.Body == "" {
			return "attachment"
		}
		return evt.Body
	}
	return ""
}

func markMessageRead(sess *fbmsgr.Session, e fbmsgr.Event) {
	msg := e.(fbmsgr.MessageEvent)
	if msg.GroupThread != "" {
		sess.SendReadReceipt(msg.GroupThread)
	} else {
		sess.SendReadReceipt(msg.SenderFBID)
	}
}

func sendTyping(sess *fbmsgr.Session, thread string, group bool, typ bool) {
	if group {
		sess.SendGroupTyping(thread, typ)
	} else {
		sess.SendTyping(thread, typ)
	}
}

func sendMessage(sess *fbmsgr.Session, thread string, group bool, msg string) {
	if group {
		sess.SendGroupText(thread, msg)
	} else {
		sess.SendText(thread, msg)
	}
}
