package chatbot

import (
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

// A Chat is a stateful conversation between some external
// entity and a Bot.
type Chat struct {
	runner *rnn.Runner
}

// NewChat creates a new chat with an empty history.
func NewChat(b *Bot) *Chat {
	b.Dropout(false)
	return &Chat{
		runner: &rnn.Runner{Block: b.Block},
	}
}

// Send adds an external message to the chat log for the
// bot to see.
// It returns true if the bot expects to receive another
// message before replying.
func (c *Chat) Send(m string) (more bool) {
	return c.sendContents(StartExternalMsg, m)
}

// ReceiveMessage tells the bot that it sent a message.
// It returns true if the bot expects to send another
// message.
func (c *Chat) ReceiveMessage(m string) (more bool) {
	return c.sendContents(StartBotMsg, m)
}

// Receive generates a message from the bot.
// The more return value indicates whether or not the bot
// wishes to send another message after this one.
func (c *Chat) Receive() (msg string, more bool) {
	lastOut := c.runner.StepTime(oneHotVector(StartBotMsg))
	lastOut[StartExternalMsg] = math.Inf(-1)
	lastOut[StartBotMsg] = math.Inf(-1)

	var msgData []byte
	for {
		byteIdx := randomSelection(lastOut)
		if byteIdx < CharCount {
			msgData = append(msgData, byte(byteIdx))
			lastOut = c.runner.StepTime(oneHotVector(byteIdx))
			continue
		}
		more = (byteIdx == StartBotMsg)
		break
	}
	msg = string(msgData)
	return
}

func (c *Chat) sendContents(start int, m string) (more bool) {
	lastOut := c.runner.StepTime(oneHotVector(start))
	for _, b := range []byte(m) {
		lastOut = c.runner.StepTime(oneHotVector(int(b)))
	}
	if start == StartExternalMsg {
		return lastOut[StartBotMsg] < lastOut[StartExternalMsg]
	} else {
		return lastOut[StartBotMsg] > lastOut[StartExternalMsg]
	}
}

func randomSelection(weightVec linalg.Vector) int {
	num := rand.Float64()
	for i, x := range weightVec {
		num -= math.Exp(x)
		if num < 0 {
			return i
		}
	}
	return len(weightVec) - 1
}
