package chatbot

import (
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	CharCount    = 256
	ControlCount = 2
	InputCount   = CharCount + ControlCount

	StartExternalMsg = CharCount
	StartBotMsg      = CharCount + 1
)

// A Bot manages a recurrent neural network that acts as a
// chat bot.
type Bot struct {
	Block rnn.Block
}

// NewBot creates a new, untrained Bot.
func NewBot() *Bot {
	stateSizes := []int{400, 300, 200}
	outNetwork := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  stateSizes[len(stateSizes)-1],
			OutputCount: InputCount,
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outNetwork.Randomize()
	outBlock := rnn.NewNetworkBlock(outNetwork, 0)

	var fullNet rnn.StackedBlock
	inSize := InputCount
	for _, outSize := range stateSizes {
		fullNet = append(fullNet, rnn.NewLSTM(inSize, outSize))
		inSize = outSize
	}
	fullNet = append(fullNet, outBlock)
	return &Bot{Block: fullNet}
}

func oneHotVector(i int) linalg.Vector {
	res := make(linalg.Vector, InputCount)
	res[i] = 1
	return res
}
