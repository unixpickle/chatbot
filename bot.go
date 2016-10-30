package chatbot

import (
	"fmt"
	"io/ioutil"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
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

// LoadBot reads a Bot from a file.
func LoadBot(path string) (*Bot, error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	decoded, err := serializer.DeserializeWithType(contents)
	if err != nil {
		return nil, err
	}
	if block, ok := decoded.(rnn.Block); ok {
		return &Bot{Block: block}, nil
	}
	return nil, fmt.Errorf("type is not an rnn.Block: %T", decoded)
}

// Save saves the Bot to a file.
func (b *Bot) Save(path string) error {
	encoded, err := serializer.SerializeWithType(b.Block.(serializer.Serializer))
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, encoded, 0755)
}

func oneHotVector(i int) linalg.Vector {
	res := make(linalg.Vector, InputCount)
	res[i] = 1
	return res
}
