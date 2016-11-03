package chatbot

import (
	"fmt"
	"io/ioutil"

	"github.com/unixpickle/neuralstruct"
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

	HiddenDropout = 0.5
)

// A Bot manages a recurrent neural network that acts as a
// chat bot.
type Bot struct {
	Block rnn.Block
}

// NewBot creates a new, untrained Bot.
func NewBot() *Bot {
	structure := neuralstruct.RAggregate{
		&neuralstruct.Stack{
			VectorSize: 10,
			NoReplace:  true,
		},
		&neuralstruct.Stack{
			VectorSize: 10,
			NoReplace:  true,
		},
	}

	stateSizes := []int{400, 300, 200}
	outNetwork := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  stateSizes[len(stateSizes)-1],
			OutputCount: structure.ControlSize() + InputCount,
		},
		&neuralstruct.PartialActivation{
			Ranges: []neuralstruct.ComponentRange{
				{Start: 0, End: structure.DataSize()},
				{Start: structure.ControlSize(), End: structure.ControlSize() + InputCount},
			},
			Activations: []neuralnet.Layer{
				&neuralnet.Sigmoid{},
				&neuralnet.LogSoftmaxLayer{},
			},
		},
	}
	outNetwork.Randomize()
	outBlock := rnn.NewNetworkBlock(outNetwork, 0)

	var fullNet rnn.StackedBlock
	inSize := InputCount + structure.DataSize()
	for _, outSize := range stateSizes {
		fullNet = append(fullNet, rnn.NewLSTM(inSize, outSize))
		fullNet = append(fullNet, rnn.NewNetworkBlock(neuralnet.Network{
			&neuralnet.DropoutLayer{
				KeepProbability: HiddenDropout,
				Training:        false,
			},
		}, 0))
		inSize = outSize
	}
	fullNet = append(fullNet, outBlock)
	return &Bot{
		Block: &neuralstruct.Block{
			Block:  fullNet,
			Struct: structure,
		},
	}
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

// Dropout enables or disables dropout in the network.
func (b *Bot) Dropout(on bool) {
	structBlock, ok := b.Block.(*neuralstruct.Block)
	if !ok {
		return
	}
	sb := structBlock.Block.(rnn.StackedBlock)
	for _, x := range sb {
		if n, ok := x.(*rnn.NetworkBlock); ok {
			net := n.Network()
			if len(net) == 1 {
				if do, ok := net[0].(*neuralnet.DropoutLayer); ok {
					do.Training = on
				}
			}
		}
	}
}

func oneHotVector(i int) linalg.Vector {
	res := make(linalg.Vector, InputCount)
	res[i] = 1
	return res
}
