package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/chatbot"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	MaxBufferChars = 400
	StepSize       = 0.005
	BatchSize      = 4
)

func main() {
	rand.Seed(time.Now().UnixNano())

	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: train <samples> <output>")
		os.Exit(1)
	}

	samples, err := chatbot.NewSampleSet(os.Args[1], MaxBufferChars)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load samples:", err)
		os.Exit(1)
	} else if samples.Len() == 0 {
		fmt.Fprintln(os.Stderr, "No samples loaded.")
		os.Exit(1)
	}

	bot, err := chatbot.LoadBot(os.Args[2])
	if os.IsNotExist(err) {
		log.Println("Creating bot...")
		bot = chatbot.NewBot()
	} else if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load bot:", err)
		os.Exit(1)
	}

	log.Println("Partitioning", samples.Len(), "samples...")
	training, validation := sgd.HashSplit(samples, 0.9)

	log.Println("Training...")

	costFunc := neuralnet.DotCost{}
	gradienter := &sgd.Adam{
		Gradienter: &seqtoseq.Gradienter{
			SeqFunc:  &rnn.BlockSeqFunc{B: bot.Block},
			Learner:  bot.Block.(sgd.Learner),
			CostFunc: costFunc,
		},
	}

	var iteration int
	var lastBatch sgd.SampleSet
	sgd.SGDMini(gradienter, training, StepSize, BatchSize, func(s sgd.SampleSet) bool {
		if iteration%4 == 0 {
			var lastCost float64
			if lastBatch != nil {
				lastCost = seqtoseq.TotalCostBlock(bot.Block, BatchSize, lastBatch, costFunc)
			}
			lastBatch = s.Copy()
			newCost := seqtoseq.TotalCostBlock(bot.Block, BatchSize, s, costFunc)

			sgd.ShuffleSampleSet(validation)
			validationCost := seqtoseq.TotalCostBlock(bot.Block, BatchSize,
				validation.Subset(0, BatchSize), costFunc)

			log.Printf("iter %d: validation=%f cost=%f last=%f", iteration, validationCost,
				newCost, lastCost)
		}

		iteration++
		return true
	})

	if err := bot.Save(os.Args[2]); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save output:", err)
		os.Exit(1)
	}
}
