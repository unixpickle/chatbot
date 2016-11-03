package main

import (
	"log"
	"math/rand"
	"net/url"
	"time"

	"github.com/unixpickle/chatbot"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/sgd/asyncsgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	MaxBufferChars = 600
	BatchSize      = 4
	SyncInterval   = 4
)

func Train(paramServer, sampleFile string) {
	rand.Seed(time.Now().UnixNano())

	log.Println("Loading samples...")
	samples, err := chatbot.NewSampleSet(sampleFile, MaxBufferChars)
	if err != nil {
		die(err)
	} else if samples.Len() == 0 {
		die("no samples")
	}

	log.Println("Partitioning", samples.Len(), "samples...")
	training, validation := sgd.HashSplit(samples, 0.9)

	bot := chatbot.NewBot()
	u, err := url.Parse(paramServer)
	if err != nil {
		die(err)
	}
	bot.Dropout(true)

	client := &asyncsgd.ParamClient{
		BaseURL: u,
	}
	params := bot.Block.(sgd.Learner).Parameters()
	costFunc := neuralnet.DotCost{}
	grad := &seqtoseq.Gradienter{
		SeqFunc:  &rnn.BlockSeqFunc{B: bot.Block},
		Learner:  bot.Block.(sgd.Learner),
		CostFunc: costFunc,
	}

	var iteration int
	slave := asyncsgd.NewSlave(grad, training, BatchSize, client, params)
	slave.Sync()
	err = slave.Loop(SyncInterval, func(next, last sgd.SampleSet) {
		if iteration%4 == 0 {
			bot.Dropout(false)
			defer bot.Dropout(true)
			var lastCost float64
			if last != nil {
				lastCost = seqtoseq.TotalCostBlock(bot.Block, BatchSize, last, costFunc)
			}
			newCost := seqtoseq.TotalCostBlock(bot.Block, BatchSize, next, costFunc)

			sgd.ShuffleSampleSet(validation)
			validationCost := seqtoseq.TotalCostBlock(bot.Block, BatchSize,
				validation.Subset(0, BatchSize), costFunc)

			log.Printf("iter %d: validation=%f cost=%f last=%f", iteration, validationCost,
				newCost, lastCost)
		}
		iteration++
	})
	if err != nil {
		die("Training error:", err)
	}
}
