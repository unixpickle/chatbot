package chatbot

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

type message struct {
	FromBot bool
	Body    string
}

type snippet struct {
	EndOfChat bool
	NextBot   bool
	Messages  []message
}

// A SampleSet stores a set of conversational training
// samples.
// A training sample consists of a message and its
// preceding messages if applicable.
type SampleSet struct {
	snippets []snippet
}

// NewSampleSet loads a sample set from a directory of
// conversation files or from a single conversation file.
//
// A conversation file must be formatted using CSV with
// two columns: the sender and the message.
// The sender is either "bot" or "human".
//
// The maxBuffer size specifies the maximum number of
// characters in a generated training sequence.
func NewSampleSet(path string, maxBuffer int) (*SampleSet, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	var convos [][]message
	if info.IsDir() {
		listing, err := ioutil.ReadDir(path)
		if err != nil {
			return nil, err
		}
		for _, entry := range listing {
			if strings.HasPrefix(entry.Name(), ".") {
				continue
			}
			convoPath := filepath.Join(path, entry.Name())
			convo, err := readConversationFile(convoPath)
			if err != nil {
				return nil, fmt.Errorf("load %s: %s", convoPath, err)
			}
			convos = append(convos, convo)
		}
	} else {
		convo, err := readConversationFile(path)
		if err != nil {
			return nil, err
		}
		convos = [][]message{convo}
	}

	return newSampleSetConvos(convos, maxBuffer)
}

// NewSampleSetReader loads a sample set by reading the
// contents of one sample data file.
func NewSampleSetReader(in io.Reader, maxBuffer int) (*SampleSet, error) {
	convo, err := readConversation(in)
	if err != nil {
		return nil, err
	}
	return newSampleSetConvos([][]message{convo}, maxBuffer)
}

func newSampleSetConvos(convos [][]message, maxBuffer int) (*SampleSet, error) {
	res := &SampleSet{}
	for _, convo := range convos {
		for i := range convo {
			sn := generateSnippet(maxBuffer, convo, i)
			if sn != nil {
				res.snippets = append(res.snippets, *sn)
			}
		}
	}
	return res, nil
}

// Len returns the number of samples.
func (s *SampleSet) Len() int {
	return len(s.snippets)
}

// Copy returns a shallow copy of the sample set.
func (s *SampleSet) Copy() sgd.SampleSet {
	res := &SampleSet{
		snippets: make([]snippet, len(s.snippets)),
	}
	copy(res.snippets, s.snippets)
	return res
}

// Swap swaps two samples.
func (s *SampleSet) Swap(i, j int) {
	s.snippets[i], s.snippets[j] = s.snippets[j], s.snippets[i]
}

// GetSample generates a seqtoseq.Sample for the snippet
// at the given index.
func (s *SampleSet) GetSample(idx int) interface{} {
	sample := s.snippets[idx]
	var inputSeq []linalg.Vector
	for _, msg := range sample.Messages {
		if msg.FromBot {
			inputSeq = append(inputSeq, oneHotVector(StartBotMsg))
		} else {
			inputSeq = append(inputSeq, oneHotVector(StartExternalMsg))
		}
		for _, chr := range []byte(msg.Body) {
			inputSeq = append(inputSeq, oneHotVector(int(chr)))
		}
	}

	var outSeq []linalg.Vector
	if sample.EndOfChat {
		nextVec := make(linalg.Vector, InputCount)
		nextVec[StartBotMsg] = 0.5
		nextVec[StartExternalMsg] = 0.5
		outSeq = append(inputSeq, nextVec)
	} else if sample.NextBot {
		outSeq = append(inputSeq, oneHotVector(StartBotMsg))
	} else {
		outSeq = append(inputSeq, oneHotVector(StartExternalMsg))
	}
	outSeq = outSeq[1:]

	return seqtoseq.Sample{Inputs: inputSeq, Outputs: outSeq}
}

// Subset returns a subset of this sample set.
func (s *SampleSet) Subset(start, end int) sgd.SampleSet {
	return &SampleSet{
		snippets: s.snippets[start:end],
	}
}

// Hash returns a hash of the given sample.
func (s *SampleSet) Hash(i int) []byte {
	return s.GetSample(i).(seqtoseq.Sample).Hash()
}

func generateSnippet(maxChars int, msgs []message, msgIdx int) *snippet {
	var count int
	for i := msgIdx; i >= 0; i-- {
		msgLen := len(msgs[i].Body)
		if msgLen == 0 {
			break
		}
		if maxChars-msgLen <= 0 {
			break
		}
		count++
		maxChars -= msgLen
	}
	if count == 0 {
		return nil
	}
	res := &snippet{
		EndOfChat: msgIdx+1 == len(msgs),
		Messages:  msgs[msgIdx+1-count : msgIdx+1],
	}
	if msgIdx+1 < len(msgs) {
		res.NextBot = msgs[msgIdx+1].FromBot
	}
	return res
}

func readConversationFile(file string) ([]message, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readConversation(f)
}

func readConversation(f io.Reader) ([]message, error) {
	r := csv.NewReader(f)

	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, nil
	}
	if len(records[0]) != 2 {
		return nil, errors.New("expected exactly two columns")
	}

	result := make([]message, len(records))
	for i, x := range records {
		record := message{Body: x[1]}
		if x[0] == "bot" {
			record.FromBot = true
		} else if x[0] != "human" {
			return nil, fmt.Errorf("record %d: unknown sender %s", i, x[0])
		}
		result[i] = record
	}

	return result, nil
}
