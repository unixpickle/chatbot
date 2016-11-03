package main

import (
	"net/http"
	"strconv"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/chatbot"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/sgd/asyncsgd"
)

const StepSize = 0.001

func Serve(port int, netFile string) {
	net, err := chatbot.LoadBot(netFile)
	if err != nil {
		die(err)
	}
	s := &Server{
		NetFile: netFile,
		Bot:     net,
		Updater: &asyncsgd.TransformerUpdater{
			StepSize:    StepSize,
			Transformer: &sgd.Adam{},
		},
	}
	s.PS = asyncsgd.NewParamServer(net.Block.(sgd.Learner).Parameters(), s)
	http.ListenAndServe(":"+strconv.Itoa(port), s)
}

type Server struct {
	PS       *asyncsgd.ParamServer
	RateLock sync.Mutex
	NetFile  string
	Bot      *chatbot.Bot
	Updater  *asyncsgd.TransformerUpdater
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/set_rate" {
		rateParam, err := strconv.ParseFloat(r.FormValue("rate"), 64)
		if err != nil {
			http.Error(w, "invalid rate", http.StatusBadRequest)
			return
		}
		s.RateLock.Lock()
		defer s.RateLock.Unlock()
		s.Updater.StepSize = rateParam
	} else {
		s.PS.ServeHTTP(w, r)
	}
}

func (s *Server) Update(g autofunc.Gradient) {
	s.RateLock.Lock()
	defer s.RateLock.Unlock()
	s.Updater.Update(g)
	s.Bot.Save(s.NetFile)
}
