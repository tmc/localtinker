package tinkertrain

import (
	"archive/tar"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	lmtrain "github.com/tmc/mlx-go-lm/lmtrain"
	"github.com/tmc/mlx-go-lm/mlxlm"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/sample"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/training"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/tuner"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/random"
)

const checkpointCompleteFile = "complete"
const checkpointMetadataFile = "checkpoint.json"
const checkpointOptimizerFile = "optimizer_state.json"

type checkpointMetadata struct {
	Format       string `json:"format"`
	Version      int    `json:"version"`
	ModelID      string `json:"model_id"`
	BaseModel    string `json:"base_model"`
	Kind         string `json:"kind"`
	Name         string `json:"name"`
	IsLoRA       bool   `json:"is_lora"`
	LoRARank     int    `json:"lora_rank"`
	TrainMLP     bool   `json:"train_mlp"`
	TrainAttn    bool   `json:"train_attn"`
	TrainUnembed bool   `json:"train_unembed"`
	HasOptimizer bool   `json:"has_optimizer"`
	Step         int    `json:"step"`
}

type optimizerState struct {
	Format         string     `json:"format"`
	Version        int        `json:"version"`
	Optimizer      string     `json:"optimizer"`
	Step           int        `json:"step"`
	LastAdam       AdamParams `json:"last_adam"`
	PendingBatches int        `json:"pending_batches"`
}

type mlxModel struct {
	mu        sync.Mutex
	id        string
	base      string
	bundle    *lmtrain.ModelBundle
	tokenizer mlxlm.Tokenizer
	adapters  *tuner.Set
	rank      int
	step      int
	pending   denseBatch
	lastLoss  float64
	lastAdam  AdamParams
}

func newMLXModel(ctx context.Context, modelID string, cfg CreateConfig) (*mlxModel, error) {
	apiBase := cfg.BaseModel
	if apiBase == "" {
		apiBase = "Qwen/Qwen3-8B"
	}
	base := resolveMLXBase(apiBase)
	bundle, err := lmtrain.LoadModel(ctx, base, lmtrain.LoadOptions{})
	if err != nil {
		return nil, fmt.Errorf("load model %q: %w", apiBase, err)
	}
	if configurable, ok := bundle.Model.(models.ExecutionConfigurable); ok {
		configurable.SetExecutionConfig(models.ExecutionConfig{
			Inference: false,
			FastSDPA:  true,
			FastRoPE:  true,
		})
	}

	rank := cfg.LoRARank
	if rank <= 0 {
		rank = 8
	}
	tcfg := tuner.Config{
		Rank:    rank,
		Alpha:   float32(rank * 4),
		Dropout: 0,
		Keys:    lmtrain.DefaultAdapterKeyPatterns(),
	}
	adapters, count, cleanup, err := lmtrain.CreateAndAttachAdapterSet(bundle, "lora", -1, tcfg)
	defer cleanup()
	if err != nil {
		return nil, fmt.Errorf("create lora adapters: %w", err)
	}
	if count == 0 {
		return nil, fmt.Errorf("create lora adapters: no trainable layers")
	}

	return &mlxModel{
		id:        modelID,
		base:      apiBase,
		bundle:    bundle,
		tokenizer: loadModelTokenizer(bundle.Path),
		adapters:  adapters,
		rank:      rank,
	}, nil
}

func loadModelTokenizer(path string) mlxlm.Tokenizer {
	tok, err := mlxlm.LoadTokenizer(path)
	if err != nil {
		return nil
	}
	return tok
}

func resolveMLXBase(base string) string {
	switch base {
	case "Qwen/Qwen3-8B":
		return "mlx-community/Qwen3-8B-4bit"
	default:
		return base
	}
}

func (m *mlxModel) forwardBackward(ctx context.Context, input ForwardBackwardInput, backward bool) (ForwardBackwardOutput, error) {
	batch, err := newDenseBatch(input)
	if err != nil {
		return ForwardBackwardOutput{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	out, err := m.evaluateDenseBatch(ctx, batch)
	if err != nil {
		return ForwardBackwardOutput{}, err
	}
	m.lastLoss = out.loss
	if backward {
		m.pending = batch
	}
	return ForwardBackwardOutput{
		LossFnOutputType: "TensorData",
		LossFnOutputs:    out.lossOutputs,
		Metrics: map[string]float64{
			"loss:mean":    out.loss,
			"tokens:sum":   batch.weightSum,
			"examples:sum": float64(len(batch.rows)),
		},
	}, nil
}

func (m *mlxModel) optimStep(ctx context.Context, params AdamParams) (OptimStepOutput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.pending.rows) == 0 {
		return OptimStepOutput{}, fmt.Errorf("optimizer step: no pending forward_backward")
	}

	lr := params.LearningRate
	if lr == 0 {
		lr = 1e-4
	}
	loss, err := m.trainDenseStep(ctx, m.pending, params, lr)
	if err != nil {
		return OptimStepOutput{}, fmt.Errorf("optimizer step: %w", err)
	}
	m.pending = denseBatch{}
	m.lastLoss = loss
	m.lastAdam = params
	m.step++

	return OptimStepOutput{Metrics: map[string]float64{
		"loss:mean":             loss,
		"optimizer_backend:mlx": 1,
		"optimizer_step:unique": float64(m.step),
	}}, nil
}

func (m *mlxModel) saveForSampler(_ context.Context, name string) (string, error) {
	return m.saveAdapter(name, "sampler_weights", false)
}

func (m *mlxModel) saveState(_ context.Context, name string) (string, error) {
	return m.saveAdapter(name, "weights", true)
}

func (m *mlxModel) saveAdapter(name, kind string, optimizer bool) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if name == "" {
		name = "adapter"
	}
	dir := filepath.Join(checkpointRoot(), m.id, kind, cleanName(name))
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create adapter dir: %w", err)
	}
	if err := m.materializeAdapterWeights(); err != nil {
		return "", err
	}
	if err := m.adapters.Save(filepath.Join(dir, "adapters.safetensors")); err != nil {
		return "", fmt.Errorf("save adapter weights: %w", err)
	}
	layers := -1
	if cfg := m.bundle.Model.Config(); cfg != nil {
		layers = cfg.NumLayers
	}
	if err := lmtrain.SaveAdapterConfig(dir, "lora", layers, m.adapters.Config()); err != nil {
		return "", fmt.Errorf("save adapter config: %w", err)
	}
	meta := checkpointMetadata{
		Format:       "localtinker.checkpoint",
		Version:      1,
		ModelID:      m.id,
		BaseModel:    m.base,
		Kind:         kind,
		Name:         cleanName(name),
		IsLoRA:       true,
		LoRARank:     m.rank,
		TrainMLP:     true,
		TrainAttn:    true,
		TrainUnembed: false,
		HasOptimizer: optimizer,
		Step:         m.step,
	}
	if err := writeJSONFile(filepath.Join(dir, checkpointMetadataFile), meta); err != nil {
		return "", fmt.Errorf("write checkpoint metadata: %w", err)
	}
	if optimizer {
		state := optimizerState{
			Format:         "localtinker.optimizer",
			Version:        1,
			Optimizer:      "adamw",
			Step:           m.step,
			LastAdam:       m.lastAdam,
			PendingBatches: len(m.pending.rows),
		}
		if err := writeJSONFile(filepath.Join(dir, checkpointOptimizerFile), state); err != nil {
			return "", fmt.Errorf("write optimizer state: %w", err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, checkpointCompleteFile), []byte("ok\n"), 0644); err != nil {
		return "", fmt.Errorf("write completion marker: %w", err)
	}
	return "tinker://" + m.id + "/" + kind + "/" + cleanName(name), nil
}

func (m *mlxModel) materializeAdapterWeights() error {
	params := m.adapters.Trainable()
	if len(params) == 0 {
		return fmt.Errorf("adapter weights: no trainable parameters")
	}
	if err := mlx.Eval(params...); err != nil {
		return fmt.Errorf("eval adapter weights: %w", err)
	}
	runtime.KeepAlive(params)
	return nil
}

func (m *mlxModel) loadState(_ context.Context, path string, optimizer bool) error {
	parsed, err := ParseTinkerPath(path)
	if err != nil {
		return err
	}
	dir := checkpointDir(parsed)
	file := filepath.Join(dir, "adapters.safetensors")
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.adapters.LoadWeights(file); err != nil {
		return fmt.Errorf("load adapter weights: %w", err)
	}
	if optimizer {
		state, err := readOptimizerState(filepath.Join(dir, checkpointOptimizerFile))
		if err != nil {
			return err
		}
		m.step = state.Step
		m.lastAdam = state.LastAdam
	}
	return nil
}

func (m *mlxModel) sample(ctx context.Context, req SampleRequest) (SampleOutput, error) {
	maxTokens := req.SamplingParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 20
	}
	numSamples := req.NumSamples
	if numSamples <= 0 {
		numSamples = 1
	}
	prompt := req.Prompt.tokens()
	if len(prompt) == 0 {
		return SampleOutput{}, fmt.Errorf("sample: empty prompt")
	}
	stop, err := stopTokenSequences(req.SamplingParams.Stop, m.tokenizer)
	if err != nil {
		return SampleOutput{}, fmt.Errorf("sample stop: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	out := SampleOutput{
		Type:      "sample",
		Sequences: make([]SampledSequence, 0, numSamples),
	}
	if req.PromptLogprobs || req.SamplingParams.PromptLogprobs {
		logprobs, err := m.tokenLogprobs(ctx, prompt)
		if err != nil {
			return SampleOutput{}, fmt.Errorf("prompt logprobs: %w", err)
		}
		out.PromptLogprobs = logprobs
	}
	if req.TopKPromptLogprobs > 0 {
		logprobs, err := m.topKPromptLogprobs(ctx, prompt, req.TopKPromptLogprobs)
		if err != nil {
			return SampleOutput{}, fmt.Errorf("top-k prompt logprobs: %w", err)
		}
		out.TopKPromptLogprobs = logprobs
	}
	for range numSamples {
		seq := append([]int(nil), prompt...)
		gen := make([]int, 0, maxTokens)
		logprobs := make([]float64, 0, maxTokens)
		stopReason := "length"
		var key *mlx.Array
		if req.SamplingParams.Seed != 0 {
			key = random.Key(uint64(req.SamplingParams.Seed))
			defer key.Free()
		}
		for range maxTokens {
			var stepKey *mlx.Array
			if key != nil {
				left, right := random.Split(key, nil)
				key.Free()
				stepKey = left
				key = right
			}
			next, logprob, err := m.nextToken(ctx, seq, req.SamplingParams, stepKey)
			if stepKey != nil {
				stepKey.Free()
			}
			if err != nil {
				return SampleOutput{}, err
			}
			seq = append(seq, next)
			gen = append(gen, next)
			logprobs = append(logprobs, logprob)
			if matchesStop(gen, stop) {
				stopReason = "stop"
				break
			}
		}
		out.Sequences = append(out.Sequences, SampledSequence{
			StopReason: stopReason,
			Tokens:     gen,
			Logprobs:   logprobs,
		})
	}
	return out, nil
}

func (m *mlxModel) nextToken(ctx context.Context, tokens []int, params SamplingParams, key *mlx.Array) (int, float64, error) {
	logits, err := m.logits(ctx, tokens)
	if err != nil {
		return 0, 0, err
	}
	defer logits.Free()
	shape := logits.Shape()
	logprobs, err := lastLogprobs(logits)
	if err != nil {
		return 0, 0, err
	}
	token, err := sampleToken(ctx, logits, params, key)
	if err != nil {
		return 0, 0, err
	}
	if token < 0 || token >= shape[2] {
		return 0, 0, fmt.Errorf("sample token %d outside vocab %d", token, shape[2])
	}
	return token, logprobs[token], nil
}

func (m *mlxModel) logits(ctx context.Context, tokens []int) (*mlx.Array, error) {
	xs := make([]int32, len(tokens))
	for i, token := range tokens {
		xs[i] = int32(token)
	}
	input, err := mlx.FromSlice(xs, []int{1, len(xs)}, mlx.Int32)
	if err != nil {
		return nil, fmt.Errorf("sample input: %w", err)
	}
	defer input.Free()

	logits, _ := m.bundle.Model.Forward(ctx, input, nil)
	shape := logits.Shape()
	if len(shape) != 3 || shape[1] == 0 || shape[2] == 0 {
		logits.Free()
		return nil, fmt.Errorf("sample logits shape %v", shape)
	}
	return logits, nil
}

func sampleToken(ctx context.Context, logits *mlx.Array, params SamplingParams, key *mlx.Array) (int, error) {
	temp := params.Temperature
	if temp == nil {
		one := 1.0
		temp = &one
	}
	topP := params.TopP
	if topP == nil {
		one := 1.0
		topP = &one
	}
	token, err := sample.TokenWithKey(ctx, logits, key, *temp, *topP, 0, params.TopK)
	if err != nil {
		return 0, fmt.Errorf("sample token: %w", err)
	}
	defer token.Free()
	mlx.Eval(token)
	v, err := mlx.ItemAs[int](token)
	if err != nil {
		return 0, fmt.Errorf("sample token: %w", err)
	}
	return v, nil
}

func (m *mlxModel) tokenLogprobs(ctx context.Context, tokens []int) ([]*float64, error) {
	if len(tokens) == 0 {
		return nil, nil
	}
	out := make([]*float64, len(tokens))
	for i := 1; i < len(tokens); i++ {
		logits, err := m.logits(ctx, tokens[:i])
		if err != nil {
			return nil, err
		}
		logprobs, err := lastLogprobs(logits)
		logits.Free()
		if err != nil {
			return nil, err
		}
		token := tokens[i]
		if token < 0 || token >= len(logprobs) {
			return nil, fmt.Errorf("token %d outside vocab %d", token, len(logprobs))
		}
		out[i] = &logprobs[token]
	}
	return out, nil
}

func (m *mlxModel) topKPromptLogprobs(ctx context.Context, tokens []int, k int) ([]any, error) {
	if len(tokens) == 0 || k <= 0 {
		return nil, nil
	}
	out := make([]any, len(tokens))
	for i := 1; i < len(tokens); i++ {
		logits, err := m.logits(ctx, tokens[:i])
		if err != nil {
			return nil, err
		}
		logprobs, err := lastLogprobs(logits)
		logits.Free()
		if err != nil {
			return nil, err
		}
		out[i] = topKLogprobs(logprobs, k)
	}
	return out, nil
}

func topKLogprobs(logprobs []float64, k int) [][]any {
	if k <= 0 || len(logprobs) == 0 {
		return nil
	}
	if k > len(logprobs) {
		k = len(logprobs)
	}
	ids := make([]int, len(logprobs))
	for i := range ids {
		ids[i] = i
	}
	sort.Slice(ids, func(i, j int) bool {
		if logprobs[ids[i]] == logprobs[ids[j]] {
			return ids[i] < ids[j]
		}
		return logprobs[ids[i]] > logprobs[ids[j]]
	})
	out := make([][]any, k)
	for i, token := range ids[:k] {
		out[i] = []any{token, logprobs[token]}
	}
	return out
}

func lastLogprobs(logits *mlx.Array) ([]float64, error) {
	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 {
		return nil, fmt.Errorf("sample logits shape %v", shape)
	}
	logits32 := mlx.Astype(logits, mlx.Float32)
	defer logits32.Free()
	if err := mlx.Eval(logits32); err != nil {
		return nil, fmt.Errorf("sample logits: %w", err)
	}
	values, err := mlx.ToSlice[float32](logits32)
	if err != nil {
		return nil, fmt.Errorf("sample logits: %w", err)
	}
	vocab := shape[2]
	start := (shape[1] - 1) * vocab
	last := values[start : start+vocab]
	max := float64(last[0])
	for _, v := range last[1:] {
		if f := float64(v); f > max {
			max = f
		}
	}
	var sum float64
	for _, v := range last {
		sum += math.Exp(float64(v) - max)
	}
	norm := max + math.Log(sum)
	out := make([]float64, len(last))
	for i, v := range last {
		out[i] = float64(v) - norm
	}
	return out, nil
}

func (m *mlxModel) Forward(ctx context.Context, input *mlx.Array, cache interface{}) (*mlx.Array, interface{}, error) {
	var c models.Cache
	if cache != nil {
		var ok bool
		c, ok = cache.(models.Cache)
		if !ok {
			return nil, nil, fmt.Errorf("invalid cache type %T", cache)
		}
	}
	logits, next := m.bundle.Model.Forward(ctx, input, c)
	return logits, next, nil
}

type denseBatch struct {
	rows      []denseRow
	seqLen    int
	weightSum float64
}

type denseRow struct {
	tokens      []int32
	targets     []int32
	weights     []float32
	outputShape []int
}

type denseEvalOutput struct {
	loss        float64
	lossOutputs []map[string]Tensor
}

func newDenseBatch(input ForwardBackwardInput) (denseBatch, error) {
	if input.LossFn != "cross_entropy" {
		return denseBatch{}, fmt.Errorf("unsupported loss function %q", input.LossFn)
	}
	if len(input.Data) == 0 {
		return denseBatch{}, fmt.Errorf("no data")
	}
	var batch denseBatch
	batch.rows = make([]denseRow, 0, len(input.Data))
	for i, datum := range input.Data {
		tokens := datum.ModelInput.tokens()
		targets, err := datum.targets()
		if err != nil {
			return denseBatch{}, fmt.Errorf("datum %d: %w", i, err)
		}
		weights, err := datum.weights(len(targets))
		if err != nil {
			return denseBatch{}, fmt.Errorf("datum %d: %w", i, err)
		}
		if len(tokens) != len(targets) {
			return denseBatch{}, fmt.Errorf("datum %d input tokens=%d target tokens=%d", i, len(tokens), len(targets))
		}
		if len(tokens) == 0 {
			return denseBatch{}, fmt.Errorf("datum %d has no tokens", i)
		}
		row := denseRow{
			tokens:      int32s(tokens),
			targets:     int32s(targets),
			weights:     float32s(weights),
			outputShape: tensorShape(datum.LossFnInputs["target_tokens"], len(targets)),
		}
		for _, w := range weights {
			if w > 0 {
				batch.weightSum += float64(w)
			}
		}
		if len(tokens) > batch.seqLen {
			batch.seqLen = len(tokens)
		}
		batch.rows = append(batch.rows, row)
	}
	if batch.weightSum == 0 {
		return denseBatch{}, fmt.Errorf("zero total weight")
	}
	return batch, nil
}

func tensorShape(t TensorData, n int) []int {
	if len(t.Shape) == 0 {
		return []int{n}
	}
	return append([]int(nil), t.Shape...)
}

func int32s(in []int) []int32 {
	out := make([]int32, len(in))
	for i, v := range in {
		out[i] = int32(v)
	}
	return out
}

func float32s(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func (b denseBatch) arrays() (inputs, targets, weights *mlx.Array, err error) {
	inputData := make([]int32, len(b.rows)*b.seqLen)
	targetData := make([]int32, len(b.rows)*b.seqLen)
	weightData := make([]float32, len(b.rows)*b.seqLen)
	for i, row := range b.rows {
		base := i * b.seqLen
		copy(inputData[base:base+len(row.tokens)], row.tokens)
		copy(targetData[base:base+len(row.targets)], row.targets)
		copy(weightData[base:base+len(row.weights)], row.weights)
	}
	inputs, err = mlx.FromSlice(inputData, []int{len(b.rows), b.seqLen}, mlx.Int32)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("inputs: %w", err)
	}
	targets, err = mlx.FromSlice(targetData, []int{len(b.rows), b.seqLen}, mlx.Int32)
	if err != nil {
		inputs.Free()
		return nil, nil, nil, fmt.Errorf("targets: %w", err)
	}
	weights, err = mlx.FromSlice(weightData, []int{len(b.rows), b.seqLen}, mlx.Float32)
	if err != nil {
		inputs.Free()
		targets.Free()
		return nil, nil, nil, fmt.Errorf("weights: %w", err)
	}
	return inputs, targets, weights, nil
}

func (m *mlxModel) evaluateDenseBatch(ctx context.Context, batch denseBatch) (denseEvalOutput, error) {
	inputs, targets, weights, err := batch.arrays()
	if err != nil {
		return denseEvalOutput{}, err
	}
	defer inputs.Free()
	defer targets.Free()
	defer weights.Free()

	logits, _ := m.bundle.Model.Forward(ctx, inputs, nil)
	defer logits.Free()

	loss, logprobs, err := denseCrossEntropy(logits, targets, weights)
	if err != nil {
		return denseEvalOutput{}, err
	}
	defer loss.Free()
	defer logprobs.Free()
	logprobs32 := mlx.Astype(logprobs, mlx.Float32)
	defer logprobs32.Free()
	if err := mlx.Eval(loss, logprobs32); err != nil {
		return denseEvalOutput{}, fmt.Errorf("eval: %w", err)
	}

	loss32, err := mlx.ItemAs[float32](loss)
	if err != nil {
		return denseEvalOutput{}, fmt.Errorf("loss: %w", err)
	}
	all, err := mlx.ToSlice[float32](logprobs32)
	if err != nil {
		return denseEvalOutput{}, fmt.Errorf("logprobs: %w", err)
	}
	out := denseEvalOutput{
		loss:        float64(loss32),
		lossOutputs: make([]map[string]Tensor, 0, len(batch.rows)),
	}
	for i, row := range batch.rows {
		data := make([]float64, len(row.targets))
		offset := i * batch.seqLen
		for j := range row.targets {
			data[j] = float64(all[offset+j])
		}
		out.lossOutputs = append(out.lossOutputs, map[string]Tensor{
			"logprobs": {Data: data, DType: "float32", Shape: row.outputShape},
		})
	}
	return out, nil
}

func (m *mlxModel) trainDenseStep(ctx context.Context, batch denseBatch, params AdamParams, lr float64) (float64, error) {
	inputs, targets, weights, err := batch.arrays()
	if err != nil {
		return 0, err
	}
	defer inputs.Free()
	defer targets.Free()
	defer weights.Free()

	trainable := m.adapters.Trainable()
	if len(trainable) == 0 {
		return 0, fmt.Errorf("no trainable parameters")
	}
	lossFn := func(ctx context.Context, adapterParams, extraInputs []*mlx.Array) (*mlx.Array, error) {
		m.adapters.UpdateParams(adapterParams)
		logits, _ := m.bundle.Model.Forward(ctx, extraInputs[0], nil)
		defer logits.Free()
		loss, _, err := denseCrossEntropy(logits, extraInputs[1], extraInputs[2])
		return loss, err
	}
	trainParams := training.DefaultTrainParameters()
	trainParams.Optimizer = "adamw"
	trainParams.TrainingMode = "separate"
	trainParams.LearningRate = float32(lr)
	trainParams.WeightDecay = float32(params.WeightDecay)
	trainParams.MaxGradNorm = float32(params.GradClipNorm)

	step, err := training.NewTrainingStep(len(trainable), 3, lossFn, trainParams)
	if err != nil {
		return 0, err
	}
	defer step.Free()
	step.InitState(trainable)

	lrTensor := mlx.NewScalar(float32(lr))
	defer lrTensor.Free()
	loss := step.Step(training.NewTrainingContext(len(trainable), 3), []*mlx.Array{inputs, targets, weights}, lrTensor)
	defer loss.Free()
	updated := step.GetParams()
	eval := append(append([]*mlx.Array{}, updated...), loss)
	if err := mlx.Eval(eval...); err != nil {
		return 0, fmt.Errorf("eval: %w", err)
	}
	m.adapters.UpdateParams(updated)
	loss32, err := mlx.ItemAs[float32](loss)
	if err != nil {
		return 0, fmt.Errorf("loss: %w", err)
	}
	return float64(loss32), nil
}

func denseCrossEntropy(logits, targets, weights *mlx.Array) (loss, logprobs *mlx.Array, err error) {
	logitShape := logits.Shape()
	targetShape := targets.Shape()
	weightShape := weights.Shape()
	if len(logitShape) != 3 {
		return nil, nil, fmt.Errorf("logits shape %v, want [batch seq vocab]", logitShape)
	}
	if len(targetShape) != 2 {
		return nil, nil, fmt.Errorf("targets shape %v, want [batch seq]", targetShape)
	}
	if len(weightShape) != 2 {
		return nil, nil, fmt.Errorf("weights shape %v, want [batch seq]", weightShape)
	}
	if logitShape[0] != targetShape[0] || logitShape[1] != targetShape[1] {
		return nil, nil, fmt.Errorf("logits shape %v does not match targets shape %v", logitShape, targetShape)
	}
	if targetShape[0] != weightShape[0] || targetShape[1] != weightShape[1] {
		return nil, nil, fmt.Errorf("weights shape %v does not match targets shape %v", weightShape, targetShape)
	}
	if logitShape[2] == 0 {
		return nil, nil, fmt.Errorf("logits shape %v has empty vocab", logitShape)
	}
	targetsExp := mlx.ExpandDims(targets, -1)
	defer targetsExp.Free()
	score := mlx.TakeAlongAxis(logits, targetsExp, -1)
	defer score.Free()
	selected := mlx.SqueezeAxis(score, -1)
	defer selected.Free()

	logsumexp := mlx.LogsumexpAxis(logits, -1, false)
	defer logsumexp.Free()
	logprobs = mlx.Subtract(selected, logsumexp)

	negLogprobs := mlx.MultiplyScalar(logprobs, float32(-1))
	defer negLogprobs.Free()
	weighted := mlx.Multiply(negLogprobs, weights)
	defer weighted.Free()
	total := mlx.Sum(weighted, false)
	defer total.Free()
	weightTotal := mlx.Sum(weights, false)
	defer weightTotal.Free()
	loss = mlx.Divide(total, weightTotal)
	return loss, logprobs, nil
}

func cleanName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "adapter"
	}
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

type TinkerPath struct {
	ModelID string
	Kind    string
	Name    string
}

func ParseTinkerPath(path string) (TinkerPath, error) {
	const prefix = "tinker://"
	if !strings.HasPrefix(path, prefix) {
		return TinkerPath{}, fmt.Errorf("invalid tinker path %q", path)
	}
	parts := strings.SplitN(strings.TrimPrefix(path, prefix), "/", 3)
	if len(parts) != 3 {
		return TinkerPath{}, fmt.Errorf("invalid tinker path %q", path)
	}
	switch parts[1] {
	case "weights", "sampler_weights":
	default:
		return TinkerPath{}, fmt.Errorf("invalid tinker checkpoint type %q", parts[1])
	}
	return TinkerPath{ModelID: parts[0], Kind: parts[1], Name: parts[2]}, nil
}

func CheckpointPathExists(path string) bool {
	file, err := CheckpointFile(path)
	if err != nil {
		return false
	}
	_, err = os.Stat(file)
	return err == nil
}

func CheckpointArchive(path string) (string, error) {
	parsed, err := ParseTinkerPath(path)
	if err != nil {
		return "", err
	}
	dir := checkpointDir(parsed)
	info, err := os.Stat(dir)
	if err != nil {
		return "", fmt.Errorf("stat checkpoint: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("checkpoint is not a directory")
	}
	archiveDir := filepath.Join(checkpointRoot(), "archives", parsed.ModelID, parsed.Kind)
	if err := os.MkdirAll(archiveDir, 0755); err != nil {
		return "", fmt.Errorf("create archive dir: %w", err)
	}
	archive := filepath.Join(archiveDir, cleanName(parsed.Name)+".tar")
	tmp := archive + ".tmp"
	if err := writeCheckpointArchive(tmp, dir); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	if err := os.Rename(tmp, archive); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("install checkpoint archive: %w", err)
	}
	return archive, nil
}

func CheckpointSize(path string) (int64, error) {
	parsed, err := ParseTinkerPath(path)
	if err != nil {
		return 0, err
	}
	var size int64
	err = filepath.WalkDir(checkpointDir(parsed), func(name string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode().IsRegular() {
			size += info.Size()
		}
		return nil
	})
	if err != nil {
		return 0, fmt.Errorf("size checkpoint: %w", err)
	}
	return size, nil
}

func CheckpointFile(path string) (string, error) {
	parsed, err := ParseTinkerPath(path)
	if err != nil {
		return "", err
	}
	return filepath.Join(checkpointDir(parsed), "adapters.safetensors"), nil
}

func DeleteCheckpoint(path string) error {
	parsed, err := ParseTinkerPath(path)
	if err != nil {
		return err
	}
	return os.RemoveAll(checkpointDir(parsed))
}

func writeJSONFile(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0644)
}

func readOptimizerState(path string) (optimizerState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return optimizerState{}, fmt.Errorf("load optimizer state: %w", err)
	}
	var state optimizerState
	if err := json.Unmarshal(data, &state); err != nil {
		return optimizerState{}, fmt.Errorf("load optimizer state: %w", err)
	}
	if state.Format != "localtinker.optimizer" || state.Version != 1 {
		return optimizerState{}, fmt.Errorf("load optimizer state: unsupported format")
	}
	return state, nil
}

func checkpointDir(path TinkerPath) string {
	return filepath.Join(checkpointRoot(), path.ModelID, path.Kind, cleanName(path.Name))
}

func writeCheckpointArchive(path, dir string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create checkpoint archive: %w", err)
	}
	defer file.Close()

	tw := tar.NewWriter(file)
	err = filepath.WalkDir(dir, func(name string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		rel, err := filepath.Rel(dir, name)
		if err != nil {
			return err
		}
		hdr, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		hdr.Name = filepath.ToSlash(rel)
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		in, err := os.Open(name)
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(tw, in)
		closeErr := in.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	})
	if err != nil {
		_ = tw.Close()
		return fmt.Errorf("write checkpoint archive: %w", err)
	}
	if err := tw.Close(); err != nil {
		return fmt.Errorf("close checkpoint archive: %w", err)
	}
	return nil
}

func checkpointRoot() string {
	if root := os.Getenv("LOCALTINKER_CHECKPOINT_ROOT"); root != "" {
		return root
	}
	return filepath.Join(os.TempDir(), "localtinker")
}

func promptOffsetFromWeights(weights []float64) (int, error) {
	offset := 0
	for offset < len(weights) && weights[offset] == 0 {
		offset++
	}
	if offset == len(weights) {
		return 0, fmt.Errorf("zero total weight")
	}
	for _, w := range weights[offset:] {
		if w != 1 {
			return 0, fmt.Errorf("weights must be all ones or a zero-prefix mask followed by ones")
		}
	}
	return offset + 1, nil
}

func stopTokenSequences(v any, tok mlxlm.Tokenizer) ([][]int, error) {
	if v == nil {
		return nil, nil
	}
	if token, ok := scalarStopToken(v); ok {
		return [][]int{{token}}, nil
	}
	if s, ok := v.(string); ok {
		return tokenizeStopStrings(tok, []string{s})
	}
	if stops, ok := v.([]string); ok {
		return tokenizeStopStrings(tok, stops)
	}
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var one []int
	if err := json.Unmarshal(data, &one); err == nil && len(one) > 0 {
		return [][]int{one}, nil
	}
	var many [][]int
	if err := json.Unmarshal(data, &many); err == nil {
		return many, nil
	}
	var oneString string
	if err := json.Unmarshal(data, &oneString); err == nil {
		return tokenizeStopStrings(tok, []string{oneString})
	}
	var manyStrings []string
	if err := json.Unmarshal(data, &manyStrings); err == nil {
		return tokenizeStopStrings(tok, manyStrings)
	}
	return nil, nil
}

func tokenizeStopStrings(tok mlxlm.Tokenizer, stops []string) ([][]int, error) {
	if tok == nil {
		return nil, fmt.Errorf("tokenizer unavailable")
	}
	out := make([][]int, 0, len(stops))
	for i, stop := range stops {
		if stop == "" {
			return nil, fmt.Errorf("stop string %d is empty", i)
		}
		tokens, err := tok.Encode(stop)
		if err != nil {
			return nil, fmt.Errorf("tokenize stop string %d: %w", i, err)
		}
		if len(tokens) == 0 {
			return nil, fmt.Errorf("stop string %d tokenized to empty sequence", i)
		}
		seq := make([]int, len(tokens))
		for j, token := range tokens {
			if token < 0 {
				return nil, fmt.Errorf("stop string %d tokenized to negative token %d", i, token)
			}
			seq[j] = int(token)
		}
		out = append(out, seq)
	}
	return out, nil
}

func scalarStopToken(v any) (int, bool) {
	switch v := v.(type) {
	case int:
		if v >= 0 {
			return v, true
		}
	case int32:
		if v >= 0 {
			return int(v), true
		}
	case int64:
		if v >= 0 && int64(int(v)) == v {
			return int(v), true
		}
	case float64:
		token := int(v)
		if v >= 0 && float64(token) == v {
			return token, true
		}
	case json.Number:
		n, err := v.Int64()
		if err == nil && n >= 0 && int64(int(n)) == n {
			return int(n), true
		}
	}
	return 0, false
}

func matchesStop(tokens []int, stops [][]int) bool {
	for _, stop := range stops {
		if len(stop) == 0 || len(stop) > len(tokens) {
			continue
		}
		start := len(tokens) - len(stop)
		ok := true
		for i := range stop {
			if tokens[start+i] != stop[i] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

type noopTokenizer struct{}

func (noopTokenizer) Encode(string) ([]int32, error) { return nil, fmt.Errorf("tokenizer unavailable") }
func (noopTokenizer) Decode([]int32) (string, error) { return "", fmt.Errorf("tokenizer unavailable") }
func (noopTokenizer) EOSToken() int32                { return -1 }
