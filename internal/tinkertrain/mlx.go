package tinkertrain

import (
	"archive/tar"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	lmtrain "github.com/tmc/mlx-go-lm/lmtrain"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/sample"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/training"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/tuner"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/random"
)

const checkpointCompleteFile = "complete"

type mlxModel struct {
	mu       sync.Mutex
	id       string
	base     string
	bundle   *lmtrain.ModelBundle
	adapters *tuner.Set
	rank     int
	step     int
	pending  []training.TrainingSample
	lastLoss float64
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
		id:       modelID,
		base:     apiBase,
		bundle:   bundle,
		adapters: adapters,
		rank:     rank,
	}, nil
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
	samples, tokens, err := trainingSamples(input)
	if err != nil {
		return ForwardBackwardOutput{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	loss, err := training.EvaluateWithAdaptersSamples(m, samples, noopTokenizer{}, len(samples), 1, maxSampleLen(samples), false, false, true)
	if err != nil {
		return ForwardBackwardOutput{}, fmt.Errorf("forward: %w", err)
	}
	m.lastLoss = float64(loss)
	if backward {
		m.pending = samples
	}
	return ForwardBackwardOutput{
		LossFnOutputType: "TensorData",
		LossFnOutputs:    lossOutputs(samples),
		Metrics: map[string]float64{
			"loss:mean":    float64(loss),
			"tokens:sum":   float64(tokens),
			"examples:sum": float64(len(samples)),
		},
	}, nil
}

func (m *mlxModel) optimStep(ctx context.Context, params AdamParams) (OptimStepOutput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.pending) == 0 {
		return OptimStepOutput{}, fmt.Errorf("optimizer step: no pending forward_backward")
	}

	lr := params.LearningRate
	if lr == 0 {
		lr = 1e-4
	}
	trainParams := training.DefaultTrainParameters()
	trainParams.BatchSize = len(m.pending)
	trainParams.Iterations = 1
	trainParams.StepsPerReport = 1
	trainParams.StepsPerEval = 0
	trainParams.ValidationBatches = 0
	trainParams.SaveEvery = 1 << 30
	trainParams.AdapterPath = ""
	trainParams.LearningRate = float32(lr)
	trainParams.WeightDecay = float32(params.WeightDecay)
	trainParams.MaxGradNorm = float32(params.GradClipNorm)
	trainParams.MaxSeqLength = maxSampleLen(m.pending)
	trainParams.AppendEOS = false
	trainParams.BatchOrder = "input"
	trainParams.DataSeed = 0
	trainParams.Optimizer = "adamw"
	trainParams.LossType = "cross-entropy"
	trainParams.FullEval = false
	trainParams.MaskPrompt = true

	loss := m.lastLoss
	err := training.TrainWithSetSamples(m, m.adapters, m.pending, nil, noopTokenizer{}, trainParams, func(p training.Progress) training.ProgressDisposition {
		if p.Type == training.ProgressTrain {
			loss = float64(p.Loss)
		}
		return training.ProgressMore
	})
	if err != nil {
		return OptimStepOutput{}, fmt.Errorf("optimizer step: %w", err)
	}
	m.pending = nil
	m.lastLoss = loss
	m.step++

	return OptimStepOutput{Metrics: map[string]float64{
		"loss:mean":             loss,
		"optimizer_backend:mlx": 1,
		"optimizer_step:unique": float64(m.step),
	}}, nil
}

func (m *mlxModel) saveForSampler(_ context.Context, name string) (string, error) {
	return m.saveAdapter(name, "sampler_weights")
}

func (m *mlxModel) saveState(_ context.Context, name string) (string, error) {
	return m.saveAdapter(name, "weights")
}

func (m *mlxModel) saveAdapter(name, kind string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if name == "" {
		name = "adapter"
	}
	dir := filepath.Join(checkpointRoot(), m.id, kind, cleanName(name))
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create adapter dir: %w", err)
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
	if err := os.WriteFile(filepath.Join(dir, checkpointCompleteFile), []byte("ok\n"), 0644); err != nil {
		return "", fmt.Errorf("write completion marker: %w", err)
	}
	return "tinker://" + m.id + "/" + kind + "/" + cleanName(name), nil
}

func (m *mlxModel) loadState(_ context.Context, path string) error {
	parsed, err := ParseTinkerPath(path)
	if err != nil {
		return err
	}
	file := filepath.Join(checkpointDir(parsed), "adapters.safetensors")
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.adapters.LoadWeights(file); err != nil {
		return fmt.Errorf("load adapter weights: %w", err)
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
	stop := stopTokenSequences(req.SamplingParams.Stop)

	m.mu.Lock()
	defer m.mu.Unlock()

	out := SampleOutput{
		Type:      "sample",
		Sequences: make([]SampledSequence, 0, numSamples),
	}
	for range numSamples {
		seq := append([]int(nil), prompt...)
		gen := make([]int, 0, maxTokens)
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
			next, err := m.nextToken(ctx, seq, req.SamplingParams, stepKey)
			if stepKey != nil {
				stepKey.Free()
			}
			if err != nil {
				return SampleOutput{}, err
			}
			seq = append(seq, next)
			gen = append(gen, next)
			if matchesStop(gen, stop) {
				stopReason = "stop"
				break
			}
		}
		out.Sequences = append(out.Sequences, SampledSequence{
			StopReason: stopReason,
			Tokens:     gen,
		})
	}
	return out, nil
}

func (m *mlxModel) nextToken(ctx context.Context, tokens []int, params SamplingParams, key *mlx.Array) (int, error) {
	xs := make([]int32, len(tokens))
	for i, token := range tokens {
		xs[i] = int32(token)
	}
	input, err := mlx.FromSlice(xs, []int{1, len(xs)}, mlx.Int32)
	if err != nil {
		return 0, fmt.Errorf("sample input: %w", err)
	}
	defer input.Free()

	logits, _ := m.bundle.Model.Forward(ctx, input, nil)
	defer logits.Free()
	shape := logits.Shape()
	if len(shape) != 3 || shape[1] == 0 || shape[2] == 0 {
		return 0, fmt.Errorf("sample logits shape %v", shape)
	}
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

func trainingSamples(input ForwardBackwardInput) ([]training.TrainingSample, int, error) {
	if input.LossFn != "cross_entropy" {
		return nil, 0, fmt.Errorf("unsupported loss function %q", input.LossFn)
	}
	if len(input.Data) == 0 {
		return nil, 0, fmt.Errorf("no data")
	}
	samples := make([]training.TrainingSample, 0, len(input.Data))
	var tokensTotal int
	for i, datum := range input.Data {
		tokens := datum.ModelInput.tokens()
		targets, err := datum.targets()
		if err != nil {
			return nil, 0, err
		}
		weights, err := datum.weights(len(targets))
		if err != nil {
			return nil, 0, err
		}
		promptOffset, err := promptOffsetFromWeights(weights)
		if err != nil {
			return nil, 0, fmt.Errorf("datum %d: %w", i, err)
		}
		activeTokens := len(weights) - promptOffset + 1
		if len(tokens) != len(targets) {
			return nil, 0, fmt.Errorf("datum %d input tokens=%d target tokens=%d", i, len(tokens), len(targets))
		}
		if len(tokens) == 0 {
			return nil, 0, fmt.Errorf("datum %d has no tokens", i)
		}
		for j := 1; j < len(tokens); j++ {
			if tokens[j] != targets[j-1] {
				return nil, 0, fmt.Errorf("datum %d target_tokens must be model_input shifted left by one token", i)
			}
		}
		full := make([]int32, 0, len(tokens)+1)
		for _, token := range tokens {
			full = append(full, int32(token))
		}
		full = append(full, int32(targets[len(targets)-1]))
		samples = append(samples, training.TrainingSample{Tokens: full, PromptOffset: promptOffset})
		tokensTotal += activeTokens
	}
	return samples, tokensTotal, nil
}

func maxSampleLen(samples []training.TrainingSample) int {
	n := 0
	for _, sample := range samples {
		if len(sample.Tokens) > n {
			n = len(sample.Tokens)
		}
	}
	if n < 2 {
		return 2
	}
	return n
}

func lossOutputs(samples []training.TrainingSample) []map[string]Tensor {
	out := make([]map[string]Tensor, 0, len(samples))
	for _, sample := range samples {
		n := len(sample.Tokens) - 1
		if n < 0 {
			n = 0
		}
		out = append(out, map[string]Tensor{
			"logprobs": {Data: make([]float64, n), DType: "float32", Shape: []int{n}},
		})
	}
	return out
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

func stopTokenSequences(v any) [][]int {
	if v == nil {
		return nil
	}
	if token, ok := scalarStopToken(v); ok {
		return [][]int{{token}}
	}
	data, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var one []int
	if err := json.Unmarshal(data, &one); err == nil && len(one) > 0 {
		return [][]int{one}
	}
	var many [][]int
	if err := json.Unmarshal(data, &many); err == nil {
		return many
	}
	return nil
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
