// Package tinkertrain provides local trainable models for localtinker.
package tinkertrain

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
)

var ErrNotFound = errors.New("model not found")

type Manager struct {
	mu       sync.Mutex
	models   map[string]trainModel
	samplers map[string]string
}

type CreateConfig struct {
	BaseModel string
	LoRARank  int
}

type Request struct {
	ModelID string
	Input   ForwardBackwardInput
}

type ForwardBackwardInput struct {
	Data         []Datum            `json:"data"`
	LossFn       string             `json:"loss_fn"`
	LossFnConfig map[string]float64 `json:"loss_fn_config"`
}

type Datum struct {
	ModelInput   ModelInput            `json:"model_input"`
	LossFnInputs map[string]TensorData `json:"loss_fn_inputs"`
}

type ModelInput struct {
	Chunks []ModelInputChunk `json:"chunks"`
}

type ModelInputChunk struct {
	Type           string `json:"type"`
	Tokens         []int  `json:"tokens,omitempty"`
	Format         string `json:"format,omitempty"`
	Data           []byte `json:"data,omitempty"`
	Location       string `json:"location,omitempty"`
	ExpectedTokens *int   `json:"expected_tokens,omitempty"`
}

type TensorData struct {
	Data              []float64 `json:"data"`
	DType             string    `json:"dtype"`
	Shape             []int     `json:"shape,omitempty"`
	SparseCrowIndices []int     `json:"sparse_crow_indices,omitempty"`
	SparseColIndices  []int     `json:"sparse_col_indices,omitempty"`
}

type ForwardBackwardOutput struct {
	LossFnOutputType string              `json:"loss_fn_output_type"`
	LossFnOutputs    []map[string]Tensor `json:"loss_fn_outputs"`
	Metrics          map[string]float64  `json:"metrics"`
}

type Tensor struct {
	Data  []float64 `json:"data"`
	DType string    `json:"dtype"`
	Shape []int     `json:"shape,omitempty"`
}

type AdamParams struct {
	LearningRate float64 `json:"learning_rate"`
	Beta1        float64 `json:"beta1"`
	Beta2        float64 `json:"beta2"`
	Eps          float64 `json:"eps"`
	WeightDecay  float64 `json:"weight_decay"`
	GradClipNorm float64 `json:"grad_clip_norm"`
}

type OptimStepOutput struct {
	Metrics map[string]float64 `json:"metrics,omitempty"`
}

type SampleRequest struct {
	SamplingSessionID  string         `json:"sampling_session_id,omitempty"`
	SeqID              int            `json:"seq_id,omitempty"`
	NumSamples         int            `json:"num_samples"`
	Prompt             ModelInput     `json:"prompt"`
	SamplingParams     SamplingParams `json:"sampling_params"`
	PromptLogprobs     bool           `json:"prompt_logprobs,omitempty"`
	TopKPromptLogprobs int            `json:"topk_prompt_logprobs,omitempty"`
	BaseModel          string         `json:"base_model,omitempty"`
	ModelPath          string         `json:"model_path,omitempty"`
}

type SamplingParams struct {
	MaxTokens      int      `json:"max_tokens,omitempty"`
	Seed           int      `json:"seed,omitempty"`
	Stop           any      `json:"stop,omitempty"`
	Temperature    *float64 `json:"temperature,omitempty"`
	TopK           int      `json:"top_k,omitempty"`
	TopP           *float64 `json:"top_p,omitempty"`
	PromptLogprobs bool     `json:"prompt_logprobs,omitempty"`
}

type SampleOutput struct {
	Type               string            `json:"type"`
	Sequences          []SampledSequence `json:"sequences"`
	PromptLogprobs     []*float64        `json:"prompt_logprobs,omitempty"`
	TopKPromptLogprobs []any             `json:"topk_prompt_logprobs,omitempty"`
}

type SampledSequence struct {
	StopReason string    `json:"stop_reason"`
	Tokens     []int     `json:"tokens"`
	Logprobs   []float64 `json:"logprobs,omitempty"`
}

type trainModel interface {
	forwardBackward(context.Context, ForwardBackwardInput, bool) (ForwardBackwardOutput, error)
	optimStep(context.Context, AdamParams) (OptimStepOutput, error)
	saveState(context.Context, string) (string, error)
	loadState(context.Context, string, bool) error
	saveForSampler(context.Context, string) (string, error)
	sample(context.Context, SampleRequest) (SampleOutput, error)
}

func NewManager() *Manager {
	return &Manager{
		models:   make(map[string]trainModel),
		samplers: make(map[string]string),
	}
}

func (m *Manager) Create(ctx context.Context, modelID string, cfg CreateConfig) error {
	if modelID == "" {
		return fmt.Errorf("create train model: empty model id")
	}
	model, err := newMLXModel(ctx, modelID, cfg)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.models == nil {
		m.models = make(map[string]trainModel)
	}
	if m.samplers == nil {
		m.samplers = make(map[string]string)
	}
	m.models[modelID] = model
	return nil
}

func (m *Manager) Delete(_ context.Context, modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.models, modelID)
	for sessionID, id := range m.samplers {
		if id == modelID {
			delete(m.samplers, sessionID)
		}
	}
}

func (m *Manager) Forward(ctx context.Context, req Request) (ForwardBackwardOutput, error) {
	return m.run(ctx, req, false)
}

func (m *Manager) ForwardBackward(ctx context.Context, req Request) (ForwardBackwardOutput, error) {
	return m.run(ctx, req, true)
}

func (m *Manager) OptimStep(ctx context.Context, modelID string, params AdamParams) (OptimStepOutput, error) {
	model, err := m.model(modelID)
	if err != nil {
		return OptimStepOutput{}, err
	}
	return model.optimStep(ctx, params)
}

func (m *Manager) SaveForSampler(ctx context.Context, modelID, name string) (string, error) {
	model, err := m.model(modelID)
	if err != nil {
		return "", err
	}
	return model.saveForSampler(ctx, name)
}

func (m *Manager) SaveState(ctx context.Context, modelID, name string) (string, error) {
	model, err := m.model(modelID)
	if err != nil {
		return "", err
	}
	return model.saveState(ctx, name)
}

func (m *Manager) LoadState(ctx context.Context, modelID, path string) error {
	return m.LoadStateWithOptimizer(ctx, modelID, path, false)
}

func (m *Manager) LoadStateWithOptimizer(ctx context.Context, modelID, path string, optimizer bool) error {
	model, err := m.model(modelID)
	if err != nil {
		return err
	}
	return model.loadState(ctx, path, optimizer)
}

func (m *Manager) CreateSamplingSession(ctx context.Context, sessionID, modelPath, baseModel string) error {
	if sessionID == "" {
		return fmt.Errorf("create sampling session: empty session id")
	}
	modelID := modelIDFromPath(modelPath)
	if modelID == "" && baseModel != "" {
		modelID = baseModel
	}
	if modelID == "" {
		return fmt.Errorf("create sampling session: missing model path")
	}
	if _, err := m.model(modelID); err != nil {
		if baseModel == "" || !errors.Is(err, ErrNotFound) {
			return err
		}
		if err := m.Create(ctx, modelID, CreateConfig{BaseModel: baseModel}); err != nil {
			return err
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.samplers == nil {
		m.samplers = make(map[string]string)
	}
	m.samplers[sessionID] = modelID
	return nil
}

func (m *Manager) Sample(ctx context.Context, req SampleRequest) (SampleOutput, error) {
	modelID := modelIDFromPath(req.ModelPath)
	if modelID == "" && req.SamplingSessionID != "" {
		var ok bool
		m.mu.Lock()
		modelID, ok = m.samplers[req.SamplingSessionID]
		m.mu.Unlock()
		if !ok {
			return SampleOutput{}, fmt.Errorf("unknown sampling session")
		}
	}
	if modelID == "" && req.BaseModel != "" {
		modelID = req.BaseModel
	}
	model, err := m.model(modelID)
	if err != nil {
		return SampleOutput{}, err
	}
	return model.sample(ctx, req)
}

func (m *Manager) run(ctx context.Context, req Request, backward bool) (ForwardBackwardOutput, error) {
	model, err := m.model(req.ModelID)
	if err != nil {
		return ForwardBackwardOutput{}, err
	}
	return model.forwardBackward(ctx, req.Input, backward)
}

func (m *Manager) model(id string) (trainModel, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	model, ok := m.models[id]
	if !ok {
		return nil, ErrNotFound
	}
	return model, nil
}

func modelIDFromPath(path string) string {
	const prefix = "tinker://"
	if len(path) < len(prefix) || path[:len(prefix)] != prefix {
		return ""
	}
	rest := path[len(prefix):]
	for i, c := range rest {
		if c == '/' {
			return rest[:i]
		}
	}
	return rest
}

func (d Datum) targets() ([]int, error) {
	target, ok := d.LossFnInputs["target_tokens"]
	if !ok {
		return nil, fmt.Errorf("missing target_tokens")
	}
	if target.SparseCrowIndices != nil || target.SparseColIndices != nil {
		return nil, fmt.Errorf("sparse target_tokens are not supported")
	}
	out := make([]int, len(target.Data))
	for i, v := range target.Data {
		if math.Trunc(v) != v {
			return nil, fmt.Errorf("target_tokens[%d] = %v is not an integer", i, v)
		}
		if v < 0 || v > float64(math.MaxInt32) {
			return nil, fmt.Errorf("target_tokens[%d] = %v is out of range", i, v)
		}
		out[i] = int(v)
	}
	return out, nil
}

func (d Datum) weights(n int) ([]float64, error) {
	weight, ok := d.LossFnInputs["weights"]
	if !ok {
		out := make([]float64, n)
		for i := range out {
			out[i] = 1
		}
		return out, nil
	}
	if weight.SparseCrowIndices != nil || weight.SparseColIndices != nil {
		return nil, fmt.Errorf("sparse weights are not supported")
	}
	if len(weight.Data) != n {
		return nil, fmt.Errorf("weights=%d target tokens=%d", len(weight.Data), n)
	}
	for i, v := range weight.Data {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return nil, fmt.Errorf("weights[%d] = %v is not a non-negative finite number", i, v)
		}
	}
	return append([]float64(nil), weight.Data...), nil
}

// tokens returns the token sequence implied by m. Image chunks contribute
// expected_tokens placeholder slots (zero ids) so the SDK contract — well-formed
// multimodal requests are accepted at the parse layer — holds. Multimodal
// execution is refused at the MLX boundary (see hasMultimodalChunks).
func (m ModelInput) tokens() ([]int, error) {
	var out []int
	for _, chunk := range m.Chunks {
		switch chunk.Type {
		case "", "encoded_text":
			out = append(out, chunk.Tokens...)
		case "image", "image_asset_pointer":
			if err := ValidateImageChunk(chunk); err != nil {
				return nil, err
			}
			n := *chunk.ExpectedTokens
			out = append(out, make([]int, n)...)
		default:
			return nil, fmt.Errorf("unknown model input chunk type %q", chunk.Type)
		}
	}
	return out, nil
}

// hasMultimodalChunks reports whether m contains any image or
// image_asset_pointer chunk. Local MLX execution refuses these even though
// the parse and token-counting layers accept them.
func (m ModelInput) hasMultimodalChunks() bool {
	return m.multimodalRefusal() != nil
}

// multimodalRefusal returns a typed boundary error describing why local
// execution cannot consume m, or nil if m has no multimodal chunks. The
// error names the boundary explicitly so callers (and tests) can tell
// "no vision backend" apart from "no image asset store".
func (m ModelInput) multimodalRefusal() error {
	for _, chunk := range m.Chunks {
		switch chunk.Type {
		case "image":
			return fmt.Errorf("image chunks require a vision backend, which the local MLX runtime does not provide")
		case "image_asset_pointer":
			return fmt.Errorf("image_asset_pointer chunks require a local image asset store, which is not configured")
		}
	}
	return nil
}

// ValidateImageChunk reports whether c is a well-formed image or
// image_asset_pointer chunk per the SDK contract. Format must be png or
// jpeg, ExpectedTokens must be set and positive, and image bytes must
// begin with the matching magic bytes. Local execution refuses any
// multimodal chunk regardless of validity (see hasMultimodalChunks).
func ValidateImageChunk(c ModelInputChunk) error {
	switch c.Format {
	case "png", "jpeg":
	default:
		return fmt.Errorf("%s chunk: format %q, want png or jpeg", c.Type, c.Format)
	}
	if c.ExpectedTokens == nil {
		return fmt.Errorf("%s chunk: expected_tokens is required", c.Type)
	}
	if *c.ExpectedTokens <= 0 {
		return fmt.Errorf("%s chunk: expected_tokens = %d, want positive", c.Type, *c.ExpectedTokens)
	}
	switch c.Type {
	case "image":
		if len(c.Data) == 0 {
			return fmt.Errorf("image chunk: data is required")
		}
		if err := checkImageMagic(c.Format, c.Data); err != nil {
			return fmt.Errorf("image chunk: %w", err)
		}
	case "image_asset_pointer":
		if c.Location == "" {
			return fmt.Errorf("image_asset_pointer chunk: location is required")
		}
	}
	return nil
}

// pngMagic and jpegMagic are the leading bytes of a PNG or JPEG file. We
// check only the magic prefix; full image decoding is left to a vision
// backend, which the local MLX runtime does not provide.
var (
	pngMagic  = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	jpegMagic = []byte{0xFF, 0xD8, 0xFF}
)

func checkImageMagic(format string, data []byte) error {
	switch format {
	case "png":
		if len(data) < len(pngMagic) || !bytes.Equal(data[:len(pngMagic)], pngMagic) {
			return fmt.Errorf("data does not start with PNG magic bytes")
		}
	case "jpeg":
		if len(data) < len(jpegMagic) || !bytes.Equal(data[:len(jpegMagic)], jpegMagic) {
			return fmt.Errorf("data does not start with JPEG magic bytes")
		}
	}
	return nil
}
