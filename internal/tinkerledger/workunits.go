package tinkerledger

// Class is the unit a Future's work is measured in. Different operations do
// fundamentally different work, so they accrue in different units; a Policy's
// Rate is applied per work unit regardless of class, but the class records what
// the count means.
type Class string

const (
	// ClassTokens counts processed tokens — the unit for forward and
	// forward/backward passes, where token count is the natural measure of work.
	ClassTokens Class = "tokens"
	// ClassSteps counts optimizer steps — the unit for optim, where one step is
	// one parameter update regardless of batch size.
	ClassSteps Class = "steps"
	// ClassBytes counts bytes written — the unit for save operations, where work
	// is proportional to checkpoint size.
	ClassBytes Class = "bytes"
	// ClassSamples counts generated sequences — the unit for sampling.
	ClassSamples Class = "samples"
	// ClassNone marks an operation that earns no work credit (control or
	// metadata operations).
	ClassNone Class = "none"
)

// ClassFor returns the work-unit class for a Future operation. An unknown
// operation defaults to [ClassNone], so a new operation earns nothing until it
// is given an explicit class — the conservative default for an accounting
// system (never credit work whose meaning is undefined).
func ClassFor(operation string) Class {
	switch operation {
	case "forward", "forward_backward", "forward_backward_custom", "compute_logprobs":
		return ClassTokens
	case "optim_step":
		return ClassSteps
	case "save_state", "save_weights", "save_weights_for_sampler":
		return ClassBytes
	case "sample":
		return ClassSamples
	default:
		return ClassNone
	}
}

// Expensive reports whether an operation produces a re-scorable trajectory and
// so warrants verify-gated accrual (an auditor re-score) rather than a sampled
// spot check. Forward/backward and logprob operations carry logprobs an auditor
// can re-derive; cheap operations (save, sample dispatch) are spot-checked.
func Expensive(operation string) bool {
	switch operation {
	case "forward", "forward_backward", "forward_backward_custom", "compute_logprobs":
		return true
	default:
		return false
	}
}
