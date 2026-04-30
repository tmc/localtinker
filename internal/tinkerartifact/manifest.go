package tinkerartifact

import "time"

// StorageKind identifies where an artifact's final files live.
type StorageKind string

const (
	StorageTinker StorageKind = "tinker"
	StorageHFHub  StorageKind = "hf_hub"
)

// ArtifactKind identifies the logical contents of an immutable artifact.
type ArtifactKind string

const (
	ArtifactBaseModel          ArtifactKind = "base_model"
	ArtifactTokenizer          ArtifactKind = "tokenizer"
	ArtifactLoRAAdapter        ArtifactKind = "lora_adapter"
	ArtifactTrainingCheckpoint ArtifactKind = "training_checkpoint"
	ArtifactSamplerCheckpoint  ArtifactKind = "sampler_checkpoint"
	ArtifactMetadataBundle     ArtifactKind = "metadata_bundle"
)

// Ref names an immutable artifact by root hash plus optional alias metadata.
type Ref struct {
	ID       string
	Kind     ArtifactKind
	Storage  StorageKind
	Name     string
	Version  string
	RootHash string
}

// Manifest describes an immutable artifact snapshot.
type Manifest struct {
	ID        string
	Kind      ArtifactKind
	Storage   StorageKind
	Name      string
	Version   string
	Created   time.Time
	Size      int64
	RootHash  string
	ChunkSize int64
	Files     []ManifestFile
	Metadata  Metadata
}

// ManifestFile describes one file in an artifact manifest.
type ManifestFile struct {
	Path   string
	Size   int64
	Mode   uint32
	SHA256 string
	BlobID string
	Chunks []ChunkRef
}

// ChunkRef describes one content-verified range of a manifest file.
type ChunkRef struct {
	Index  int64
	Offset int64
	Size   int64
	SHA256 string
}

// Metadata carries model and checkpoint details used by scheduling.
type Metadata struct {
	BaseModel    string
	TokenizerID  string
	RepoID       string
	RepoType     string
	Revision     string
	CommitHash   string
	DType        string
	Quant        string
	IsLoRA       bool
	LoRARank     int
	TrainMLP     bool
	TrainAttn    bool
	TrainUnembed bool
	HasOptimizer bool
}

const (
	MinChunkSize     int64 = 16 << 20
	DefaultChunkSize int64 = 32 << 20
	MaxChunkSize     int64 = 128 << 20
)
