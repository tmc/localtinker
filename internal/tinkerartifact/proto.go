package tinkerartifact

import "github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"

// ToProto converts a manifest to its protobuf representation.
func ToProto(m Manifest) *tinkerv1.Manifest {
	pm := &tinkerv1.Manifest{
		Id:        m.ID,
		Kind:      string(m.Kind),
		Storage:   string(m.Storage),
		Name:      m.Name,
		Version:   m.Version,
		RootHash:  m.RootHash,
		ChunkSize: m.ChunkSize,
		Metadata:  metadataMap(m.Metadata),
	}
	for _, f := range m.Files {
		pf := &tinkerv1.ManifestFile{
			Path:   f.Path,
			Size:   f.Size,
			Mode:   f.Mode,
			Sha256: f.SHA256,
			BlobId: f.BlobID,
		}
		for _, c := range f.Chunks {
			pf.Chunks = append(pf.Chunks, &tinkerv1.ChunkRef{
				Index:  c.Index,
				Offset: c.Offset,
				Size:   c.Size,
				Sha256: c.SHA256,
			})
		}
		pm.Files = append(pm.Files, pf)
	}
	return pm
}

// FromProto converts a protobuf manifest to a local manifest.
func FromProto(pm *tinkerv1.Manifest) Manifest {
	if pm == nil {
		return Manifest{}
	}
	m := Manifest{
		ID:        pm.GetId(),
		Kind:      ArtifactKind(pm.GetKind()),
		Storage:   StorageKind(pm.GetStorage()),
		Name:      pm.GetName(),
		Version:   pm.GetVersion(),
		RootHash:  pm.GetRootHash(),
		ChunkSize: pm.GetChunkSize(),
		Metadata:  metadataFromMap(pm.GetMetadata()),
	}
	for _, pf := range pm.GetFiles() {
		f := ManifestFile{
			Path:   pf.GetPath(),
			Size:   pf.GetSize(),
			Mode:   pf.GetMode(),
			SHA256: pf.GetSha256(),
			BlobID: pf.GetBlobId(),
		}
		m.Size += f.Size
		for _, pc := range pf.GetChunks() {
			f.Chunks = append(f.Chunks, ChunkRef{
				Index:  pc.GetIndex(),
				Offset: pc.GetOffset(),
				Size:   pc.GetSize(),
				SHA256: pc.GetSha256(),
			})
		}
		m.Files = append(m.Files, f)
	}
	return m
}

func metadataMap(m Metadata) map[string]string {
	out := make(map[string]string)
	set := func(k, v string) {
		if v != "" {
			out[k] = v
		}
	}
	set("base_model", m.BaseModel)
	set("tokenizer_id", m.TokenizerID)
	set("repo_id", m.RepoID)
	set("repo_type", m.RepoType)
	set("revision", m.Revision)
	set("commit_hash", m.CommitHash)
	set("dtype", m.DType)
	set("quant", m.Quant)
	if m.IsLoRA {
		out["is_lora"] = "true"
	}
	if m.LoRARank != 0 {
		out["lora_rank"] = itoa(m.LoRARank)
	}
	if m.TrainMLP {
		out["train_mlp"] = "true"
	}
	if m.TrainAttn {
		out["train_attn"] = "true"
	}
	if m.TrainUnembed {
		out["train_unembed"] = "true"
	}
	if m.HasOptimizer {
		out["has_optimizer"] = "true"
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func metadataFromMap(in map[string]string) Metadata {
	if len(in) == 0 {
		return Metadata{}
	}
	return Metadata{
		BaseModel:    in["base_model"],
		TokenizerID:  in["tokenizer_id"],
		RepoID:       in["repo_id"],
		RepoType:     in["repo_type"],
		Revision:     in["revision"],
		CommitHash:   in["commit_hash"],
		DType:        in["dtype"],
		Quant:        in["quant"],
		IsLoRA:       in["is_lora"] == "true",
		LoRARank:     atoi(in["lora_rank"]),
		TrainMLP:     in["train_mlp"] == "true",
		TrainAttn:    in["train_attn"] == "true",
		TrainUnembed: in["train_unembed"] == "true",
		HasOptimizer: in["has_optimizer"] == "true",
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func atoi(s string) int {
	var n int
	for _, r := range s {
		if r < '0' || r > '9' {
			return 0
		}
		n = n*10 + int(r-'0')
	}
	return n
}
