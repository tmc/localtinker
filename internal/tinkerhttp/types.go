package tinkerhttp

import "encoding/json"

type ConfigResponse struct {
	PJWTAuthEnabled                    bool           `json:"pjwt_auth_enabled"`
	CredentialDefaultSource            string         `json:"credential_default_source"`
	SampleDispatchBytesSemaphoreSize   int            `json:"sample_dispatch_bytes_semaphore_size"`
	InflightResponseBytesSemaphoreSize int            `json:"inflight_response_bytes_semaphore_size"`
	UseJWT                             bool           `json:"use_jwt"`
	Auth                               map[string]any `json:"auth"`
	ParallelFWDBWDChunks               bool           `json:"parallel_fwdbwd_chunks"`
	MaxRequestBytes                    int            `json:"max_request_bytes"`
	Features                           map[string]any `json:"features"`
}

type CreateSessionResponse struct {
	Type      string `json:"type"`
	SessionID string `json:"session_id"`
	ID        string `json:"id"`
	Status    string `json:"status"`
}

type HeartbeatRequest struct {
	SessionID string `json:"session_id"`
	ID        string `json:"id"`
}

type HeartbeatResponse struct {
	Type       string `json:"type"`
	SessionID  string `json:"session_id"`
	Status     string `json:"status"`
	HeartbeatN int64  `json:"heartbeat_n"`
}

type FutureResponse struct {
	FutureID  string `json:"future_id"`
	RequestID string `json:"request_id"`
	ID        string `json:"id,omitempty"`
	ModelID   string `json:"model_id,omitempty"`
}

type RetrieveFutureRequest struct {
	FutureID          string `json:"future_id"`
	RequestID         string `json:"request_id"`
	ID                string `json:"id"`
	AllowMetadataOnly bool   `json:"allow_metadata_only"`
}

type RetrieveFutureResponse struct {
	Status              string          `json:"status"`
	State               string          `json:"state"`
	FutureID            string          `json:"future_id"`
	RequestID           string          `json:"request_id,omitempty"`
	Type                string          `json:"type,omitempty"`
	QueueState          string          `json:"queue_state,omitempty"`
	Result              json.RawMessage `json:"result,omitempty"`
	Error               json.RawMessage `json:"error,omitempty"`
	Metadata            json.RawMessage `json:"metadata,omitempty"`
	Category            string          `json:"category,omitempty"`
	ResponsePayloadSize int             `json:"response_payload_size,omitempty"`
}

type TelemetryResponse struct {
	Status string `json:"status"`
}

type ErrorResponse struct {
	Status  string `json:"status"`
	Code    string `json:"code"`
	Message string `json:"message"`
}
