package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkerartifact"
	"github.com/tmc/localtinker/internal/tinkernode"
	"github.com/tmc/localtinker/internal/tinkernodecap"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

const rpcMaxBytes = 128 << 20

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return usage()
	}
	switch args[0] {
	case "run":
		return runNode(args[1:])
	case "cache":
		return cache(args[1:])
	case "-h", "--help", "help":
		_ = usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func runNode(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	coordinator := fs.String("coordinator", "http://127.0.0.1:8080", "coordinator URL")
	id := fs.String("id", "", "stable node ID")
	name := fs.String("name", hostname(), "node name")
	root := fs.String("root", defaultRoot(), "node state directory")
	heartbeat := fs.Duration("heartbeat", 5*time.Second, "heartbeat interval")
	peerAddr := fs.String("peer-addr", "", "listen address for artifact peer service")
	peerURL := fs.String("peer-url", "", "advertised artifact peer URL")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	store, err := tinkerartifact.OpenStore(filepath.Join(*root, "artifact-store"))
	if err != nil {
		return err
	}
	advertisedPeer, closePeer, err := startArtifactPeer(*peerAddr, *peerURL, store)
	if err != nil {
		return err
	}
	defer closePeer()

	caps, err := tinkernodecap.Probe(tinkernodecap.ProbeOptions{Root: *root})
	if err != nil {
		return err
	}
	if advertisedPeer != "" {
		if caps.Labels == nil {
			caps.Labels = make(map[string]string)
		}
		caps.Labels["artifact_peer_url"] = advertisedPeer
	}

	client := tinkerv1connect.NewTinkerCoordinatorClient(
		http.DefaultClient,
		*coordinator,
		connect.WithReadMaxBytes(rpcMaxBytes),
		connect.WithSendMaxBytes(rpcMaxBytes),
	)
	tracker := tinkerv1connect.NewArtifactTrackerClient(
		http.DefaultClient,
		*coordinator,
		connect.WithReadMaxBytes(rpcMaxBytes),
		connect.WithSendMaxBytes(rpcMaxBytes),
	)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	reg, err := client.RegisterNode(ctx, connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId:       *id,
		Name:         *name,
		Hostname:     hostname(),
		Version:      "dev",
		TrustTier:    "trusted-local",
		Capabilities: protoCapabilities(caps),
		Labels:       caps.Labels,
	}))
	if err != nil {
		return fmt.Errorf("register node: %w", err)
	}
	nodeID := reg.Msg.GetAssignedNodeId()
	log.Printf("registered node %s with coordinator %s", nodeID, reg.Msg.GetCoordinatorId())
	if artifacts, err := artifactInventory(store); err == nil {
		resp, err := client.Heartbeat(ctx, connect.NewRequest(&tinkerv1.HeartbeatRequest{
			NodeId:   nodeID,
			UnixNano: time.Now().UnixNano(),
			Load: &tinkerv1.NodeLoad{
				ActiveLeases:         0,
				QueuedOperations:     0,
				MemoryAvailableBytes: caps.Memory.AvailableBytes,
			},
			Artifacts: artifacts,
		}))
		if err != nil {
			log.Printf("initial heartbeat: %v", err)
		} else if resp.Msg.GetDrainRequested() {
			log.Printf("drain requested")
			return nil
		} else if err := handlePrewarm(ctx, tracker, store, nodeID, resp.Msg.GetPrewarmRoots()); err != nil {
			log.Printf("prewarm: %v", err)
		}
	} else {
		log.Printf("artifact inventory: %v", err)
	}

	ticker := time.NewTicker(*heartbeat)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			artifacts, err := artifactInventory(store)
			if err != nil {
				log.Printf("artifact inventory: %v", err)
			}
			resp, err := client.Heartbeat(ctx, connect.NewRequest(&tinkerv1.HeartbeatRequest{
				NodeId:   nodeID,
				UnixNano: time.Now().UnixNano(),
				Load: &tinkerv1.NodeLoad{
					ActiveLeases:         0,
					QueuedOperations:     0,
					MemoryAvailableBytes: caps.Memory.AvailableBytes,
				},
				Artifacts: artifacts,
			}))
			if err != nil {
				if errors.Is(ctx.Err(), context.Canceled) {
					return nil
				}
				log.Printf("heartbeat: %v", err)
				continue
			}
			if resp.Msg.GetDrainRequested() {
				log.Printf("drain requested")
				return nil
			}
			if err := handlePrewarm(ctx, tracker, store, nodeID, resp.Msg.GetPrewarmRoots()); err != nil {
				log.Printf("prewarm: %v", err)
			}
		}
	}
}

func cache(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: localtinker-node cache root")
	}
	switch args[0] {
	case "root":
		root, err := tinkerartifact.HFCacheRoot()
		if err != nil {
			return err
		}
		fmt.Println(root)
		return nil
	case "import":
		return cacheImport(args[1:])
	case "sync":
		return cacheSync(args[1:])
	case "-h", "--help", "help":
		_ = usage()
		return nil
	default:
		return fmt.Errorf("unknown cache command %q", args[0])
	}
}

func cacheImport(args []string) error {
	fs := flag.NewFlagSet("cache import", flag.ContinueOnError)
	root := fs.String("root", defaultRoot(), "node state directory")
	src := fs.String("src", "", "source directory")
	kind := fs.String("kind", string(tinkerartifact.ArtifactTrainingCheckpoint), "artifact kind")
	storage := fs.String("storage", string(tinkerartifact.StorageTinker), "artifact storage kind")
	name := fs.String("name", "", "artifact alias/name")
	repoID := fs.String("repo-id", "", "Hugging Face repo ID for hf_hub storage")
	repoType := fs.String("repo-type", "model", "Hugging Face repo type for hf_hub storage")
	revision := fs.String("revision", "", "Hugging Face revision/ref for hf_hub storage")
	commitHash := fs.String("commit-hash", "", "Hugging Face commit hash for hf_hub storage")
	coordinator := fs.String("coordinator", "", "optional coordinator URL for manifest publish")
	alias := fs.String("alias", "", "optional coordinator alias")
	chunkSize := fs.Int64("chunk-size", tinkerartifact.DefaultChunkSize, "chunk size")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *src == "" {
		return fmt.Errorf("cache import: missing -src")
	}
	store, err := tinkerartifact.OpenStore(filepath.Join(*root, "artifact-store"))
	if err != nil {
		return err
	}
	opts := tinkerartifact.ManifestOptions{
		Kind:      tinkerartifact.ArtifactKind(*kind),
		Storage:   tinkerartifact.StorageKind(*storage),
		Name:      *name,
		ChunkSize: *chunkSize,
		Metadata: tinkerartifact.Metadata{
			RepoID:     *repoID,
			RepoType:   *repoType,
			Revision:   *revision,
			CommitHash: *commitHash,
		},
	}
	var m tinkerartifact.Manifest
	if opts.Storage == tinkerartifact.StorageHFHub {
		m, err = store.InstallHFHubFromDirectory(context.Background(), *src, opts)
	} else {
		m, err = store.AddDirectory(context.Background(), *src, opts)
	}
	if err != nil {
		return err
	}
	if *coordinator != "" {
		publishAlias := *alias
		if publishAlias == "" {
			publishAlias = *name
		}
		if err := publishManifest(context.Background(), *coordinator, tinkerartifact.ToProto(m), publishAlias); err != nil {
			return err
		}
	}
	fmt.Println(m.RootHash)
	return nil
}

func cacheSync(args []string) error {
	fs := flag.NewFlagSet("cache sync", flag.ContinueOnError)
	root := fs.String("root", defaultRoot(), "node state directory")
	coordinator := fs.String("coordinator", "http://127.0.0.1:8080", "coordinator URL")
	rootHash := fs.String("root-hash", "", "artifact root hash or alias")
	nodeID := fs.String("node-id", "", "optional node ID for inventory report after sync")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *rootHash == "" {
		return fmt.Errorf("cache sync: missing -root-hash")
	}
	store, err := tinkerartifact.OpenStore(filepath.Join(*root, "artifact-store"))
	if err != nil {
		return err
	}
	tracker := tinkerv1connect.NewArtifactTrackerClient(
		http.DefaultClient,
		*coordinator,
		connect.WithReadMaxBytes(rpcMaxBytes),
		connect.WithSendMaxBytes(rpcMaxBytes),
	)
	ctx := context.Background()
	syncedRoot, err := syncArtifact(ctx, tracker, store, *rootHash)
	if err != nil {
		return err
	}
	if *nodeID != "" {
		inv, err := artifactInventory(store)
		if err != nil {
			return err
		}
		if _, err := tracker.ReportInventory(ctx, connect.NewRequest(&tinkerv1.ReportInventoryRequest{
			NodeId:    *nodeID,
			Artifacts: inv,
		})); err != nil {
			return err
		}
	}
	fmt.Println(syncedRoot)
	return nil
}

func handlePrewarm(ctx context.Context, tracker tinkerv1connect.ArtifactTrackerClient, store *tinkerartifact.Store, nodeID string, roots []string) error {
	if len(roots) == 0 {
		return nil
	}
	synced := false
	for _, root := range roots {
		if root == "" || store.Has(root) {
			continue
		}
		if _, err := syncArtifact(ctx, tracker, store, root); err != nil {
			return fmt.Errorf("%s: %w", root, err)
		}
		synced = true
	}
	if !synced {
		return nil
	}
	inv, err := artifactInventory(store)
	if err != nil {
		return err
	}
	_, err = tracker.ReportInventory(ctx, connect.NewRequest(&tinkerv1.ReportInventoryRequest{
		NodeId:    nodeID,
		Artifacts: inv,
	}))
	return err
}

func syncArtifact(ctx context.Context, tracker tinkerv1connect.ArtifactTrackerClient, store *tinkerartifact.Store, rootHashOrAlias string) (string, error) {
	got, err := tracker.GetManifest(ctx, connect.NewRequest(&tinkerv1.GetManifestRequest{RootHashOrAlias: rootHashOrAlias}))
	if err != nil {
		return "", err
	}
	manifest := tinkerartifact.FromProto(got.Msg.GetManifest())
	peers, err := tracker.ListPeers(ctx, connect.NewRequest(&tinkerv1.ListPeersRequest{RootHash: manifest.RootHash}))
	if err != nil {
		return "", err
	}
	var urls []string
	for _, peer := range peers.Msg.GetPeers() {
		urls = append(urls, peer.GetAddress())
	}
	if err := tinkernode.SyncArtifact(ctx, store, manifest, urls); err != nil {
		return "", err
	}
	return manifest.RootHash, nil
}

func publishManifest(ctx context.Context, coordinator string, manifest *tinkerv1.Manifest, alias string) error {
	tracker := tinkerv1connect.NewArtifactTrackerClient(
		http.DefaultClient,
		coordinator,
		connect.WithReadMaxBytes(rpcMaxBytes),
		connect.WithSendMaxBytes(rpcMaxBytes),
	)
	_, err := tracker.PublishManifest(ctx, connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: manifest,
		Alias:    alias,
	}))
	return err
}

func protoCapabilities(c tinkernodecap.Capabilities) *tinkerv1.NodeCapabilities {
	out := &tinkerv1.NodeCapabilities{
		Memory: &tinkerv1.MemoryInfo{
			TotalBytes:     c.Memory.TotalBytes,
			AvailableBytes: c.Memory.AvailableBytes,
		},
		Disk: &tinkerv1.DiskInfo{
			RootBytes:      c.Disk.RootBytes,
			AvailableBytes: c.Disk.AvailableBytes,
		},
		MaxConcurrency: int32(c.MaxConcurrency),
		Features: &tinkerv1.NodeFeatures{
			Lora:               c.Features.LoRA,
			OptimizerState:     c.Features.OptimizerState,
			Sampling:           c.Features.Sampling,
			TopKPromptLogprobs: c.Features.TopKPromptLogprobs,
			CustomLossArrays:   c.Features.CustomLossArrays,
		},
	}
	for _, backend := range c.Backends {
		out.Backends = append(out.Backends, &tinkerv1.Backend{
			Name:          backend.Name,
			Device:        backend.Device,
			UnifiedMemory: backend.UnifiedMem,
		})
	}
	for _, model := range c.Models {
		out.Models = append(out.Models, &tinkerv1.NodeModel{
			Name:        model.Name,
			TokenizerId: model.Tokenizer,
			MaxContext:  int32(model.MaxContext),
			Dtype:       model.DType,
			Quant:       model.Quant,
			CanTrain:    model.CanTrain,
			CanSample:   model.CanSample,
		})
	}
	return out
}

func usage() error {
	fmt.Fprintf(os.Stderr, "usage: localtinker-node run [-coordinator url] [-id id] [-name name] [-root dir] [-peer-addr addr]\n")
	fmt.Fprintf(os.Stderr, "       localtinker-node cache root\n")
	fmt.Fprintf(os.Stderr, "       localtinker-node cache import -src dir [-root dir] [-name alias] [-coordinator url]\n")
	fmt.Fprintf(os.Stderr, "       localtinker-node cache sync -root-hash hash-or-alias [-coordinator url] [-root dir] [-node-id id]\n")
	return flag.ErrHelp
}

func defaultRoot() string {
	if root := os.Getenv("LOCALTINKER_NODE_ROOT"); root != "" {
		return root
	}
	if cache, err := os.UserCacheDir(); err == nil {
		return filepath.Join(cache, "localtinker-node")
	}
	return ".localtinker-node"
}

func hostname() string {
	name, err := os.Hostname()
	if err != nil || name == "" {
		return "localtinker-node"
	}
	return name
}

func startArtifactPeer(addr, advertised string, store *tinkerartifact.Store) (string, func(), error) {
	if addr == "" {
		return advertised, func() {}, nil
	}
	mux := http.NewServeMux()
	tinkernode.RegisterArtifactPeer(mux, store, connect.WithReadMaxBytes(rpcMaxBytes), connect.WithSendMaxBytes(rpcMaxBytes))
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return "", nil, err
	}
	if advertised == "" {
		advertised = advertisedURL(ln.Addr().String())
	}
	server := &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		if err := server.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("artifact peer: %v", err)
		}
	}()
	log.Printf("artifact peer serving on %s", advertised)
	return advertised, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}, nil
}

func advertisedURL(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "http://" + addr
	}
	if host == "" || host == "::" || host == "0.0.0.0" || host == "[::]" {
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, port)
}

func artifactInventory(store *tinkerartifact.Store) ([]*tinkerv1.ArtifactInventory, error) {
	refs, err := store.Inventory()
	if err != nil {
		return nil, err
	}
	out := make([]*tinkerv1.ArtifactInventory, 0, len(refs))
	for _, ref := range refs {
		m, err := store.Manifest(ref.RootHash)
		if err != nil {
			return nil, err
		}
		out = append(out, &tinkerv1.ArtifactInventory{
			RootHash:     ref.RootHash,
			State:        "complete",
			BytesPresent: uint64(m.Size),
		})
	}
	return out, nil
}
