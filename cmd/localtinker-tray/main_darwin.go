//go:build darwin

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/apple/appkit"
	"github.com/tmc/apple/dispatch"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
)

var (
	delegateSerial uint64
	appDelegate    appkit.NSApplicationDelegateObject
)

type trayApp struct {
	app        appkit.NSApplication
	statusBar  appkit.NSStatusBar
	statusItem appkit.NSStatusItem
	menu       appkit.NSMenu
	delegateID objc.ID

	coordinator string
	nodeID      string
	interval    time.Duration
	client      *http.Client

	mu       sync.Mutex
	snap     snapshot
	closed   chan struct{}
	closeMux sync.Once
}

func main() {
	runtime.LockOSThread()

	coordinator := flag.String("coordinator", "http://127.0.0.1:8080", "coordinator base URL")
	nodeID := flag.String("node-id", "", "node id or name to highlight")
	interval := flag.Duration("interval", 5*time.Second, "poll interval")
	flag.Parse()

	app := appkit.GetNSApplicationClass().SharedApplication()
	app.SetActivationPolicy(appkit.NSApplicationActivationPolicyAccessory)

	tray, err := newTrayApp(app, normalizeCoordinator(*coordinator), *nodeID, *interval)
	if err != nil {
		log.Fatal(err)
	}
	appDelegate = appkit.NewNSApplicationDelegate(appkit.NSApplicationDelegateConfig{
		WillTerminate: func(_ foundation.NSNotification) {
			tray.close()
		},
	})
	app.SetDelegate(appDelegate)
	tray.start()
	app.Run()
}

func normalizeCoordinator(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "http://127.0.0.1:8080"
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	return strings.TrimRight(s, "/")
}

func newTrayApp(app appkit.NSApplication, coordinator, nodeID string, interval time.Duration) (*trayApp, error) {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	t := &trayApp{
		app:         app,
		coordinator: coordinator,
		nodeID:      nodeID,
		interval:    interval,
		client:      &http.Client{Timeout: 3 * time.Second},
		closed:      make(chan struct{}),
		snap: snapshot{
			Coordinator: coordinator,
			CheckedAt:   time.Now(),
		},
	}
	if err := t.registerDelegate(); err != nil {
		return nil, err
	}
	t.statusBar = appkit.GetNSStatusBarClass().SystemStatusBar()
	item := t.statusBar.StatusItemWithLength(appkit.VariableStatusItemLength)
	t.statusItem = appkit.NSStatusItemFromID(item.GetID())
	t.statusItem.SetAutosaveName(appkit.NSStatusItemAutosaveName("com.tmc.localtinker.tray"))
	t.statusItem.SetVisible(true)

	menu := appkit.NewMenuWithTitle("localtinker")
	menu.SetDelegate(appkit.NSMenuDelegateObjectFromID(t.delegateID))
	t.menu = menu
	t.statusItem.SetMenu(&menu)
	t.refreshStatusItem()
	return t, nil
}

func (t *trayApp) registerDelegate() error {
	className := fmt.Sprintf("LocalTinkerTrayDelegate_%d", atomic.AddUint64(&delegateSerial, 1))
	cls, err := objc.RegisterClass(
		className,
		objc.GetClass("NSObject"),
		nil, nil,
		[]objc.MethodDef{
			{Cmd: objc.RegisterName("menuNeedsUpdate:"), Fn: t.handleMenuNeedsUpdate},
			{Cmd: objc.RegisterName("refreshNow:"), Fn: t.handleRefreshNow},
			{Cmd: objc.RegisterName("openDashboard:"), Fn: t.handleOpenDashboard},
			{Cmd: objc.RegisterName("openRuns:"), Fn: t.handleOpenRuns},
			{Cmd: objc.RegisterName("openCheckpoints:"), Fn: t.handleOpenCheckpoints},
			{Cmd: objc.RegisterName("openNodes:"), Fn: t.handleOpenNodes},
			{Cmd: objc.RegisterName("openArtifacts:"), Fn: t.handleOpenArtifacts},
			{Cmd: objc.RegisterName("quit:"), Fn: t.handleQuit},
		},
	)
	if err != nil {
		return fmt.Errorf("register status item delegate: %w", err)
	}
	t.delegateID = objc.Send[objc.ID](objc.ID(cls), objc.Sel("alloc"))
	t.delegateID = objc.Send[objc.ID](t.delegateID, objc.Sel("init"))
	return nil
}

func (t *trayApp) start() {
	t.refresh()
	go func() {
		ticker := time.NewTicker(t.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				t.refresh()
			case <-t.closed:
				return
			}
		}
	}()
}

func (t *trayApp) close() {
	t.closeMux.Do(func() {
		close(t.closed)
		if t.statusItem.ID != 0 && t.statusBar.ID != 0 {
			t.statusBar.RemoveStatusItem(t.statusItem)
		}
	})
}

func (t *trayApp) refresh() {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	s := fetchSnapshot(ctx, t.client, t.coordinator)
	cancel()

	t.mu.Lock()
	t.snap = s
	t.mu.Unlock()

	dispatch.MainQueue().Async(func() {
		t.refreshStatusItem()
	})
}

func (t *trayApp) currentSnapshot() snapshot {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.snap
}

func (t *trayApp) refreshStatusItem() {
	if t.statusItem.ID == 0 {
		return
	}
	s := t.currentSnapshot()
	button := t.statusItem.Button()
	button.SetTitle(s.title(t.nodeID))
	button.SetToolTip(s.tooltip(t.nodeID))
	t.menu.SetTitle("localtinker")
}

func (t *trayApp) handleMenuNeedsUpdate(_ objc.ID, _ objc.SEL, menuID objc.ID) {
	menu := appkit.NSMenuFromID(menuID)
	menu.RemoveAllItems()

	s := t.currentSnapshot()
	addDisabledItem(menu, "localtinker")
	addDisabledItem(menu, "Poll: "+t.interval.String())
	addSeparator(menu)
	for _, line := range s.menuLines(t.nodeID) {
		addDisabledItem(menu, line)
	}
	addSeparator(menu)
	addActionItem(menu, "Open Dashboard", "openDashboard:", t.delegateID)
	addActionItem(menu, "Open Runs", "openRuns:", t.delegateID)
	addActionItem(menu, "Open Checkpoints", "openCheckpoints:", t.delegateID)
	addActionItem(menu, "Open Nodes", "openNodes:", t.delegateID)
	addActionItem(menu, "Open Artifacts", "openArtifacts:", t.delegateID)
	addActionItem(menu, "Refresh Now", "refreshNow:", t.delegateID)
	addActionItem(menu, "Quit", "quit:", t.delegateID)
}

func (t *trayApp) handleRefreshNow(_ objc.ID, _ objc.SEL, _ objc.ID) {
	go t.refresh()
}

func (t *trayApp) handleOpenDashboard(_ objc.ID, _ objc.SEL, _ objc.ID) {
	openDashboard(t.coordinator, "/")
}

func (t *trayApp) handleOpenRuns(_ objc.ID, _ objc.SEL, _ objc.ID) {
	openDashboard(t.coordinator, "/runs")
}

func (t *trayApp) handleOpenCheckpoints(_ objc.ID, _ objc.SEL, _ objc.ID) {
	openDashboard(t.coordinator, "/checkpoints")
}

func (t *trayApp) handleOpenNodes(_ objc.ID, _ objc.SEL, _ objc.ID) {
	openDashboard(t.coordinator, "/nodes")
}

func (t *trayApp) handleOpenArtifacts(_ objc.ID, _ objc.SEL, _ objc.ID) {
	openDashboard(t.coordinator, "/artifacts")
}

func (t *trayApp) handleQuit(_ objc.ID, _ objc.SEL, _ objc.ID) {
	t.close()
	t.app.Terminate(nil)
}

func openDashboard(coordinator, path string) bool {
	url := foundation.NewURLWithString(dashboardURL(coordinator, path))
	return appkit.GetNSWorkspaceClass().SharedWorkspace().OpenURL(url)
}

func addDisabledItem(menu appkit.NSMenu, title string) {
	item := appkit.NewMenuItemWithTitleActionKeyEquivalent(title, 0, "")
	item.SetEnabled(false)
	menu.AddItem(&item)
}

func addActionItem(menu appkit.NSMenu, title, selector string, target objc.ID) {
	item := appkit.NewMenuItemWithTitleActionKeyEquivalent(title, objc.Sel(selector), "")
	item.SetTarget(objectivec.ObjectFromID(target))
	menu.AddItem(&item)
}

func addSeparator(menu appkit.NSMenu) {
	item := appkit.GetNSMenuItemClass().SeparatorItem()
	menu.AddItem(&item)
}
