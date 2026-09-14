import { createSignal, onMount, For } from "solid-js"

interface ImportanceEntry {
    file: string
    score: number
}

interface Stats {
    turnCount: number
    tokensSaved: number
    filesPruned: number
}

function App() {
    const [activeTab, setActiveTab] = createSignal<"files" | "stats" | "config">("files")
    const [files, setFiles] = createSignal<ImportanceEntry[]>([])
    const [stats, setStats] = createSignal<Stats>({ turnCount: 0, tokensSaved: 0, filesPruned: 0 })
    const [expanded, setExpanded] = createSignal(false)

    onMount(() => {
        // Initial data load would go here
    })

    return (
        <div>
            <div>
                <span>Context Manager</span>
                <button onClick={() => setExpanded(!expanded())}>
                    {expanded() ? "−" : "+"}
                </button>
            </div>

            {expanded() && (
                <div>
                    <div>
                        <button onClick={() => setActiveTab("files")}>Files</button>
                        <button onClick={() => setActiveTab("stats")}>Stats</button>
                        <button onClick={() => setActiveTab("config")}>Config</button>
                    </div>

                    {activeTab() === "files" && (
                        <div>
                            <div>Importance Scores</div>
                            <For each={files()}>
                                {(entry) => (
                                    <div>
                                        <span>{entry.file}</span>
                                        <span>{entry.score.toFixed(1)}</span>
                                    </div>
                                )}
                            </For>
                            {files().length === 0 && (
                                <div>No importance data yet.</div>
                            )}
                        </div>
                    )}

                    {activeTab() === "stats" && (
                        <div>
                            <div>Turns: {stats().turnCount}</div>
                            <div>Tokens Saved: {stats().tokensSaved}</div>
                            <div>Files Pruned: {stats().filesPruned}</div>
                        </div>
                    )}

                    {activeTab() === "config" && (
                        <div>
                            <div>Quick Settings</div>
                            <div>Pruning Enabled: yes</div>
                            <div>Min Importance: 3.0</div>
                            <div>Protected Threshold: 7.0</div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default App
