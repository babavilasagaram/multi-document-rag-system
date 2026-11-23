import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import {
  Activity,
  ArrowRight,
  BookOpen,
  Bot,
  Check,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Clock3,
  CloudUpload,
  Database,
  FileCheck2,
  FileText,
  FolderOpen,
  Grid2X2,
  HelpCircle,
  LayoutDashboard,
  List,
  LoaderCircle,
  Menu,
  MessageSquareText,
  PanelRightClose,
  Plus,
  RefreshCw,
  Search,
  Send,
  Server,
  Settings,
  Sparkles,
  Upload,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react'
import { API_BASE_URL, askQuestion, getDocuments, getErrorMessage, getHealth, uploadDocument } from './api/client'
import type {
  ConnectionState,
  DocumentRecord,
  QueryResponse,
  UploadProgress,
  UploadResponse,
} from './api/types'
import { Brand } from './components/Brand'

type Page = 'overview' | 'research' | 'documents' | 'activity' | 'settings'
type ViewMode = 'grid' | 'list'

interface ChatMessage {
  id: number
  question: string
  response?: QueryResponse
  error?: string
}

const navItems: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'research', label: 'AI Research', icon: MessageSquareText },
  { id: 'documents', label: 'Documents', icon: FolderOpen },
  { id: 'activity', label: 'Activity', icon: Activity },
  { id: 'settings', label: 'Settings', icon: Settings },
]

const pageTitles: Record<Page, string> = {
  overview: 'Overview',
  research: 'AI Research',
  documents: 'Document library',
  activity: 'Activity',
  settings: 'Settings',
}

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [connection, setConnection] = useState<ConnectionState>('checking')
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [documentsLoading, setDocumentsLoading] = useState(true)
  const [documentsError, setDocumentsError] = useState('')

  const loadData = useCallback(async () => {
    setConnection('checking')
    setDocumentsLoading(true)
    const healthResult = getHealth()
      .then(() => setConnection('connected'))
      .catch(() => setConnection('disconnected'))
    const documentsResult = getDocuments()
      .then((result) => {
        setDocuments(result.documents)
        setDocumentsError('')
      })
      .catch((error) => setDocumentsError(getErrorMessage(error)))
      .finally(() => setDocumentsLoading(false))
    await Promise.allSettled([healthResult, documentsResult])
  }, [])

  useEffect(() => {
    void loadData()
  }, [loadData])

  function navigate(next: Page) {
    setPage(next)
    setMobileOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="app-shell">
      <Sidebar page={page} open={mobileOpen} onNavigate={navigate} onClose={() => setMobileOpen(false)} />
      <div className="workspace">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
            <Menu size={20} />
          </button>
          <div>
            <span className="topbar-eyebrow">RAG WORKSPACE</span>
            <h1>{pageTitles[page]}</h1>
          </div>
          <ConnectionBadge state={connection} />
        </header>

        <main>
          {page === 'overview' && (
            <OverviewPage
              documents={documents}
              loading={documentsLoading}
              error={documentsError}
              connection={connection}
              onNavigate={navigate}
              onRefresh={loadData}
            />
          )}
          {page === 'research' && <ResearchPage documents={documents} />}
          {page === 'documents' && (
            <DocumentsPage
              documents={documents}
              loading={documentsLoading}
              error={documentsError}
              onRefresh={loadData}
            />
          )}
          {page === 'activity' && (
            <ActivityPage documents={documents} connection={connection} loading={documentsLoading} />
          )}
          {page === 'settings' && <SettingsPage connection={connection} onRefresh={loadData} />}
        </main>
      </div>
    </div>
  )
}

function Sidebar({
  page,
  open,
  onNavigate,
  onClose,
}: {
  page: Page
  open: boolean
  onNavigate: (page: Page) => void
  onClose: () => void
}) {
  return (
    <>
      <button className={`sidebar-scrim ${open ? 'is-open' : ''}`} onClick={onClose} aria-label="Close navigation" />
      <aside className={`sidebar ${open ? 'is-open' : ''}`}>
        <div className="sidebar-top">
          <Brand />
          <button className="sidebar-close" onClick={onClose} aria-label="Close navigation">
            <X size={20} />
          </button>
        </div>
        <nav aria-label="Primary navigation">
          <p className="nav-label">Workspace</p>
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <button
                className={`nav-item ${page === item.id ? 'active' : ''}`}
                key={item.id}
                onClick={() => onNavigate(item.id)}
                aria-current={page === item.id ? 'page' : undefined}
              >
                <Icon size={18} />
                <span>{item.label}</span>
                {page === item.id && <span className="active-dot" />}
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer">
          <div className="sidebar-help-icon"><HelpCircle size={18} /></div>
          <div>
            <strong>Local intelligence</strong>
            <span>Grounded in your documents</span>
          </div>
        </div>
      </aside>
    </>
  )
}

function ConnectionBadge({ state }: { state: ConnectionState }) {
  const Icon = state === 'connected' ? Wifi : state === 'checking' ? LoaderCircle : WifiOff
  return (
    <div className={`connection-badge ${state}`}>
      <Icon size={14} className={state === 'checking' ? 'spin' : ''} />
      <span>{state === 'connected' ? 'Backend connected' : state === 'checking' ? 'Checking connection' : 'Backend offline'}</span>
    </div>
  )
}

function OverviewPage({
  documents,
  loading,
  error,
  connection,
  onNavigate,
  onRefresh,
}: {
  documents: DocumentRecord[]
  loading: boolean
  error: string
  connection: ConnectionState
  onNavigate: (page: Page) => void
  onRefresh: () => void
}) {
  const chunks = documents.reduce((total, document) => total + document.chunks, 0)
  const maxChunks = Math.max(...documents.map((document) => document.chunks), 1)

  return (
    <div className="page page-overview">
      <section className="overview-hero">
        <div className="hero-copy">
          <span className="section-kicker"><Sparkles size={15} /> DOCUMENT INTELLIGENCE</span>
          <h2>Your knowledge,<br /><em>intelligently connected.</em></h2>
          <p>Explore your documents and discover precise, source-grounded insights with AI.</p>
          <div className="hero-actions">
            <button className="button primary" onClick={() => onNavigate('research')}>
              Start researching <ArrowRight size={17} />
            </button>
            <button className="button secondary" onClick={() => onNavigate('documents')}>
              <Upload size={17} /> Upload document
            </button>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="visual-grid">
            <span /><span /><span /><span /><span /><span />
          </div>
          <div className="insight-window">
            <div className="insight-top"><span /><span /></div>
            <div className="insight-symbol"><Bot size={25} /></div>
            <p>Connected knowledge</p>
            <div className="answer-lines"><span /><span /><span /></div>
            <div className="source-pill"><FileText size={12} /> Source grounded</div>
          </div>
          <div className="floating-doc doc-one"><FileText size={17} /></div>
          <div className="floating-doc doc-two"><Database size={17} /></div>
        </div>
      </section>

      <section className="metrics-strip" aria-label="Workspace statistics">
        <div className="metric">
          <span className="metric-icon"><FileCheck2 size={19} /></span>
          <div><strong>{loading ? '—' : documents.length}</strong><span>Indexed documents</span></div>
        </div>
        <div className="metric">
          <span className="metric-icon"><Database size={19} /></span>
          <div><strong>{loading ? '—' : chunks.toLocaleString()}</strong><span>Searchable chunks</span></div>
        </div>
        <div className="metric">
          <span className="metric-icon"><Server size={19} /></span>
          <div>
            <strong className={`status-text ${connection}`}>
              {connection === 'connected' ? 'Online' : connection === 'checking' ? 'Checking' : 'Offline'}
            </strong>
            <span>Backend status</span>
          </div>
        </div>
      </section>

      {error && <InlineError message={error} onRetry={onRefresh} />}

      <section className="overview-lower">
        <div className="documents-preview">
          <div className="section-heading">
            <div>
              <span className="eyebrow">YOUR KNOWLEDGE BASE</span>
              <h3>Indexed documents</h3>
            </div>
            <button className="text-button" onClick={() => onNavigate('documents')}>View library <ChevronRight size={16} /></button>
          </div>
          {loading ? (
            <DocumentSkeleton rows={3} />
          ) : documents.length ? (
            <div className="document-bars">
              {documents.slice(0, 5).map((document, index) => (
                <div className="document-bar-row" key={`${document.filename}-${index}`}>
                  <div className="document-title">
                    <span className="file-icon"><FileText size={17} /></span>
                    <span title={document.filename}>{document.filename}</span>
                  </div>
                  <div className="chunk-bar"><span style={{ width: `${Math.max(8, (document.chunks / maxChunks) * 100)}%` }} /></div>
                  <strong>{document.chunks}</strong>
                  <span className="row-unit">chunks</span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyDocuments onUpload={() => onNavigate('documents')} />
          )}
        </div>

        <div className="research-promo">
          <span className="promo-number">01</span>
          <div className="promo-icon"><BookOpen size={24} /></div>
          <div>
            <span className="eyebrow">AI RESEARCH</span>
            <h3>Turn documents into decisions.</h3>
            <p>Ask complex questions and get clear answers connected to their original sources.</p>
          </div>
          <button onClick={() => onNavigate('research')} aria-label="Open AI Research"><ArrowRight size={20} /></button>
        </div>
      </section>
    </div>
  )
}

function ResearchPage({ documents }: { documents: DocumentRecord[] }) {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedMessage, setSelectedMessage] = useState<number | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  async function submit(rawQuestion: string) {
    const cleanQuestion = rawQuestion.trim()
    if (!cleanQuestion || loading) return
    const id = Date.now()
    setQuestion('')
    setLoading(true)
    setMessages((current) => [...current, { id, question: cleanQuestion }])
    try {
      const response = await askQuestion(cleanQuestion)
      setMessages((current) => current.map((message) => message.id === id ? { ...message, response } : message))
      setSelectedMessage(id)
    } catch (error) {
      setMessages((current) => current.map((message) => message.id === id ? { ...message, error: getErrorMessage(error) } : message))
    } finally {
      setLoading(false)
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
    }
  }

  const activeSources = messages.find((message) => message.id === selectedMessage)?.response?.sources || []

  return (
    <div className="research-layout">
      <section className="conversation">
        <div className="conversation-context">
          <div>
            <span className="context-dot" />
            <span>{documents.length} indexed {documents.length === 1 ? 'document' : 'documents'} available</span>
          </div>
          <span>Session only</span>
        </div>

        <div className={`message-canvas ${messages.length ? 'has-messages' : ''}`}>
          {!messages.length ? (
            <div className="research-empty">
              <div className="research-glyph"><Sparkles size={28} /></div>
              <span className="section-kicker">GROUNDED AI RESEARCH</span>
              <h2>Ask your knowledge anything.</h2>
              <p>Explore the information inside your documents with AI-powered research and traceable sources.</p>
              <div className="suggestions">
                {[
                  'Summarize the key ideas across my documents',
                  'What topics appear most frequently?',
                  'Compare the main conclusions in my sources',
                ].map((suggestion) => (
                  <button key={suggestion} onClick={() => setQuestion(suggestion)}>
                    <span>{suggestion}</span><ArrowRight size={15} />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message) => (
                <article className="message-pair" key={message.id}>
                  <div className="user-question">
                    <span>You</span>
                    <p>{message.question}</p>
                  </div>
                  <div className="ai-answer">
                    <div className="answer-avatar"><Sparkles size={17} /></div>
                    <div className="answer-body">
                      <div className="answer-label">RAG Workspace</div>
                      {!message.response && !message.error && <ThinkingState />}
                      {message.error && (
                        <div className="answer-error"><CircleAlert size={17} /><span>{message.error}</span></div>
                      )}
                      {message.response && (
                        <>
                          <MarkdownText text={message.response.answer} />
                          {message.response.sources.length > 0 ? (
                            <button className="source-summary" onClick={() => setSelectedMessage(message.id)}>
                              <span className="source-stack"><FileText size={14} /></span>
                              {message.response.sources.length} {message.response.sources.length === 1 ? 'source' : 'sources'} used
                              <PanelRightClose size={15} />
                            </button>
                          ) : (
                            <p className="no-sources">No source references were returned for this answer.</p>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </article>
              ))}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <form className="composer" onSubmit={(event) => { event.preventDefault(); void submit(question) }}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                void submit(question)
              }
            }}
            placeholder="Ask a question about your documents..."
            rows={2}
            disabled={loading}
            aria-label="Research question"
          />
          <div className="composer-footer">
            <span><Sparkles size={13} /> Answers include available citations</span>
            <button type="submit" disabled={!question.trim() || loading} aria-label="Send question">
              {loading ? <LoaderCircle size={18} className="spin" /> : <Send size={18} />}
            </button>
          </div>
        </form>
      </section>

      <aside className="source-panel">
        <div className="source-panel-head">
          <div><span className="eyebrow">REFERENCE PANEL</span><h3>Sources</h3></div>
          <span className="source-count">{activeSources.length}</span>
        </div>
        {activeSources.length ? (
          <div className="source-list">
            {activeSources.map((source, index) => (
              <div className="source-card" key={`${source.filename}-${source.page}-${index}`}>
                <div className="source-index">{String(index + 1).padStart(2, '0')}</div>
                <FileText size={19} />
                <div>
                  <strong title={source.filename}>{source.filename}</strong>
                  <span>{source.page != null ? `Page ${source.page}` : 'Page not provided'}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="source-empty">
            <div><FileText size={24} /></div>
            <h4>Sources will appear here</h4>
            <p>Ask a question to see the documents and pages used to form an answer.</p>
          </div>
        )}
        <div className="source-note"><CircleCheck size={16} /><span>Only references returned by your backend are shown.</span></div>
      </aside>
    </div>
  )
}

function MarkdownText({ text }: { text: string }) {
  const lines = text.split('\n')
  const nodes: ReactNode[] = []
  let list: string[] = []
  const flushList = () => {
    if (!list.length) return
    nodes.push(<ul key={`list-${nodes.length}`}>{list.map((item, index) => <li key={index}>{inlineMarkdown(item)}</li>)}</ul>)
    list = []
  }
  lines.forEach((line, index) => {
    if (/^[-*]\s+/.test(line)) {
      list.push(line.replace(/^[-*]\s+/, ''))
      return
    }
    flushList()
    if (!line.trim()) return
    if (/^#{1,3}\s/.test(line)) {
      nodes.push(<h4 key={index}>{inlineMarkdown(line.replace(/^#{1,3}\s/, ''))}</h4>)
    } else {
      nodes.push(<p key={index}>{inlineMarkdown(line)}</p>)
    }
  })
  flushList()
  return <div className="markdown-answer">{nodes}</div>
}

function inlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={index}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index}>{part.slice(1, -1)}</code>
    return part
  })
}

function ThinkingState() {
  return (
    <div className="thinking">
      <span /><span /><span />
      <p>Searching your knowledge base</p>
    </div>
  )
}

function DocumentsPage({
  documents,
  loading,
  error,
  onRefresh,
}: {
  documents: DocumentRecord[]
  loading: boolean
  error: string
  onRefresh: () => void
}) {
  const [search, setSearch] = useState('')
  const [view, setView] = useState<ViewMode>('grid')
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState<UploadProgress | null>(null)
  const [uploadResult, setUploadResult] = useState<{ type: 'stored' | 'skipped' | 'error'; message: string } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const filtered = useMemo(
    () => documents.filter((document) => document.filename.toLowerCase().includes(search.trim().toLowerCase())),
    [documents, search],
  )

  async function handleFile(file?: File) {
    if (!file || uploading) return
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setUploadResult({ type: 'error', message: 'Choose a PDF file to add to your knowledge base.' })
      return
    }
    setUploading(true)
    setProgress({ loaded: 0, total: file.size, percent: 0 })
    setUploadResult(null)
    try {
      const result: UploadResponse = await uploadDocument(file, setProgress)
      const status = String(result.status || '').toLowerCase()
      const skipped = status === 'skipped' || /already|duplicate|exist/i.test(result.message || '')
      setUploadResult({
        type: skipped ? 'skipped' : 'stored',
        message: result.message || (skipped ? `${file.name} is already indexed.` : `${result.filename || file.name} was indexed successfully.`),
      })
      await onRefresh()
    } catch (uploadError) {
      setUploadResult({ type: 'error', message: getErrorMessage(uploadError) })
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="page documents-page">
      <section className="page-intro split-intro">
        <div>
          <span className="section-kicker">KNOWLEDGE LIBRARY</span>
          <h2>Build your source of truth.</h2>
          <p>Upload PDF documents, index their contents, and make every page searchable with AI.</p>
        </div>
        <div className="library-stat">
          <span>Indexed</span>
          <strong>{loading ? '—' : documents.length}</strong>
          <small>{documents.length === 1 ? 'document' : 'documents'}</small>
        </div>
      </section>

      <section
        className={`upload-zone ${dragging ? 'is-dragging' : ''} ${uploading ? 'is-uploading' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => { if (event.currentTarget === event.target) setDragging(false) }}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          void handleFile(event.dataTransfer.files[0])
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          hidden
          onChange={(event) => { void handleFile(event.target.files?.[0]); event.currentTarget.value = '' }}
        />
        <div className="upload-icon">{uploading ? <LoaderCircle size={25} className="spin" /> : <CloudUpload size={25} />}</div>
        <div>
          <h3>{uploading ? 'Indexing your document' : 'Drop a PDF to start indexing'}</h3>
          <p>{uploading ? 'Keep this page open while your document is processed.' : 'PDF files are sent directly to your local RAG backend.'}</p>
        </div>
        {uploading ? (
          <div className="upload-progress">
            <div><span style={{ width: `${progress?.percent || 0}%` }} /></div>
            <strong>{progress?.percent || 0}%</strong>
          </div>
        ) : (
          <button className="button secondary" onClick={() => inputRef.current?.click()}><Plus size={17} /> Select PDF</button>
        )}
      </section>

      {uploadResult && (
        <div className={`upload-result ${uploadResult.type}`}>
          {uploadResult.type === 'error' ? <CircleAlert size={18} /> : uploadResult.type === 'skipped' ? <Clock3 size={18} /> : <CircleCheck size={18} />}
          <div><strong>{uploadResult.type.toUpperCase()}</strong><span>{uploadResult.message}</span></div>
          <button onClick={() => setUploadResult(null)} aria-label="Dismiss upload status"><X size={16} /></button>
        </div>
      )}
      {error && <InlineError message={error} onRetry={onRefresh} />}

      <section className="library-section">
        <div className="library-toolbar">
          <div className="search-box">
            <Search size={17} />
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search documents" aria-label="Search documents" />
            {search && <button onClick={() => setSearch('')} aria-label="Clear search"><X size={15} /></button>}
          </div>
          <div className="toolbar-right">
            <span>{filtered.length} {filtered.length === 1 ? 'file' : 'files'}</span>
            <div className="view-toggle">
              <button className={view === 'grid' ? 'active' : ''} onClick={() => setView('grid')} aria-label="Grid view"><Grid2X2 size={17} /></button>
              <button className={view === 'list' ? 'active' : ''} onClick={() => setView('list')} aria-label="List view"><List size={18} /></button>
            </div>
            <button className="icon-button" onClick={onRefresh} aria-label="Refresh documents"><RefreshCw size={17} /></button>
          </div>
        </div>

        {loading ? <DocumentSkeleton rows={4} /> : filtered.length ? (
          <div className={`document-library ${view}`}>
            {filtered.map((document, index) => (
              <article className="document-card" key={`${document.filename}-${index}`}>
                <div className="document-card-top">
                  <div className="pdf-icon"><FileText size={22} /></div>
                  <span className="indexed-label"><Check size={12} /> Indexed</span>
                </div>
                <div className="document-card-body">
                  <h3 title={document.filename}>{document.filename}</h3>
                  <p>PDF document</p>
                </div>
                <div className="document-card-meta">
                  <span><Database size={14} /> {document.chunks.toLocaleString()} chunks</span>
                  <span>Ready for research</span>
                </div>
              </article>
            ))}
          </div>
        ) : search ? (
          <div className="simple-empty"><Search size={24} /><h3>No matching documents</h3><p>Try a different filename or clear your search.</p></div>
        ) : (
          <EmptyDocuments onUpload={() => inputRef.current?.click()} />
        )}
      </section>
    </div>
  )
}

function ActivityPage({
  documents,
  connection,
  loading,
}: {
  documents: DocumentRecord[]
  connection: ConnectionState
  loading: boolean
}) {
  const chunks = documents.reduce((sum, document) => sum + document.chunks, 0)
  return (
    <div className="page activity-page">
      <section className="page-intro">
        <span className="section-kicker">WORKSPACE SIGNAL</span>
        <h2>What your backend can see.</h2>
        <p>A transparent view of currently available system data—without invented analytics.</p>
      </section>
      <div className="activity-composition">
        <section className="system-snapshot">
          <div className="section-heading"><div><span className="eyebrow">LIVE SNAPSHOT</span><h3>Current state</h3></div><ConnectionBadge state={connection} /></div>
          <div className="snapshot-rows">
            <div><span><FileCheck2 size={18} /> Indexed documents</span><strong>{loading ? '—' : documents.length}</strong></div>
            <div><span><Database size={18} /> Available chunks</span><strong>{loading ? '—' : chunks.toLocaleString()}</strong></div>
            <div><span><Server size={18} /> API connection</span><strong>{connection === 'connected' ? 'Available' : connection === 'checking' ? 'Checking' : 'Unavailable'}</strong></div>
          </div>
        </section>
        <aside className="tracking-note">
          <div className="tracking-icon"><Activity size={23} /></div>
          <span className="eyebrow">DATA POLICY</span>
          <h3>Activity history is not tracked.</h3>
          <p>The current backend does not provide query counts, processing times, retrieval accuracy, or persistent user events.</p>
          <div className="not-tracked-list">
            {['Query history', 'Processing metrics', 'Retrieval analytics'].map((item) => <span key={item}><X size={13} /> {item}</span>)}
          </div>
        </aside>
      </div>
      <section className="indexed-ledger">
        <div className="section-heading"><div><span className="eyebrow">AVAILABLE RECORDS</span><h3>Document index</h3></div><span>{documents.length} total</span></div>
        {documents.length ? documents.map((document, index) => (
          <div className="ledger-row" key={`${document.filename}-${index}`}>
            <span className="ledger-index">{String(index + 1).padStart(2, '0')}</span>
            <FileText size={17} />
            <strong>{document.filename}</strong>
            <span>{document.chunks} chunks</span>
            <span className="ledger-status"><span /> Indexed</span>
          </div>
        )) : <div className="ledger-empty">No indexed document records are currently available.</div>}
      </section>
    </div>
  )
}

function SettingsPage({ connection, onRefresh }: { connection: ConnectionState; onRefresh: () => void }) {
  return (
    <div className="page settings-page">
      <section className="page-intro">
        <span className="section-kicker">CONFIGURATION</span>
        <h2>Workspace settings.</h2>
        <p>Connection and application details for this local document intelligence workspace.</p>
      </section>
      <div className="settings-layout">
        <nav className="settings-nav" aria-label="Settings sections">
          <button className="active"><Server size={17} /> Connection</button>
          <button><BookOpen size={17} /> Application</button>
        </nav>
        <div className="settings-content">
          <section className="settings-group">
            <div className="settings-heading"><div><span className="eyebrow">BACKEND</span><h3>API connection</h3></div><ConnectionBadge state={connection} /></div>
            <div className="setting-row">
              <div><strong>Base URL</strong><span>The FastAPI service used for documents and research.</span></div>
              <code>{API_BASE_URL}</code>
            </div>
            <div className="setting-row">
              <div><strong>Connection status</strong><span>Health is checked against the configured backend.</span></div>
              <button className="button secondary small" onClick={onRefresh}><RefreshCw size={15} /> Check again</button>
            </div>
          </section>
          <section className="settings-group">
            <div className="settings-heading"><div><span className="eyebrow">ABOUT</span><h3>Application</h3></div></div>
            <div className="setting-row">
              <div><strong>RAG Workspace</strong><span>Your documents. Connected intelligence.</span></div>
              <span className="version">Frontend 1.0</span>
            </div>
            <div className="setting-row">
              <div><strong>Conversation storage</strong><span>Research messages live in browser memory for this session only.</span></div>
              <span className="local-label">Local only</span>
            </div>
          </section>
          <div className="settings-callout">
            <CircleAlert size={18} />
            <div><strong>Backend configuration is external</strong><p>Authentication, API keys, and server settings are intentionally not exposed in this frontend.</p></div>
          </div>
        </div>
      </div>
    </div>
  )
}

function InlineError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="inline-error">
      <CircleAlert size={18} />
      <span>{message}</span>
      <button onClick={onRetry}><RefreshCw size={14} /> Retry</button>
    </div>
  )
}

function DocumentSkeleton({ rows }: { rows: number }) {
  return <div className="skeleton-list">{Array.from({ length: rows }, (_, index) => <div key={index}><span /><span /><span /></div>)}</div>
}

function EmptyDocuments({ onUpload }: { onUpload: () => void }) {
  return (
    <div className="empty-documents">
      <div><FolderOpen size={25} /></div>
      <h3>Your library is ready</h3>
      <p>Upload the first PDF to begin building a searchable knowledge base.</p>
      <button className="text-button" onClick={onUpload}>Upload a document <ArrowRight size={15} /></button>
    </div>
  )
}

export default App
