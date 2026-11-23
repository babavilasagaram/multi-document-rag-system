export type ConnectionState = 'checking' | 'connected' | 'disconnected'

export interface HealthResponse {
  status?: string
  [key: string]: unknown
}

export interface DocumentRecord {
  filename: string
  chunks: number
}

export interface DocumentsResponse {
  documents: DocumentRecord[]
  total: number
}

export interface SourceReference {
  filename: string
  page?: number | null
}

export interface QueryResponse {
  question: string
  answer: string
  sources: SourceReference[]
}

export interface UploadResponse {
  status?: string
  filename?: string
  chunks?: number
  message?: string
  [key: string]: unknown
}

export interface UploadProgress {
  loaded: number
  total: number
  percent: number
}
