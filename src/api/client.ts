import type {
  DocumentsResponse,
  HealthResponse,
  QueryResponse,
  SourceReference,
  UploadProgress,
  UploadResponse,
} from './types'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

function safeMessage(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== 'object') return fallback
  const data = payload as Record<string, unknown>
  if (typeof data.detail === 'string') return data.detail
  if (typeof data.message === 'string') return data.message
  if (Array.isArray(data.detail)) {
    const first = data.detail[0]
    if (first && typeof first === 'object' && typeof (first as Record<string, unknown>).msg === 'string') {
      return (first as Record<string, string>).msg
    }
  }
  return fallback
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError('The RAG backend is unavailable. Check that the API is running and try again.')
  }

  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ApiError(safeMessage(payload, `The request failed with status ${response.status}.`), response.status)
  }
  return payload as T
}

export function getHealth() {
  return request<HealthResponse>('/health')
}

export async function getDocuments(): Promise<DocumentsResponse> {
  const payload = await request<unknown>('/documents')
  if (!payload || typeof payload !== 'object' || !Array.isArray((payload as DocumentsResponse).documents)) {
    throw new ApiError('The backend returned an unexpected document response.')
  }
  const data = payload as DocumentsResponse
  return {
    documents: data.documents.filter(
      (document) =>
        document &&
        typeof document.filename === 'string' &&
        typeof document.chunks === 'number',
    ),
    total: typeof data.total === 'number' ? data.total : data.documents.length,
  }
}

export function uploadDocument(
  file: File,
  onProgress?: (progress: UploadProgress) => void,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const formData = new FormData()
    formData.append('file', file)

    xhr.open('POST', `${API_BASE_URL}/documents/upload`)
    xhr.responseType = 'json'
    xhr.upload.addEventListener('progress', (event) => {
      if (!event.lengthComputable) return
      onProgress?.({
        loaded: event.loaded,
        total: event.total,
        percent: Math.round((event.loaded / event.total) * 100),
      })
    })
    xhr.addEventListener('load', () => {
      const payload = xhr.response
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve((payload || {}) as UploadResponse)
        return
      }
      reject(new ApiError(safeMessage(payload, `Upload failed with status ${xhr.status}.`), xhr.status))
    })
    xhr.addEventListener('error', () => {
      reject(new ApiError('The upload could not reach the backend. Check the API connection and CORS settings.'))
    })
    xhr.send(formData)
  })
}

export async function askQuestion(question: string): Promise<QueryResponse> {
  const payload = await request<unknown>('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!payload || typeof payload !== 'object') {
    throw new ApiError('The backend returned an unexpected query response.')
  }
  const data = payload as Record<string, unknown>
  if (typeof data.answer !== 'string') {
    throw new ApiError('The query completed without a readable answer.')
  }
  const sources = Array.isArray(data.sources)
    ? data.sources.filter(
        (source): source is SourceReference =>
          !!source &&
          typeof source === 'object' &&
          typeof (source as SourceReference).filename === 'string',
      )
    : []
  return {
    question: typeof data.question === 'string' ? data.question : question,
    answer: data.answer,
    sources,
  }
}

export function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Something unexpected happened. Please try again.'
}
