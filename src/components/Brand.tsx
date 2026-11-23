import { Boxes } from 'lucide-react'

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand" aria-label="RAG Workspace">
      <span className="brand-mark" aria-hidden="true">
        <Boxes size={18} strokeWidth={2.2} />
      </span>
      {!compact && (
        <span className="brand-name">
          RAG <strong>WORKSPACE</strong>
        </span>
      )}
    </div>
  )
}
