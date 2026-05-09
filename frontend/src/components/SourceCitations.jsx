import styles from './SourceCitations.module.css'

function formatTimestamp(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function scoreColor(score) {
  if (score >= 0.8) return 'var(--green)'
  if (score >= 0.6) return 'var(--amber)'
  return 'var(--text-muted)'
}

export default function SourceCitations({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <div className={styles.container}>
      <p className={styles.label}>Sources</p>
      <div className={styles.list}>
        {sources.map((src, i) => (
          <div
            key={src.chunk_id}
            className={styles.card}
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <div className={styles.cardHeader}>
              <span className={styles.videoTitle}>{src.video_title}</span>
              <span className={styles.timestamp}>~{formatTimestamp(src.timestamp_approx)}</span>
              <span
                className={styles.score}
                style={{ color: scoreColor(src.score) }}
              >
                {Math.round(src.score * 100)}%
              </span>
            </div>
            <p className={styles.excerpt}>{src.text}</p>
          </div>
        ))}
      </div>
    </div>
  )
}