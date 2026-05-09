import { useState } from 'react'
import styles from './VideoInput.module.css'

function YouTubeIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
    </svg>
  )
}

function VideoCard({ label, url, setUrl, meta, index }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardLabel}>
        <span className={styles.videoTag}>Video {label}</span>
      </div>

      {meta ? (
        <div className={styles.metaPreview}>
          {meta.thumbnail_url && (
            <img
              src={meta.thumbnail_url}
              alt={meta.title}
              className={styles.thumbnail}
            />
          )}
          <div className={styles.metaInfo}>
            <p className={styles.metaTitle}>{meta.title}</p>
            <p className={styles.metaAuthor}>{meta.author}</p>
          </div>
        </div>
      ) : (
        <div className={styles.inputWrapper}>
          <YouTubeIcon />
          <input
            type="text"
            className={styles.input}
            placeholder="Paste YouTube URL..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>
      )}
    </div>
  )
}

export default function VideoInput({ onIngest, loading, videoMeta }) {
  const [urlA, setUrlA] = useState('')
  const [urlB, setUrlB] = useState('')

  const canSubmit = urlA.trim() && urlB.trim() && !loading

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>CreatorJoy</h1>
        <p className={styles.subtitle}>Drop two videos. Find out why one crushed it.</p>
      </div>

      <div className={styles.cards}>
        <VideoCard
          label="A"
          url={urlA}
          setUrl={setUrlA}
          meta={videoMeta?.video_a}
        />
        <div className={styles.vsLabel}>VS</div>
        <VideoCard
          label="B"
          url={urlB}
          setUrl={setUrlB}
          meta={videoMeta?.video_b}
        />
      </div>

      {!videoMeta && (
        <button
          className={styles.analyzeBtn}
          disabled={!canSubmit}
          onClick={() => onIngest(urlA, urlB)}
        >
          {loading ? (
            <span className={styles.loadingRow}>
              <span className={styles.spinner} />
              Analysing...
            </span>
          ) : (
            'Analyse Videos →'
          )}
        </button>
      )}

      {videoMeta && (
        <div className={styles.readyBadge}>
          ✦ Ready — ask anything below
        </div>
      )}
    </div>
  )
}