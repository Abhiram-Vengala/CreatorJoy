import { useState } from 'react'
import VideoInput from './components/VideoInput'
import ChatPanel from './components/ChatPanel'
import { ingestVideos } from './api'
import styles from './App.module.css'

export default function App() {
  const [sessionId, setSessionId]   = useState(null)
  const [videoMeta, setVideoMeta]   = useState(null)
  const [loading, setLoading]       = useState(false)
  const [ingestError, setIngestError] = useState(null)

  async function handleIngest(urlA, urlB) {
    setLoading(true)
    setIngestError(null)
    try {
      const result = await ingestVideos(urlA, urlB)
      setSessionId(result.session_id)
      setVideoMeta({ video_a: result.video_a, video_b: result.video_b })
    } catch (err) {
      setIngestError(err?.response?.data?.detail || 'Ingestion failed. Check URLs and backend.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.layout}>
      {/* Left panel — video input */}
      <aside className={styles.sidebar}>
        <VideoInput
          onIngest={handleIngest}
          loading={loading}
          videoMeta={videoMeta}
        />
        {ingestError && (
          <div className={styles.ingestError}>{ingestError}</div>
        )}
      </aside>

      {/* Divider */}
      <div className={styles.divider} />

      {/* Right panel — chat */}
      <main className={styles.chat}>
        <ChatPanel
          sessionId={sessionId}
          disabled={!sessionId}
        />
      </main>
    </div>
  )
}
