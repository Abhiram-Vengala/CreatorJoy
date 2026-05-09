import { useState, useRef, useEffect } from 'react'
import { streamChat } from '../api'
import SourceCitations from './SourceCitations'
import styles from './ChatPanel.module.css'

const SUGGESTED_PROMPTS = [
  "Why did one video outperform the other?",
  "Compare the hooks in the first 10 seconds",
  "What specific improvements would you suggest?",
  "Which video has better audience retention signals?",
]

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`${styles.messageRow} ${isUser ? styles.userRow : styles.assistantRow}`}>
      {!isUser && (
        <div className={styles.avatar}>CJ</div>
      )}
      <div className={styles.bubble}>
        <p className={styles.messageText}>{msg.content}</p>
        {msg.sources && msg.sources.length > 0 && (
          <SourceCitations sources={msg.sources} />
        )}
      </div>
    </div>
  )
}

function StreamingMessage({ content, sources }) {
  return (
    <div className={`${styles.messageRow} ${styles.assistantRow}`}>
      <div className={styles.avatar}>CJ</div>
      <div className={styles.bubble}>
        <p className={styles.messageText}>
          {content}
          <span className={styles.cursor} />
        </p>
        {sources && sources.length > 0 && (
          <SourceCitations sources={sources} />
        )}
      </div>
    </div>
  )
}

export default function ChatPanel({ sessionId, disabled }) {
  const [messages, setMessages]         = useState([])
  const [input, setInput]               = useState('')
  const [streaming, setStreaming]       = useState(false)
  const [streamContent, setStreamContent] = useState('')
  const [streamSources, setStreamSources] = useState([])
  const [error, setError]               = useState(null)
  const bottomRef                        = useRef(null)
  const inputRef                         = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamContent])

  async function send(text) {
    const message = text || input.trim()
    if (!message || streaming || !sessionId) return

    setInput('')
    setError(null)

    const userMsg = { role: 'user', content: message }
    const nextHistory = [...messages, userMsg]
    setMessages(nextHistory)

    setStreaming(true)
    setStreamContent('')
    setStreamSources([])

    let fullContent = ''

    try {
      await streamChat({
        sessionId,
        message,
        history: messages.map(m => ({ role: m.role, content: m.content })),
        onToken: (token) => {
          fullContent += token
          setStreamContent(fullContent)
        },
        onSources: (sources) => {
          setStreamSources(sources)
        },
        onDone: () => {
          setMessages(prev => [
            ...prev,
            {
              role: 'assistant',
              content: fullContent,
              sources: streamSources,
            },
          ])
          setStreaming(false)
          setStreamContent('')
          setStreamSources([])
          inputRef.current?.focus()
        },
      })
    } catch (err) {
      setError('Something went wrong. Check the backend and try again.')
      setStreaming(false)
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const isEmpty = messages.length === 0 && !streaming

  return (
    <div className={styles.panel}>
      {/* Messages area */}
      <div className={styles.messages}>
        {isEmpty && !disabled && (
          <div className={styles.emptyState}>
            <p className={styles.emptyTitle}>Ask anything about the videos</p>
            <p className={styles.emptySubtitle}>Powered by transcript analysis + RAG</p>
            <div className={styles.suggestions}>
              {SUGGESTED_PROMPTS.map((p, i) => (
                <button
                  key={i}
                  className={styles.suggestion}
                  style={{ animationDelay: `${i * 80}ms` }}
                  onClick={() => send(p)}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        {disabled && (
          <div className={styles.disabledState}>
            <div className={styles.lockIcon}>⬡</div>
            <p>Analyse two videos first to unlock the chat</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}

        {streaming && (
          <StreamingMessage content={streamContent} sources={streamSources} />
        )}

        {error && (
          <div className={styles.error}>{error}</div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className={styles.inputBar}>
        <input
          ref={inputRef}
          className={styles.chatInput}
          placeholder={disabled ? 'Analyse videos first...' : 'Ask about hooks, retention, improvements...'}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled || streaming}
        />
        <button
          className={styles.sendBtn}
          onClick={() => send()}
          disabled={disabled || streaming || !input.trim()}
        >
          {streaming ? <span className={styles.spinner} /> : '↑'}
        </button>
      </div>
    </div>
  )
}