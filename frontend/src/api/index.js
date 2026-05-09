import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export async function ingestVideos(urlA, urlB) {
  const response = await api.post('/ingest', {
    url_a: urlA,
    url_b: urlB,
  })
  return response.data
}

export async function streamChat({ sessionId, message, history, onToken, onSources, onDone }) {
  const response = await fetch(`${BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      history,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Chat request failed: ${errorText}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let doneCalled = false

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''

    for (const part of parts) {
      const trimmed = part.trim()
      if (!trimmed) continue
      const line = trimmed.startsWith('data:') ? trimmed.slice(5).trim() : trimmed

      if (!line) continue

      try {
        const payload = JSON.parse(line)
        if (payload.type === 'token') {
          onToken(payload.content)
        } else if (payload.type === 'sources') {
          onSources(payload.sources)
        } else if (payload.type === 'done') {
          doneCalled = true
          onDone()
        }
      } catch (err) {
        console.warn('Unable to parse SSE chunk:', err)
      }
    }
  }

  if (!doneCalled) {
    onDone()
  }
}
