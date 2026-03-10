/**
 * Features.jsx
 * ------------
 * WHY useRef + requestAnimationFrame over setInterval for typewriter:
 *
 * setInterval suffers from "stale closure" in React hooks — the interval
 * callback captures the `currentIndex` value at the time of creation.
 * Even with a deps array, clearing and re-creating the interval on each
 * render can cause double-firing in React StrictMode (which double-invokes
 * effects) and produces visible flicker. requestAnimationFrame + useRef
 * stores mutable state in a ref (not React state), so the callback always
 * reads the latest value. The rAF loop also automatically pauses when the
 * tab is hidden (via the Page Visibility API's effect on rAF), saving CPU
 * for background tabs without extra code. For the diagnostic shuffler and
 * cursor scheduler we use setInterval but with a cleanup that safely clears
 * on unmount/re-run, avoiding the stale-closure issue by resetting on param change.
 */

import { useEffect, useRef, useState } from 'react'

/* ─── Card 1: Diagnostic Shuffler ─────────────────────────────────────── */
const SHUFFLE_CARDS = [
  { label: 'Pose',        color: 'bg-signal/10 border-signal/30',  textColor: 'text-signal' },
  { label: 'Range',       color: 'bg-ink/5 border-ink/20',         textColor: 'text-ink' },
  { label: 'Constraints', color: 'bg-paper border-ink/15',         textColor: 'text-ink/70' },
]

function DiagnosticShuffler() {
  const [stack, setStack] = useState(SHUFFLE_CARDS)

  useEffect(() => {
    const id = setInterval(() => {
      setStack((prev) => {
        const next = [...prev]
        next.unshift(next.pop())
        return next
      })
    }, 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="relative h-36 mt-4">
      {stack.map((card, i) => (
        <div
          key={card.label}
          className={`absolute inset-x-0 border rounded-xl px-5 py-3 flex items-center justify-between transition-all duration-500
            ${card.color}`}
          style={{
            top: `${i * 10}px`,
            zIndex: stack.length - i,
            transform: `scale(${1 - i * 0.04}) translateY(${i * 4}px)`,
            transitionTimingFunction: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
            opacity: 1 - i * 0.25,
          }}
        >
          <span className={`font-mono-data text-xs font-semibold uppercase tracking-widest ${card.textColor}`}>
            {card.label}
          </span>
          <span className="text-ink/30 text-xs font-mono-data">●●●</span>
        </div>
      ))}
    </div>
  )
}

/* ─── Card 2: Telemetry Typewriter ─────────────────────────────────────── */
const TYPEWRITER_MESSAGES = [
  'Torque ripple reduced 12% after phase tuning',
  'Offset increased dwell time at TDC',
  'Constraint window validated for manufacturing',
]

function TelemetryTypewriter() {
  const [displayed, setDisplayed] = useState('')
  const [blink, setBlink] = useState(true)
  const msgIndexRef = useRef(0)
  const charIndexRef = useRef(0)
  const dirRef = useRef('typing') // 'typing' | 'deleting' | 'waiting'
  const frameRef = useRef(null)
  const lastRef = useRef(0)

  useEffect(() => {
    function tick(ts) {
      const delay = dirRef.current === 'typing' ? 45 : 25

      if (ts - lastRef.current >= delay) {
        lastRef.current = ts
        const msg = TYPEWRITER_MESSAGES[msgIndexRef.current]

        if (dirRef.current === 'typing') {
          charIndexRef.current += 1
          setDisplayed(msg.slice(0, charIndexRef.current))
          if (charIndexRef.current >= msg.length) {
            dirRef.current = 'waiting'
            lastRef.current = ts + 1800 // pause before deleting
          }
        } else if (dirRef.current === 'waiting') {
          // After pause, switch to deleting
          if (ts >= lastRef.current) {
            dirRef.current = 'deleting'
          }
        } else {
          // deleting
          charIndexRef.current = Math.max(0, charIndexRef.current - 1)
          setDisplayed(msg.slice(0, charIndexRef.current))
          if (charIndexRef.current === 0) {
            msgIndexRef.current = (msgIndexRef.current + 1) % TYPEWRITER_MESSAGES.length
            dirRef.current = 'typing'
          }
        }
      }

      frameRef.current = requestAnimationFrame(tick)
    }

    frameRef.current = requestAnimationFrame(tick)

    // Cursor blink on a separate interval is fine — it's visual only,
    // no stale-closure risk since it doesn't read component state.
    const blinkId = setInterval(() => setBlink((v) => !v), 530)

    return () => {
      cancelAnimationFrame(frameRef.current)
      clearInterval(blinkId)
    }
  }, [])

  return (
    <div className="mt-4 rounded-xl bg-ink/5 border border-ink/10 p-4">
      {/* Live feed header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="w-2 h-2 rounded-full bg-signal animate-pulse" />
        <span className="font-mono-data text-xs text-ink/50 uppercase tracking-widest">Live Feed</span>
      </div>
      {/* Typewriter output */}
      <p className="font-mono-data text-sm text-ink min-h-[3rem] leading-relaxed">
        {displayed}
        <span
          className="ml-px inline-block w-2 h-4 bg-signal rounded-sm align-middle"
          style={{ opacity: blink ? 1 : 0, transition: 'opacity 0.1s' }}
        />
      </p>
    </div>
  )
}

/* ─── Card 3: Cursor Protocol Scheduler ────────────────────────────────── */
const DAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

function CursorScheduler() {
  const [activeDay, setActiveDay] = useState(null)
  const [publishActive, setPublishActive] = useState(false)
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 })
  const [clicking, setClicking] = useState(false)
  const stepRef = useRef(0)
  const timerRef = useRef(null)

  const DAY_TARGETS = [
    { x: 14, y: 24 },
    { x: 42, y: 24 },
    { x: 70, y: 24 },
    { x: 98, y: 24 },
    { x: 126, y: 24 },
    { x: 154, y: 24 },
    { x: 182, y: 24 },
  ]
  const PUBLISH_TARGET = { x: 96, y: 76 }

  function runStep() {
    const step = stepRef.current
    if (step < 7) {
      const target = DAY_TARGETS[step]
      setCursorPos(target)
      setActiveDay(null)
      setPublishActive(false)
      timerRef.current = setTimeout(() => {
        setClicking(true)
        setActiveDay(step)
        setTimeout(() => setClicking(false), 180)
        stepRef.current += 1
        timerRef.current = setTimeout(runStep, 600)
      }, 400)
    } else {
      // Move to Publish
      setCursorPos(PUBLISH_TARGET)
      timerRef.current = setTimeout(() => {
        setClicking(true)
        setPublishActive(true)
        setTimeout(() => setClicking(false), 180)
        timerRef.current = setTimeout(() => {
          stepRef.current = 0
          setActiveDay(null)
          setPublishActive(false)
          runStep()
        }, 1500)
      }, 400)
    }
  }

  useEffect(() => {
    timerRef.current = setTimeout(runStep, 800)
    return () => clearTimeout(timerRef.current)
  }, [])

  return (
    <div className="mt-4 rounded-xl bg-ink/5 border border-ink/10 p-4 select-none" style={{ position: 'relative', overflow: 'hidden' }}>
      {/* Week grid */}
      <div className="flex gap-1.5 mb-4" style={{ position: 'relative', zIndex: 1 }}>
        {DAYS.map((d, i) => (
          <div
            key={i}
            className={`w-8 h-8 rounded-lg flex items-center justify-center font-mono-data text-xs font-semibold transition-all duration-200
              ${activeDay === i
                ? 'bg-signal text-offwhite scale-95'
                : 'bg-paper border border-ink/15 text-ink/60'
              }`}
          >
            {d}
          </div>
        ))}
      </div>

      {/* Publish button */}
      <button
        className={`w-full py-2 rounded-lg font-grotesk font-semibold text-sm transition-all duration-200
          ${publishActive
            ? 'bg-signal text-offwhite scale-95'
            : 'bg-ink/10 text-ink/60 border border-ink/15'
          }`}
        style={{ position: 'relative', zIndex: 1 }}
      >
        {publishActive ? 'Published!' : 'Publish'}
      </button>

      {/* Animated SVG cursor overlay */}
      <svg
        className="pointer-events-none absolute inset-0 w-full h-full"
        style={{ zIndex: 2 }}
        viewBox="0 0 210 100"
        preserveAspectRatio="xMidYMid meet"
      >
        <g
          transform={`translate(${cursorPos.x}, ${cursorPos.y}) scale(${clicking ? 0.85 : 1})`}
          style={{ transition: 'transform 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94)' }}
        >
          <path
            d="M 0 0 L 0 12 L 3 9 L 5.5 14.5 L 7 13.8 L 4.5 8.3 L 8.5 8.3 Z"
            fill="#E63B2E"
            stroke="#111"
            strokeWidth="0.5"
          />
        </g>
      </svg>
    </div>
  )
}

/* ─── Main Features Section ─────────────────────────────────────────────── */
export default function Features() {
  return (
    <section id="why-it-works" className="bg-offwhite py-24 md:py-32">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        {/* Section header */}
        <div className="mb-16">
          <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-3">
            Why It Works
          </p>
          <h2 className="font-grotesk font-bold text-ink text-4xl md:text-5xl lg:text-6xl leading-tight">
            Three instruments.<br />One clear signal.
          </h2>
        </div>

        {/* Cards grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          {/* Card 1 — Mechanism Clarity */}
          <div className="rounded-[2rem] bg-paper border border-ink/10 drop-shadow-sm p-8 flex flex-col">
            <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-2">01</p>
            <h3 className="font-grotesk font-bold text-ink text-2xl mb-2">Mechanism Clarity</h3>
            <p className="font-grotesk text-ink/60 text-sm leading-relaxed">
              Complex linkages become readable, interactive motion stories.
            </p>
            <DiagnosticShuffler />
          </div>

          {/* Card 2 — Performance Narratives */}
          <div className="rounded-[2rem] bg-paper border border-ink/10 drop-shadow-sm p-8 flex flex-col">
            <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-2">02</p>
            <h3 className="font-grotesk font-bold text-ink text-2xl mb-2">Performance Narratives</h3>
            <p className="font-grotesk text-ink/60 text-sm leading-relaxed">
              Live insights turned into executive-ready language.
            </p>
            <TelemetryTypewriter />
          </div>

          {/* Card 3 — Sales-Ready Proof */}
          <div className="rounded-[2rem] bg-paper border border-ink/10 drop-shadow-sm p-8 flex flex-col">
            <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-2">03</p>
            <h3 className="font-grotesk font-bold text-ink text-2xl mb-2">Sales-Ready Proof</h3>
            <p className="font-grotesk text-ink/60 text-sm leading-relaxed">
              Model outputs mapped to buyer objections and outcomes.
            </p>
            <CursorScheduler />
          </div>

        </div>
      </div>
    </section>
  )
}
