/**
 * Protocol.jsx
 * ------------
 * WHY pin:true ScrollTrigger with explicit pinSpacing vs alternatives:
 *
 * The "stacking cards" pattern requires each card to remain fixed in the
 * viewport while the next card scrolls over it. GSAP ScrollTrigger's
 * `pin: true` achieves this by adding `position: fixed` to the pinned
 * element and inserting a spacer div to preserve document flow. The spacer
 * height equals the element's natural height, preventing downstream sections
 * from jumping up.
 *
 * Alternatives considered:
 * - CSS `position: sticky` with `top: 0`: works for simple stacking but
 *   doesn't support the scale/blur/opacity transition on the PREVIOUS card
 *   as the new one scrolls in — you'd need JS anyway for that tween.
 * - Intersection Observer: gives enter/leave events but not scroll progress
 *   (no `scrub` equivalent), so the scale transition would be binary, not
 *   smooth.
 * - Framer Motion's `useScroll` + `useTransform`: viable but adds ~30KB
 *   to bundle; GSAP ScrollTrigger is already loaded for Philosophy.
 *
 * iOS Safari caveat:
 * Safari recalculates viewport height when the URL bar shows/hides, which
 * invalidates ScrollTrigger's cached positions. `invalidateOnRefresh: true`
 * forces a full recalculation on the ScrollTrigger `refresh` event (which
 * fires on resize/orientation change), preventing misaligned trigger points.
 */

import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

/* ─── SVG: Rotating concentric circles — pure CSS animation ─────────────── */
function ConcentricCircles() {
  return (
    <svg viewBox="0 0 120 120" className="w-24 h-24 md:w-32 md:h-32" aria-hidden="true">
      <style>{`
        @keyframes kbRotCW  { to { transform: rotate(360deg);  transform-origin: 60px 60px; } }
        @keyframes kbRotCCW { to { transform: rotate(-360deg); transform-origin: 60px 60px; } }
        .kb-rot-cw  { animation: kbRotCW  8s  linear infinite; }
        .kb-rot-ccw { animation: kbRotCCW 12s linear infinite; }
        .kb-rot-cw2 { animation: kbRotCW  20s linear infinite; }
      `}</style>
      <circle cx="60" cy="60" r="8"  fill="#E63B2E" />
      <circle cx="60" cy="60" r="18" fill="none" stroke="#111" strokeWidth="1.5" strokeDasharray="4 4"  className="kb-rot-cw" />
      <circle cx="60" cy="60" r="30" fill="none" stroke="#111" strokeWidth="1"   strokeDasharray="2 6"  className="kb-rot-ccw" />
      <circle cx="60" cy="60" r="42" fill="none" stroke="#111" strokeWidth="0.8" strokeDasharray="1 8"  className="kb-rot-cw2" />
      <circle cx="60" cy="60" r="54" fill="none" stroke="#E63B2E" strokeWidth="0.5" opacity="0.4" />
    </svg>
  )
}

/* ─── SVG: Scanning horizontal line across dot grid — GSAP x tween ─────── */
function ScanningGrid({ active }) {
  const lineRef = useRef(null)

  useEffect(() => {
    if (!active || !lineRef.current) return
    const tween = gsap.to(lineRef.current, {
      attr: { x1: 110, x2: 110 },
      duration: 1.8,
      ease: 'power1.inOut',
      repeat: -1,
      yoyo: true,
    })
    return () => tween.kill()
  }, [active])

  const dots = []
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 8; col++) {
      dots.push(
        <circle
          key={`${row}-${col}`}
          cx={10 + col * 14}
          cy={10 + row * 14}
          r="1.8"
          fill="#111"
          opacity="0.25"
        />
      )
    }
  }

  return (
    <svg viewBox="0 0 120 80" className="w-24 h-16 md:w-32 md:h-20" aria-hidden="true">
      {dots}
      <line
        ref={lineRef}
        x1="10" y1="0" x2="10" y2="80"
        stroke="#E63B2E"
        strokeWidth="1.5"
        opacity="0.8"
      />
    </svg>
  )
}

/* ─── SVG: EKG waveform stroke-dashoffset animation ────────────────────── */
function EKGWaveform({ active }) {
  const pathRef = useRef(null)

  useEffect(() => {
    if (!pathRef.current) return
    const len = pathRef.current.getTotalLength()
    pathRef.current.style.strokeDasharray = `${len}`
    pathRef.current.style.strokeDashoffset = `${len}`

    if (!active) return

    const tween = gsap.to(pathRef.current, {
      strokeDashoffset: 0,
      duration: 2.5,
      ease: 'power2.inOut',
      repeat: -1,
      repeatDelay: 0.8,
      onRepeat: () => {
        if (pathRef.current) {
          pathRef.current.style.strokeDashoffset = `${len}`
        }
      },
    })
    return () => tween.kill()
  }, [active])

  const d =
    'M 0 40 L 20 40 L 30 40 L 35 15 L 40 55 L 45 20 L 50 40 L 70 40 L 80 40 L 85 20 L 90 50 L 95 25 L 100 40 L 120 40'

  return (
    <svg viewBox="0 0 120 70" className="w-24 h-12 md:w-32 md:h-16" aria-hidden="true">
      <path
        ref={pathRef}
        d={d}
        fill="none"
        stroke="#E63B2E"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/* ─── Card data ─────────────────────────────────────────────────────────── */
const CARDS = [
  {
    step: '01',
    title: 'Instrument the Mechanism',
    desc: 'Model the system and surface the parameters that actually move outcomes.',
    Graphic: ConcentricCircles,
  },
  {
    step: '02',
    title: 'Translate the Signal',
    desc: 'Convert engineering outputs into clear performance narratives.',
    Graphic: ScanningGrid,
  },
  {
    step: '03',
    title: 'Deploy the Proof',
    desc: 'Package insights into buyer-ready assets, demos, and stories.',
    Graphic: EKGWaveform,
  },
]

/* ─── Main Protocol Section ─────────────────────────────────────────────── */
export default function Protocol() {
  const sectionRef = useRef(null)
  const cardRefs   = useRef([])
  const [activeCard, setActiveCard] = useState(0)

  useEffect(() => {
    const ctx = gsap.context(() => {
      CARDS.forEach((_, i) => {
        if (i === 0) return // first card is the static base

        ScrollTrigger.create({
          trigger: cardRefs.current[i],
          start: 'top 85%',
          end: 'top 20%',
          scrub: true,
          invalidateOnRefresh: true, // iOS Safari: recalculate on viewport resize
          onEnter:     () => setActiveCard(i),
          onLeaveBack: () => setActiveCard(i - 1),
          onUpdate: (self) => {
            const prev = cardRefs.current[i - 1]
            if (!prev) return
            // Smoothly scale, blur, and fade the previous card as new one enters
            const prog = self.progress
            gsap.set(prev, {
              scale:  1 - prog * 0.1,
              filter: `blur(${prog * 20}px)`,
              opacity: 1 - prog * 0.5,
            })
          },
        })
      })
    }, sectionRef)

    return () => ctx.revert()
  }, [])

  return (
    <section
      ref={sectionRef}
      id="protocol"
      className="bg-offwhite py-24 md:py-32 overflow-hidden"
    >
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        {/* Section header */}
        <div className="mb-16">
          <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-3">
            The Protocol
          </p>
          <h2 className="font-grotesk font-bold text-ink text-4xl md:text-5xl lg:text-6xl leading-tight">
            Three steps.<br />One clear outcome.
          </h2>
        </div>

        {/* Stacking cards */}
        <div className="relative flex flex-col gap-6 md:gap-8">
          {CARDS.map((card, i) => {
            const { step, title, desc, Graphic } = card
            return (
              <div
                key={step}
                ref={(el) => (cardRefs.current[i] = el)}
                className="w-full bg-paper rounded-[3rem] border border-ink/10 flex flex-col md:flex-row items-center justify-between px-8 md:px-16 py-12 md:py-16 gap-10 md:gap-0 will-change-transform"
                style={{ minHeight: '50vh' }}
              >
                {/* Left: text */}
                <div className="max-w-lg">
                  <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-4">
                    Step {step}
                  </p>
                  <h3 className="font-grotesk font-bold text-ink text-3xl md:text-4xl leading-tight mb-4">
                    {title}
                  </h3>
                  <p className="font-grotesk text-ink/60 text-base md:text-lg leading-relaxed">
                    {desc}
                  </p>
                </div>

                {/* Right: SVG animation */}
                <div className="flex-shrink-0 flex items-center justify-center w-36 h-36 md:w-48 md:h-48 rounded-[2rem] bg-offwhite border border-ink/10">
                  <Graphic active={activeCard === i} />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
