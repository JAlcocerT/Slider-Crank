/**
 * Philosophy.jsx
 * --------------
 * WHY word-span splitting without GSAP SplitText plugin:
 *
 * GSAP SplitText is a Club GreenSock (paid membership) plugin. Using it in
 * an open project or any project without a valid license violates the GSAP
 * licensing terms. Manual span-wrapping achieves the identical visual effect:
 * each word is wrapped in a <span> and GSAP targets the NodeList with a
 * stagger. The only trade-off is that non-breaking hyphenation across word
 * boundaries isn't handled — acceptable here since the statements are fixed
 * copy, not dynamic user content.
 *
 * The GSAP ScrollTrigger here uses `start: "top 80%"` so the reveal begins
 * when the section is 80% into the viewport — a comfortable reading trigger
 * that doesn't fire too early on large monitors.
 */

import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const CONTRAST_A =
  'Most industrial tech focuses on: shipping models that are accurate but unreadable to everyone else.'

const CONTRAST_B_BEFORE = 'We focus on: turning mechanics into'
const CONTRAST_B_ACCENT = 'conviction'
const CONTRAST_B_AFTER  = 'with visual, data-backed stories.'

/** Wrap each word in a span for GSAP stagger targeting */
function WordSpans({ text, className }) {
  const words = text.split(' ')
  return (
    <>
      {words.map((word, i) => (
        <span
          key={i}
          className={`word-span inline-block ${className ?? ''}`}
          style={{ willChange: 'transform, opacity' }}
        >
          {word}{i < words.length - 1 ? '\u00A0' : ''}
        </span>
      ))}
    </>
  )
}

export default function Philosophy() {
  const sectionRef = useRef(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // All .word-span elements inside the section start hidden
      gsap.set('.word-span', { opacity: 0, y: 20 })

      // ScrollTrigger: reveal words as section scrolls into view
      gsap.to('.word-span', {
        opacity: 1,
        y: 0,
        duration: 0.6,
        stagger: 0.04,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 80%',
          toggleActions: 'play none none none',
        },
      })
    }, sectionRef)

    return () => ctx.revert()
  }, [])

  return (
    <section
      ref={sectionRef}
      id="philosophy"
      className="relative bg-ink text-paper py-32 md:py-48 overflow-hidden"
    >
      {/* Low-opacity parallax texture — CSS background-attachment: fixed
          creates the parallax effect without GSAP for a purely decorative
          element. Safari ignores background-attachment:fixed on mobile but
          it gracefully degrades to a static texture. */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "url('https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&q=60')",
          backgroundAttachment: 'fixed',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          opacity: 0.08,
        }}
      />

      <div className="relative max-w-5xl mx-auto px-6 md:px-12">
        {/* Contrast statement A — smaller, muted */}
        <p className="font-grotesk font-medium text-paper/40 text-lg md:text-xl leading-relaxed mb-16 md:mb-24 max-w-2xl">
          <WordSpans text={CONTRAST_A} />
        </p>

        {/* Contrast statement B — massive */}
        <p className="font-grotesk font-bold text-paper text-2xl md:text-4xl lg:text-5xl leading-snug">
          <WordSpans text={CONTRAST_B_BEFORE} />
          {' '}
          <span
            className="word-span font-serif-drama italic text-signal inline-block"
            style={{
              fontSize: 'clamp(3.5rem, 10vw, 8rem)',
              lineHeight: 1,
              willChange: 'transform, opacity',
            }}
          >
            {CONTRAST_B_ACCENT}
          </span>{' '}
          <WordSpans text={CONTRAST_B_AFTER} />
        </p>
      </div>
    </section>
  )
}
