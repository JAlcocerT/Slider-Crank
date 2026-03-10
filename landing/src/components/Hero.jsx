/**
 * Hero.jsx
 * --------
 * WHY 100dvh not 100vh:
 * On iOS Safari, `100vh` is calculated using the full viewport height
 * including the retractable URL bar. When the page first loads with the
 * URL bar visible, 100vh overflows the screen — the bottom content is
 * clipped or requires scrolling before the hero fills the view.
 * `100dvh` (dynamic viewport height) is recalculated after the URL bar
 * retracts, so it always fills the visible area exactly. All modern
 * browsers (Safari 15.4+, Chrome 108+, Firefox 101+) support dvh units.
 * We keep a `min-h-screen` fallback class in addition for older browsers.
 */

import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'

const HERO_IMAGE =
  'https://images.unsplash.com/photo-1486325212027-8081e485255e?w=1600&q=80'

export default function Hero() {
  const containerRef = useRef(null)

  useEffect(() => {
    // GSAP stagger fade-up on mount — uses gsap.context() for safe cleanup
    // in React StrictMode (double-invocation in dev). ctx.revert() kills all
    // tweens and reverts inline styles when the component unmounts.
    const ctx = gsap.context(() => {
      gsap.fromTo(
        '.hero-animate',
        { y: 40, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.9,
          stagger: 0.15,
          ease: 'power3.out',
          delay: 0.2,
        }
      )
    }, containerRef)

    return () => ctx.revert()
  }, [])

  return (
    <section
      id="hero"
      ref={containerRef}
      className="relative min-h-screen overflow-hidden"
      style={{ height: '100dvh' }}
    >
      {/* Hero sentinel: observed by Navbar's IntersectionObserver to
          detect when user scrolls past the fold */}
      <div id="hero-sentinel" className="absolute top-[90%] left-0 w-full h-px pointer-events-none" />

      {/* Background image */}
      <div
        className="absolute inset-0 bg-center bg-cover"
        style={{ backgroundImage: `url(${HERO_IMAGE})` }}
      />

      {/* Heavy gradient overlay — Preset C requires ink-dominant overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/70 to-transparent" />

      {/* Content — bottom-left anchored */}
      <div className="absolute bottom-0 left-0 p-12 md:p-20 max-w-4xl">
        {/* Line 1: editorial label */}
        <p className="hero-animate font-grotesk font-medium text-paper/60 text-sm md:text-base tracking-widest uppercase mb-3">
          Computational Mechanics — Translated
        </p>

        {/* Line 2: "Decode the" — Space Grotesk bold */}
        <h1 className="hero-animate font-grotesk font-bold text-paper leading-none text-5xl md:text-7xl lg:text-8xl">
          Decode the
        </h1>

        {/* Line 3: "Machine." — DM Serif Display italic, signal accent */}
        <h1 className="hero-animate font-serif-drama italic text-signal leading-none text-7xl md:text-[9rem] lg:text-[12rem]">
          Machine.
        </h1>

        {/* Subhead */}
        <p className="hero-animate font-grotesk text-paper/75 text-base md:text-xl max-w-2xl mt-6 leading-relaxed">
          We translate computational mechanics into clear, credible stories
          that win trust, align teams, and accelerate adoption.
        </p>

        {/* CTA */}
        <div className="hero-animate mt-8">
          <a
            href="#get-started"
            className="btn-magnetic inline-block bg-signal text-offwhite font-grotesk font-semibold px-8 py-4 rounded-full text-base md:text-lg"
          >
            <span className="relative z-10">Book a strategy session</span>
            <span className="btn-bg bg-ink rounded-full" />
          </a>
        </div>
      </div>
    </section>
  )
}
