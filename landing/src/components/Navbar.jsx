/**
 * Navbar.jsx
 * ----------
 * WHY fixed pill navbar with IntersectionObserver:
 *
 * A fixed pill navbar keeps the brand and primary CTA accessible at all times
 * without fighting scroll — critical for a consulting landing page where the
 * "Book" button must never be more than one glance away.
 *
 * We use IntersectionObserver (not a scroll event listener) to detect when
 * the user has scrolled past the hero section. IntersectionObserver fires
 * asynchronously via the browser's intersection engine — it does NOT block
 * the main thread on every scroll tick. A naive `window.addEventListener
 * ('scroll', ...)` would require a debounce/throttle and still runs in the
 * main-thread JS microtask queue, causing jank on slower devices.
 *
 * We deliberately avoid using GSAP's ScrollTrigger for this UI-only concern
 * because: (1) GSAP ScrollTrigger is already loaded for the Protocol section —
 * adding it here creates a temporal coupling where the navbar breaks if the
 * Protocol section is removed; (2) IntersectionObserver is a browser primitive
 * with zero JS payload; (3) the only state we need is a boolean (past/not-past),
 * which maps perfectly to a React useState toggle.
 */

import { useState, useEffect, useRef } from 'react'
import { Menu, X } from 'lucide-react'

const NAV_LINKS = [
  { label: 'Why It Works',    href: '#why-it-works' },
  { label: 'The Instruments', href: '#why-it-works' },
  { label: 'The Protocol',    href: '#protocol' },
  { label: 'Get Started',     href: '#get-started' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const heroRef = useRef(null)

  useEffect(() => {
    // Observe the hero section sentinel — when it leaves the viewport we
    // know the user has scrolled past the fold.
    const heroEl = document.getElementById('hero-sentinel')
    if (!heroEl) return

    const observer = new IntersectionObserver(
      ([entry]) => setScrolled(!entry.isIntersecting),
      { threshold: 0 }
    )
    observer.observe(heroEl)
    return () => observer.disconnect()
  }, [])

  const navBase =
    'fixed top-4 left-1/2 -translate-x-1/2 z-50 transition-all duration-500 rounded-full px-6 py-3 flex items-center gap-8'
  const navScrolled =
    'bg-paper/80 backdrop-blur-xl border border-ink/10 shadow-lg'
  const navTransparent = 'bg-transparent'

  return (
    <nav className={`${navBase} ${scrolled ? navScrolled : navTransparent}`}>
      {/* Logo */}
      <a
        href="#"
        className={`font-grotesk font-bold text-lg tracking-tight select-none ${
          scrolled ? 'text-ink' : 'text-paper'
        }`}
      >
        KineoBridge
      </a>

      {/* Desktop links */}
      <ul className="hidden md:flex items-center gap-6">
        {NAV_LINKS.map(({ label, href }) => (
          <li key={label}>
            <a
              href={href}
              className={`font-grotesk text-sm font-medium transition-colors hover:text-signal ${
                scrolled ? 'text-ink/70' : 'text-paper/80'
              }`}
            >
              {label}
            </a>
          </li>
        ))}
      </ul>

      {/* Desktop CTA */}
      <a
        href="#get-started"
        className="hidden md:block btn-magnetic bg-signal text-offwhite font-grotesk font-semibold text-sm px-5 py-2 rounded-full"
      >
        <span className="relative z-10">Book a Session</span>
        <span className="btn-bg bg-ink rounded-full" />
      </a>

      {/* Mobile hamburger */}
      <button
        className={`md:hidden ml-auto ${scrolled ? 'text-ink' : 'text-paper'}`}
        onClick={() => setMenuOpen((v) => !v)}
        aria-label="Toggle menu"
      >
        {menuOpen ? <X size={22} /> : <Menu size={22} />}
      </button>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div className="absolute top-full left-0 right-0 mt-2 mx-2 bg-paper/95 backdrop-blur-xl border border-ink/10 rounded-2xl p-4 shadow-xl md:hidden">
          <ul className="flex flex-col gap-3">
            {NAV_LINKS.map(({ label, href }) => (
              <li key={label}>
                <a
                  href={href}
                  onClick={() => setMenuOpen(false)}
                  className="font-grotesk text-sm font-medium text-ink/80 hover:text-signal block py-1"
                >
                  {label}
                </a>
              </li>
            ))}
            <li>
              <a
                href="#get-started"
                onClick={() => setMenuOpen(false)}
                className="block bg-signal text-offwhite font-grotesk font-semibold text-sm px-4 py-2 rounded-full text-center mt-1"
              >
                Book a Session
              </a>
            </li>
          </ul>
        </div>
      )}
    </nav>
  )
}
