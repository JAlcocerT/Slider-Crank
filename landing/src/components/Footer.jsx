/**
 * Footer.jsx
 * ----------
 * WHY rounded-t-[4rem] on footer:
 *
 * The large top-radius creates a "platform" or "stage" effect — as if the
 * footer is a physical object the content rests on top of. This is a
 * deliberate choice in the Brutalist Signal design system to add tactile
 * dimensionality to what is otherwise a flat digital surface. It also
 * visually distinguishes the footer from the preceding CTA section (also
 * bg-ink) by introducing a geometric break, preventing the two sections
 * from bleeding together. The 4rem radius is consistent with the
 * rounded-[3rem] used on Protocol cards and the rounded-[2rem] on Feature
 * cards — a harmonic radius scale that creates system cohesion.
 */

const NAV_LINKS = [
  { label: 'Why It Works',    href: '#why-it-works' },
  { label: 'The Protocol',    href: '#protocol' },
  { label: 'The Instrument',  href: '#simulator' },
  { label: 'Get Started',     href: '#get-started' },
]

export default function Footer() {
  const year = new Date().getFullYear()

  return (
    <footer className="bg-ink text-paper rounded-t-[4rem] px-8 md:px-16 pt-16 pb-10">
      <div className="max-w-7xl mx-auto">

        {/* Main grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-8 mb-16">

          {/* Left col: brand */}
          <div className="flex flex-col gap-4">
            <p className="font-grotesk font-bold text-paper text-2xl">KineoBridge</p>
            <p className="font-grotesk text-paper/50 text-sm leading-relaxed max-w-xs">
              Computational mechanics, translated for the real world.
            </p>
          </div>

          {/* Center col: nav links */}
          <nav aria-label="Footer navigation">
            <p className="font-mono-data text-xs text-paper/30 uppercase tracking-widest mb-5">
              Navigation
            </p>
            <ul className="flex flex-col gap-3">
              {NAV_LINKS.map(({ label, href }) => (
                <li key={label}>
                  <a
                    href={href}
                    className="font-grotesk text-sm text-paper/60 hover:text-paper transition-colors"
                  >
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          {/* Right col: legal + system status */}
          <div className="flex flex-col gap-4">
            <p className="font-mono-data text-xs text-paper/30 uppercase tracking-widest mb-1">
              System Status
            </p>

            {/* Pulsing green dot + operational label */}
            <div className="flex items-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
              </span>
              <span className="font-mono-data text-xs text-paper/60">System Operational</span>
            </div>

            <div className="mt-2 flex flex-col gap-1.5">
              <p className="font-mono-data text-xs text-paper/30">
                Simulator Engine: Online
              </p>
              <p className="font-mono-data text-xs text-paper/30">
                Kinematics Core: v0.1.0
              </p>
            </div>
          </div>

        </div>

        {/* Bottom bar */}
        <div className="border-t border-paper/10 pt-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <p className="font-mono-data text-xs text-paper/25">
            © {year} KineoBridge. All rights reserved.
          </p>
          <div className="flex gap-6">
            <a href="#" className="font-mono-data text-xs text-paper/25 hover:text-paper/50 transition-colors">
              Privacy
            </a>
            <a href="#" className="font-mono-data text-xs text-paper/25 hover:text-paper/50 transition-colors">
              Terms
            </a>
            <a href="mailto:hello@kineobridge.com" className="font-mono-data text-xs text-paper/25 hover:text-paper/50 transition-colors">
              Contact
            </a>
          </div>
        </div>

      </div>
    </footer>
  )
}
