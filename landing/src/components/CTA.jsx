/**
 * CTA.jsx
 * -------
 * Single-CTA conversion section — no pricing tiers needed for a consulting brand.
 *
 * The consulting model works on trust and specificity, not comparison shopping.
 * A single, bold call-to-action removes the cognitive load of choosing between
 * plans or tiers, and communicates confidence: we know what we offer and who
 * it's for. The copy "Bring one mechanism. Leave with a clear path from model
 * to market." makes the value exchange concrete without over-promising.
 *
 * The bg-ink background creates a strong visual break after the Simulator
 * section (bg-offwhite), signalling a transition from "proof of capability"
 * to "take action" — a standard conversion funnel structure.
 */

export default function CTA() {
  return (
    <section id="get-started" className="bg-ink py-32 md:py-48">
      <div className="max-w-5xl mx-auto px-6 md:px-12 text-center flex flex-col items-center gap-10">

        {/* Label */}
        <p className="font-mono-data text-xs text-paper/30 uppercase tracking-widest">
          Ready to bridge the gap?
        </p>

        {/* Main headline */}
        <h2 className="font-grotesk font-bold text-paper text-5xl md:text-7xl lg:text-8xl leading-[0.95] tracking-tight">
          Start the Bridge
        </h2>

        {/* Body copy */}
        <p className="font-grotesk text-paper/60 text-lg md:text-xl max-w-lg leading-relaxed">
          Bring one mechanism. Leave with a clear path from model to market.
        </p>

        {/* CTA button with magnetic hover */}
        <a
          href="mailto:hello@kineobridge.com?subject=Strategy Session Request"
          className="btn-magnetic bg-signal text-offwhite font-grotesk font-semibold text-lg md:text-xl px-10 py-5 rounded-full"
        >
          <span className="relative z-10">Book a strategy session</span>
          <span className="btn-bg bg-paper rounded-full" />
        </a>

        {/* Trust signal */}
        <p className="font-mono-data text-xs text-paper/20 mt-4">
          No commitment required — just a conversation.
        </p>

      </div>
    </section>
  )
}
