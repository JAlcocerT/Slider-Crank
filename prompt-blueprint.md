# Prompt Blueprint: Cinematic Landing Page for Computational Mechanics

## Project Intent
Create a premium, cinematic landing page for a sub-brand that bridges computational mechanics, data analytics, and modern sales/marketing. The site should feel like a digital instrument: precise, weighted, and confident, while translating engineering rigor into business value.

## Brand Seed
- Brand name: KineoBridge
- One-line purpose: Bridge computational mechanics and data analytics to turn physical systems into clear, marketable performance stories.

## Audience
- Mechanical engineers and applied scientists
- Product leaders in industrial tech
- Sales and marketing teams selling physical systems
- Founders building hardware + software businesses

## Positioning
- Technical credibility with executive readability
- Mechanisms and models, translated into customer outcomes
- A distinct bridge between lab-grade rigor and market-ready narratives

## Tone and Voice
- Cinematic, precise, and grounded
- Minimal hype; high signal
- Data-forward, not jargon-heavy

## Desired Actions (Primary CTA)
- Book a strategy session

## Aesthetic Direction
- Preset C: Brutalist Signal (Raw Precision)

## Copy Inputs (for builder questions)
1. Brand name and purpose:
   KineoBridge - Bridge computational mechanics and data analytics to turn physical systems into clear, marketable performance stories.
2. Aesthetic direction:
   Preset C (Brutalist Signal)
3. Three key value propositions:
   - Mechanisms made visual and intuitive
   - Data-driven performance narratives for decision makers
   - Sales-ready insights from physical models
4. Primary CTA:
   Book a strategy session

## Hero Copy Guidance
- Emphasize precision, translation, and clarity.
- Suggested hero line pattern (Preset C):
  - First line: "Decode the" (bold sans)
  - Second line: "Machine." (massive serif italic)
- Subhead:
  "We translate computational mechanics into clear, credible stories that win trust, align teams, and accelerate adoption."

## Nav Labels
- Why It Works
- The Instruments
- The Protocol
- Get Started

## Feature Card Details
Card 1: Diagnostic Shuffler
- Title: "Mechanism Clarity"
- Descriptor: "Complex linkages become readable, interactive motion stories."
- Sub-labels: "Pose", "Range", "Constraints"

Card 2: Telemetry Typewriter
- Title: "Performance Narratives"
- Descriptor: "Live insights turned into executive-ready language."
- Feed snippets:
  - "Torque ripple reduced 12% after phase tuning"
  - "Offset increased dwell time at TDC"
  - "Constraint window validated for manufacturing"

Card 3: Cursor Protocol Scheduler
- Title: "Sales-Ready Proof"
- Descriptor: "Model outputs mapped to buyer objections and outcomes."
- Week labels: S M T W T F S
- Save button label: "Publish"

## Philosophy Section Inputs
- Industry: "industrial tech"
- Common approach: "shipping models that are accurate but unreadable to everyone else"
- Differentiated approach: "turning mechanics into conviction with visual, data-backed stories"
- Accent keyword: "conviction"

## Protocol Steps (3)
1. Step 01
   Title: "Instrument the Mechanism"
   Description: "Model the system and surface the parameters that actually move outcomes."
2. Step 02
   Title: "Translate the Signal"
   Description: "Convert engineering outputs into clear performance narratives."
3. Step 03
   Title: "Deploy the Proof"
   Description: "Package insights into buyer-ready assets, demos, and stories."

## Membership / Pricing Guidance
- Replace pricing with a single CTA section.
- Section title: "Start the Bridge"
- CTA button: "Book a strategy session"
- Supporting line: "Bring one mechanism. Leave with a clear path from model to market."

## Footer Details
- Tagline: "Computational mechanics, translated for the real world."
- Status indicator label: "System Operational"

## Imagery / Texture Keywords (Unsplash)
- brutalist concrete, industrial control room, raw materials, precision instruments, mechanical components, grid textures

## Visual Emphasis Notes
- Keep type tight and data-forward.
- Use large scale contrast for hero and manifesto.
- Prioritize kinetic, instrument-like motion in feature cards.

## Constraints
- Avoid overselling. Keep claims grounded.
- No heavy formulas; focus on outcomes and clarity.
- Ensure mobile layout is clean and scannable.

## Technical Requirements (From Original Prompt)
- Stack: React 19, Tailwind CSS v3.4.17, GSAP 3 (with ScrollTrigger), Lucide React.
- Fonts: Load via Google Fonts <link> tags in index.html based on the selected preset.
- Images: Use real Unsplash URLs matching the preset image mood.
- File structure: Single App.jsx with components defined in the same file (or split into components/ if >600 lines). Single index.css for Tailwind directives + noise overlay + custom utilities.
- No placeholders. Every card, label, and animation must be implemented and functional.
- Responsive: Mobile-first. Stack cards vertically on mobile, reduce hero font sizes, collapse navbar into a minimal version.

## Fixed Design System (Never Change)
- Global CSS noise overlay using inline SVG <feTurbulence> at 0.05 opacity.
- Use rounded-[2rem] to rounded-[3rem] radius system for containers.
- Buttons: magnetic hover scale(1.03) with cubic-bezier(0.25, 0.46, 0.45, 0.94).
- Buttons: overflow-hidden with a sliding background span for hover color transitions.
- Links/interactive elements: translateY(-1px) lift on hover.
- GSAP: use gsap.context() within useEffect; return ctx.revert() in cleanup.
- Easing: power3.out for entrances, power2.inOut for morphs.
- Stagger: 0.08 for text, 0.15 for cards/containers.

## Component Architecture (Never Change Structure)
- Navbar: fixed, pill-shaped, centered; morphs on scroll using IntersectionObserver or ScrollTrigger.
- Hero: 100dvh, full-bleed Unsplash image + primary-to-black gradient; bottom-left content; hero line pattern by preset; GSAP stagger fade-up.
- Features: 3 cards with required interaction patterns (Shuffler, Typewriter, Scheduler).
- Philosophy: dark section with parallax texture; contrasting statements; GSAP SplitText-style reveal.
- Protocol: 3 full-screen stacking cards with ScrollTrigger pin; unique canvas/SVG animations per card.
- Membership/Pricing: 3-tier grid (or single CTA if pricing not applicable).
- Footer: rounded-t-[4rem], grid layout, "System Operational" status indicator.

## Build Sequence
1. Map selected preset to full design tokens.
2. Generate hero copy from brand + purpose + hero line pattern.
3. Map value props to the 3 feature card patterns.
4. Generate Philosophy section contrast statements.
5. Generate Protocol steps from brand process.
6. Scaffold project: npm create vite@latest, install deps, write all files.
7. Ensure animations and interactions work and images load.
