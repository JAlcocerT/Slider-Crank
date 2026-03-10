/**
 * Simulator.jsx
 * -------------
 * The KineoBridge slider-crank simulator — pure CSR JavaScript, zero Python.
 *
 * ARCHITECTURE DECISIONS
 * ----------------------
 *
 * WHY CLOSED-FORM OVER fsolve:
 * The Python backend (Python_Symbolic.ipynb) uses scipy.optimize.fsolve on
 * 11 symbolic equations — correct for a general multibody solver. For this
 * canonical 1-DOF slider-crank the loop equations reduce analytically to
 * β = asin((l2·sin(θ₂) − offset) / l3) and s = l2·cos(θ₂) + l3·cos(β).
 * This is O(1) per angle vs O(k·n) Newton iterations. See src/lib/kinematics.js
 * for the full derivation.
 *
 * WHY canvas FOR MECHANISM ANIMATION, recharts FOR PLOTS:
 * - canvas: gives frame-perfect 60fps control via requestAnimationFrame.
 *   Clearing and redrawing the mechanism each frame (crank arm, con-rod,
 *   slider, rail, pivot) takes < 0.1ms at this complexity level. React
 *   re-renders on every animation frame would be prohibitive (~16ms budget
 *   shared with layout/paint).
 * - recharts: ~300KB vs Plotly.js ~3MB. React-native API, no extra globals,
 *   responsive out of the box via ResponsiveContainer. The position/velocity
 *   charts are static relative to animation (only update on param change),
 *   so re-render cost is negligible.
 *
 * WHY useMemo FOR DATA ARRAYS:
 * The 360-point position/velocity/beta arrays are pure functions of l2, l3,
 * and offset. useMemo(fn, [l2, l3, offset]) caches the result and only
 * recomputes when a slider moves — not on every animation frame. Without
 * this, the recharts data arrays would be rebuilt 60x per second, causing
 * GC pressure and chart flicker.
 *
 * WHY cancelAnimationFrame ON UNMOUNT:
 * The rAF loop holds a reference to the component's canvas and state.
 * Without cancellation on unmount, the loop continues running after the
 * component is removed from the DOM, causing: (1) attempted draws to a
 * detached canvas, (2) memory leaks from closures over stale refs.
 */

import { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { sliderCrankKinematics, computeFullCycle } from '../lib/kinematics'

/* ─── Constants ─────────────────────────────────────────────────────────── */
const INK     = '#111111'
const SIGNAL  = '#E63B2E'
const PAPER   = '#E8E4DD'

/* ─── Slider control component ──────────────────────────────────────────── */
function ParamSlider({ label, unit, value, min, max, step, onChange, warning }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-baseline">
        <label className="font-grotesk font-medium text-ink text-sm">{label}</label>
        <span className="font-mono-data text-xs text-ink/60">
          {value.toFixed(1)}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, ${SIGNAL} 0%, ${SIGNAL} ${
            ((value - min) / (max - min)) * 100
          }%, #11111120 ${((value - min) / (max - min)) * 100}%, #11111120 100%)`,
          accentColor: SIGNAL,
        }}
      />
      {warning && (
        <p className="font-mono-data text-xs text-signal">{warning}</p>
      )}
    </div>
  )
}

/* ─── Canvas mechanism renderer ─────────────────────────────────────────── */
function MechanismCanvas({ l2, l3, offset, theta2Rad }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    // Scale: fit the mechanism to canvas with padding
    // Origin (fixed pivot) at (W*0.25, H*0.5 - offset*scale)
    const maxReach = l2 + l3 + 2
    const scale = (W * 0.7) / maxReach
    const originX = W * 0.18
    const originY = H / 2

    ctx.clearRect(0, 0, W, H)

    const kin = sliderCrankKinematics(l2, l3, offset, theta2Rad)
    if (!kin) return

    const { pinX, pinY, sliderX, sliderY } = kin

    // Convert mechanism coords to canvas coords
    // Mechanism: x right, y up. Canvas: x right, y down.
    const cx = (mx) => originX + mx * scale
    const cy = (my) => originY - my * scale  // flip y

    const crankPinCX  = cx(pinX)
    const crankPinCY  = cy(pinY)
    const sliderCX    = cx(sliderX)
    const sliderCY    = cy(sliderY)
    const railY       = cy(sliderY)

    // ── Slider rail (dashed horizontal line) ──
    ctx.save()
    ctx.setLineDash([6, 4])
    ctx.strokeStyle = `${INK}30`
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(originX - 10, railY)
    ctx.lineTo(W - 20, railY)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.restore()

    // ── Connecting rod ──
    ctx.save()
    ctx.strokeStyle = `${INK}80`
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.beginPath()
    ctx.moveTo(crankPinCX, crankPinCY)
    ctx.lineTo(sliderCX, sliderCY)
    ctx.stroke()
    ctx.restore()

    // ── Crank arm ──
    ctx.save()
    ctx.strokeStyle = INK
    ctx.lineWidth = 5
    ctx.lineCap = 'round'
    ctx.beginPath()
    ctx.moveTo(cx(0), cy(0))
    ctx.lineTo(crankPinCX, crankPinCY)
    ctx.stroke()
    ctx.restore()

    // ── Fixed pivot (filled circle at origin) ──
    ctx.save()
    ctx.fillStyle = INK
    ctx.beginPath()
    ctx.arc(cx(0), cy(0), 6, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()

    // ── Crank pin (red circle) ──
    ctx.save()
    ctx.fillStyle = SIGNAL
    ctx.beginPath()
    ctx.arc(crankPinCX, crankPinCY, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()

    // ── Slider body (rounded rectangle) ──
    const sw = 24  // slider width
    const sh = 18  // slider height
    ctx.save()
    ctx.fillStyle = INK
    ctx.beginPath()
    const rx = sliderCX - sw / 2
    const ry = sliderCY - sh / 2
    const r  = 5
    ctx.moveTo(rx + r, ry)
    ctx.lineTo(rx + sw - r, ry)
    ctx.quadraticCurveTo(rx + sw, ry, rx + sw, ry + r)
    ctx.lineTo(rx + sw, ry + sh - r)
    ctx.quadraticCurveTo(rx + sw, ry + sh, rx + sw - r, ry + sh)
    ctx.lineTo(rx + r, ry + sh)
    ctx.quadraticCurveTo(rx, ry + sh, rx, ry + sh - r)
    ctx.lineTo(rx, ry + r)
    ctx.quadraticCurveTo(rx, ry, rx + r, ry)
    ctx.closePath()
    ctx.fill()
    ctx.restore()

    // ── Slider pin (small circle connecting con-rod to slider) ──
    ctx.save()
    ctx.fillStyle = PAPER
    ctx.beginPath()
    ctx.arc(sliderCX, sliderCY, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()

  }, [l2, l3, offset, theta2Rad])

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={300}
      className="w-full rounded-2xl bg-offwhite border border-ink/10"
      style={{ height: 'clamp(180px, 30vw, 300px)' }}
    />
  )
}

/* ─── Custom recharts tooltip ────────────────────────────────────────────── */
function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-paper border border-ink/15 rounded-xl px-3 py-2 shadow-lg">
      <p className="font-mono-data text-xs text-ink/50 mb-1">{label}°</p>
      <p className="font-mono-data text-sm text-ink font-semibold">
        {payload[0].value.toFixed(4)} {unit}
      </p>
    </div>
  )
}

/* ─── Stats row ─────────────────────────────────────────────────────────── */
function StatsRow({ positions, velocities, betas, angles }) {
  if (!positions.length) return null

  const sMax = Math.max(...positions)
  const sMin = Math.min(...positions)
  const range = (sMax - sMin).toFixed(4)

  const intakeIdx = positions.indexOf(sMin)
  const intakeAngle = angles[intakeIdx]?.toFixed(1) ?? '—'
  const compressionAngle = intakeIdx != null
    ? (360 - parseFloat(intakeAngle)).toFixed(1)
    : '—'

  const betaMax = Math.max(...betas)
  const betaMin = Math.min(...betas)
  const betaRange = (betaMax - betaMin).toFixed(3)

  const stats = [
    { label: 'Range of Motion',       value: range,            unit: 'L'  },
    { label: 'Min Position Angle',    value: `${intakeAngle}`, unit: '°'  },
    { label: 'Complement Angle',      value: compressionAngle, unit: '°'  },
    { label: 'Con-rod Angle Range',   value: betaRange,        unit: '°'  },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
      {stats.map(({ label, value, unit }) => (
        <div key={label} className="bg-paper rounded-2xl border border-ink/10 p-4">
          <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-2 leading-tight">
            {label}
          </p>
          <p className="font-mono-data text-lg text-ink font-semibold">
            {value}<span className="text-ink/40 text-sm ml-1">{unit}</span>
          </p>
        </div>
      ))}
    </div>
  )
}

/* ─── Main Simulator component ──────────────────────────────────────────── */
export default function Simulator() {
  const [l2,     setL2]     = useState(5)
  const [l3,     setL3]     = useState(7)
  const [offset, setOffset] = useState(1)
  const [rpm,    setRpm]    = useState(30)
  const [theta2, setTheta2] = useState(0)

  const animRef   = useRef(null)
  const lastTsRef = useRef(null)

  // Validate constraints: offset < l2  (enforced in Python Dash app as well)
  const offsetWarning = offset >= l2
    ? `Offset must be less than crank length (l2 = ${l2.toFixed(1)})`
    : null

  // l3 must be long enough to reach: l3 > l2 (simplification; full check is l3 > |l2 ± offset|)
  const l3Warning = l3 <= l2
    ? `Con-rod should be longer than crank (l3 > ${l2.toFixed(1)}) for full rotation`
    : null

  const geometryValid = !offsetWarning

  // ── Animation loop ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!geometryValid) {
      cancelAnimationFrame(animRef.current)
      return
    }

    function loop(ts) {
      if (lastTsRef.current !== null) {
        const dt = (ts - lastTsRef.current) / 1000  // seconds
        const dTheta = (2 * Math.PI * rpm) / 60 * dt
        setTheta2((prev) => (prev + dTheta) % (2 * Math.PI))
      }
      lastTsRef.current = ts
      animRef.current = requestAnimationFrame(loop)
    }

    lastTsRef.current = null
    animRef.current = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(animRef.current)
    }
  }, [rpm, geometryValid])

  // ── Precomputed chart data (memoized — only recompute on param change) ──
  const cycleData = useMemo(() => {
    if (!geometryValid) return { angles: [], positions: [], velocities: [], betas: [], chartData: [] }
    const { angles, positions, velocities, betas } = computeFullCycle(l2, l3, offset)
    const chartData = angles.map((angle, i) => ({
      angle,
      position: positions[i],
      velocity: velocities[i],
    }))
    return { angles, positions, velocities, betas, chartData }
  }, [l2, l3, offset, geometryValid])

  // ── Handle offset exceeding l2: clamp it ────────────────────────────────
  const handleL2Change = useCallback((val) => {
    setL2(val)
    if (offset >= val) setOffset(parseFloat((val - 0.1).toFixed(1)))
  }, [offset])

  const handleOffsetChange = useCallback((val) => {
    if (val >= l2) return  // silently block — warning is shown
    setOffset(val)
  }, [l2])

  return (
    <section id="simulator" className="bg-offwhite py-24 md:py-32">
      <div className="max-w-7xl mx-auto px-6 md:px-12">

        {/* Header */}
        <div className="mb-12">
          <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-3">
            Live Simulation
          </p>
          <h2 className="font-grotesk font-bold text-ink text-4xl md:text-5xl lg:text-6xl leading-tight">
            The Instrument
          </h2>
          <p className="font-grotesk text-ink/60 text-base md:text-lg mt-3">
            Adjust parameters. Watch the machine respond.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-10">

          {/* Left column: canvas + charts */}
          <div className="flex flex-col gap-8">

            {/* Canvas */}
            {geometryValid ? (
              <MechanismCanvas
                l2={l2}
                l3={l3}
                offset={offset}
                theta2Rad={theta2}
              />
            ) : (
              <div className="w-full rounded-2xl bg-offwhite border border-signal/40 flex items-center justify-center text-signal font-mono-data text-sm p-8"
                   style={{ minHeight: 200 }}>
                Invalid geometry — adjust parameters
              </div>
            )}

            {/* Charts */}
            {geometryValid && cycleData.chartData.length > 0 && (
              <div className="flex flex-col gap-8">

                {/* Position chart */}
                <div>
                  <p className="font-mono-data text-xs text-ink/50 uppercase tracking-widest mb-3">
                    Piston Position (s) vs Crank Angle
                  </p>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={cycleData.chartData} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#11111115" />
                      <XAxis
                        dataKey="angle"
                        tick={{ fontFamily: 'Space Mono', fontSize: 10, fill: '#11111160' }}
                        tickFormatter={(v) => `${v}°`}
                        ticks={[0, 60, 120, 180, 240, 300, 360]}
                      />
                      <YAxis
                        tick={{ fontFamily: 'Space Mono', fontSize: 10, fill: '#11111160' }}
                        tickFormatter={(v) => v.toFixed(1)}
                        width={40}
                      />
                      <Tooltip content={<ChartTooltip unit="L" />} />
                      <Line
                        type="monotone"
                        dataKey="position"
                        stroke={SIGNAL}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Velocity chart */}
                <div>
                  <p className="font-mono-data text-xs text-ink/50 uppercase tracking-widest mb-3">
                    Velocity (ds/dθ) vs Crank Angle
                  </p>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={cycleData.chartData} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#11111115" />
                      <XAxis
                        dataKey="angle"
                        tick={{ fontFamily: 'Space Mono', fontSize: 10, fill: '#11111160' }}
                        tickFormatter={(v) => `${v}°`}
                        ticks={[0, 60, 120, 180, 240, 300, 360]}
                      />
                      <YAxis
                        tick={{ fontFamily: 'Space Mono', fontSize: 10, fill: '#11111160' }}
                        tickFormatter={(v) => v.toFixed(1)}
                        width={40}
                      />
                      <ReferenceLine y={0} stroke={`${INK}30`} strokeDasharray="4 4" />
                      <Tooltip content={<ChartTooltip unit="L/rad" />} />
                      <Line
                        type="monotone"
                        dataKey="velocity"
                        stroke={`${INK}99`}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

              </div>
            )}

            {/* Stats row */}
            {geometryValid && (
              <StatsRow
                positions={cycleData.positions}
                velocities={cycleData.velocities}
                betas={cycleData.betas}
                angles={cycleData.angles}
              />
            )}
          </div>

          {/* Right column: parameter sliders */}
          <div className="bg-paper rounded-[2rem] border border-ink/10 p-8 flex flex-col gap-6 self-start">
            <p className="font-grotesk font-semibold text-ink text-lg">Parameters</p>

            <ParamSlider
              label="Crank Length"
              unit=" L"
              value={l2}
              min={1}
              max={10}
              step={0.1}
              onChange={handleL2Change}
            />

            <ParamSlider
              label="Connecting Rod"
              unit=" L"
              value={l3}
              min={1}
              max={20}
              step={0.1}
              onChange={setL3}
              warning={l3Warning}
            />

            <ParamSlider
              label="Piston Offset"
              unit=" L"
              value={offset}
              min={0}
              max={Math.max(0.1, l2 - 0.1)}
              step={0.1}
              onChange={handleOffsetChange}
              warning={offsetWarning}
            />

            <div className="border-t border-ink/10 pt-4">
              <ParamSlider
                label="Animation Speed"
                unit=" RPM"
                value={rpm}
                min={10}
                max={120}
                step={5}
                onChange={setRpm}
              />
            </div>

            {/* Current angle readout */}
            <div className="rounded-xl bg-offwhite border border-ink/10 p-4">
              <p className="font-mono-data text-xs text-ink/40 uppercase tracking-widest mb-1">
                Crank Angle
              </p>
              <p className="font-mono-data text-2xl text-ink">
                {((theta2 * 180) / Math.PI).toFixed(1)}<span className="text-ink/40 text-sm ml-1">°</span>
              </p>
            </div>

            {/* Constraint note */}
            <p className="font-mono-data text-xs text-ink/30 leading-relaxed">
              Constraint: offset &lt; l2<br />
              (matches Python Dash app)
            </p>
          </div>

        </div>
      </div>
    </section>
  )
}
