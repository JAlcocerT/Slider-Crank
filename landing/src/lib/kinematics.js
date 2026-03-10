/**
 * src/lib/kinematics.js
 * ---------------------
 * Closed-form slider-crank kinematics for the KineoBridge landing simulator.
 *
 * DERIVATION SOURCE
 * -----------------
 * The Python backend (Python_Symbolic.ipynb) sets up 11 symbolic constraint
 * equations for the full open-loop chain (including the fixed-pivot frame),
 * then calls scipy.optimize.fsolve to numerically solve the system at each
 * crank angle. That approach is correct for a general multibody solver where
 * you may add more links, cams, or mixed kinematic pairs at runtime.
 *
 * WHY CLOSED-FORM HERE INSTEAD OF fsolve
 * ----------------------------------------
 * For the canonical 1-DOF slider-crank (3 moving links, 4 joints), the
 * kinematic loop equation reduces analytically to two scalar constraints:
 *
 *   Loop closure (x):  l2*cos(θ₂) + l3*cos(β) = s
 *   Loop closure (y):  l2*sin(θ₂) - l3*sin(β) = offset
 *
 * Solving the y-equation directly gives β = asin((l2·sin(θ₂) − offset) / l3).
 * Substituting back into the x-equation gives s immediately. No iteration.
 *
 * This is O(1) per crank angle vs O(k·n) for fsolve (k Newton iterations,
 * n equations). At 60 fps with a 360-point precomputation pass per param
 * change, the closed-form runs in < 0.1 ms. fsolve for 360 points would be
 * ~50–200 ms in Python (acceptable for Dash callbacks) but would block the
 * JS main thread unacceptably.
 *
 * VELOCITY (first-order kinematics)
 * ----------------------------------
 * Differentiating s with respect to θ₂ using the chain rule:
 *
 *   ds/dθ₂ = −l2·sin(θ₂) − l3·sin(β)·(dβ/dθ₂)
 *
 * where  dβ/dθ₂ = (l2·cos(θ₂)) / √(l3² − (l2·sin(θ₂) − offset)²)
 *
 * This matches the velocity expression in the Python symbolic solution.
 *
 * CONSTRAINT ENFORCEMENT
 * -----------------------
 * The mechanism is geometrically valid only when |sinβ| ≤ 1, i.e.
 *   |l2·sin(θ₂) − offset| ≤ l3
 * This is always satisfied when l3 > l2 + |offset|, the standard design
 * guideline. The UI enforces offset < l2 (same constraint as the Python app).
 * Returns null for degenerate configurations so the caller can skip/warn.
 */

/**
 * Compute slider-crank kinematics at a single crank angle.
 *
 * @param {number} l2        Crank length (link 2)
 * @param {number} l3        Connecting rod length (link 3)
 * @param {number} offset    Slider offset (perpendicular distance of rail from crank pivot)
 * @param {number} theta2Rad Crank angle in radians (measured from positive x-axis)
 * @returns {{ s, beta, pinX, pinY, sliderX, sliderY, dsdt } | null}
 */
export function sliderCrankKinematics(l2, l3, offset, theta2Rad) {
  // Connecting rod angle β from the y-loop closure equation
  const sinBeta = (l2 * Math.sin(theta2Rad) - offset) / l3
  // Guard against asin domain error (invalid / locked geometry)
  if (Math.abs(sinBeta) > 1) return null
  const beta = Math.asin(sinBeta)

  // Piston position s from the x-loop closure equation
  const s = l2 * Math.cos(theta2Rad) + l3 * Math.cos(beta)

  // Joint positions for canvas animation
  const pinX = l2 * Math.cos(theta2Rad)   // crank pin (joint between link 2 and link 3)
  const pinY = l2 * Math.sin(theta2Rad)
  const sliderX = s
  const sliderY = offset                   // slider is constrained to this y-rail

  // First-order velocity ds/dθ₂
  const radical = l3 * l3 - Math.pow(l2 * Math.sin(theta2Rad) - offset, 2)
  const denom = Math.sqrt(Math.max(radical, 0))
  const dBetaDTheta = denom > 1e-10 ? (l2 * Math.cos(theta2Rad)) / denom : 0
  const dsdt = -l2 * Math.sin(theta2Rad) - l3 * Math.sin(beta) * dBetaDTheta

  return { s, beta, pinX, pinY, sliderX, sliderY, dsdt }
}

/**
 * Precompute position + velocity arrays over 0–360° for charting.
 * Returns arrays of length `steps` (default 360).
 *
 * @param {number} l2
 * @param {number} l3
 * @param {number} offset
 * @param {number} [steps=360]
 * @returns {{ angles: number[], positions: number[], velocities: number[], betas: number[] }}
 */
export function computeFullCycle(l2, l3, offset, steps = 360) {
  const angles = []
  const positions = []
  const velocities = []
  const betas = []

  for (let i = 0; i <= steps; i++) {
    const theta2Deg = (i / steps) * 360
    const theta2Rad = (theta2Deg * Math.PI) / 180
    const result = sliderCrankKinematics(l2, l3, offset, theta2Rad)
    if (result) {
      angles.push(theta2Deg)
      positions.push(parseFloat(result.s.toFixed(4)))
      velocities.push(parseFloat(result.dsdt.toFixed(4)))
      betas.push(parseFloat(((result.beta * 180) / Math.PI).toFixed(3)))
    }
  }

  return { angles, positions, velocities, betas }
}
