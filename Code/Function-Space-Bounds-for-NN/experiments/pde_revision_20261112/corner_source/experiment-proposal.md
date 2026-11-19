# Experiment proposal: certified integration of a Cartesian ReLU network on polar boxes

Status: implementation specification; no implementation or results are claimed. This document records the current experiment, which uses one unchanged network across arithmetic precisions. The earlier high-precision-network-to-low-precision-surrogate pipeline is not part of this experiment.

## 1. Research question and fixed decisions

Can binary32 interval arithmetic reach a specified accuracy in a certified PDE solution-error estimator faster than binary64 interval arithmetic, when both certify the same frozen network with the same enclosure algorithms?

We fix:

- A Cartesian-input ReLU network with no prescribed corner factors, coordinate transformation inside the network, exact-solution features, supervised exact-solution loss, or hard boundary ansatz.
- Training with sampled Dirichlet energy plus a soft boundary penalty.
- Polar integration boxes, parameterized by radius and normalized angle.
- Maxgap selection: refine the cell with the largest width of its local integral enclosure; bisect its longest parameter side; choose the first axis on equal lengths.
- Certified activation-pattern detection and analytic integration of affine pieces.
- The exact real-valued function defined by the frozen binary32 weights as the verification target. This is not a certificate for arbitrary floating-point executions of the network.
- Fixed arithmetic precision during each run. Spatial refinement is allowed; precision adaptation is not required.

Known PDE coefficients and domain geometry may be used. The explicit solution and its normal flux may be used only in the verification/error-estimation and diagnostic stages, not as network features, training targets, or factors in the trial function.

## 2. Model problem and quantities to certify

Let $\omega\coloneqq3\pi/2$ and

$$
\Omega\coloneqq\{(r\cos\varphi,r\sin\varphi):0<r<1,\ 0<\varphi<\omega\}.
$$

Let $\Gamma\coloneqq\partial\Omega$, let $\Gamma_{\mathrm a}$ denote its unit-circle arc, and let $\gamma$ denote the trace. The radial edges have angles $0$ and $\omega$. Set

$$
s(r,\varphi)\coloneqq\frac{32}{9}\sin(2\varphi/3).
$$

We consider $-\Delta u=s$ in $\Omega$ with homogeneous Dirichlet data. For $v\in H^1(\Omega)$ define

$$
J(v)\coloneqq\int_\Omega\left(\frac12|\nabla v|^2-sv\right)\,dx.
$$

The weak formulation and energy minimization framework are supported by [Q, Section 3.3.1, pp. 41–43, especially equation (3.18) and the energy formulation on p. 42; Section 3.4.1, pp. 50–51, Lemma 3.1 and Remark 3.6]. Trace conventions are supported by [Q, Section 2.4.3, pp. 23–24, Theorem 2.3 and Property 2.6].

The following model-specific formulas are derived in the project's `LaTeX/main.tex`, Propositions 1–3; they are not attributed to the cited books:

$$
u(r,\varphi)=(r^{2/3}-r^2)\sin(2\varphi/3),\qquad J(u)=-\pi/6.
$$

The outward normal flux $q$ is

$$
q(r,0)=q(r,\omega)=-\frac23(r^{-1/3}-r),\qquad
q(1,\varphi)=-\frac43\sin(2\varphi/3).
$$

For the fixed trained network $v$, define

$$
C(v)\coloneqq\int_\Gamma q\,\gamma v\,ds,\qquad
B_{\mathrm a}(v)^2\coloneqq\int_{\Gamma_{\mathrm a}}(\gamma v)^2\,ds,
\qquad B(v)^2\coloneqq\int_\Gamma(\gamma v)^2\,ds.
$$

The estimator to enclose is

$$
G(v)\coloneqq2J(v)+\pi/3-2C(v)=\|\nabla(v-u)\|_{L^2(\Omega)}^2,
$$

$$
T(v)\coloneqq\frac32G(v)+B_{\mathrm a}(v)^2
=3J(v)+\pi/2-3C(v)+B_{\mathrm a}(v)^2.
$$

Our proved sector inequality gives

$$
\|v-u\|_{H^1(\Omega)}^2\le T(v).
$$

For a valid interval $I_T$ containing $T(v)$, the reported solution-error certificate is the upward-rounded quantity $\sqrt{\sup I_T}$. We may intersect $I_T$ with $[0,\infty)$ using the proved nonnegativity of $T(v)$. An empty intersection is an implementation failure. The lower endpoint of $I_T$ is not a lower bound on the true squared solution error.

Report $J(v)$, $C(v)$, $B_{\mathrm a}(v)^2$, and $B(v)^2$ separately. As an auxiliary, more conservative certificate, also report the manuscript's bound using $Q\coloneqq\sqrt{256/135+4\pi/3}$ and $B(v)$. The signed-flux estimator $T(v)$ is the primary benchmark target.

## 3. Training specification

### Network

Use the Cartesian network

$$
v_\theta(x,y)\coloneqq b_0+\sum_{j=1}^{64}a_j\operatorname{ReLU}(w_{j1}x+w_{j2}y+b_j).
$$

This is architecture `[2, 64, 1]`, with 257 scalar parameters. Store every parameter in binary32. Use He initialization for the hidden weights, zero hidden biases, and independent output weights uniform on $[-1/8,1/8]$ and zero output bias. Record the exact initialization implementation and software version. Do not insert a boundary multiplier or corner feature. A finite Cartesian ReLU network is Lipschitz on the bounded domain and belongs to $H^1(\Omega)$.

### Loss and sampling

Fix $\beta\coloneqq100$ before the precision comparison, and train on

$$
J_\beta(v)\coloneqq J(v)+\frac\beta2 B(v)^2.
$$

This is soft enforcement. Over unrestricted $H^1(\Omega)$ the corresponding variational problem has Robin rather than exact Dirichlet conditions; the network is nevertheless evaluated against the Dirichlet solution by the estimator in Section 2. See [Q, Section 3.2.5, p. 40] for the boundary-term interpretation of Robin conditions.

For each update draw fresh, independent interior and boundary batches. For interior samples draw independent $U,V\sim\mathrm{Uniform}(0,1)$ and set $r\coloneqq\sqrt U$, $\varphi\coloneqq\omega V$, and $(x,y)\coloneqq(r\cos\varphi,r\sin\varphi)$. This is uniform area sampling. For boundary samples select the two rays and arc with probabilities $1/P,1/P,\omega/P$, where $P\coloneqq2+\omega$, then sample uniformly in arclength on the selected piece. Interior and boundary batches are independent.

For batch sizes $n_\Omega,n_\Gamma$, minimize the sampled loss

$$
\widehat J_\beta(\theta)\coloneqq
\frac{3\pi/4}{n_\Omega}\sum_{i=1}^{n_\Omega}
\left(\frac12|\nabla v_\theta(X_i)|^2-s(X_i)v_\theta(X_i)\right)
+\frac{\beta P}{2n_\Gamma}\sum_{j=1}^{n_\Gamma}v_\theta(Y_j)^2.
$$

The area and perimeter factors must not be omitted. Evaluate the forcing specified by the PDE, without using $u$. Choose sampled points away from endpoints when an expression requires an angular coordinate; exact endpoint values do not affect the integrals.

Initial configuration:

| Setting | Value |
|---|---|
| Training seeds | 0, 1, 2 |
| Optimizer | Adam |
| Updates | 10,000 |
| Learning rate | $10^{-3}$ for 7,000 updates, then $10^{-4}$ |
| Interior batch | 8,192 |
| Boundary batch | 2,048 |
| Parameters and network forward pass | binary32 |
| Loss reductions | binary64; gradients are accumulated into binary32 parameters |
| Main presentation network | seed 0, final checkpoint |

Differentiate the sampled loss using input gradients with the computational graph retained. Training is a heuristic optimization procedure: do not claim interchange of parameter differentiation and integration for the nonsmooth ReLU gradient-energy loss. We do not need such a claim for the certificate.

Use independent validation samples to record sampled energy, boundary mismatch, and diagnostic solution error. Exact-solution diagnostics must not contribute to optimization, checkpoint selection, or architecture features. Freeze the final checkpoint of every seed and hash the exported model. If training is inadequate, report this and revise training in a separately identified pilot before freezing a new benchmark; never select networks according to favorable IA32 runtime.

Export weights as IEEE binary32 bit patterns, dimensions, and a manifest. The verifier must not read a decimal serialization that changes weights. Promotion to binary64 is exact. No higher-precision-to-lower-precision surrogate is created.

## 4. Polar boxes and deterministic geometry

Use the parameter map

$$
F(r,z)\coloneqq(r\cos(\omega z),r\sin(\omega z)),\qquad(r,z)\in[0,1]^2.
$$

Store box endpoints as exact dyadic rationals, using an integer numerator and depth. The first axis is $r$ and the second is $z$. Boxes include their boundaries for enclosure purposes; overlaps of physical cell boundaries have measure zero. The collapsed edge $r=0$ also has area zero.

For a parameter box $[r_0,r_1]\times[z_0,z_1]$, write $\varphi_i\coloneqq\omega z_i$. The physical cell area is

$$
A_K\coloneqq\frac12(r_1^2-r_0^2)(\varphi_1-\varphi_0).
$$

Construct a Cartesian bounding rectangle by enclosing $\cos\varphi$ and $\sin\varphi$ on the whole angular interval, including all interior extrema, then multiplying their intervals by $[r_0,r_1]$. This is an enclosure, not an assumption that the extrema occur at parameter corners. Enclose $s$ using $(32/9)\sin(\pi[z_0,z_1])$.

Boundary partitions consist of the two radial intervals in $r$ and the arc interval in $z$. Their exact measures are $r_1-r_0$ and $\omega(z_1-z_0)$ respectively. Do not add the collapsed origin as a boundary component with positive length.

Start the interior partition with four angular strips $[0,1]\times[k/4,(k+1)/4]$, $k=0,1,2,3$, to avoid one unnecessarily broad initial Cartesian enclosure. Start each ray with one interval and the arc with four equal normalized-angle intervals. These roots are fixed for all runs.

## 5. Certified activation-pattern detection

A cell is accepted as affine only when a complete activation pattern is proved valid throughout a containing Cartesian rectangle. For the proposed shallow network each preactivation is affine, so its extrema occur at rectangle vertices. Evaluate all preactivations at all four Cartesian vertices using outward-rounded IA. A neuron is certified active if all four lower bounds are strictly positive, and inactive if all four upper bounds are strictly negative. If every hidden neuron is resolved, accept the cell. Otherwise use the unresolved-cell routine.

Uncertain signs are not evidence of a true activation crossing. Record them as inconclusive. Initially do not implement special rules for exact-zero preactivations. Such rules can be added only with an explicit validity argument.

The justification is convexity of fixed activation regions [HR, Section 2.1, p. 4, Lemma 1; Appendix B.1, pp. 12–13]. The vertex-based region-cutting procedure is described in [HR, Appendix A, p. 12]. For deeper networks, agreement must concern complete hidden-layer patterns, not just output values or gradients at vertices. Checking the mapped parameter corners is insufficient because their convex hull need not contain the curved cell.

For this one-hidden-layer architecture, a direct interval range of each affine preactivation over the Cartesian rectangle is mathematically equivalent to a vertex-extrema test and may be used as an optimized implementation. Both precision backends must use the same version. For boundary segments the containing Cartesian rectangle may be degenerate on a ray; test its distinct vertices.

For the certified active set $S$, enclose the exact coefficients

$$
c_0\coloneqq b_0+\sum_{j\in S}a_jb_j,\qquad
c_1\coloneqq\sum_{j\in S}a_jw_{j1},\qquad
c_2\coloneqq\sum_{j\in S}a_jw_{j2}.
$$

Then $v(x,y)=c_0+c_1x+c_2y$ on the cell. Cache coefficient enclosures by activation-pattern bitset, separately for each precision, and include cache construction in end-to-end timing. These coefficients may need interval enclosures even though the original weights are exactly representable.

## 6. Local integral enclosures

### 6.1 Interior: certified affine cells

Define geometric moments

$$
m_0\coloneqq\int_Ks\,dx,\qquad m_1\coloneqq\int_Ksx\,dx,\qquad m_2\coloneqq\int_Ksy\,dx.
$$

Set

$$
S_0\coloneqq\int_{\varphi_0}^{\varphi_1}\sin(2\varphi/3)\,d\varphi,
\quad S_x\coloneqq\int_{\varphi_0}^{\varphi_1}\sin(2\varphi/3)\cos\varphi\,d\varphi,
\quad S_y\coloneqq\int_{\varphi_0}^{\varphi_1}\sin(2\varphi/3)\sin\varphi\,d\varphi.
$$

The moments are

$$
m_0=\frac{32}{9}\frac{r_1^2-r_0^2}{2}S_0,\qquad
m_1=\frac{32}{9}\frac{r_1^3-r_0^3}{3}S_x,\qquad
m_2=\frac{32}{9}\frac{r_1^3-r_0^3}{3}S_y.
$$

Use endpoint differences of these antiderivatives:

$$
P_0(\varphi)\coloneqq-\frac32\cos(2\varphi/3),
$$

$$
P_x(\varphi)\coloneqq\frac12\left(-\frac35\cos(5\varphi/3)+3\cos(\varphi/3)\right),
$$

$$
P_y(\varphi)\coloneqq\frac12\left(3\sin(\varphi/3)-\frac35\sin(5\varphi/3)\right).
$$

The exact energy contribution is

$$
J_K=\frac{A_K}{2}(c_1^2+c_2^2)-(c_0m_0+c_1m_1+c_2m_2).
$$

Enclose the contribution $3J_K$ to $T(v)$. This formula eliminates quadrature overestimation on an affine cell, but not rounding uncertainty. A midpoint rule is exact for affine values on Cartesian rectangles and for constant gradients; it is not a valid shortcut for the nonconstant forcing times the network, or for a Cartesian affine function composed with the polar map.

### 6.2 Interior: unresolved cells

Use the Cartesian bounding rectangle and ordinary interval forward propagation to enclose $v$ by $V$ and each component of its almost-everywhere gradient by $D_1,D_2$. For each hidden unit use slope $[1,1]$ if its preactivation lower bound is positive, $[0,0]$ if its upper bound is negative, and $[0,1]$ otherwise. The shallow gradient formula is the interval enclosure of $\sum_j a_jw_j\mathbf1_{\{w_j\cdot x+b_j>0\}}$; a zero preactivation with zero input gradient causes no validity problem for this enclosure.

Use a dedicated interval square operation, with lower bound zero when the interval contains zero. Let $S_K$ enclose the forcing on the polar cell. An enclosure of $J_K$ is

$$
A_K\left(\frac12(D_1^2+D_2^2)-S_KV\right).
$$

No factors $1/r$ occur because the gradient is computed in Cartesian coordinates. No second derivatives are used. Bounds are required almost everywhere, which suffices for integration. See [M, Section 9.1, pp. 129–132] for interval integral enclosures; we use the elementary integration-of-bounds argument here, not Theorem 9.2 as a convergence-rate theorem for discontinuous ReLU energy densities.

### 6.3 Radial boundary edges

On either ray, let $[a,b]\subseteq[0,1]$. Define the nonnegative weight moments

$$
H_0(a,b)\coloneqq\frac32(b^{2/3}-a^{2/3})-\frac12(b^2-a^2),
$$

$$
H_1(a,b)\coloneqq\frac35(b^{5/3}-a^{5/3})-\frac13(b^3-a^3).
$$

They equal $\int_a^b(r^{-1/3}-r)\,dr$ and $\int_a^b(r^{-1/3}-r)r\,dr$. Evaluate these formulas rigorously, including intervals adjoining zero; never evaluate $q(0)$. Compute fractional powers through directed enclosures of the nonnegative cube root followed by integer powers, or another justified rational-power enclosure. Do not substitute a floating-point approximation of the exponent $2/3$ or $5/3$ without enclosing its effect.

For a certified affine restriction $v(r)=d_0+d_1r$, the local contribution to $C(v)$ is

$$
C_K=-\frac23(d_0H_0+d_1H_1).
$$

Here $(d_0,d_1)=(c_0,c_1)$ on the angle-zero ray and $(d_0,d_1)=(c_0,-c_2)$ on the angle-$\omega$ ray. Use the exact ray coordinates $(r,0)$ and $(0,-r)$, rather than approximate trigonometric coordinates. The local contribution to $T(v)$ is $-3C_K$.

For an unresolved segment with value enclosure $V$, use

$$
C_K\in-\frac23H_0V.
$$

This uses the known nonnegative weight and remains valid at the corner. Boundary-square diagnostics can be enclosed by $(b-a)V^2$, or integrated as a quadratic on affine segments.

### 6.4 Circular arc

Use the angular interval $[a,b]\subseteq[0,\omega]$; arclength is $d\varphi$. On a certified affine piece, the restriction is

$$
v(\varphi)=c_0+c_1\cos\varphi+c_2\sin\varphi.
$$

Compute its square integral using the moments of $1,\cos\varphi,\sin\varphi,\cos^2\varphi,\sin^2\varphi,\sin\varphi\cos\varphi$. Explicit antiderivatives are respectively

$$
\varphi,\quad \sin\varphi,\quad-\cos\varphi,\quad
\varphi/2+\sin(2\varphi)/4,\quad
\varphi/2-\sin(2\varphi)/4,\quad \sin^2\varphi/2.
$$

The exact flux integral is $C_K=-\frac43(c_0S_0+c_1S_x+c_2S_y)$, with angular moments from Section 6.1. Enclose the combined contribution $\int_K(v^2-3qv)\,ds$ to $T(v)$ and also its separate diagnostic components.

For an unresolved arc with value enclosure $V$, use $(b-a)V^2$ for its square integral and $-\frac43S_0V$ for $C_K$. Both $S_0$ and the radial $H_0$ are nonnegative; known sign constraints may be intersected with their computed intervals. In the combined estimator the arc expression is $v^2+4\sin(2\varphi/3)v$.

## 7. Maxgap refinement and stopping

Use one queue containing the interior cells, the two ray partitions, and the arc partition. The local estimator contributions are $3J_K$ in the interior, $-3C_K$ on rays, and $B_{\mathrm a,K}^2-3C_K$ on the arc. All priorities therefore refer to the same global quantity $T(v)$.

Define each cell priority as the width of its current local estimator interval. For a mathematically exact priority ordering of stored endpoints, compute width differences and comparisons as exact dyadic rational operations, or with an equivalently exact comparison routine. Do not silently change maxgap ordering through rounded width comparisons. Treat an infinite width as larger than every finite width; a NaN endpoint is an error. Break equal priorities by a deterministic root index and binary tree path.

For an interior parameter box split the longer side in $(r,z)$; split $r$ on equality. A boundary interval is bisected in its own parameter. Maintain exact dyadic endpoints. Use a binary aggregation tree with outward-rounded interval sums; updating one leaf must update all affected ancestors. Do not subtract a parent interval from the global sum, which needlessly loses dependency information. Add the interval enclosure of $\pi/2$ at the root to obtain $I_T$.

Primary width targets are $10^{-2},10^{-3},10^{-4}$ for $I_T$. Interpret them as exact decimal rationals when deciding whether the width target is met. Run once per configuration toward the tightest target and record the first time each target is reached. Also record energy and boundary interval widths at every checkpoint. A target on $I_T$ does not automatically impose the same target on every diagnostic interval.

The initial limits are 10 minutes of end-to-end verification time, $2^{20}$ active interior leaves, $2^{18}$ active boundary leaves in total, and dyadic depth 40 in any coordinate. Stop when any limit is reached and record which limit caused the stop. Check limits between refinement steps and record any step overrun. If the selected maxgap cell cannot be bisected under the declared depth/representation limit, stop with a limit status; do not silently skip it and claim to follow the same rule.

A single unchanged enclosure, or several unchanged iterations, do not prove permanent stagnation. An affine cell can retain rounding width and be selected repeatedly. Record such behavior but do not claim the tolerance is impossible. Finite convergence definitions and nested-intersection facts are in [M, Section 6.3, pp. 57–58, Lemmas 6.4–6.5 and Definition 6.6]. No minimum-width or rate claim follows from those facts for this implementation.

## 8. Arithmetic configurations and implementation

| Name | Local network, coefficients, and integrand arithmetic | Global interval sums |
|---|---|---|
| IA32 | IEEE binary32 | IEEE binary32 |
| IA64 | IEEE binary64 | IEEE binary64 |
| IA32/64 | IEEE binary32 | IEEE binary64 |

IA32 versus IA64 is the primary precision comparison. IA32/64 is a separately labeled accumulation experiment. Do not call IA32/64 fully single-precision verification.

Implement the verifier in compiled C++ with the same algorithm and memory layout for both endpoint types. Use rigorous directed rounding for every required operation. Use strict floating-point compilation; disable unsafe reassociation and unaccounted contraction, verify the actual rounding environment, preserve gradual underflow, and handle infinities/NaNs explicitly. An overflow can produce a valid unbounded enclosure, but it must never become a finite certificate through unchecked NaN arithmetic. Ordinary PyTorch/NumPy arithmetic is not the certification backend.

Use MPFR at 256-bit precision with directed rounding for geometric and transcendental enclosures and outward conversion to the target endpoint format. This is a common high-precision geometry service, not a claim that every operation in IA32 is binary32. For endpoint expressions involving $\pi$, use interval propagation with both bounds, not one rounded angle. Native standard-library trigonometric functions are not assumed correctly directed-rounded.

Compute moments from their analytic formulas in the common geometry service, then convert each moment outward to binary32 or binary64. Include geometry-service time in primary end-to-end timings. Cache geometry by exact dyadic cell identifiers; use separate fresh caches for independent end-to-end runs. Cache hits and misses must be reported. Cancellation in differences of antiderivatives remains enclosed, even if the enclosure is wide; known positivity and independent range bounds may sharpen it. Do not replace a small moment by zero without a proof. Fixed 256-bit geometry precision is not assumed sufficient for every imaginable cell; its interval width is part of the certificate.

Outward rounding is defined in [M, Section 3.2, p. 22, Definition 3.1]. MPFR rounding modes and conversions are specified in [P, Sections 4.4, 5.2, and 5.4]. Compiler settings and floating-point-environment requirements are described in [C, “Controlling Floating Point Behavior,” especially “-ffp-model,” “Accessing the floating point environment,” and “A note about crtfastmath.o”]. Use documentation matching the installed compiler release when implementing; archive any additional version-specific reference in Zotero and record it in `../Literature/sources.json` before citing it.

## 9. Fair runtime comparisons

### A. Identical-partition replay

Generate a deterministic partition history using IA64 maxgap. Save its initial roots and split sequence. Replay that same sequence using every backend, with and without affine detection. Replaying an IA64 history is a controlled-workload experiment, not an assertion that the other configurations follow their own maxgap rule on that history.

Record at fixed leaf-count checkpoints: local integration time, geometry time, activation-test time, coefficient-cache time, aggregation time, final-estimator time, enclosure widths, and error upper bound. Reuse of geometry across backends is permitted only in an explicitly labeled cached-geometry replay; do not present it as end-to-end performance.

### B. Independent time-to-target runs

Each backend follows its own maxgap priorities under the same splitting and tie rules. Report first-passage runtime and leaf count for each width target, as well as the certificate $\sqrt{\sup I_T}$. Different partitions are expected and are part of the cost of reaching the target.

### C. Affine-detection ablation

Compare general IA against IA with certified affine detection and analytic moments, in both IA32 and IA64. In the general-IA variant use the general cell bounds even when a cell happens to be affine. Both variants retain the same rigorous geometry and boundary-weight handling. This separates benefits of structure detection from benefits of arithmetic precision.

Use one CPU thread initially, identical batching, and the same machine. Record CPU, operating system, compiler and options, MPFR version, and power/thread settings. Perform one warm-up and five measured repetitions per case, with fresh end-to-end caches. Alternate backend order deterministically. Report medians and ranges. Exclude training, file export, and exact-solution diagnostic calculations from verification timing; include verifier model loading and geometry/cache construction in end-to-end timing, with their separate breakdowns. Record total and peak memory.

Report certified-affine leaf count and physical area, unresolved leaf count, sign-inconclusive counts, cumulative cell evaluations including discarded parents, and depth distributions. Activation-test failures are not necessarily true region crossings. Runtime claims must use equally optimized IA32 and IA64 implementations.

## 10. Validation and acceptance checks

Before full experiments:

1. Check interval basic arithmetic, squares, conversions, signed zero, underflow, and overflow against exact dyadic/rational cases and MPFR reference operations. Use cases with known exact results as well as random cases; testing is not a substitute for the arithmetic validity argument.
2. Check exported weight hashes and exact promotion. Every backend must certify the same mathematical network.
3. Test affine networks and one-ReLU networks whose switching lines give independently computable pieces. Include lines through a corner and cells where zero cannot be resolved.
4. Check the geometric area formula and the moments in Section 6 against independently derived expressions. Check angular extrema inside cells, not just endpoints.
5. Check radial weighted integrals at $a=0$. Never evaluate the singular flux pointwise there.
6. Check analytic affine-cell formulas against independently computed high-precision interval enclosures and exact polynomial test cases. Merely comparing point quadrature approximations is not a proof.
7. Verify that affine acceptance at Cartesian bounding-box vertices encloses every physical cell, and that a test using only mapped polar corners is never used.
8. Verify maxgap ordering, deterministic ties, longest-side splitting, root coverage, aggregation updates, resource statuses, and exact-decimal target comparisons.
9. Evaluate the model-specific exact energy and flux constants independently. Use $u$ only here and for diagnostics. Do not use its finite-energy identity to excuse an invalid enclosure implementation.
10. Check that the final error upper bound is computed with outward rounding. A negative upper bound for $G(v)$ or $T(v)$ is an implementation failure. Any clamping uses a proved nonnegativity constraint and cannot conceal an empty intersection.

Keep fast meaningful unit tests plus small integration tests. Certification claims require documented enclosure arguments for each implemented kernel; empirical agreement alone is insufficient.

## 11. Implementation deliverables and run interface

Create the following files during implementation; the commands below are the target interface and are not claimed to exist yet:

```text
Code/
  experiment-proposal.md
  config/sector_relu.yaml
  train.py
  export_model.py
  verifier/                 # C++ sources and CMake configuration
  tests/
  run_benchmark.py
  plot_results.py
  models/                   # frozen checkpoints, bit-exact weights, manifests
  results/                  # JSON/CSV records, split histories, hardware metadata
```

Target execution sequence, from `Code/`:

```sh
python train.py --config config/sector_relu.yaml --seeds 0 1 2
python export_model.py --config config/sector_relu.yaml
cmake -S verifier -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
python run_benchmark.py --config config/sector_relu.yaml --mode replay
python run_benchmark.py --config config/sector_relu.yaml --mode independent
python run_benchmark.py --config config/sector_relu.yaml --mode ablation
python plot_results.py --input results --output results/figures
```

Pin Python dependencies and record native dependencies before execution. Training and certification should run in separate processes so that floating-point-environment changes cannot leak between them. The benchmark runner records failures as structured results rather than omitting them.

Minimum per-run output: model hash and seed; backend and algorithm variant; target; status; elapsed time; cell counts and cumulative evaluations; precision and compiler settings; raw endpoint representations for $I_J,I_C,I_{B_{\mathrm a}^2},I_{B^2},I_T$; upward-rounded error upper bound; timing breakdown; cache counts; maximum depths; peak memory. Decimal display endpoints must themselves be outward-rounded or supplemented with exact hexadecimal values.

Primary figures: end-to-end time versus estimator enclosure width; time-to-target by backend; certified error upper bound versus time; enclosure width versus evaluated cells; and affine-detection cost/benefit. Show unmet targets and limits. Separate estimator overestimation, quadrature/enclosure width, and arithmetic effects in the interpretation.

Success means a lower-precision configuration reaches a comparable certified target faster, with all setup and refinement costs included. Faster arithmetic per cell alone is insufficient. No success, target attainment, or speedup is assumed in advance.

## 12. References and locators

All references below have been inspected in the relevant portions. Page numbers refer to printed pagination unless otherwise stated. Do not use blanket citations in subsequent documents; cite the specific locators used.

- **[Q] Primary PDE reference:** Alfio Quarteroni, *Numerical Models for Differential Problems*, third edition, Springer, 2017. Zotero PDF `96N9WTHR` via `../Literature/sources.json`. The file's title/copyright pages identify 2017 despite the filename. Used here: Section 2.4.3, pp. 23–24; Section 3.2.5, p. 40; Section 3.3.1, pp. 41–43; Section 3.4.1, pp. 50–51.
- **[M] Primary interval-arithmetic reference:** Ramon E. Moore, R. Baker Kearfott, and Michael J. Cloud, *Introduction to Interval Analysis*, SIAM, 2009. Zotero PDF `SWLX67UD` via `../Literature/sources.json`. Used here: Section 3.2, p. 22; Section 6.3, pp. 57–58; Section 9.1, pp. 129–132. The exact-arithmetic convergence theorem is not asserted for our discontinuous ReLU energy integrand or rounded implementation.
- **[HR] Activation-region reference:** Boris Hanin and David Rolnick, *Deep ReLU Networks Have Surprisingly Few Activation Patterns*, NeurIPS 2019, arXiv:1906.00904v2. Zotero PDF `8NTV9EBG` via `../Literature/sources.json`. Used here: Section 2.1, p. 4, Lemma 1; Appendix A, p. 12; Appendix B.1, pp. 12–13. These support region convexity and vertex tests, not a proof that our IA implementation resolves signs.
- **[P] Arithmetic implementation reference:** *GNU MPFR 4.2.2 Reference Manual*. Zotero PDF `BI5G9YK6` via `../Literature/sources.json`. Used here: Section 4.4, “Rounding,” printed pp. 9–10; Section 5.2, “Assignment Functions,” and Section 5.4, “Conversion Functions,” printed pp. 19–22, for directed assignment/conversion semantics. Verify each chosen conversion function against its documented entry.
- **[C] Compiler reference:** *Clang Compiler User's Manual*. Local file: `../Literature/Clang - Users Manual.html`. Used here: “Controlling Floating Point Behavior,” specifically “-ffp-model,” “Accessing the floating point environment,” and “A note about crtfastmath.o.” This archived web manual is not assumed to match every installed compiler release.

The exact solution, flux, analytic cell formulas, and full $H^1$ estimator are model-specific derivations. They must be checked as mathematics and implemented with rigorous rounding; they are not consequences of an empirical training result.
