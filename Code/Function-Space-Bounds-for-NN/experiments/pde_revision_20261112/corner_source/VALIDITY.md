# Enclosure arguments and implementation scope

This note explains the enclosure logic implemented in `interval.hpp` and `main.cpp`. It does not claim formal machine-checked verification of the C++ program.

## Arithmetic

The native kernels assume IEEE binary32 or binary64 operations, source-type rounding to nearest, gradual underflow, and no reassociation or unaccounted contraction. The build uses Clang strict floating-point settings. At startup the program checks round-to-nearest; tests also check gradual underflow. The current benchmark is on the recorded Apple ARM64 compiler/runtime combination.

For a correctly rounded finite basic operation, the exact result lies between the two adjacent representable neighbours of the rounded result. The implementation encloses each endpoint result by `nextafter` toward negative or positive infinity. This is deliberately wider than optimal directed rounding. Exact zero operands are simplified only where the exact operation is known. Multiplication uses all four endpoint products; squaring handles zero crossings separately. Unsupported NaN/unbounded priority calculations fail closed instead of issuing a finite certificate. Arithmetic validation includes exact rational comparisons, subnormal operands, signs, and randomized point products. Tests alone do not replace the stated arithmetic assumptions.

MPFR interval endpoints use RNDD/RNDU for all service operations. Products take all endpoint products. Trigonometric ranges combine endpoint bounds with every possible critical point in the explicitly guarded angle range [-16,16]. Critical-point locations are themselves enclosures of multiples of pi/2; an uncertain intersection includes the extremum. Fractional powers use a directed cube root followed by interval integer powers. No rounded exponent is treated as the exact number 2/3 or 5/3.

Converting MPFR interval endpoints to native formats uses directed `mpfr_get_flt`/`mpfr_get_d`. Every geometric quantity therefore encloses the exact sector geometry. This shared 256-bit service is present even in the configuration named IA32. Geometry cost is included in end-to-end timing. These runs are not pure binary32 calculations of all geometric constants.

## Network and local bounds

The target is the real-valued shallow ReLU network with the stored binary32 weights interpreted exactly. Native interval operations enclose affine preactivations, ReLU values, and almost-everywhere gradients. On an uncertain activation, each gradient contribution uses a slope interval [0,1]. Constant-zero preactivations cannot invalidate that enclosure: for a shallow neuron with zero input weights its gradient contribution is zero.

A common strictly certified activation pattern throughout a containing Cartesian rectangle implies an affine restriction on the physical cell. Extrema of shallow affine preactivations occur at rectangle vertices; the implementation uses the equivalent interval-range test. The containing rectangle is obtained from certified polar angular ranges, including interior trigonometric extrema. Agreement at mapped polar corners is never used as a sufficient condition.

Affine coefficients are sums/products of exact stored weights, enclosed using the selected native precision. Their coefficient intervals are not mistaken for exact point coefficients. Analytic energy and boundary formulas are evaluated with these coefficient intervals and certified geometric moments. The formulas are derived in `experiment-proposal.md`, Section 6. Nonnegative square integrals may be intersected with [0,infinity).

For unresolved cells, integrating an almost-everywhere enclosure of the Cartesian energy density and multiplying by the cell area gives an integral enclosure. On rays, the singular nonnegative flux weight is integrated analytically and then multiplied by a value interval. The verifier never evaluates the singular flux at the origin. Arc bounds use the corresponding integrated nonnegative sine weight. No quadrature convergence-rate claim is used.

## Global bound and refinement

Local intervals enclose contributions to T = 3 J + pi/2 - 3 C + Ba^2. A binary aggregation tree sums all disjoint-cell contributions with outward enclosure at each node. Replacing a parent by its children updates ancestor sums, rather than subtracting an unrelated interval sum. Exact rational comparison of native endpoint differences implements maxgap. Dyadic coordinate counters implement longest-side splitting with the first-axis tie rule.

The manuscript proves ||v-u||_H1^2 <= T for the sector. After adding the certified pi/2, the implementation intersects T with its known nonnegative range. An upper endpoint below zero is an error. An MPFR upward square root gives the reported H1-error upper bound. The lower endpoint of the T enclosure is not a lower bound for the actual H1 error.

In the current version the program records bit-exact interval endpoints, finite resource stops, and noncertified timing/area diagnostics. A resource stop does not prove that a tolerance is impossible at the chosen arithmetic precision.

## Reference locators

- Moore, Kearfott, and Cloud (2009), Section 3.2, p. 22: outward rounding; Section 9.1, pp. 129–132: interval integration; Section 6.3, pp. 57–58: nested intervals and finite convergence. Primary interval reference in `../../Literature/`.
- Hanin and Rolnick (2019), Section 2.1, p. 4, Lemma 1; Appendix A, p. 12; Appendix B.1, pp. 12–13: activation-region convexity and vertex tests. Local PDF in `../../Literature/`.
- GNU MPFR 4.2.2 manual, Section 4.4, pp. 9–10, and Section 5.4, starting p. 19: rounding and native conversions. Local manual in `../../Literature/`.
- Clang manual, “Controlling Floating Point Behavior,” subsections “-ffp-model” and “Accessing the floating point environment”: compiler semantics. The archived manual and installed compiler version are recorded separately; actual-platform validation is required.
