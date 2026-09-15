# Conditional-to-rollout theory for trajectory iMF

**Version:** 1.1, 2026-09-11
**Scope:** the fixed-context trajectory-iMF transition objective implemented in this repository
**Claim type:** conditional theorem with explicit, presently unverified calibration and coverage assumptions

## 1. Status and claim boundary

This note proves a conditional-to-rollout implication. It does **not** infer endpoint accuracy from
the scalar loss currently optimized by the code. The logical chain is

\[
\boxed{\text{certified raw conditional residual}}
\Longrightarrow
\boxed{\text{one-step endpoint }W_1\text{ error}}
\Longrightarrow
\boxed{\text{generated-context error}}
\Longrightarrow
\boxed{\text{fixed-policy rollout error}}.
\]

The first implication is proved for an endpoint-supported, raw-square iMF population objective.
For a general trained network it is represented by an **endpoint-calibration assumption**. The
current registered objective has neither the required `r=0` schedule atom nor a raw endpoint loss,
and its stopped adaptive weighting does not dominate raw squared residuals. Consequently, the
headline rollout result is mathematically valid but is not yet an unconditional guarantee for the
current checkpoint.

The theorem compares two rollout laws under a **common policy**. It is a fixed-policy result and is
**not a policy-improvement guarantee**. It does not bound an actor optimized inside the learned
model, planner exploitation, out-of-support actions, or epistemic uncertainty. All constants are
stage-specific and must be measured or upper-bounded before the theorem is numerically informative.

The proved statements below are new only as a theorem package specialized to this causal objective;
their ingredients—regression projection, coupling, change of measure, and kernel perturbation—are
standard. The honest theoretical contribution is the precise interface showing what a trajectory
iMF experiment must certify, plus the identification of assumptions that the present loss does not
satisfy.

## 2. Measurable setup

Let \((\mathsf X,d_X)\) and \((\mathsf A,d_A)\) be Polish state/latent and action spaces with their
Borel sigma fields. At stage \(h\), let \((\mathsf C_h,d_h)\) be a Polish, Markovized causal-context
space. A point \(c_h\) contains everything on which the transition and policy depend: for example,
the full state-action history, or a recurrent state together with the current observation. If a
finite history is used, one may take
\(\mathsf C_h=\mathsf X^{h+1}\times\mathsf A^h\) with a weighted product metric.
These are, in particular, standard Borel spaces, so regular conditional laws and the kernel
composition below have the usual measurable versions.

For probability laws \(\lambda,\zeta\in\mathcal P_1(\mathsf S)\) on a metric space
\((\mathsf S,d)\), write

\[
W_{1,d}(\lambda,\zeta)
=\inf_{\Gamma\in\Pi(\lambda,\zeta)}\int d(s,\tilde s)\,\Gamma(ds,d\tilde s), \tag{2.0}
\]

where \(\Pi(\lambda,\zeta)\) is the set of couplings. Subscripts are omitted when the metric is
unambiguous.

A **Borel probability kernel** \(P_h^X(dx'\mid c,a)\) is the true next-state conditional and
\(Q_{\theta,h}^X(dx'\mid c,a)\) is the learned conditional. For every Borel set \(B\), their values
on \(B\) are Borel functions of \((c,a)\). Let \(\pi_h(da\mid c)\) be the policy kernel used by both
rollouts. Let

\[
\Psi_h:\mathsf C_h\times\mathsf A\times\mathsf X\to\mathsf C_{h+1}
\]

be the Borel context-update map. It appends the transition to a history or updates the recurrent
state. The true and learned closed-loop kernels are

\[
\begin{aligned}
K_h^P(B\mid c)
 &=\int\pi_h(da\mid c)\int P_h^X(dx'\mid c,a)
      \mathbf 1\{\Psi_h(c,a,x')\in B\},\\
K_h^Q(B\mid c)
 &=\int\pi_h(da\mid c)\int Q_{\theta,h}^X(dx'\mid c,a)
      \mathbf 1\{\Psi_h(c,a,x')\in B\}.
\end{aligned} \tag{2.1}
\]

Assume throughout that all laws used in a \(W_1\) distance have finite first moment and that the
corresponding first-moment maps are measurable. Equivalently, the transition kernels may be viewed
as Borel maps into \(\mathcal P_1(\mathsf X)\) equipped with its \(W_1\) topology; this makes
\((c,a)\mapsto W_1(P_h^X(\cdot\mid c,a),Q_{\theta,h}^X(\cdot\mid c,a))\) measurable. Starting from
initial laws \(\eta_0\) and \(\nu_0\), define
\(\eta_{h+1}=\eta_hK_h^P\) and \(\nu_{h+1}=\nu_hK_h^Q\). The generated decision-context law is

\[
\nu_h^\pi(dc,da)=\nu_h(dc)\pi_h(da\mid c). \tag{2.2}
\]

The training law \(\mu_h\) is a probability measure on the same realized conditioning space
\(\mathsf C_h\times\mathsf A\). This equality of spaces is substantive: a corruption schedule must
produce conditioning objects on which the same regular conditional \(P_h^X\) is defined. Merely
calling both tensors “contexts” is insufficient.

**Proposition 1 (rollout-law existence).** Status: **proved**. If the kernels and update maps above
are Borel, then \(K_h^P,K_h^Q\) are probability kernels and the sequential products define unique
path laws. In particular, normalized one-step conditionals induce a normalized autoregressive joint
law.

*Proof.* Measurability of parameterized integration shows that (2.1) is a kernel. The
Ionescu--Tulcea extension theorem applied successively to the initial law, policy kernel, transition
kernel, and deterministic update kernel gives a unique probability measure on every finite path;
the finite-dimensional laws are consistent. The same construction applies to \(Q_\theta\). This
proves normalization only, not agreement with the data path law. \(\square\)

## 3. The fixed-context, per-token iMF derivative

For a realized token context \(c_k\), query interval \(0\le r_k\le t_k\le1\), target \(x_k\), and
Gaussian source \(\epsilon_k\), the straight interpolation is

\[
z_k(t_k)=(1-t_k)x_k+t_k\epsilon_k.
\]

The average-velocity head is \(u_\theta(z_k,c_k,r_k,t_k)\); the instantaneous-velocity head is
\(v_\theta(z_k,c_k,t_k,t_k)\). The conditional iMF derivative moves the query token and its query
time while holding the realized context and lower endpoint fixed:

\[
\mathcal D_k u_\theta
=\partial_tu_\theta+J_zu_\theta\,v_\theta,
\qquad
(\dot z_k,\dot c_k,\dot r_k,\dot t_k)
=(v_\theta,0,0,1). \tag{3.1}
\]

Thus the iMF regression output for token \(k\) is

\[
F_{\theta,k}=u_\theta+(t_k-r_k)\mathcal D_ku_\theta. \tag{3.2}
\]

Every token may have a different \((r_k,t_k)\). Equation (3.1) is still tokenwise: independent
times change the evaluation points but do not introduce cross-token tangent terms.

**Proposition 2 (conditional versus joint JVP).** Status: **proved**. Suppose
\(c_k=C_k(h_0(\tau_0),\ldots,h_{k-1}(\tau_{k-1}),a_{<k})\). Differentiating a joint path that also
moves the history adds

\[
J_cu_{\theta,k}\sum_{j<k}J_{h_j}C_k\,\dot h_j
\quad\text{and, if exposure times move,}\quad
J_cu_{\theta,k}\sum_{j<k}\partial_{\tau_j}C_k\,\dot\tau_j. \tag{3.3}
\]

These terms are absent from the derivative of the conditional field at fixed \(c_k\).

*Proof.* Apply the multivariate chain rule to
\(u_\theta(z_k,C_k(h_{<k},\tau_{<k},a_{<k}),r_k,t_k)\). Setting all context tangents to zero leaves
(3.1); retaining them gives (3.3). No probabilistic assumption is needed. \(\square\)

This establishes the correct derivative for a conditional objective, not novelty: fixed-context
causal tangents already appear in Causal-rCM. Nor does it prove that teacher-corrupted contexts and
self-generated contexts have the same law.

## 4. From raw iMF regression to an endpoint certificate

This section isolates the one part that cannot be silently inferred from the implemented scalar
loss.

Unlike the kernel theorem in Sections 2, 5, and 6, the Gaussian interpolation and its Jacobian are
linear-space constructions. Accordingly, throughout this section we specialize to

\[
\mathsf X=\mathbb R^d,
\qquad d_X(x,y)=\|x-y\|_2, \tag{4.0}
\]

with the Borel sigma field. The learned and oracle fields take values in \(\mathbb R^d\), their
Jacobians are Euclidean Fréchet derivatives represented by \(d\times d\) matrices, and operator
norms are induced by \(\|\cdot\|_2\). No vector addition, Gaussian source, or Jacobian in this
section is asserted for an arbitrary Polish space.

If a downstream rollout uses another state metric \(\widetilde d_X\), the bridge must be explicit.

**Lemma 0 (metric domination bridge).** Status: **proved**. If, on the support of both kernels,

\[
\widetilde d_X(x,y)\le B_h\|x-y\|_2, \tag{4.0a}
\]

then every Euclidean coupling gives

\[
W_{1,\widetilde d_X}(P,Q)\le B_hW_{1,\|\cdot\|_2}(P,Q), \tag{4.0b}
\]

Consequently, A3 holds for \(\widetilde d_X\) after replacing \(\kappa_h\) by
\(B_h^2\kappa_h\).

*Proof.* For every \(\Gamma\in\Pi(P,Q)\), integrating (4.0a) gives
\(\int\widetilde d_X\,d\Gamma\le B_h\int\|x-y\|_2d\Gamma\). Taking the infimum over the same
coupling set proves (4.0b), and squaring supplies the factor \(B_h^2\). \(\square\)

Without (4.0a), the Euclidean endpoint certificate does not imply the metric error used by
Theorem 1.

Fix \((c,a)\). Draw \(X\sim P_h^X(\cdot\mid c,a)\) and \(\Xi\sim\gamma=N(0,I)\) independently, set
\(Z_s=(1-s)X+s\Xi\), and let \(Y=\Xi-X\). Write

\[
v_h^\star(z,c,a,s)=\mathbb E[Y\mid Z_s=z,c,a]. \tag{4.1}
\]

**Assumption A1 (conditional flow regularity).** The following conditions hold; statements about a
fixed \((c,a)\) are interpreted for \(\mu_h\)-almost every \((c,a)\):

1. the integrated conditional second moment is finite,
   \[
   \int\mu_h(dc,da)\int\|x\|_2^2P_h^X(dx\mid c,a)<\infty,
   \]
   so \(Y\in L^2\) jointly under \(\mu_h\), the conditional target, and \(\gamma\);
2. \(v_h^\star\) has a unique measurable flow \(\phi^\star_{s\leftarrow t}\) carrying the
   interpolation marginal at time \(t\) to that at time \(s\);
3. \(u_h^\star(z,c,a,r,t)=\{z-\phi^\star_{r\leftarrow t}(z,c,a)\}/(t-r)\) for \(r<t\), with the
   continuous boundary \(u_h^\star(z,c,a,t,t)=v_h^\star(z,c,a,t)\);
4. for almost every \((c,a,\xi)\), along the **oracle characteristic**
   \(z_s^\star=\phi^\star_{s\leftarrow1}(\xi,c,a)\), the function
   \[
   g(s)=s\{u_\theta-u^\star\}(z_s^\star,c,a,0,s)
   \]
   has a representative in \(W^{1,1}([0,1];\mathbb R^d)\), with traces
   \(g(0)=0\) and
   \(g(1)=\{u_\theta-u^\star\}(\xi,c,a,0,1)\); the chain rule (4.7) holds almost
   everywhere on \((0,1)\). In particular, this assumes the required finite trace at \(s=1\), not
   merely local absolute continuity on \((0,1)\);
5. \(\|J_zu_\theta\|_{\mathrm{op}}\le G_h<\infty\) along those characteristics, and the residuals
   \(F_\theta-F^\star\), \(v_\theta-v^\star\), and their Jacobian-weighted product in (4.7) are
   square-integrable under \(\mu_h\otimes\gamma\otimes ds\).

These are analytical assumptions. Neural-network smoothness alone does not prove that the
population conditional field has the stated unique global flow. In particular, a globally
Lipschitz deterministic flow from a Gaussian cannot produce an arbitrary singular or atomic target;
such conditionals require weaker transport maps or a different endpoint argument.

For \(r=0,t=s\), define

\[
F_\theta=u_\theta+s(\partial_su_\theta+J_zu_\theta v_\theta),
\qquad
F^\star=u^\star+s(\partial_su^\star+J_zu^\star v^\star)=v^\star. \tag{4.2}
\]

**Assumption A2 (raw endpoint-slice regression).** With probability \(\alpha_h>0\), the training
schedule selects \(r=0\). This component has conditioning marginal \(\mu_h\), and conditional on
every \((c,a)\) it samples \(s\) with density
\(q_h(s\mid c,a)\ge q_{h,\min}>0\) relative to Lebesgue measure on \((0,1)\). On this component it
uses the untransformed population loss

\[
\mathcal R_h(\theta)=
\mathbb E\big[\|F_\theta(Z_s,c,a,s)-Y\|^2
+\lambda_{v,h}\|v_\theta(Z_s,c,a,s)-Y\|^2\big],
\quad\lambda_{v,h}>0, \tag{4.3}
\]

without a residual-dependent normalization. The displayed loss and its Bayes comparator are assumed
finite. Let \(\mathcal E_h<\infty\) be its schedule-weighted **Bayes excess risk**, including the
mass \(\alpha_h\).

**Lemma 1 (regression projection).** Status: **proved**. If \(Y\in L^2\), \(f(W)\in L^2\), and
\(f^\star(W)=\mathbb E[Y\mid W]\), then

\[
\mathbb E\|f(W)-Y\|^2-\mathbb E\|f^\star(W)-Y\|^2
=\mathbb E\|f(W)-f^\star(W)\|^2. \tag{4.4}
\]

Consequently, under A2, the raw Bayes excess is exactly the weighted sum of the
\(F_\theta-F^\star\) and \(v_\theta-v^\star\) squared residuals. This is the
**Pythagorean regression identity**.

More explicitly, with all fields evaluated at \((Z_s,c,a,0,s)\),

\[
\mathcal E_h=\alpha_h\,
\mathbb E_{(c,a)\sim\mu_h}\int_0^1q_h(s\mid c,a)
\mathbb E\!\left[
\|F_\theta-F^\star\|^2+\lambda_{v,h}\|v_\theta-v^\star\|^2
\mid c,a,s\right]ds. \tag{4.4a}
\]

*Proof.* Expand \(f-Y=(f-f^\star)+(f^\star-Y)\). The cross term vanishes because
\(f-f^\star\) is \(\sigma(W)\)-measurable and
\(\mathbb E[f^\star-Y\mid W]=0\). \(\square\)

This identity is relative to the unrestricted Bayes predictor. It does not say optimization finds
that predictor, and it ceases to be a raw \(L^2\) certificate after the implemented
residual-dependent adaptive transformation. A stop-gradient through the JVP does not change the
forward residual entering Lemma 1, but it does change the optimization vector field; convergence of
that stopped update to a small population residual is a separate assumption.

**Lemma 2 (iMF differential certificate).** Status: **proved** under A1--A2. Couple the learned and
oracle endpoint maps with the **same base-noise draw** \(\xi\), and define

\[
T_{\theta,h}(c,a,\xi)=\xi-u_{\theta,h}(\xi,c,a,0,1),
\qquad
T_h^\star(c,a,\xi)=\xi-u_h^\star(\xi,c,a,0,1). \tag{4.5}
\]

Then

\[
\mathbb E_{\mu_h,\gamma}\|T_{\theta,h}-T_h^\star\|^2
\le
\frac{2}{\alpha_hq_{h,\min}}
\max\!\left\{1,\frac{G_h^2}{\lambda_{v,h}}\right\}\mathcal E_h. \tag{4.6}
\]

*Proof.* Along \(z_s^\star\), put
\(e(s)=u_\theta(z_s^\star,c,a,0,s)-u^\star(z_s^\star,c,a,0,s)\). Since
\(\dot z_s^\star=v^\star\), the chain rule and (4.2) give the exact identity

\[
\frac{d}{ds}\{s e(s)\}
=F_\theta-F^\star-sJ_zu_\theta(v_\theta-v^\star). \tag{4.7}
\]

Because \(g\in W^{1,1}([0,1])\), it is absolutely continuous on the closed interval. The
fundamental theorem of calculus therefore includes both endpoint traces, including the trace at
\(s=1\), and (4.7) implies

\[
e(1)=\int_0^1
\left[F_\theta-F^\star-sJ_zu_\theta(v_\theta-v^\star)\right]ds. \tag{4.8}
\]

Jensen's inequality, \(\|a-b\|^2\le2\|a\|^2+2\|b\|^2\), and the Jacobian bound yield

\[
\mathbb E\|e(1)\|^2
\le2\int_0^1\mathbb E
\left[\|F_\theta-F^\star\|^2+G_h^2\|v_\theta-v^\star\|^2\right]ds. \tag{4.9}
\]

At every \(s\), the oracle-flow marginal is the interpolation marginal used by the regression.
Lemma 1 identifies the integrand with raw Bayes residuals. The endpoint-slice component contributes
at least \(\alpha_hq_{h,\min}\) times the uniform-in-\(s\) integral. Finally,
\(T_\theta-T^\star=-(u_\theta-u^\star)(\xi,c,a,0,1)=-e(1)\). \(\square\)

The velocity term in (4.7) is essential. Matching \(F_\theta\) alone does not control the endpoint
when the learned tangent \(v_\theta\) differs from the oracle marginal velocity.

**Lemma 3 (common-noise endpoint coupling).** Status: **proved**. If
\(P_h^X(\cdot\mid c,a)=T_h^\star(c,a,\cdot)_\#\gamma\) and
\(Q_{\theta,h}^X(\cdot\mid c,a)=T_{\theta,h}(c,a,\cdot)_\#\gamma\), then

\[
W_1^2(P_h^X(\cdot\mid c,a),Q_{\theta,h}^X(\cdot\mid c,a))
\le\mathbb E_{\xi\sim\gamma}\|T_h^\star(c,a,\xi)-T_{\theta,h}(c,a,\xi)\|^2. \tag{4.10}
\]

*Proof.* Push the single draw \(\xi\) through both maps. This is a valid coupling, so the optimal
\(W_1\) cost is no larger than its expected distance. Squaring and Jensen give (4.10). \(\square\)

Here \(W_1\) uses the Euclidean metric from (4.0). Equation (4.0b) is the only metric conversion
claimed for a non-Euclidean rollout.

Independent samples from the two endpoint laws do not give (4.10); they give the cost of a
generally nonoptimal coupling.

For the remainder, the preceding sufficient route is summarized as a portable assumption.

**Assumption A3 (endpoint calibration).** Status: **assumption, not established for the current trained model**.
For a registered nonnegative residual certificate \(\mathcal E_h\) and known finite \(\kappa_h\),
let

\[
e_h(c,a)=W_1(P_h^X(\cdot\mid c,a),Q_{\theta,h}^X(\cdot\mid c,a)),
\qquad
\mathbb E_{\mu_h}e_h^2\le\kappa_h\mathcal E_h. \tag{4.11}
\]

Here \(W_1\) is Euclidean by default. If A4 and the rollout evaluation use
\(\widetilde d_X\), Lemma 0 must first be invoked and its \(B_h^2\) factor absorbed into the
\(\kappa_h\) appearing in (4.11).

A1--A2 and Lemmas 1--3 imply A3 with
\(\kappa_h=2(\alpha_hq_{h,\min})^{-1}\max\{1,G_h^2/\lambda_{v,h}\}\). A direct raw endpoint loss,
a separately validated trace inequality, or another calibrated certificate could also imply A3.
For independently paired data and source draws, the raw endpoint MSE itself is a valid but generally
loose coupling certificate. Subtracting its Bayes risk destroys that guarantee, while the raw value
retains irreducible conditional variance even when the two endpoint laws agree. The scalar loss
logged by the present run does not supply either certificate automatically.

## 5. Generated-context transfer

The world model is trained under \(\mu_h\) but evaluated under \(\nu_h^\pi\). Assume
\(\nu_h^\pi\ll\mu_h\), let its **Radon--Nikodym** density be
\(w_h=d\nu_h^\pi/d\mu_h\), and define

\[
M_h^2=\mathbb E_{\mu_h}w_h^2
=1+\chi^2(\nu_h^\pi\Vert\mu_h)<\infty. \tag{5.1}
\]

This is a chi-square coverage coefficient. Finite \(M_h\) is substantially stronger than mere
overlap; absolute continuity alone can hold while \(M_h=\infty\).

**Lemma 4 (generated-context change of measure).** Status: **proved** under A3 and (5.1). The
generated-context mean transition error obeys

\[
\mathbb E_{\nu_h^\pi}e_h
\le M_h\sqrt{\kappa_h\mathcal E_h}. \tag{5.2}
\]

*Proof.* Change measure and apply Cauchy--Schwarz:

\[
\mathbb E_{\nu_h^\pi}e_h
=\mathbb E_{\mu_h}[w_he_h]
\le(\mathbb E_{\mu_h}w_h^2)^{1/2}
    (\mathbb E_{\mu_h}e_h^2)^{1/2}.
\]

Insert A3 and (5.1). \(\square\)

The direction matters: the expectation is under the learned model's generated contexts because
that is the second kernel in the perturbation decomposition below. Replacing it by a teacher-forced
context average without (5.1) is invalid.

## 6. Multi-step rollout theorem

Add the following stagewise conditions.

**Assumption A4 (closed-loop regularity).** There are finite \(L_h,\rho_h\ge0\) such that:

1. for fixed \((c,a)\), \(x'\mapsto\Psi_h(c,a,x')\) is \(L_h\)-Lipschitz;
2. the **true closed-loop kernel** is Wasserstein-Lipschitz,
   \[
   W_{1,d_{h+1}}(K_h^P(\cdot\mid c),K_h^P(\cdot\mid \tilde c))
   \le\rho_h d_h(c,\tilde c). \tag{6.1}
   \]

The policy's sensitivity is already inside \(K_h^P\); a discontinuous actor can therefore make
\(\rho_h\) infinite even when environment dynamics are smooth.

**Theorem 1 (conditional residual to rollout error).** Status: **proved** under A3--A4 and finite
coverage (5.1). Let

\[
D_h=W_{1,d_h}(\eta_h,\nu_h),
\qquad
\delta_h=L_h\mathbb E_{\nu_h^\pi}e_h.
\]

Then

\[
D_{h+1} \le \rho_hD_h+\delta_h,
\qquad
\delta_h\le L_hM_h\sqrt{\kappa_h\mathcal E_h}, \tag{6.2}
\]

and for every horizon \(H\ge1\),

\[
D_H\le
\left(\prod_{j=0}^{H-1}\rho_j\right)D_0
+\sum_{h=0}^{H-1}
\left(\prod_{j=h+1}^{H-1}\rho_j\right)
L_hM_h\sqrt{\kappa_h\mathcal E_h}, \tag{6.3}
\]

where an empty product equals one.

*Proof.* Insert an intermediate rollout using the true kernel from the learned context law:

\[
\begin{aligned}
D_{h+1}
&=W_1(\eta_hK_h^P,\nu_hK_h^Q)\\
&\le W_1(\eta_hK_h^P,\nu_hK_h^P)
    +W_1(\nu_hK_h^P,\nu_hK_h^Q). \tag{6.4}
\end{aligned}
\]

The first term is at most \(\rho_hD_h\): integrate an optimal (or arbitrarily near-optimal)
coupling of \(\eta_h,\nu_h\), then use (6.1) and the Kantorovich dual characterization. For the
second term, couple the action using the same policy draw. The dual characterization and the
\(L_h\)-Lipschitz append map give

\[
W_1(\nu_hK_h^P,\nu_hK_h^Q)
\le L_h\int e_h(c,a)\,\nu_h(dc)\pi_h(da\mid c)=\delta_h. \tag{6.5}
\]

Lemma 4 proves the second inequality in (6.2). Repeated substitution yields (6.3). \(\square\)

For constants \(\rho_h\le\rho<1\) and
\(L_hM_h\sqrt{\kappa_h\mathcal E_h}\le b\), (6.3) gives
\(D_H\le\rho^HD_0+b(1-\rho^H)/(1-\rho)\). For \(\rho=1\) the bound is linear, and for
\(\rho>1\) it may grow exponentially. Calling (6.3) “stable” without controlling these constants
would be misleading.

If the reported state at stage \(h\) is a 1-Lipschitz projection of \(c_h\), its marginal \(W_1\)
error is no larger than \(D_h\).

**Corollary 1 (fixed-policy return error).** Status: **proved**. If the stage reward
\(R_h:\mathsf C_h\to\mathbb R\) is \(\ell_h\)-Lipschitz and integrable under both path laws, then

\[
\left|\sum_{h=0}^{H-1}\mathbb E_{\eta_h}R_h
-\sum_{h=0}^{H-1}\mathbb E_{\nu_h}R_h\right|
\le\sum_{h=0}^{H-1}\ell_hD_h. \tag{6.6}
\]

*Proof.* Kantorovich--Rubinstein duality gives
\(|\mathbb E_{\eta_h}R_h-\mathbb E_{\nu_h}R_h|\le\ell_hD_h\) at each stage. Sum the inequalities.
\(\square\)

This corollary holds for the same fixed policy in both systems. An actor trained to maximize the
learned return is selected adaptively from model errors, so (6.6) alone does not control its real
performance. It also assumes the same reward function is evaluated on both context laws. A learned
reward head requires an additional generated-context reward-error term.

## 7. Reductions

### 7.1 Ordinary iMF

Set the trajectory length to one. There is no causal prefix, the history-exposure schedule
disappears, and the per-token interval is a single \((r,t)\). Equations (3.1)--(3.2) are the
conditional **ordinary iMF** objective. Making \(c\) deterministic removes conditioning and gives
the unconditional objective. This is an algebraic reduction, not a claim that the training
distribution or architecture matches every published iMF implementation.

### 7.2 Conditional flow matching

Restrict all intervals to the boundary \(r=t\). Then the multiplier \(t-r\) in (3.2) is zero, so
the predicted quantity reduces to \(u_\theta(z,c,t,t)\). With the boundary identification
\(u_\theta(z,c,t,t)=v_\theta(z,c,t,t)\), raw regression to \(\epsilon-x\) is **conditional flow matching**
on the straight path. The auxiliary velocity loss is a duplicate at this boundary. This is the
flow-matching component of the schedule; it supplies no direct \((0,1)\) endpoint trace.

### 7.3 One-step dynamics

At inference choose \((r,t)=(0,1)\) and draw \(\epsilon\sim\gamma\). The transport rule is

\[
\widehat x'=\epsilon-u_\theta(\epsilon,c,a,0,1), \tag{7.1}
\]

which is one network evaluation and defines the learned conditional transition kernel. This
**one-step dynamics** reduction is an inference identity. It does not say that training at interior
times identifies the endpoint value.

### 7.4 The three context schedules

Let \(S\in\{\mathrm{clean},\mathrm{corrupt},\mathrm{suffix}\}\) be sampled with fixed probabilities
\(p_S\). By total expectation the objective is

\[
\mathcal L=\sum_Sp_S\,\mathbb E[\mathcal L_S\mid S]. \tag{7.2}
\]

Thus a single estimator supervises clean contexts, independently corrupted contexts, and a clean
prefix with a teacher-corrupted future suffix. Equation (7.2) is bookkeeping, not proof that these
conditionals are mutually compatible or that it replaces shortcut-consistency, self-forcing, or
overshooting objectives. The suffix is teacher-corrupted, not generated by the current model.

## 8. What the present implementation does not satisfy

The frozen matched-objective protocol currently declares:

- `imf_endpoint_scale = 0.0`, so no raw endpoint reconstruction term directly certifies (4.11);
- `imf_adaptive_power = 1.0`, so the main residual is divided by a stopped
  residual-dependent factor;
- the logit-normal query sampler has support strictly inside \((0,1)\), plus an atom at `r=t`, but
  **no atom at `r=0`**;
- boundary-velocity supervision helps identify the tangent head but does not create the missing
  endpoint slice;
- finite minibatch loss and successful optimization do not establish a population Bayes excess;
- no bound on \(G_h\), \(M_h\), \(L_h\), or \(\rho_h\) is currently certified.

With an unweighted full-history metric, the append operation retains every old discrepancy and
typically prevents \(\rho_h<1\). A contraction claim would need a proved sufficient recurrent state
or a prespecified fading-memory metric; evaluating only the latest-state projection can still give
a useful noncontractive finite-horizon bound.

Therefore A2 is false for the registered schedule as written, and A3 is unverified. Possible
honest routes are: add a prespecified raw \(r=0\) slice; turn on a raw endpoint term with known
weight; or prove and validate a trace inequality from interior \(r\)-risk using uniform derivative
bounds. Any such change is a new arm or preregistered ablation, not a retrospective proof about old
checkpoints.

The implementation's query/history **shared-noise augmentation** uses the same \(\epsilon_k\) to
form two corrupted views of a data token. That coupling may reduce variance or improve compatibility,
but no such result is proved. It is also distinct from Lemma 3, which couples the learned endpoint
map and an oracle endpoint map using the same source draw. Shared Gaussian sampling represents
aleatoric variation and **does not establish epistemic uncertainty**.

Finally, finite \(M_h\) is not guaranteed by mixing clean, corrupted, and suffix contexts. A
generated recurrent state can lie outside the support of every teacher-derived context. Likewise,
a low average \(D_H\) under one fixed policy does not protect a planner that deliberately searches
for rare model errors.

## 9. Counterexamples

These examples show that each major hypothesis carries mathematical content.

### 9.1 Coverage failure

Take context space \(\{0,1\}\), training law \(\mu=\delta_0\), and generated law \(\nu=\delta_1\).
Let \(P=Q\) at context zero but let their next-state Bernoulli laws be \(\delta_0\) and \(\delta_1\)
at context one. Training error is zero while generated-context \(W_1\) error is one. Here
\(\nu\not\ll\mu\), so no finite change-of-measure factor exists.

### 9.2 Endpoint-support failure

Suppose the schedule observes only \(r\ge1/4\). The continuous scalar field
\(u_M(r)=M\max\{1-4r,0\}\) has zero scheduled squared error but endpoint error
\(|u_M(0)|^2=M^2\). Letting \(M\) grow rules out any schedule-risk-to-endpoint constant unless one
adds endpoint mass or controls a trace regularity norm. Interior support that merely approaches zero
has the same problem: if the schedule has no atom at zero, the bounded continuous sequence
\(u_n(r)=\max\{1-nr,0\}\) converges to zero at every sampled \(r>0\), so dominated convergence makes
its scheduled squared risk tend to zero while \(u_n(0)^2=1\) for all \(n\). A uniform derivative or
Sobolev-trace bound would prevent this spike, but no such bound is currently certified.

### 9.3 Adaptive-weight failure

For adaptive power one, a scalar raw squared residual \(q\) contributes
\(q/(q+\varepsilon)\), which approaches one as \(q\to\infty\). There is no global finite constant
\(C\) for which \(q\le Cq/(q+\varepsilon)\). Stop-gradient changes derivatives, not this missing
value domination. Hence a small or bounded adaptive objective is not a raw \(L^2\) certificate.

### 9.4 Kernel-regularity failure

Even a perfect learned one-step kernel can amplify an initial mismatch. The deterministic kernel
\(x\mapsto2x\) multiplies \(W_1\) by two each step. More sharply, the threshold kernel
\(x\mapsto\mathbf1\{x\ge0\}\) sends \(\delta_{-\epsilon}\) and \(\delta_{+\epsilon}\), initially
distance \(2\epsilon\), to laws distance one. No uniform finite \(\rho\) exists as
\(\epsilon\downarrow0\). The factor in (6.1) cannot be dropped.

### 9.5 Independent-pairing failure

Let \(P=Q=\operatorname{Bernoulli}(1/2)\). Then \(W_1(P,Q)=0\). Independently drawn
\(X\sim P,Y\sim Q\) nevertheless satisfy \(\mathbb E|X-Y|=1/2\). Pairing an oracle target and a
generated sample independently therefore cannot validate the endpoint coupling in Lemma 3. The
same base bit gives zero cost.

### 9.6 Policy-selection failure

Let two actions agree everywhere in the training support. Outside it, make the learned model assign
one action reward \(+M\) while the environment assigns \(-M\). A planner picks that action for every
\(M>0\), although teacher-policy rollout error can remain zero. A fixed-policy perturbation theorem
does not become a uniform-over-policies guarantee without coverage and complexity control over the
policy class.

## 10. Executable controls

The offline verifier contains the following positive and negative controls:

1. A smooth scalar example evaluates the exact differential identity (4.8) by Gauss--Legendre
   quadrature and verifies that deleting the learned-versus-oracle velocity correction breaks it.
2. A two-state Bernoulli Markov system evaluates \(W_1\), the chi-square transfer factor, every
   one-step recursion, and the iterated bound exactly.
3. A scalar affine Gaussian system evaluates exact one-dimensional Gaussian \(W_1\), independently
   checks it by Gauss--Hermite quantile integration, and verifies both change of measure and the
   stable linear recursion for twelve steps.
4. The finite-state coverage example has zero training risk and unit generated error.
5. Continuous fields supported near \(r=0\) demonstrate endpoint-support failure.
6. Residuals from \(10^{-8}\) to \(10^{12}\) demonstrate adaptive-weight failure.
7. Bernoulli same-noise and independent couplings demonstrate independent-pairing failure; expansive
   and discontinuous transitions demonstrate the regularity failure.

Run:

```sh
/private/tmp/trajectory-imf-venv/bin/python \
  dreamer_imf_comparison/scripts/verify_trajectory_imf_theory.py --numerics --json
```

The success token is `TRAJECTORY_IMF_THEORY_NUMERICS_VERIFIED`. These controls catch algebraic and
implementation errors in the displayed inequalities. They are not evidence that a learned world
model satisfies A1--A4.

## 11. Relation to prior results

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) establishes conditional
  vector-field regression and the marginal-flow construction used in (4.1).
- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) introduces the
  interval-average transport identity behind (4.2) and (7.1).
- [Improved Mean Flows](https://arxiv.org/abs/2512.02012) supplies the predicted marginal-velocity
  tangent and v-regression formulation used by (3.1)--(4.3).
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392) is prior art for independently corrupted
  causal tokens and provides an all-subsequence variational interpretation that is not proved here.
- [Causal-rCM](https://arxiv.org/abs/2606.25473) is prior art for fixed-context causal JVPs and noisy
  histories. Holding context fixed in (3.1) is a correctness condition, not an originality claim.

The change-of-measure and Wasserstein-kernel arguments are standard probability tools. The present
package's value is to compose them without conflating training contexts, generated contexts,
endpoint couplings, and rollout kernels. It does not claim a new form of Cauchy--Schwarz or
Kantorovich duality.

## 12. Publication-safe theorem statement

The following wording matches what is proved:

> For any fixed policy, if the learned conditional endpoint kernels admit a certified squared
> \(W_1\) error under the training-context law, the learned generated-context law has finite
> chi-square density ratio relative to that law, and the true closed-loop kernel is
> Wasserstein-Lipschitz, then horizon-wise rollout \(W_1\) error is bounded by a geometric sum of
> square-root conditional residual certificates. For an endpoint-supported raw-square iMF
> population objective, we derive one sufficient endpoint certificate through the iMF differential
> identity and a shared source-noise coupling.

Immediately follow it with:

> The registered trajectory-iMF loss does not currently meet the sufficient endpoint-slice and raw
> weighting conditions; the theorem therefore specifies a falsifiable certification target rather
> than a guarantee for the reported checkpoints.

An experimental paper may report estimated residuals, coverage coefficients, or local Lipschitz
diagnostics, but estimates must not be presented as deterministic upper bounds unless their
estimation uncertainty and domain of validity are controlled.
