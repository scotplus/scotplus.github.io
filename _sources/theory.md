---
header-includes:
    - \usepackage{amsmath}
    - \usepackage{amsfonts}
    - \usepackage{bbm}  
---

# The Evolution of SCOOTR

We will begin by examining the evolution of optimal transport formulations, and then
enter into a brief description of our application of these formulations.

## Formulations

In this section, we will examine the theoretical formulations for each optimal
transport problem that has been applied to the problem of single-cell multi-omics
alignment by our lab. We will begin with the standard formulation for optimal
transport, and then move onto Gromov-Wasserstein, co-optimal transport, and our
most recent formulation.

### Optimal Transport (OT)

This formulation is applied to many problems in machine learning and was first
proposed by ?Kantorovich?. While not very applicable to data alignment, OT sets
up our understanding of the following alignment methods.

#### Formulation

The original formulation for optimal transport seeks to minimize the cost of moving
mass from one probability measure, $\mu$ ($n_{\mu}$ outcomes), to another probability 
measure, $\nu$ ($n_{\nu}$ outcomes). In the case of all formulations related to 
SCOOTR and its predecessors, these measures are discrete; so, for each outcome 
$x$ in $\mu$, OT seeks to find a cost-minimizing way (given a cost function 
$C(x_i \in \mu, y_i \in \nu)$) to move its mass to the set 
of outcomes in $\nu$. As a result, a solved OT problem will result in a coupling 
matrix, $\pi$, supported on $\mu$ and $\nu$; i.e., if a given $\pi$ results from 
this formulation of optimal transport, we know it will be in the set 
$\Pi(\mu, \nu) = \{\pi | \pi 𝟙_{n_{\nu}} = \mu, \pi^{T}𝟙_{n_{\mu}} = \nu\}$.
Therefore, the problem this basic OT formulation faces is:

$min_{\pi \in \Pi(\mu, \nu)} (\Sigma_{i = 1}^{n_\mu}\Sigma_{j = 1}^{n_\nu} (C(x_i, y_j)\pi_{ij}))$

or 

$min_{\pi \in \Pi(\mu, \nu)} (\langle C, \pi \rangle)$. Sometimes, we also refer to
$\pi$ as $\Gamma$.

Clearly, this problem requires us to find $n_{\mu}*n_{\nu}$ unknowns, which will quickly
lead to a computationally infeasible optimization as we scale. In order to remedy this issue,
we add an entropic regularization term to the objective function:

$min_{\pi \in \Pi(\mu, \nu)} (\langle C, \pi \rangle + \epsilon \langle \pi, \log \pi \rangle)$

Note that $H(\pi) = \langle \pi, \log \pi \rangle$ is a measure of entropy of $\pi$; so,
the higher the entropy, the lower the cost. As a result, as $\epsilon$ grows, the optimal $\pi$
gets more dense (favoring entropy even more), whereas if $\epsilon$ falls, we approach our original
OT formulation.

#### Algorithm

Although $\epsilon$ adds entropy to the OT formulation, it makes solving for $\pi$ computationally
tractable. In particular, it enables Sinkhorn's algorithm, as we can reframe our objective function
as the equivalent:

$min_{\pi \in \Pi(\mu, \nu)} (KL(\pi, \pi_{\textit{init}}))$

Where $\pi_{\textit{init}} = e^{\frac{C}{\epsilon}}$. Application of the KL divergence to these
two matrices will reveal the equivalence of these two minimization problems. From here,
Sinkhorn's algorithm solves this new problem iteratively; considering the problem now seeks
to project $\pi_{\textit{init}}$ onto $\Pi$, we can solve for some vectors $u, v$ in the space of
$\mu$ and $\nu$ respectively such that $\pi = diag(u)\pi_{\textit{init}}diag(v)$. Given we know the
supports of $\pi$ should be $\mu$ and $\nu$, we can iterate between the two equations:

$u = \mu \oslash \pi_{\textit{init}}v$ and 

$v = \nu \oslash \pi_{\textit{init}}^{T}u$

where $\oslash$ is element-wise division. This algorithm leads to a much faster determination of $\pi$ than possible before,
at the cost of more entropy in the final solution. Note that, the smaller $\epsilon$
gets, the slower $u$ and $v$ will coverge as a result of the new initial matrix.
Sinkhorn's algorithm thus introduces a speed-entropy tradeoff in the final coupling matrix.

While we will not go over the specifics of different algorithms used to solve the more complex cost
functions later in this document, keep in mind that variations of Sinkhorn's algorithm are
generally applicable. Looking at Sinkhorn's algorithm gives a particular understanding of how
$\epsilon$ works that is uniquely useful.

### Unbalanced Optimal Transport (UOT)

Expanding on this original optimization problem, we can relax the constraint $\pi \in \Pi$.
In particular, rather than forcing the marginals of $\pi$ to equal $\mu$ and $\nu$ respectively,
we can add cost terms encouraging the marginals to approach $\mu$ and $\nu$. These terms will
take the form of:

$\rho_x KL(\pi𝟙_{n_\nu}, \mu)$ and 

$\rho_y KL(\pi^{T}𝟙_{n_\mu}, \nu)$

Clearly, these terms will encourage $\pi$ towards the constraint $\pi \in \Pi$, but will not
force it. As $\rho_x$ and $\rho_y$ approach infinitely, we recover the OT formulation. As
they decrease, the marginal distributions we recover from $\pi$ will begin to vary more
from the constraint; in effect, we are unbalancing the initial distribution of mass in
both measures. As such, this formulation is unblanaced optimal transport. To get a sense
for what this unbalancing might mean, consider a measure $\mu$ where the outcomes are
4 different types of dog and a type of cat, and a measure $\nu$ where the outcomes are 4 
different types of cat and a type of dog. Were these measures to be uniform, under the OT formulation, some of the
dogs in $\mu$ would have to transport some of their mass to cats; otherwise, we would 
recover too much mass in the column of $\pi$ relating to $\nu$'s dog. However, if we 
use UOT with a smaller $\rho$ for each measure, we might allow the single cat to transport
more mass from $\mu$, and the single dog to transport more mass from $\nu$ in order to prevent
cross-species mass transportation (assuming varying species are farther apart). 
While a silly example, it illustrates why $\rho$ is a helpful extra parameter for OT problems.

As another example, suppose we have 5 mines (outcomes of $\mu$) transporting to
5 factories (outcomes of $\nu$). In the regular OT case, each mine produces a fixed
amount of mass, decided before optimization. However, if we have some flexibility on how
much each mine produces or each factory processes, we might add in $\rho$ as a hyperparameter
($\rho = (\rho_x, \rho_y)$).

Note that the addition of these terms is called "marginal relaxation." Many of our cost
functions from here will have a balanced and an unbalanced variant; the latter will
always have the addition of these marginal relaxation terms.

Our final optimization problem for UOT is:

$min_{\pi} (\langle C, \pi \rangle + \epsilon \langle \pi, \log \pi \rangle) + \rho_x KL(\pi 𝟙_{n_\nu}, \mu) + \rho_y KL(\pi^{T}𝟙_{n_\mu}, \nu)$

### Gromov-Wasserstein Optimal Transport (GW)

In previous cases, the way we decide how much it costs to transport mass
from $\mu$ to $\nu$ depends on the distance between each outcome in the two distributions,
defined by some distance function $D$. In other words, the goal is to minimize how
"far" the mass must travel for each outcome. 

However, suppose we instead considered two pairwise distance matrices $D_\mu$ and 
$D_\nu$ (size $n_{\mu} \times n_{\mu}$ and $n_{\nu} \times n_{\nu}$ respectively) 
within $\mu$ and $\nu$'s space of outcomes, and tried to minimize how far mass 
must travel between these pairwise distances. In this new case, if we have $x_a$ 
and $x_b$ in $\mu$ and $y_a$ and $y_b$ in $\nu$, we want to transport less mass 
from $x_a$ to $y_a$ and $x_b$ to $y_b$ if the pairwise distances between 
($x_a$ and $x_b$) and ($y_a$ and $y_b$) are farther apart. This objective encourages 
a transport plan $\pi$ that conserves the pairwise geometry (as defined by D) of each
domain; for example, a given $x \in \mu$ will likely transport much of its mass to
the $y \in \nu$ that shares the most similar relative position to points in its domain
as $x$ does. The new minimization problem this presents is called "Gromov-Wasserstein 
Optimal Transport", hence the abbreviation GW. 

In order to write this new objective as a minimization problem, we define the 
fourth order tensor L, which decides the distance between two intra-domain 
pairwise distances, $D_{\mu_{ij}}$ and $D_{\nu_{kl}}$. L leads use to a new objective 
function and minimization problem for GW:

$min_{\pi \in \Pi(\mu, \nu)} (\Sigma_{i = 1}^{n_\mu}\Sigma_{j = 1}^{n_\mu}\Sigma_{k = 1}^{n_\nu}\Sigma_{l = 1}^{n_\nu} (L(D_{\mu_{ij}}, D_{\nu_{kl}})\pi_{ik}\pi_{jl}))$

which can also be expressed as the inner product: 

$min_{\pi \in \Pi(\mu, \nu)} (\langle L(D_\mu, D_\nu) \otimes \pi, \pi \rangle)$,

where $\otimes$ is the tensor product.

Just as in OT, adding entropic regularization allows us to use Sinkhorn-like iterations
when we find the new cost-minimizing $\pi$:

$min_{\pi \in \Pi(\mu, \nu)} (\langle L(D_\mu, D_\nu) \otimes \pi, \pi \rangle) + \epsilon \langle \pi, \log \pi \rangle$

Now, we move onto the unbalanced version of this formulation.

### Unbalanced Gromov-Wasserstein Optimal Transport (UGW)

Just as with the transition from OT to UOT, we relax the $\pi \in \Pi$ marginal constraint
using $\rho_x$ and $\rho_y$:

$min_{\pi} (\langle L(D_\mu, D_\nu) \otimes \pi, \pi \rangle) + \epsilon \langle \pi, 
\log \pi \rangle + \rho_x KL(\pi𝟙_{n_\nu}, \mu) + \rho_y KL(\pi^{T}𝟙_{n_\mu}, \nu)$

As with UOT, the $\rho$ parameters allow outcomes of either mesaure to transport
more or less mass than they originally would have given $\mu$ and $\nu$. This time,
if a given outcome $x$ of $\mu$ is quite similar to multiple outcomes in $\nu$ 
in terms of relative intra-domain position, it might end up transporting more mass
than it originally had, outweighing the new KL term (and moving away from the 
constraint of OT/GW). Again, if we let $rho_x$ and $rho_y$ approach infinity, we
recover GW.

### Co-Optimal Transport (COOT)

Co-optimal transport expands on the original OT formulation by adding two new
measures, which it couples in a joint optimization problem with our original
two measures.

#### Formulation

We can now introduce the COOT formulation. Although conceptually similar to OT, 
this new concept is also a generalization of GW. Recall that in GW, we had an 
objective function of the form:

$(\langle L(D_\mu, D_\nu) \otimes \pi, \pi \rangle)$

Suppose that, instead of having distance matrices here, we had matrices $A$ and $B$ of
information on two different sets of outcomes. We define new probability measures
$\mu^s$, $\mu^f$, $\nu^s$, $\nu^f$. Our new goal is to find a way to transport
mass from $\mu^s$ to $\nu^s$ and $\mu^f$ to $\nu^f$ with the least joint cost;
in other words, we have information on the relationship between $\mu^s$ and $\mu^f$
($A$) as well as $\nu^s$ and $\nu^f$ ($B$) to inform our two transport plans, $\pi_s$ and
$\pi_f$. 

In order to leverage this information we have between measures across
these separate transport problems, we use a concept similar to GW. For each $x_s$
in the space of outcomes of $\mu^s$, we determine how much mass to transport to
each $y_s$ in the outcome space of $\nu^s$ based on the similarity of the relative 
positioning of $x_s$ to $\mu^f$ and $y_s$ to $\nu^f$. This relative positioning,
however, naturally depends on information on the correspondence between $\mu^f$
and $\nu^f$; otherwise, the positioning itself would be meaningless.

In order to get information on this correspondence, we decide how much mass to
transport from each $x_f$ in the outcome space of $\mu^f$ to each $y_f$ in the outcome
space of $\nu^f$ based on the difference between the relative positioning of
$x_f$ to $\mu^s$ and the relativing positioning of $y_f$ to $\nu^s$. However,
this positioning depends on information on the correspondence between $\mu^s$ and 
$\nu^s$, which brings us right back to where we were in the beginning of the 
previous paragraph. As we now see, there is a strong interdependence between the 
determination of $\pi_s$ and $\pi_f$; each decision in either matrix depends
on accurate correspondence information from the other matrix. Thus, we call
the procedure of finding $\pi_s$ and $\pi_f$ "co-optimal transport" (COOT).

Now, let's build the objective function for COOT from what we know about its
similarity to GW. In the GW case, we have $\mu^s = \mu^f$ and $\nu^s = \nu^f$,
if we treat $A = D_\mu$ and $B = D_\nu$. The distance matrices mimic having
information on the relationships between two pairs of measures. As a result,
in the GW case (formulated in terms of COOT), we would have $\pi_s = \pi_f$.
Generalizing this reformulation of GW, we get

$(\langle L(A, B) \otimes \pi_f, \pi_s \rangle)$

as a new objective function. In order to get a closer look, let's expand this:

$(\Sigma_{i = 1}^{n_{\mu^s}}\Sigma_{j = 1}^{n_{\mu^f}}\Sigma_{k = 1}^{n_{\nu^s}}\Sigma_{l = 1}^{n_{\nu^f}} (L(A_{ij}, B_{kl})\pi_{s_{ij}}\pi_{f_{j,l}}))$

This new objective function describes the exact interdependence we realized earlier
in this section – we simultaneously optimize $\pi_s$ and $\pi_f$, relying on joint
information on the relative positioning of i and k to their corresponding f-superscripted
measures and the relative positioning of j and l to their corresponding s-superscripted
measures. From here, we can reframe this function as a minimization problem and add
entropic regularization (allowing for Sinkhorn iterations):

$min_{\pi_s \in \Pi(\mu^s, \nu^s), \pi_f \in \Pi(\mu^f, \nu^f)} (\langle L(A, B) \otimes \pi_f, \pi_s \rangle) + \epsilon_s \langle \pi_s, \log \pi_s \rangle + \epsilon_f \langle \pi_f, \log \pi_f \rangle$

We can also do joint entropic regularization, which is equivalent to the $\epsilon_s = \epsilon_f$ case:

$min_{\pi_s \in \Pi(\mu^s, \nu^s), \pi_f \in \Pi(\mu^f, \nu^f)} (\langle L(A, B) \otimes \pi_f, \pi_s \rangle) + \epsilon \langle \pi_s \otimes \pi_f, \log (\pi_s \otimes \pi_f) \rangle$

Note that, in the case of our work, we gerenally use euclidean distance (l2)
for L. Now, we can move on to how we solve this new transport problem; clearly,
we will now need more than Sinkhorn iterations.

#### Algorithm

In order to jointly solve for these two transport plans, we employ an algorithm
called "block coordinate descent," or BCD. Without getting too into the details
(of which there are many), we will try to give some intuition on what this algorithm
is doing.

Consider the cost function we derived for standard COOT above and suppose we hold
$\pi_f$ constant. If we do so, we uncover a new minimization problem of the form (CHECK THIS, CITE):

$min_{\pi_s \in \Pi(\mu^s, \nu^s)} (\langle L_c(A, B, \pi_f), \pi_s \rangle) + \epsilon \langle \pi_s, 
\log \pi_s \rangle$

Where L is a function of $A$, $B$, and $\pi_f$ outside the scope of this document.
With this new minimization problem, we have an opportunity to optimize $\pi_s$
given $\pi_f$ using Sinkhorn iterations, given we now have the same form as a standard
OT problem (with a more complex cost function).

Since the COOT cost function is symmetric with respect to $\pi_s$ and $\pi_f$, we
can recover the same class of problem, except with $\pi_f$ as the subject of optimization.
This reformulation of the COOT problem allows us to employ BCD - we do Sinkhorn
iterations on $\pi_s$ given $\pi_f$, and then switch to $\pi_f$ given $\pi_s$
until both of our transport plans converge.

In many co-optimal transport tools, you may find hyperparameters that seem redundant,
like nits_bcd and nits_uot. These hyperparameters refer to BCD, as well as the
subprocess that BCD employs for individual plan optimization (like Sinkhorn). Hopefully,
this section has helped you understand that solving a COOT problem has multiple layers
of iteration, each of which must be considered when selecting optimization parameters.

### Unbalanced Co-Optimal Transport (UCOOT)

No surprises here – we will now unbalance COOT using four marginal relaxation
parameters: $\rho^s_x, \rho^s_y, \rho^f_x, \rho^f_y$:

$min_{\pi_s, \pi_f} (\langle L(A, B) \otimes \pi_f, \pi_s \rangle) + \epsilon \langle \pi_s \otimes \pi_f, \log (\pi_s \otimes \pi_f) \rangle + \rho^s_x KL(\pi_s 𝟙_{n_{\nu^s}}, \mu^s)$ 

$+ \rho^s_y KL(\pi_s^{T}𝟙_{n_{\mu^s}}, \nu^s) + \rho^f_x KL(\pi_f 𝟙_{{n_\nu^f}}, \mu^f) + \rho^f_y KL(\pi_f^{T}𝟙_{n_{\mu^f}}, \nu^f)$

These parameters each allow outcomes in any of our four measures to transport more or less mass
than originally allocated, provided $\rho <$ infinity. As all four $\rho$ approach infinity,
we recover COOT. We tend to relax the marginals of our transport plans with $\rho$ the most
when there is some disproportionality among outcomes of any of the measures that we wish to
correct. Note that we have now introduced a large number of new hyperparameters. These
marginal relaxation terms can be joined for less complexity, either by transport:

$min_{\pi_s, \pi_f} (\langle L(A, B) \otimes \pi_f, \pi_s \rangle) + \epsilon \langle \pi_s \otimes \pi_f, \log (\pi_s \otimes \pi_f) \rangle + \rho^s KL(\pi_s 𝟙_{n_{\nu^s}} \otimes \pi_s^{T}𝟙_{n_{\mu^s}}, \mu^s \otimes \nu^s)$

$+ \rho^f KL(\pi_f 𝟙_{n_{\nu^f}} \otimes \pi_f^{T}𝟙_{n_{\mu^f}}, \mu^f \otimes \nu^f)$

or by domain:

$min_{\pi_s, \pi_f} (\langle L(A, B) \otimes \pi_f, \pi_s \rangle) + \epsilon \langle \pi_s \otimes \pi_f, \log (\pi_s \otimes \pi_f) \rangle + \rho_x KL(\pi_s 𝟙_{n_{\nu^s}} \otimes \pi_f 𝟙_{n_{\nu^f}}, \mu^s \otimes \mu^f)$ 

$+ \rho_y KL(\pi_s^{T}𝟙_{n_{\mu^s}} \otimes \pi_f^{T}𝟙_{n_{\mu^f}}, \nu^s \otimes \nu^f)$

Transport marginal relaxation ties together the relaxation of the pairs of measures
that transport mass back and forth (are tied by a transport plan), whereas domain
marginal relaxation ties together the relaxation of the pairs of measures that have
shared information on $A$ and $B$. Both ways of joining lead to less flexibility;
in SCOOTR we chose the tie together the marginal relaxation according to the transport plans.
In other words, in SCOOTR, we marginally relax both marginals of each $\pi$ with the same
value for $\rho$ specific to each $\pi$. We chose this strategy rather than tying by
domain, as for single-cell applications, we expect sets of samples and features to have
comparable levels of representation. More on this in the application section.

### Augmented Gromov-Wasserstein (AGW)

With GW and UGW, we were able to recover non-linear relationships between outcomes
by preserving geometry; with COOT and UCOOT, we were able to map two sets of outcomes,
albeit using a formulation similar to the more linear OT and UOT problems. AGW (what we use in SCOOTR)
seeks to find the dual mapping of UCOOT with the potential for nonlinear relationships
of UGW. In effect, AGW pushes together the cost functions of UCOOT and UGW, and
assigns a hyperparameter $\alpha$ that determines the relative usage of each cost function.
As a result, we have the following minimization problem:

$min_{\pi_s, \pi_f} \alpha(\langle L(A, B) \otimes \pi_f, \pi_s \rangle) + (1 - \alpha) (\langle L(D_{\mu_s}, D_{\nu_s}), \pi_s \otimes \pi_s \rangle) + \epsilon \langle \pi_s \otimes \pi_f, \log (\pi_s \otimes \pi_f) \rangle$

$ + \rho_x KL(\pi_s 𝟙_{n_{\nu^s}} \otimes \pi_f 𝟙_{n_{\nu^f}}, \mu^s \otimes \mu^f) + \rho_y KL(\pi_s^{T}𝟙_{n_{\mu^s}} \otimes \pi_f^{T}𝟙_{n_{\mu^f}}, \nu^s \otimes \nu^f)$

While less theoretical than the previous formulations we have looked at, AGW
allows for an improvement on UCOOT by allowing for the recovery of nonlinear
relationships among the original measures we sought to couple, $\mu_s$ and $\nu_s$.
Now, we will move onto why we use these methods in single-cell multi-omics alignment. 

## Applications

In all of the formulations above, we have thought about transporting mass between
probability measures, which seems fairly removed from aligning single-cell datasets.
However, if we build up from the OT formulation's application, we can see how it
might be useful.

### OT/UOT

Suppose we treat the set of samples of a given dataset $A \in \mathbb{R}^2$ as a probability measure,
and another set of samples of a given dataset $B \in \mathbb{R}^2$ as another probability measure.
Suppose we call each row in $A$ $a_i$, and each row in $B$ $b_i$, for all i respective rows ($n_A$ and $n_B$) in each matrix.
By applying the OT/UOT framework to this problem, we would be trying to minimize:

$min_{\pi \in \Pi(\mu, \nu)} (\Sigma_{i = 1}^{n_A}\Sigma_{j = 1}^{n_B} (C(a_i, b_j)\pi_{ij}))$

If we allowed C to be some measure of distance between these two vectors, like
the dot product, we would recover a transport plan that matches rows by their
explicit similarity (i.e. $C(v, v) = 0$ for vector $v$). In terms of matching samples,
this type of transport plan would not benefit us much – unless we had the same set
of features (columns), the function $C$ would not be possible to construct, as the
two sets of samples would not lie in the same metric space. Even if we did construct
$C$, it would not make sense; if we are looking at different datasets, we don't
care if two samples have the same values for each ordered feature if those features are
not the same. So, we conclude that the OT formulation can align samples for which
we have data on the same set of features – a sort of vacuous problem, when we
can just use all available samples from $A$ and $B$ for analysis by concatenating
$A$ and $B$ (given they have the same features).

### GW/UGW

Now, we move to GW/UGW using the same intuition from the development of our
formulations above. Now, rather than directly comparing the two matrices,
we begin by calculating intra-domain distance matrices $D_A$ and $D_B$ of size
$n_A \times n_A$ and $n_B \times n_B$. These matrices can be calculated in a number of
ways; using euclidean distance, using a nearest neighbors graph (distance via
connectivity or by Dijkstra's, assuming the graph has weights), or any other
intra-domain distance metric. From here, we apply GW to these distance matrices
with any form of cost function, but usually euclidean distance.

According to our intuition on GW, our new resulting transport plan will match
samples based on their relative position to their own domain (by their pairwise
distances with all other intra-domain samples). As a result, our transport plan
will uncover the sample-sample matching that most preserves domain geometry.
For example, if we took a dataset $A$ and applied some rotation, translation, and
scaling to get a new dataset $A'$, OT/UOT would give us a transport plan that does
not necessarily match samples correctly, considering the transport plan is built
from raw dataset values. However, GW/UGW would give a perfect matching from sample
to sample in $A$ and $A'$, considering that they share an exact intra-domain geometry.
As a result, GW/UGW applied to datasets $A$ and $B$ find the geometry-preserving
coupling between the two, which has better applications to aligning datasets.

However, it also has its pitfalls; suppose we have a simple domain, $A$, which has
two features. $A$'s samples all fall within a circle of radius 1 about the origin.
Now, suppose we have $A'$, which just adds some small, random translation to each
point in $A$. GW/UGW would not necessarily align matching samples in $A$ and $A'$ –
since the geometry of $A$ can be roughly conserved with any rotation, GW/UGW might
find a plan that conserves each local geometry, but results in a rotation of $A'$.

Note also that GW/UGW, in its geometry conservation, can recover nonlinear feature
relationships, provided they are strong enough; it does not have an inherently
linear matching. In addition, since its only goal is to conserve geometry, its
alignments assume some underlying manifold structure or latent embedding that these
datasets share; if our two datasets are just clouds, GW will not (and should not) accomplish any
meaningful coupling.

### COOT/UCOOT

With COOT, on the same datasets $A$ and $B$, we seek to find co-optimal coupling matrices $\pi_s$ and
$\pi_f$. COOT ensures that the highly related features of highly related samples
(according to the optimal coupling matrices) have explicit values close together 
(based on the cost function); i.e., for large $\pi_{s_{ik}}$ and $\pi_{f_{jl}}$,
$A_ij$ must be close to $B_kl$, which is consistent with our COOT objective function:

$\langle L(A, B) \otimes \pi_f, \pi_s \rangle$

This feature of COOT allows us to generate sample coupling matrices that provide meaningful
alignments, even though we are comparing direct feature values as in OT/UOT. We achieve this new result because
when we are comparing these feature values (distinct samples with distinct features), 
we have a sense of how the features are related. Since the problem is symmetric,
we also get a feature coupling matrix that gives us meaningful relationships between features,
an added bonus.

Note, however, that these relationships are all inherently linear, since the information
we gain from features is now in the form of a matrix (clearly linear) when generating a sample 
coupling matrix. So, while our new sample alignment will be informed by feature relationships
more directly, as well as allowing for feature supervision (can penalize how far $\pi_f$
diverges from some prior), it will recover strictly linear results. Again, we see that
COOT also assumes some underlying manifold structure – without meaningful relationships between
features, COOT will not accomplish a meaningful alignment.

In addition, COOT is clearly not robust to transformations like our GW alignment.
If we scale, rotate, and translate a given matrix $A$ into $A'$, COOT will not necessarily
produce a perfect alignment between $A$ and $A'$, as it compares direct values instead of distances.
For example, while a translation may not change the optimal coupling matrix, the
minimum COOT cost will directly rise, unlike in the case of GW.

### AGW

The pitfalls of GW and COOT motivated us to see if we could find the best of both
alignments. By combining GW and COOT in our formulation above, we allow for
the determination and recovery (in the form of $\pi_f$) of nonlinear feature
relationships. In addition, we now get to distinguish between geometry-conserving
alignments from GW (our pitfall before) using the extra COOT term. As a result,
SCOOTR (AGW) allows us to align two datasets $A$ and $B$ using optimal transport better 
than ever before; we get the alignment power of GW without its pitfalls. In
addition, we can utilize the potential feature supervision from COOT to further
improve GW's results.

Note that, since GW and COOT share the assumption of an underlying manifold structure
shared between $A$ and $B$, SCOOTR also shares this assumption. Without some underlying
embedding to learn, SCOOTR would not be able to produce coupling matrices; it assumes
that a geometry-conserving coupling has meaning, which is not the case for many
datasets. Otherwise, SCOOTR really would be a silver bullet for data alignment.

## Conclusion

If this information is all a little confusing, don't worry – seeing these different
OT formulations in practice will help consolidate how you might use all of the
different OT tools available to align your data. In addition, seeing what $\epsilon$, 
$\rho$, and other hyperparameters do to our alignments will help make this document
more clear. You're more than ready to move onto some code if you've made it this far!

