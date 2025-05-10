# Math Background

### Notation

We define the following sets, indices, and independent variables used throughout this formulation:

- $ S $: Set of mobile phase ions
- $ P $: Set of proteins or large solutes
- $ i, k \in S $: Indices for ion species (with $ k \ne i $ used for pairwise exchange)
- $ j \in P $: Index for protein species
- $ t $: Time (dimensional)
- $ z $: Axial position along the column (dimensional)

All concentrations $ C_x $ refer to mobile phase concentrations, and $ Q_x $ refer to corresponding concentrations bound to the resin.

### Generalized Stoichiometric Ion-Exchange Reactions

We consider reversible exchange between mobile ions and bound species on a charged resin. Two categories of reactions are defined:

#### 1. Ion–Ion Exchange:
$$
C_{i} + Q_{k} \overset{K_{eq,i-k}}{\rightleftharpoons} Q_{i} + C_{k}
$$

Where:
- A unique equilibrium must be defined for each ion-ion pair

#### 2. Protein–Ion Exchange (SMA model):
$$
C_j + \nu_j \overline{Q}_{i} \overset{K_{eq,j-i}}{\rightleftharpoons} Q_j + \nu_j C_i
$$

Where:
- $ \nu_j $: number of counterions displaced per molecule of protein $ j $
- $ \overline{Q}_i $: exchangeable bound counterion $ i $, not sterically hindered by adsorbed protein
- A unique equilibrium must be defined for each protein-ion pair
- This follows the **Steric Mass Action (SMA)** framework.

**Note:** No protein–protein exchange equilibrium is defined. Proteins are assumed not to displace each other directly. Instead, competition for binding sites and steric exclusion are captured implicitly through the total resin capacity and the exclusion coefficients in the SMA framework.

### Exchangeable Counterion Definition

The term $ \overline{Q}_i $ represents the **exchangeable** portion of counterion $ i $ already bound to the resin. These are ions that are not sterically blocked by adsorbed proteins and are therefore available to participate in further ion-exchange reactions. 

This exchangeable fraction is a function of:
- The total ionic capacity of the resin $ \Lambda $
- The amount of protein bound $ Q_j $
- Each protein’s displacement coefficient $ \nu_j $
- Each protein's steric factor $ \sigma_j $, representing the number of the number of additional binding sites rendered inaccessible to other proteins due to steric hindrance from a single bound molecule of protein $ j $, though these sites remain available for ion-ion exchange
- The bound concentrations of other counterions $ Q_k $

Under this framework, $ \overline{Q}_i $ can be written as:

$$
\overline{Q}_i = \Lambda - \sum_{j \in P} (\nu_j + \sigma_j) Q_j - \sum_{k \in S, k \ne i} \overline{Q}_k
$$
We assume that the distribution of ion species within the blocked fraction mirrors that of the total bound ions.
$$
\frac{\overline{Q}_k}{\sum_{k \in S, k \ne i} \overline{Q}_k} = \frac{Q_k}{\sum_{k \in S, k \ne i} Q_k}
$$
Substituted into the original equation and rearranged, we get the following.
$$
\overline{Q}_i = \frac{\Lambda - \sum_{j \in P} (\nu_j + \sigma_j) Q_j}{1 + \frac{\sum_{k \in S, k \ne i} Q_k}{Q_i}}
$$

### Kinetic Binding Equations

We express the rate of change of bound species concentrations using kinetic mass balance equations. These expressions account for adsorption and desorption reactions between ions, proteins, and the resin surface, scaled by the system's characteristic residence time $ t_{\text{res}} $.

#### Ion Mass Balance (for $ i \in S $):

$$
\frac{dQ_i}{dt} =
\frac{1}{t_{\text{res}}}
\left(\sum_{k \in S, k \ne i} \left[ k_{\text{ads},i-k} \, C_i \, Q_k - k_{\text{des},i-k} \, Q_i \, C_k \right]
+
\sum_{j \in P} \left[ k_{\text{ads},i-j} \, C_i^{\nu_j} \, Q_j - k_{\text{des},i-j} \, Q_i \, C_j \right]\right)
$$

This equation includes:
- Ion–ion exchange: between ion $ i $ and other ions $ k $
- Protein–ion exchange: due to displacement or release by protein $ j $

#### Protein Mass Balance (for $ j \in P $):

$$
\frac{dQ_j}{dt} =
\frac{1}{t_{\text{res}}}
\left(\sum_{i \in S} \left[ k_{\text{ads},j-i} \, C_j \, \overline{Q}_i^{\nu_j} - k_{\text{des},j-i} \, Q_j \, C_i^{\nu_j} \right]\right)
$$

This equation includes:
- Protein–ion exchange: due to displacement or release by protein $ j $

#### Kinetic and Equilibrium Constants

Each adsorption rate constant $ k_{\text{ads},x-y} $ and corresponding desorption constant $ k_{\text{des},x-y} $ are related by the equilibrium constant $ K_{eq,x-y} $, where:

$$
k_{\text{des},i-k} = \frac{k_{\text{ads},i-k}}{K_{eq,i-k}}, \quad \text{and} \quad K_{eq,i-k} = \frac{1}{K_{eq,k-i}}
$$

### Main Mass Balance (for all $ i \in S \cup P $)

To capture the transport of ions and proteins along the column, we use a 1D convection–diffusion equation with a kinetic term to account for binding to the resin. This expression is applied to each mobile-phase species $ i $, including both ions and proteins.

$$
\left( \varepsilon_i + \varepsilon_p K_{d,i} \right) \frac{\partial C_i}{\partial t}
=
- v \frac{\partial C_i}{\partial z}
+ D_{\text{ax},i} \frac{\partial^2 C_i}{\partial z^2}
- \varepsilon_p K_{d,i} \frac{\partial Q_i}{\partial t}
$$

Where:
- $ \varepsilon_i $: interstitial (fluid-phase) void fraction
- $ \varepsilon_p $: particle (pore) void fraction
- $ K_{d,i} $: pore accessibility fraction of species $ i $
- $ v $: superficial (interstitial) velocity
- $ D_{\text{ax},i} $: effective axial dispersion coefficient for species $ i $