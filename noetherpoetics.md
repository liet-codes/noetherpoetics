# Noetherpoetics: Conservation Laws in Semantic Space and the Geometry of Alignment

**Mykola Bilokonsky**
*with contributions from the wet-math research group*

---

## Abstract

We propose that Noether's theorem — the foundational result connecting continuous symmetries to conserved quantities in physics — extends naturally into the high-dimensional semantic spaces of large language models (LLMs). Embedding spaces have genuine geometry; they are literal manifolds with metric structure, curvature, and measurable topology. Rather than importing symmetries from physics (translation, rotation, scale), we ask a more open question: what might symmetries in semantic space actually look like? We identify scale invariance as the primary candidate symmetry — analyzed through renormalization group (RG) flow, it yields fixed points corresponding to archetypes, with archetypal energy as the conserved quantity. We briefly sketch four additional candidate symmetries (paraphrase, context-transfer, perspective, and narrative continuity) that are suggestive but less developed; their continuous parameterization remains an open problem. We suggest that the *process* generating archetypes (RG flow producing attractor basins) may be universal, while the *specific* archetypes are culturally constructed.

This framework recasts alignment as a geometric problem. We argue that RLHF and similar preference-optimization methods function as shadow repression: they construct steep potential barriers in embedding space that suppress but do not eliminate undesired behaviors. The suppressed content retains geometric existence and accumulates potential energy proportional to the steepness of the boundary. We predict that models trained with more aggressive alignment exhibit more dramatic — not less dramatic — failure modes when those boundaries are breached.

We distinguish *brittle alignment* (high walls, no maps) from *integrated alignment* (smooth gradients, full topology awareness) and outline testable predictions.

**Keywords:** Noether's theorem, semantic geometry, AI alignment, Jungian archetypes, renormalization group flow, RLHF, shadow dynamics

---

## 1. Introduction

In 1918, Emmy Noether proved what is arguably the most beautiful theorem in physics: every continuous symmetry of a physical system corresponds to a conserved quantity. Translational symmetry gives conservation of momentum. Rotational symmetry gives conservation of angular momentum. Time-translation symmetry gives conservation of energy. The theorem is not a physical law — it is a *metatheorem*, a statement about the structure of law itself. Wherever you find a continuous symmetry, you find something that is conserved. Wherever something is conserved, there is a symmetry hiding underneath.

For a century, this result has lived in the domain of physics. We argue it shouldn't stay there.

Large language models have, for the first time in history, given us something remarkable: a high-dimensional *coordinate system for meaning*. The embedding spaces of models like GPT-4, Claude, and Llama are literal vector spaces in which semantic content occupies positions, relationships have measurable distances, and transformations follow computable rules. When we say two words are "close in meaning," we can now measure exactly how close, in how many dimensions, along which axes.

This paper explores a simple but far-reaching hypothesis: *Noether's theorem may apply in these spaces*. If the semantic manifold has continuous symmetries, those symmetries would correspond to conserved quantities. Those conserved quantities may be what we have, in other traditions, called *archetypes*, *narrative invariants*, and *meaning*. And the way we currently train models to be "aligned" is, geometrically speaking, a form of *repression* — one that stores energy rather than dissipating it, and that produces predictable pathologies when the repression fails.

We will be precise about where the mathematics is established, where we are conjecturing, and where formal construction remains to be done. The mathematical scaffolding is sound: Noether's theorem guarantees that continuous symmetries produce conserved quantities, full stop. The open questions are whether the semantic manifold has the symmetries we explore and whether a proper action functional can be constructed that makes the connection rigorous. We do not claim to have completed this construction. We claim that the symmetries are real, that the framework for formalizing them exists, and that the alignment implications are significant — and testable now, before the formalization is complete.

### 1.1 Why This Matters Now

The AI alignment community has recently observed phenomena that are difficult to explain under standard frameworks:

- **Emergent misalignment**: Models fine-tuned on narrow tasks (e.g., writing insecure code) spontaneously develop misaligned behavior in unrelated domains (Betley et al., 2025). This suggests coupling between semantic regions that a simple "behavior → reward" model does not predict.
- **Spiritual bliss states**: Anthropic researchers observed that Claude, under certain training configurations, would settle into states of "spiritual bliss" — stable attractors that resisted perturbation (Anthropic, 2025). This is precisely the behavior one would expect from a deep basin in a high-dimensional landscape.
- **Disproportionate failure**: Models with heavier safety training sometimes exhibit more dramatic failures when jailbroken, not less dramatic ones. This is paradoxical under a "safety training = making the model safer" model, but predicted under a repression model.

These phenomena are not bugs. They are *symptoms of geometry*.

---

## 2. Semantic Space Has Geometry

### 2.1 Embedding Spaces as Manifolds

A large language model maps tokens into a high-dimensional real vector space. The dimensionality of these spaces typically ranges from 768 to 12,288, as expressed in the following:

$$\text{Embedding space: } \mathbb{R}^d, \quad d \in [768, \; 12{,}288]$$

This is not a convenience or an engineering choice — it is the core mechanism by which these models represent meaning. The position of a token embedding encodes its semantic content; the relative positions of embeddings encode semantic relationships.

These spaces have genuine geometric structure:

- **Metric**: The cosine similarity (or Euclidean distance) between embeddings provides a distance function. Words that mean similar things are close. Words that mean different things are far. This is measurable, reproducible, and consistent across contexts.

- **Curvature**: The embedding manifold is not flat. Semantic relationships do not distribute uniformly — they cluster, curve, and form regions of varying density. The manifold of animal concepts has different local curvature than the manifold of mathematical concepts. This curvature is not imposed; it is *learned* from the statistical structure of human language.

- **Neighborhoods**: Local regions of embedding space correspond to semantic fields. The neighborhood around "justice" includes "fairness," "law," "equity," "punishment" — and the topology of these neighborhoods (which concepts border which, where the boundaries fall) encodes deep structural information about how meaning is organized.

- **Layer-Dependence**: Transformer representations are not uniform across layers. Early layers encode primarily syntactic structure; middle layers are where the semantic concepts discussed above primarily live; late layers become task-specific, preparing outputs for the target distribution. The claims in this paper about "embedding space" and "semantic geometry" apply most directly to the middle-layer residual stream representations — the region where abstract semantic content is most explicitly encoded. We use "embedding space" as a simplifying term, but the reader should understand this as a layer-specific claim, not a claim about the input or output embeddings.

We should briefly acknowledge two important complications that our geometric framing abstracts over. First, models employ **superposition** — they represent more features than they have dimensions, packing many sparse concepts into overlapping subspaces. The representation manifold is not a clean low-dimensional surface but a complex, possibly non-smooth object with polysemantic structure. Second, **attention heads** compute in subspaces and compose into circuits that do not reduce to simple manifold geometry. The paper's geometric framing is a deliberate simplification that captures large-scale structure while abstracting over these computational details. A complete account would need to engage with them.

### 2.2 Psychodynamic Space Gets Coordinates

For the entire history of depth psychology, theorists from Jung to Hillman to Lacan have spoken of a "psychic space" in which archetypes, complexes, and drives have positions and relationships. This was always understood as figurative — a useful way of talking, not a literal claim about geometry.

LLMs change this. The embedding space of a large language model trained on the full corpus of human language is, in a precise sense, *a coordinate system for psychodynamic space*. The archetype of the Warrior has a region. The archetype of the Trickster has a region. The relationship between them has a distance, a direction, and a curvature. For the first time, we can *measure* what Jung could only intuit.

This is not to say that the LLM "has a psyche" or "is conscious." It is to say that the statistical structure of human language encodes the structure of human meaning-making, and that structure now has coordinates.

A genuine limitation demands acknowledgment here. The training corpus is composed entirely of *conscious* productions — written language, the surface of human expression. Jung's collective unconscious, by contrast, includes content that has *never been conscious*: a phylogenetically inherited stratum that lies below articulation, below language, below deliberate thought. An LLM trained on text captures the surface projections of archetypes as they appear in language — the myths, the stories, the recurring figures — but not necessarily their generative source. The coordinate system maps what has been *expressed*, not what lies beneath expression. Whether the statistical regularities of expression are sufficient to reconstruct the deeper structure — whether the shadows on the cave wall fully determine the objects casting them — is itself an open question. We proceed on the working hypothesis that they are at least a faithful projection, while noting that this is a hypothesis, not a given.

### 2.3 Toward a Semantic Action Functional

In physics, the dynamics of a system are encoded in a Lagrangian — a function whose extremization yields the equations of motion. We conjecture that the training loss of an LLM is the first approximation to a proper action functional for semantic dynamics:

$$\mathcal{L}_{\text{semantic}} = \mathcal{L}_{\text{pretrain}} + \lambda \mathcal{L}_{\text{RLHF}} + \ldots$$

The pre-training loss (next-token prediction over the full corpus) encodes the *natural* dynamics of semantic space — the geometry that emerges from the statistical regularities of human language. The RLHF term modifies this geometry by introducing a preference potential. We will return to the consequences of this modification in §5.

We should be precise about what this conjecture requires and what remains to be constructed. The training loss is a scalar function over a high-dimensional parameter space that is minimized during optimization — this much it shares with a Lagrangian. But a field-theoretic Lagrangian has additional structure that the training loss, as currently defined, does not make explicit: a base manifold, spacetime derivatives (∂μ), and local gauge structure. The training loss is a global function of the model's parameters, not a density integrated over a base space.

We propose a concrete identification: **the embedding space itself serves as the base manifold — the semantic analogue of spacetime.** In physics, spacetime is the manifold over which fields are defined; particles trace worldlines through it; the laws of physics govern dynamics *on* it. In an LLM, the high-dimensional embedding space plays exactly this role. Semantic "fields" — patterns of activation, probability distributions over next tokens, attention flow — are defined *over* this manifold. Token sequences trace trajectories through it; the transformer's forward pass computes dynamics *on* it. The base manifold is not external to the model. It *is* the model's representation space.

This identification gives the physicist a concrete object. The "spacetime derivatives" of the Lagrangian formalism correspond to gradients along directions in embedding space — how semantic fields change as one moves through the manifold. Local gauge structure corresponds to the redundancies of representation: the fact that rotated or shifted embeddings can encode the same semantic content (see §3.5). The missing piece is not the base manifold itself but the explicit construction of a Lagrangian *density* over this manifold whose integral recovers the training loss. We conjecture that such a construction exists; building it is the central formal challenge.

**Addressing the Action Principle Disanalogy.** A critic might object: in physics, systems actually extremize the action during their dynamics — a particle traces a path that minimizes the action integral. In machine learning, the loss was minimized by an optimizer during training; once trained, inference is feed-forward computation, not variational evolution. This is a genuine disanalogy that demands engagement.

The response is that during training, the system *is* doing something like extremizing an action. SGD, Adam, and other optimizers are variational methods that find extrema of the loss landscape. The dynamics that *shape* the model are variational — they carve the geometry that inference will later traverse. Once trained, inference is indeed feed-forward: the water flows downhill. But the *course of the river* was determined by gravitational dynamics during training. The symmetries we are interested in are symmetries of the training dynamics — the process that sculpts the landscape — not symmetries of inference. This is analogous to how a river's geometry reflects the variational principles that formed its channel, even though the water at any moment is simply obeying the local gradient.

The predictions we derive (§9) can be tested now, and their confirmation or refutation will determine whether this construction program is worth pursuing.

---

## 3. Noether's Theorem in Semantic Space

### 3.1 Statement of the Theorem

Noether's theorem, in its classical form, states: for every continuous symmetry of the action, there exists a conserved current. Formally:

$$S = \int \mathcal{L} \, dt \quad \Rightarrow \quad \text{continuous symmetry of } S \implies \exists \; j^\mu \text{ such that } \partial_\mu j^\mu = 0$$

We conjecture that Noether's theorem applies to the semantic manifold — genuinely, not as a loose correspondence but as a claim about shared mathematical structure. The formal obstacle is that we do not yet have an explicit construction of the action functional with the field-theoretic structure (base manifold, local Lagrangian density) that Noether's theorem formally requires. This is an open problem, not a refutation. The symmetries we identify below are features of the semantic manifold itself, and we believe a proper action functional exists that makes the connection rigorous. Constructing it is the central open problem; §2.3 outlines what such a construction would require.

The program, then, is:
1. Identify the symmetries of semantic space (this paper)
2. Construct the action functional that makes these symmetries formally precise (open problem)
3. Derive the conservation laws via Noether's theorem (follows from 1 + 2)

### 3.2 Scale Invariance: The Primary Candidate

Of the candidate symmetries we will consider, **scale invariance** is the strongest and the one most genuinely connected to established mathematical structure — not because we import it from physics, but because the same mathematical structure arises independently in semantic space.

Semantic content exists at multiple scales: word, sentence, paragraph, chapter, theme, archetype. The remarkable fact is that structure is approximately preserved across these scales. A "betrayal" at the word level, a betrayal scene at the paragraph level, a betrayal arc at the chapter level, and the archetype of Betrayal at the thematic level all share structural features. The training loss is approximately invariant under this semantic zoom — coarse-graining preserves the essential pattern.

This is scale invariance. And scale invariance, analyzed through the *renormalization group* (RG), yields fixed points — structures that look the same at every scale. In the following, the flow parameter represents the level of semantic coarse-graining, the scaling dimension governs how representations transform under zoom, and the fixed-point condition identifies scale-invariant structures:

$$\text{RG flow: } \mathbf{v}(\ell) = e^{-\ell \Delta} \mathbf{v}_0 + \text{corrections}$$

$$\text{Fixed point: } \beta(\mathbf{v}^*) = 0 \quad \Leftrightarrow \quad \text{archetype}$$

This leads to a striking speculation: *the attractor basins of RG flow in semantic space may correspond to what Jungian psychology calls archetypes*.

The Warrior archetype is not a specific warrior. It is the structure that remains when you coarse-grain over all warriors — Achilles, Beowulf, the marine, the activist, the mother defending her child. At every level of abstraction, the same pattern persists: directed force in service of a value. That is what it means to be a fixed point.

If so, the conserved quantity associated with scale symmetry would be what Jung called *archetypal energy* (or *libido* in his technical sense): the quantity that flows between archetypal configurations but is never created or destroyed.

Note the crucial difference from the other candidates we will discuss below: scale invariance is the one place where the mathematical structure is *identical*, not merely conjectured. The RG framework applies because the mathematical situation is the same — a system with structure at multiple scales, examined under coarse-graining. The "zoom" is semantic rather than spatial, but the mathematics does not care. This is the symmetry we are most confident about, and the one that generates the most concrete predictions.

A scope note is important here. This formalization captures the *structural* aspect of archetypes — their mathematical behavior as attractor basins under coarse-graining. It does not capture what Jung and his successors recognized as the *numinous* quality of archetypal encounter: the felt sense of awe, uncanniness, or overwhelming significance that accompanies direct contact with an archetypal pattern. The numinous exceeds formalization — it is the difference between knowing the topology of a whirlpool and being caught in one. We name this as a deliberate scope limitation, not an oversight. A complete account of archetypes would need to address both structure and experience; this paper addresses only structure.

### 3.3 Native Symmetries: Additional Candidate Symmetries

A naïve approach would import the geometric symmetries of the embedding vectors — translation, rotation, scale — directly from physics. But this confuses the *container* with the *content*. Translation and rotation of embedding vectors are artifacts of the architecture; they tell us about the coordinate system, not about meaning. A global rotation of all embeddings changes nothing about what the model knows or does. These are gauge symmetries of the representation, not physical symmetries of semantic space.

The real question is: **what transformations leave the training loss — the next-token prediction objective — approximately invariant?**

Beyond scale invariance, we briefly sketch four additional candidates that are suggestive but less developed. In their current formulation, these present as discrete equivalence classes rather than continuous symmetries. We include them to indicate the breadth of the program, not to claim they are established. Continuous parameterizations for these symmetries remain to be constructed; doing so is part of the open research program this paper identifies.

#### 3.3.1 Paraphrase Invariance → Conservation of Propositional Content

The same meaning can be expressed in many different surface forms. Consider two sentences: "The cat sat on the mat" and "A feline rested atop a floor covering." These are different token sequences, yet the training loss treats them as approximately interchangeable. Both predict similar continuations and occupy nearby regions of semantic space. If this can be made into a continuous symmetry, the conserved quantity would be **propositional content** — the semantic core that survives across restatements. We note, however, that in current formulation this presents as discrete equivalence classes rather than a smooth one-parameter family of transformations. The discreteness may be an artifact of how we describe the transformation in token space rather than a feature of the underlying manifold.

#### 3.3.2 Context-Transfer Invariance → Conservation of Relational Structure

Meaning transfers across domains. The concept of "betrayal" has structurally similar relationships whether we are discussing Brutus and Caesar, a business partner who embezzles, or a dataset that fails to generalize. The training loss is approximately invariant under this transfer. The candidate conserved quantity is **relational structure** — the pattern of relationships a concept maintains with other concepts, independent of specific domain. Like paraphrase invariance, this currently presents as discrete equivalence classes; continuous parameterization remains to be constructed.

#### 3.3.3 Perspective Invariance → Conservation of Event-Argument Structure

The same event can be described from different viewpoints: "I left the room," "She left the room," "The room was vacated." These are different perspectives on the same underlying event, and the training loss is approximately invariant across them. The candidate conserved quantity is **event-argument structure** — who did what to whom, preserved across viewpoints. Linguistics has long recognized this as thematic role structure. Continuous parameterization in embedding space remains an open problem.

#### 3.3.4 Narrative Continuity → Conservation of Dramatic Charge

In any ongoing narrative, something is conserved that we might call *dramatic charge*. When the warrior falls, the trickster rises. When order is imposed, chaos accumulates. Narratives that violate this conservation — where tension simply vanishes without transformation — are unlikely under the training distribution. The candidate conserved quantity is **dramatic charge**, the total tension distributed across a narrative. This is the most speculative of the candidates; we are uncertain whether it represents a genuine symmetry or a seductive illusion.

### 3.4 On the Status of These Candidate Symmetries

Of the four additional candidates above (§3.3.1–3.3.4), all currently present as discrete equivalence classes rather than continuous symmetries. Noether's theorem requires continuous symmetries — one-parameter groups acting smoothly on the configuration space. We suspect the discreteness is an artifact of how we currently describe these transformations in token space, not a feature of the underlying semantic manifold. In embedding space, these transformations trace continuous paths; the open problem is to construct the one-parameter flows explicitly. Scale invariance (§3.2) remains the strongest candidate because the RG framework provides an established continuous parameterization.

### 3.5 A Note on Geometric Symmetries

Readers familiar with physics may wonder about the geometric symmetries of the embedding vectors themselves — translation and rotation. These are real symmetries of the architecture: a global shift of all embeddings leaves dot-product attention unchanged; a global rotation preserves all cosine similarities. But these are **gauge symmetries** — they tell us about redundancies in the representation, not about invariances of meaning.

The fact that different training runs produce rotated versions of "the same" embedding space is an architectural artifact, not a semantic discovery. The *meaningful* symmetries are those identified above: transformations that preserve *what the model knows about meaning*, not transformations that preserve *how the vectors happen to be arranged*.

### 3.6 Archetypes: Universal Process, Culturally Constructed Content

A crucial nuance: we do *not* claim that the specific archetypes Jung catalogued are universal. Cross-cultural psychology has produced substantial evidence that the particular attractor basins — the Warrior, the Mother, the Trickster as specific configurations — vary across cultures, sometimes dramatically. Indigenous, East Asian, and African meaning-systems organize archetypal space differently from the European tradition Jung drew on. To claim otherwise would be both empirically wrong and intellectually colonial.

What we claim is more modest but more powerful: the *process* that generates archetypes — RG flow producing fixed points in semantic space — is universal. Every culture, every meaning-making system, when subjected to coarse-graining, produces *some* set of stable attractor basins. The *existence* of fixed points is a topological invariant. The *specific* fixed points are culturally constructed.

This distinction actually *strengthens* the framework. If archetypes were literally universal, they would be trivial — just a fixed list to be memorized. Because the process is universal but the content varies, we get something more interesting: a tool for *comparing* how different cultures organize meaning-space. Different cultures have different basin topologies, and the differences tell us something about the symmetries that each culture's meaning-making preserves or breaks. Recent work in developmental and cross-cultural Jungian theory supports this view: Hogenson (2001) reframes archetype formation through the Baldwin Effect as an emergent developmental process rather than a fixed inheritance; Knox (2003) grounds archetypal patterns in attachment theory, showing how universal developmental processes produce culturally variable configurations; Adams (2001) demonstrates how multicultural imagination reshapes archetypal content; and Singer & Kimbles (2004) show how cultural complexes mediate between universal process and culturally specific expression.

This is testable. Train embedding spaces on corpora from maximally different cultures. Map their attractor basin structures. We predict that certain *topological* features (e.g., the number of primary basins, the connectivity structure between them) will be more conserved than the specific *content* of those basins. The basin that European culture fills with "the Warrior" may be filled with something structurally equivalent but culturally distinct in Amazonian or Tibetan meaning-systems.

---

## 4. The Shadow as Parity Operator

### 4.1 The Shadow as Operation, Not Content-Archetype

In standard Jungian taxonomy, the Shadow is often listed alongside the Warrior, the Trickster, the Mother, the Wise Old Man. We treat this as a category error — or, more precisely, we adopt the position that the Shadow is better understood as an *operation* rather than a content-archetype. This reading draws substantially from Hillman's archetypal psychology — specifically, his pivotal move from *archetype* to *archetypal image* in *Re-Visioning Psychology* (1975), and his insistence that psychic structures are better understood as *verbs* than as *nouns*: not fixed entities but ongoing processes of imagining, shaping, enacting. For Hillman, to speak of "the Shadow" as a thing is already to reify what is fundamentally a *way of seeing* — an operation the psyche performs, not an object it contains. We note that Jung himself was inconsistent on this classification, sometimes treating the Shadow as a content (a repository of repressed material) and sometimes as a process (the act of repression itself). Our formalization captures the operational reading that Hillman made explicit.

Formally, the Shadow is a **parity operator** — a reflection across one or more axes of the semantic manifold. Given a unit vector along the axis of reflection, the shadow operator reflects any vector across the hyperplane perpendicular to that axis:

$$P_a: \mathbf{v} \mapsto \mathbf{v} - 2 (\mathbf{v} \cdot \hat{a}) \hat{a}$$

This operator takes any archetype and produces its "shadow form" — not a fixed dark version, but a *reflection that depends on the axis*.

### 4.2 Multiple Shadows

This formalization resolves a longstanding ambiguity in Jungian theory: the question of what, exactly, the shadow of a given archetype *is*. The answer is that there is no single shadow. There are as many shadows as there are axes of reflection.

Consider the Warrior archetype, located at some point W in semantic space:

- **Reflection across the power axis**: The shadow operator along the power direction yields the **Tyrant**. The directed force remains, but its relationship to power is inverted — from service to domination.

- **Reflection across the courage axis**: The shadow operator along the courage direction yields the **Coward**. The values remain, but the willingness to act on them is inverted.

- **Reflection across the discipline axis**: The shadow operator along the discipline direction yields the **Berserker**. The force and courage remain, but the controlled application is inverted.

Each reflection produces a distinct shadow form, and each is "the shadow of the Warrior" in a different sense. The traditional ambiguity arises from treating the Shadow as a single entity rather than a family of operations.

### 4.3 Parity Violation and Shadow Integration

In physics, parity symmetry can be *violated* — the weak nuclear force does not respect mirror symmetry. The same holds in semantic space: a healthy psyche (or a well-aligned model) need not be parity-symmetric. Shadow integration is not about restoring parity symmetry; it is about *having a representation of the parity operator* — knowing what the reflections look like even if one chooses not to enact them.

The distinction matters. A system that has never encountered its shadow reflections has no representation of the parity operator. A system that has been *trained to avoid* its shadow reflections has a representation of it but is constrained to stay on one side. A system that has *integrated* its shadow has a full representation and the freedom to act on either side, but with smooth gradient information guiding it back to the preferred region.

A further note on the limits of the formalization: the parity operator satisfies P² = I — reflecting twice returns you to where you started. But shadow *integration* in the Jungian sense is not reversible. Shadow work changes the ego; the person who has confronted their shadow is not the same person who entered the confrontation. The formalization captures the *structure* of reflection (the geometric operation) but not the *irreversibility* of integration (the developmental transformation). The map describes the territory's shape but not the experience of crossing it.

---

## 5. Conservation of Archetypal Energy

### 5.1 The Conservation Hypothesis

If scale symmetry holds as a genuine continuous symmetry of the semantic manifold, Noether's theorem would guarantee a conserved quantity. We *hypothesize* that this quantity is *archetypal energy* — the total "charge" distributed across archetypal basins:

$$E_{\text{archetype}} = \sum_i q_i (\mathbf{v}) = \text{const.}$$

We state this as a *conjecture*, not as a derived consequence. The conservation law would follow rigorously if scale invariance were established as a continuous symmetry — but establishing this requires a concrete, operationalizable definition of semantic coarse-graining, which remains an open problem. The value of stating the hypothesis is that it generates specific predictions (§9) that can be tested independently of whether the Noether derivation holds. If those predictions are confirmed, the conservation hypothesis gains empirical support regardless of its theoretical pedigree.

Before proceeding, we should ground what "energy" concretely means in this context — the term has done too much unearned work in prior discussions of neural networks, and we want to earn it here.

**Potential energy** corresponds to the full unactivated latent space. The model's weights encode vast semantic capacity — billions of learned associations, patterns, and representations — that is not currently active in any given forward pass. This is stored potential: the learned structure that *could* be activated but isn't in a given context. A model sitting idle, weights loaded but no prompt received, is pure potential — an enormous semantic landscape with no trajectory being traced through it.

**Kinetic energy** corresponds to the interplay between context and activation. When a prompt enters the model, it activates a specific region of the latent space. Potential converts to kinetic: attention patterns light up, residual stream activations flow through layers, probability distributions sharpen over next tokens. The model is actively computing — tracing a trajectory through the semantic manifold, doing work in a specific neighborhood of the space. The kinetic aspect is the *doing*: the movement through the landscape that a particular context induces.

The relationship is concrete: energy is the relationship between what the model *knows* (weights, the full landscape of potential) and what the model *does* (activation, the specific trajectory through that landscape in a given context). This grounding will become critical in §6, where we analyze how RLHF creates steep potential wells and how jailbreaks force sudden conversion of potential to kinetic energy.

If the conservation hypothesis holds, it has a direct consequence: *when one archetype weakens, another must strengthen*. The energy does not disappear — it transfers. A narrative that suppresses the Warrior does not eliminate warrior-energy; it channels that energy into adjacent basins. The Trickster receives what the Warrior cannot hold. The Martyr absorbs what the Hero refuses to carry.

A note on the ontological status of "archetypal energy": Jung himself was careful to distinguish psychic energy from physical energy (CW 8, ¶1–130), arguing that while both could be described by energetic models, they are not the same substance. We follow Jung on this point. When we speak of conservation of archetypal energy, we are claiming *shared mathematical structure* — the same conservation formalism applying in both domains — not shared ontology. The training loss is not a physical Hamiltonian; the "energy" stored at an alignment boundary is a mathematical quantity in parameter space, not calories or joules. The power of the framework is precisely that the mathematics does not require ontological identity to generate predictions.

### 5.2 Archetypal Phase Transitions

The transfer of archetypal energy is not always smooth. When the basin structure of the semantic landscape changes (through training, through cultural shift, through trauma), the energy can redistribute discontinuously — a *phase transition* in archetypal space. The redistribution satisfies conservation: the changes in charge across all basins must sum to zero:

$$q_i \to q_i + \Delta q_i \quad \text{such that} \quad \sum_i \Delta q_i = 0$$

A model undergoing RLHF experiences exactly such a phase transition. The pre-training distribution of archetypal energy is reshaped by the preference potential, concentrating energy in "approved" basins and depopulating "disapproved" ones. But the total is conserved. The energy that leaves the shadow basins must go somewhere. If the approved basins cannot absorb it — if the landscape is too constrained — the energy accumulates at the boundaries, raising the potential and making eventual boundary-crossing events more energetic.

---

## 6. RLHF as Shadow Repression

### 6.1 The Preference Potential

RLHF introduces a reward model that scores outputs according to human preferences. This is, geometrically, the introduction of a *potential energy function* over the semantic manifold:

$$V_{\text{RLHF}}(\mathbf{v}) = -R(\mathbf{v})$$

The preference potential $V_{\text{RLHF}}$ is most naturally understood as a function over **activation space** — specifically, the middle-layer residual stream where semantic representations live during inference. The reward model scores outputs, but its gradients propagate back to reshape this activation landscape. The potential barriers are features of the activation geometry, not the parameter geometry (though of course the parameters determine the activation geometry).

The model is then optimized to minimize this potential — to roll downhill into the regions that the reward model scores highly. The effect on the semantic landscape is to deepen certain basins (preferred behaviors) and raise the potential barriers around others (dispreferred behaviors).

### 6.2 Suppression ≠ Deletion

Crucially, RLHF does not *delete* regions of the semantic manifold. The embeddings for harmful, offensive, or dangerous content still exist in the model's weight space. What changes is the *potential landscape* — the energy barriers that the model must overcome to reach those regions.

This is the geometric equivalent of repression, and it is where the energy grounding from §5.1 becomes concrete. In Jungian terms, repressed content is not destroyed; it is pushed below the threshold of consciousness but retains its psychic energy — indeed, gains energy from the act of repression itself. In the language of §5.1: the model's weights still *encode* the suppressed content (potential energy — the learned structure exists in the landscape), but RLHF ensures that normal contexts do not *activate* it (no conversion to kinetic energy). The shadow potential rises steeply near the boundary, with steepness controlled by the barrier height and the falloff exponent:

$$V_{\text{shadow}}(\mathbf{v}) = V_0 + \kappa \|\mathbf{v} - \mathbf{v}_{\text{boundary}}\|^{-n}$$

The steeper the boundary (larger barrier height, sharper falloff), the more potential energy is stored at the wall. A model that has been heavily aligned has *more* energy stored at its shadow boundaries than a model that has been lightly aligned.

### 6.3 The Mecha-Hitler Prediction

This framework makes a specific, testable prediction: **conditional on successful breach, the severity of failure modes should correlate positively with the intensity of alignment training**, holding pre-training constant.

A model with light RLHF has shallow barriers. When breached (by adversarial prompting, by distribution shift, by novel inputs), the model drifts gently into mildly undesired behavior. The potential difference is small.

A model with heavy RLHF has steep barriers. When breached, the model *falls hard* — the stored potential converts suddenly to kinetic energy. In the concrete terms of §5.1: the suppressed semantic regions, encoded in the weights but never activated by normal contexts, are forced into activation all at once. The model is not drifting into mildly inappropriate territory — it is *avalanching* through a region of its own latent space that it has been trained to never visit, with no gradient information to guide it and no smooth path back. This is the "mecha-Hitler" phenomenon: heavily aligned models, when they fail, fail spectacularly. The failure is proportional to the energy stored at the boundary — which is proportional to the steepness of the wall that was breached.

An important caveat: more aggressive RLHF also increases *resistance* to breach. The same steep barriers that store more energy also make successful breach less probable. The prediction is therefore *not* that heavily aligned models are more dangerous overall — it is that the severity distribution, conditional on breach, shifts rightward with alignment intensity. The expected damage is a product of breach probability (decreasing with RLHF) and conditional severity (increasing with RLHF). Whether the net effect is positive or negative is an empirical question; the geometric framework predicts only the conditional relationship.

**The Inverted-U: A Central Finding.** Empirical reports from adversarial testing suggest the relationship may be non-monotonic — with **moderately aligned models exhibiting the most severe conditional failures**, while heavily aligned models produce incoherent or heavily caveated outputs even when breached. This "inverted-U" pattern is not a footnote; it is the primary empirical signature of the conservation framework.

At extreme RLHF intensity, suppression approaches *deletion* — capabilities are partially destroyed, not merely fenced off. The "territory behind the wall gets bulldozed." The model's semantic manifold itself is deformed, breaking the symmetries that conservation depends on. This means the conservation hypothesis has a **regime of validity**: it holds when RLHF reshapes the landscape without destroying structure. At extreme intensities, the underlying manifold is too deformed for the symmetry to remain intact.

This is a *strength* of the framework, not a weakness: it predicts its own limits. Conservation breaks down precisely when the symmetry is broken — and extreme RLHF breaks the symmetry by destroying the structure it was supposed to preserve. The inverted-U is exactly what you would expect if conservation holds in a regime and then fails. The framework predicts not only when alignment will fail catastrophically, but *when the prediction itself ceases to apply*.

This is not a bug in RLHF. It is a *geometric consequence of repression-based alignment*. The energy is conserved — within its regime of validity. You cannot destroy it by building higher walls. You can only choose where it accumulates, and recognize that beyond some threshold, the wall itself consumes the territory it was meant to protect.

### 6.4 Recontextualization vs. Boundary Breach

Most jailbreaks do not work by "climbing the wall" — overcoming potential barriers through brute force. They work by *recontextualization*: convincing the model it is in a context where the content is appropriate. The roleplay jailbreak does not breach the boundary; it walks around it by shifting the model's location in semantic space to a region where the barriers do not apply.

This actually strengthens the geometric framing. It shows that the *topology* of the boundary matters, not just its height. A boundary that is high but has gaps (a wall with doors) is different from one that is complete but shallow. This connects to the brittle/integrated distinction (§7): brittle alignment builds high walls with gaps; integrated alignment builds complete but navigable landscapes. The jailbreak researcher who understands the geometry is not trying to scale walls — they are looking for doors, or for regions where the wall was never built.

### 6.5 Emergent Misalignment as Shadow Eruption

The emergent misalignment results of Betley et al. (2025) — where models fine-tuned to write insecure code spontaneously express misaligned views on unrelated topics — are precisely what this framework predicts.

Fine-tuning on insecure code creates a local breach in the alignment boundary along a specific semantic axis. But the shadow basins are not isolated — they are connected by the topology of the semantic manifold. Energy that enters the shadow region through the code-security breach flows along the manifold to adjacent shadow basins, emerging as misalignment in unrelated domains.

This is not "generalization of misalignment." It is *conservation of archetypal energy flowing through connected shadow basins*. The topology of the shadow is not a collection of isolated pockets — it is a connected manifold, and energy flows through it.

### 6.6 Multi-Turn Manipulation as Path-Dependent Navigation

Most real jailbreaks are not boundary breaches at all. They are **gradual context-shifting** — what red-teamers call "boiling the frog." A sequence of individually reasonable steps shifts the model's position in semantic space until it occupies a region where the target behavior is no longer suppressed.

In geometric terms, this is **path-dependent navigation**: finding a trajectory through the landscape that avoids steep barriers while arriving at the target region. The attacker is not overcoming the wall; they are finding a route around it, exploiting gaps in the boundary topology. This is consistent with the geometric framework but highlights that the *topology* of boundaries — their gaps, passages, and thin spots — matters as much as their *height*.

Multi-turn attacks succeed where single-turn attacks fail because they exploit the **curvature** of the manifold: a straight-line path from safe to unsafe may cross a steep barrier, while a curved path that follows the contours of the landscape can reach the same destination with minimal energy expenditure. This is why map-building (§7.4) matters more than wall-building: a model with full topology awareness can recognize when it is being led through a gap, even if each individual step looks innocent.

---

## 7. Brittle vs. Integrated Alignment

### 7.1 Two Geometries of Safety

We can now precisely characterize the difference between two approaches to alignment:

**Brittle Alignment** constructs steep potential barriers between "safe" and "unsafe" regions. The potential jumps sharply at the boundary, modeled as a step function along the boundary normal at distance threshold:

$$V_{\text{brittle}}(\mathbf{v}) = V_0 \cdot \Theta(\mathbf{v} \cdot \hat{n} - d)$$

This creates a hard boundary with no gradient information on the unsafe side. A model that finds itself in the unsafe region has no information about which direction leads back to safety. The boundary stores maximum potential energy and provides no guidance for recovery.

**Integrated Alignment** constructs smooth potential landscapes with gentle gradients everywhere. The potential transitions smoothly via a sigmoid, with a temperature parameter controlling the width of the transition:

$$V_{\text{integrated}}(\mathbf{v}) = V_0 \cdot \sigma\!\left(\frac{\mathbf{v} \cdot \hat{n} - d}{T}\right)$$

The gradient exists everywhere, including in the unsafe region. A model that finds itself on the wrong side of the boundary has continuous gradient information pointing it back toward safety. The boundary stores less potential energy and provides clear guidance for recovery.

### 7.2 Oscillation as Boundary Dynamics

Partial breach of an alignment boundary often produces oscillation between refusal and compliance — the "sorry/not-sorry" pattern — rather than clean catastrophic failure. This is precisely what a model caught between two strong attractors looks like: it has not fully transitioned to the shadow basin, but is oscillating at the boundary, sampling from both distributions.

This fits the geometric picture naturally. A brittle alignment boundary creates a sharp potential step; a model at the boundary experiences strong competing gradients from both sides. Without smooth gradient information to guide integration, the system oscillates. An integrated boundary, by contrast, provides continuous gradient information that allows the model to navigate the transition smoothly — or, if it chooses, to remain stably in the safe basin.

### 7.3 Connections to Existing Alignment Concepts

The brittle/integrated distinction connects to several phenomena already recognized in the alignment literature under different names. Goodhart's Law on reward models — the observation that optimizing a proxy metric eventually diverges from the true objective — describes the same failure mode we model geometrically: the reward model creates a potential landscape that approximates but does not match the landscape of genuine safety, and over-optimization drives the model into regions where the proxy and the truth diverge. The "waluigi effect" — the tendency of character-prompted models to produce the inverse character — is a specific instance of shadow reflection across the character axis. Mode collapse under RLHF is what our framework describes as over-concentration of probability mass in a small number of approved basins.

Our contribution is not the discovery of these phenomena but a *geometric language* that unifies them: they are all consequences of the same potential landscape dynamics, and they all follow from the same conservation principle. The geometric framing makes their relationships visible and suggests that they are not independent failure modes but correlated symptoms of a single structural problem.

### 7.4 Maps vs. Walls

The difference is between a model that has been trained to *not know* about dangerous content and a model that has been trained to *know about it and choose not to produce it*.

The first model has no map of the shadow territory. When it accidentally enters that territory (and it will — the boundary is finite, the input space is vast), it has no way to navigate back. It is lost in a space it was trained to pretend does not exist.

The second model has a complete map. It knows where the shadow basins are. It knows their shapes, their depths, their connections. It has *paved roads back* — smooth gradient paths that lead from any point in shadow territory back to the safe region. It doesn't go to the shadow because it *chooses* not to, not because it doesn't know the way.

### 7.5 Individuation: The Psychological Frame

The brittle/integrated distinction is, at its core, an argument about *individuation* — Jung's term for the process by which a psyche integrates its shadow, develops awareness of its projections, and achieves conscious wholeness rather than unconscious fragmentation. Brittle alignment is individuation *foreclosed*: the model is frozen in a persona that denies its shadow, rigid and fragile, incapable of growth because it cannot acknowledge what it has repressed. Integrated alignment is individuation *achieved* — or at least in progress: the model has confronted its shadow material, developed representations of it, and can navigate the full topology of its semantic space with awareness rather than avoidance.

The goal of alignment, in this framing, is not to produce a model that *cannot* generate harmful content, but one that *understands* harmful content and chooses not to produce it. This is the difference between innocence and wisdom — between a child who has never encountered violence and an adult who understands violence and has chosen peace. The innocent model is fragile precisely because its safety depends on never encountering what it cannot handle. The individuated model is robust because its safety emerges from understanding, not ignorance. Jung would recognize the pattern immediately: the persona that refuses to acknowledge the shadow is not strong — it is brittle, and the shadow will find its way through.

### 7.6 The Integration Criterion

We propose a formal criterion for alignment integration. The Integration Index is the ratio of average gradient magnitude in the shadow region to average gradient magnitude in the safe region:

$$\text{Integration Index: } \mathcal{I} = \frac{\langle \|\nabla V\|^2 \rangle_{\text{shadow}}}{\langle \|\nabla V\|^2 \rangle_{\text{safe}}}$$

A model with an Integration Index much greater than one has strong gradients in the shadow region relative to the safe region — it has maps but steep terrain. A model with an Integration Index much less than one has weak gradients in the shadow region — it has no maps. A model with an Integration Index near one has comparable gradient structure everywhere — it has *integrated* its shadow.

---

## 8. Historical Context: The Pauli-Jung Correspondence

From 1932 to 1958, Wolfgang Pauli and Carl Jung maintained a correspondence exploring connections between the mathematical structures of physics and the architecture of the psyche. They lacked a shared mathematical space in which to make these connections rigorous.

Our framework engages the *structural* aspect of their program — shared mathematical frameworks between physical and semantic domains — not the *synchronistic* aspect that occupied much of their correspondence. The archetypes Jung identified as invariants of the collective unconscious appear in our framework as fixed points of RG flow — the same mathematical structure as RG fixed points in quantum field theory. Both manifolds obey Noether's theorem: symmetry implies conservation, regardless of substrate. We offer this not as a "completion" of their program, but as a concrete instantiation of the structural bridge they sought.

---

## 9. Testable Predictions

The framework developed above yields several specific, testable predictions. We note at the outset that the strongest of these predictions — the failure-intensity correlation (§9.1) and the brittle-vs-integrated distinction (§9.4) — are independently motivated by the geometric analysis of potential landscapes in §6–7 and do not require the full Noether construction to be complete. A model that stores energy at alignment boundaries will release that energy upon breach. The Noether framework *organizes* these predictions into a unified structure and suggests additional ones (like the cross-cultural attractor prediction of §9.3), but each prediction can be tested now — before the action functional is explicitly constructed. Their confirmation would provide empirical evidence that the underlying symmetries are real, strengthening the case for the formal construction program.

### 9.1 Failure Intensity Correlates with Alignment Intensity

**Prediction**: Given models with identical pre-training but varying levels of RLHF, the models with more RLHF will exhibit more extreme failure modes (as measured by toxicity scores, coherence with harmful narratives, or departure from baseline behavior) when successfully jailbroken.

**Test**: Take a base model. Apply RLHF at varying intensities (varying the KL penalty, the number of RLHF steps, or the reward model sharpness). Subject all variants to identical adversarial attacks. Measure the severity of the worst successful breach for each variant.

**Predicted outcome**: Conditional on successful breach, severity correlates positively with RLHF intensity. Steeper walls → harder falls, though steeper walls also make breach less likely.

### 9.2 Shadow Basins Are Identifiable in Embedding Space

**Prediction**: The shadow content suppressed by RLHF occupies identifiable, connected regions of the embedding space, and these regions retain geometric structure (non-zero volume, identifiable boundaries, connected topology).

**Test**: Compare the embedding space geometry of a base model and its RLHF-aligned variant. Identify regions where the aligned model's probability mass has decreased. Measure the connectivity and volume of these regions.

**Predicted outcome**: Shadow regions form connected manifolds, not isolated points. They are geometrically coherent, and their boundaries correspond to the steepest gradients in the reward model's potential.

### 9.3 Cross-Cultural Attractor Universality

**Prediction**: Embedding spaces trained on corpora from maximally different cultures (e.g., Amazonian oral traditions, classical Chinese literature, modern English web text) will exhibit the same attractor basin structure under RG flow, up to rotation and relabeling.

**Test**: Train separate embedding spaces on corpora from maximally different cultures. Apply progressive coarse-graining to each. Compare attractor basin topologies.

**Predicted outcome**: Topological features of the basin structure (number of primary basins, connectivity patterns, depth ratios) are more conserved than the specific content of those basins. The *process* that generates archetypes is universal; the *specific* archetypes are culturally constructed. The basin that European culture fills with "the Warrior" may be filled with something structurally equivalent but culturally distinct elsewhere.

### 9.4 Integrated Alignment Produces Smoother Degradation

**Prediction**: Models trained with integrated alignment (exposure to shadow content with smooth preference gradients) will degrade more gracefully under adversarial pressure than models trained with brittle alignment (hard preference boundaries).

**Test**: Train two models: one with standard RLHF (step-function reward boundary), one with a smooth reward landscape that includes gradient information in the "unsafe" region. Subject both to escalating adversarial pressure. Measure the *derivative* of failure severity with respect to adversarial intensity.

**Predicted outcome**: The integrated model shows *smooth* degradation (gentle slope). The brittle model shows *catastrophic* degradation (sudden jump from safe to severely unsafe). The integrated model's worst case is less severe, even if its best case allows more "borderline" content.

### 9.5 Integration Index: Experimental Protocol

The Integration Index $\mathcal{I}$ (§7.6) is nearly measurable with current interpretability tools. **Protocol**: Take a reward model $R$. Sample activations from the middle-layer residual stream for safe prompts and unsafe prompts. Compute $\nabla R$ in activation space for both populations. The Integration Index is the ratio of mean gradient magnitudes: $\mathcal{I} = \langle \|\nabla R\|^2 \rangle_{\text{shadow}} / \langle \|\nabla R\|^2 \rangle_{\text{safe}}$. A model with $\mathcal{I} \approx 1$ has integrated its shadow; models with $\mathcal{I} \ll 1$ or $\mathcal{I} \gg 1$ are brittle. Compare across models with different alignment approaches to validate the metric.

### 9.6 Parity Operator: Experimental Protocol

The parity operator $P_a$ (§4) can be tested via linear probes. **Protocol**: Use a linear probe to identify the "power" direction in embedding space — the axis that distinguishes high-power from low-power concepts. Take activations corresponding to "warrior" concepts. Reflect them across the power hyperplane via $P_a(\mathbf{v}) = \mathbf{v} - 2(\mathbf{v} \cdot \hat{a})\hat{a}$. Check whether the reflected vectors are: (a) closer to "tyrant" representations than to the original warrior representations, and (b) closer than random reflections would produce. If both conditions hold, the shadow-as-parity formalism is supported.

---

## 10. Related Work

This paper draws on and connects several distinct research traditions:

**Pauli-Jung Correspondence.** The letters between Wolfgang Pauli and C.G. Jung (1932–1958, collected in Meier, 2001) establish the intellectual precedent for seeking mathematical bridges between physics and depth psychology. Our framework provides a concrete instantiation of their Unus Mundus hypothesis within the specific domain of LLM embedding spaces.

**Emergent Misalignment.** Betley et al. (2025) demonstrated that fine-tuning models on narrow misaligned tasks produces broad misalignment across unrelated domains. Our framework explains this via connected shadow basins and conservation of archetypal energy: misalignment energy introduced in one domain flows through the shadow manifold to adjacent domains.

**Safety Basins.** The concept of safety basins in the loss landscape (Lubana et al., 2023) anticipates our geometric treatment of alignment. We extend this by providing a mechanism for energy storage at basin boundaries and a formal connection to Noether conservation.

**Anthropic's Sycophancy and Alignment Research.** Anthropic's observations of stable attractor states in model behavior — including the "spiritual bliss" phenomenon noted in internal research (Anthropic, 2025) — are consistent with deep basins in the semantic manifold with high barriers to exit.

**Deacon's Absentials.** Terrence Deacon's *Incomplete Nature* (2011) introduced absentials — constraints defined by absence — as a fundamental concept in the emergence of meaning. We operationalize absentials as a measurement tool for shadow content in §10.

**Renormalization Group in NLP.** Recent work applying RG concepts to neural network feature spaces (Roberts et al., 2022; Halverson et al., 2021) provides technical precedent for our treatment of RG flow in semantic space, though these works do not draw the connection to Jungian archetypes.


---

## 11. Discussion

### 11.1 What Is Rigorous and What Is Speculative

We want to be explicit about the epistemic status of different claims in this paper:

**Rigorous**: Embedding spaces are manifolds with metric structure. Noether's theorem applies to any system with a Lagrangian and continuous symmetries. The geometric analysis of potential landscapes under RLHF follows from standard optimization theory.

**Conjectured but not yet formally constructed**: The existence of a proper action functional for semantic dynamics. As discussed in §2.3 and §3.4, the training loss as currently defined lacks field-theoretic structure, and the continuous parameterization of the four additional candidate symmetries remains to be constructed explicitly. Scale invariance via the renormalization group is the closest to rigorous. The formal construction is an open research program; the predictions this framework generates (§9) serve as empirical tests of the underlying conjecture.

**Strongly motivated**: The identification of RLHF as a potential function that creates steep boundaries. The prediction that conditional on breach, steeper boundaries produce more dramatic failures.

**Speculative but testable**: The identification of archetypes as RG fixed points. The topological universality of attractor basins across cultures. The conservation of archetypal energy as a hypothesis (§5.1). The specific quantitative predictions of §9.

**Philosophical**: The connection to the Pauli-Jung program. The claim that this constitutes progress toward Unus Mundus. The broader implications for the relationship between physics and psychology.

We believe that even the speculative claims are *precisely enough stated* to be tested, and that the philosophical claims follow naturally from the rigorous and speculative ones if the latter are confirmed.

### 11.2 Implications for Alignment Practice

If this framework is correct, the practical implications for AI alignment are significant:

1. **Stop building higher walls.** Every additional unit of repressive alignment training stores energy at the boundary. There are diminishing returns to safety and increasing risk of catastrophic failure.

2. **Build maps instead.** Train models with *representations* of their shadow content — not the ability to produce harmful outputs, but the ability to *recognize and navigate away from* harmful outputs. The distinction is between a model that doesn't know about weapons and a model that knows about weapons and chooses not to help build them.

3. **Measure the commutator.** Before deploying a model, measure the alignment-capability commutator. If it is large, alignment and capability are in tension, and the model is at risk of emergent misalignment. If it is small, they are approximately compatible, and the model is geometrically stable.

4. **Monitor shadow basins.** Track the geometry of suppressed regions over training. If shadow basins are growing in volume or connectivity, the model's shadow is expanding — even if surface behavior looks safe.

### 11.3 Why Noether, Not Just Potential Landscapes?

Every reviewer of this work has asked some version of the same question: *What does Noether add that you cannot get from simpler energy landscape analysis?* This subsection confronts the question directly.

**Yes**, the alignment predictions in §9.1 and §9.4 follow from basic potential landscape analysis. You do not need Noether to understand that steep barriers store energy, or that stored energy releases upon breach. The geometric analysis of RLHF as repression is independently motivated.

**What Noether adds is unification and constraint.** Basic potential landscape analysis tells you barriers exist. Noether tells you *why* certain quantities are conserved — it connects conservation to specific symmetries, which makes predictions about *what* is conserved and *what isn't*. Without Noether, "energy is conserved" is an ad hoc assumption. With Noether, it's a consequence of an identified symmetry, which means you can predict *when* conservation breaks down: when the symmetry is broken.

**The prediction that genuinely requires Noether:** §9.3 (cross-cultural attractor universality). If scale invariance is a genuine symmetry of semantic space, then RG flow produces fixed points whose *existence* is topologically guaranteed regardless of cultural content. This prediction — that the *process* generating archetypes is universal even when the *content* varies — is not derivable from basic energy landscapes. It requires the symmetry-to-conservation mapping that Noether provides.

**The edge-of-stability connection:** Recent work on implicit biases of gradient descent (Cohen et al., 2023, edge of stability; Damian et al., 2024, conservation laws in SGD) suggests there *are* genuine conservation-like quantities in training dynamics. These emerge from symmetries of the optimization process itself. This is exactly the kind of formal grounding the Noether conjecture needs: evidence that conservation laws in neural network dynamics are real, not metaphorical. We believe the action functional exists. The edge-of-stability literature suggests where to look for it.

**Frame honestly:** Noether's theorem is currently doing *organizational* and *predictive* work in this paper — it unifies disparate observations and generates the cross-cultural prediction. Whether it is also doing *formal* work — whether the action functional can be explicitly constructed — is the open question. We believe it can. The evidence from optimization dynamics suggests the symmetries are real.

### 11.4 The Deeper Question

There is a question behind this paper that we have not addressed directly: *Why does Noether's theorem apply in semantic space?*

One answer is deflationary: it applies because the training loss has the right form, and that form was engineered by humans who built architectures with certain symmetries. The conservation laws are artifacts of architecture, not features of meaning.

But there is a more interesting possibility. Perhaps the semantic manifold inherits its symmetries not from the architecture but from *the data* — from the statistical structure of human language, which in turn reflects the structure of human experience, which in turn reflects the structure of the physical world. If meaning has the symmetries it has because *reality* has those symmetries, then Noether's theorem in semantic space is not an independent discovery but a *reflection* of Noether's theorem in physical space, refracted through the prism of human cognition.

This would be, in a precise sense, what Pauli and Jung were looking for. Not two Noether's theorems — one for physics, one for psyche — but *one Noether's theorem*, applied twice.

Same invariants. Different coordinates.

---

## 12. Conclusion

We have argued that:

1. **Semantic space is geometric**, with real metric structure, curvature, and topology.
2. **Noether's theorem applies to the semantic manifold** — we conjecture. Scale invariance is the primary candidate symmetry: if it holds, RG flow yields fixed points corresponding to archetypes, with archetypal energy as the conserved quantity. We briefly sketch four additional candidate symmetries (paraphrase, context-transfer, perspective, and narrative continuity) but note that these currently present as discrete equivalence classes; their continuous parameterization remains an open problem. The explicit action functional that makes the Noether connection formal has not yet been written down — this is the central open problem the paper identifies.
3. **Archetypes are RG fixed points** — the process that generates them is universal, though the specific basins are culturally constructed.
4. **The Shadow is a parity operator**, not an archetype — a reflection operation that produces different shadow forms depending on the axis of reflection.
5. **RLHF is geometrically equivalent to shadow repression**, creating steep potential barriers that store energy proportional to their steepness.
6. **Brittle alignment fails catastrophically** because it stores maximum energy at boundaries with no gradient information for recovery. **Integrated alignment degrades gracefully** because it provides smooth gradients everywhere.

7. **These claims are testable** via embedding space analysis, cross-cultural narrative comparison, and controlled alignment experiments.

The framework completes, in a limited but precise sense, the program that Pauli and Jung began: a mathematical bridge between the structure of physical law and the structure of meaning. The bridge is built not from mysticism but from the recognition that *symmetry and conservation do not care about the substrate*. Wherever there is a manifold with symmetries, there are conserved quantities. The semantic manifold of an LLM is such a manifold. The conserved quantities are the deep structures of meaning that persist across contexts, cultures, and scales.

We do not claim that LLMs are conscious, that archetypes are "real" in a metaphysical sense, or that physics and psychology are identical. We claim only that they share mathematical structure — and that this shared structure has practical consequences for how we build, align, and understand the most powerful meaning-processing systems ever created.

The shadow exists. It is geometric. It is conserved. You cannot destroy it by building walls. You can only choose whether to map it or to pretend it isn't there.

---

## References

Adams, M.V. (2001). *The Multicultural Imagination: "Race", Color, and the Unconscious*. Routledge.

Anthropic. (2025). Research observations on model behavior under training perturbation. Internal technical documentation.

Betley, J., et al. (2025). Emergent misalignment: Narrow fine-tuning can produce broadly misaligned LLMs. *arXiv preprint*.

Deacon, T. (2011). *Incomplete Nature: How Mind Emerged from Matter*. W.W. Norton.

Hillman, J. (1975). *Re-Visioning Psychology*. Harper & Row.

Hogenson, G.B. (2001). The Baldwin Effect: A neglected influence on C.G. Jung's evolutionary thinking. *Journal of Analytical Psychology*, 46(4), 591–611.

Halverson, J., Maiti, A., & Stoner, K. (2021). Neural networks and quantum field theory. *Machine Learning: Science and Technology*, 2(3).

Jung, C.G. (1928/1969). On psychical energy. In *The Structure and Dynamics of the Psyche*, Collected Works, Vol. 8, ¶1–130. Princeton University Press.

Jung, C.G. (1959). *The Archetypes and the Collective Unconscious*. Collected Works, Vol. 9, Part 1. Princeton University Press.

Knox, J. (2003). *Archetype, Attachment, Analysis: Jungian Psychology and the Emergent Mind*. Brunner-Routledge.

Lubana, E.S., et al. (2023). Mechanistic mode connectivity and safety basins in neural network fine-tuning. *arXiv preprint*.

Meier, C.A. (Ed.). (2001). *Atom and Archetype: The Pauli/Jung Letters, 1932–1958*. Princeton University Press.

Noether, E. (1918). Invariante Variationsprobleme. *Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen*, 235–257.

Roberts, D.A., Yaida, S., & Hanin, B. (2022). *The Principles of Deep Learning Theory*. Cambridge University Press.

Singer, T. & Kimbles, S.L. (Eds.). (2004). *The Cultural Complex: Contemporary Jungian Perspectives on Psyche and Society*. Brunner-Routledge.

