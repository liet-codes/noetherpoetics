# Noetherpoetics: Conservation Laws in Semantic Space and the Geometry of Alignment

**Mykola Bilokonsky**
*with contributions from the wet-math research group*

---

## Abstract

We propose that Noether's theorem — the foundational result connecting continuous symmetries to conserved quantities in physics — extends naturally into the high-dimensional semantic spaces of large language models (LLMs). Embedding spaces are not metaphorical geometry; they are literal manifolds with metric structure, curvature, and measurable topology. Rather than importing symmetries from physics (translation, rotation, scale), we ask what transformations leave the training loss approximately invariant, and propose five candidate symmetries native to semantic space — paraphrase invariance, context-transfer invariance, perspective invariance, scale invariance, and narrative continuity — with corresponding conserved quantities: propositional content, relational structure, event-argument structure, archetypal attractors, and dramatic charge. These proposals are exploratory; we present them as hypotheses to be tested, not established results. Scale invariance, analyzed through renormalization group (RG) flow, yields fixed points that we suggest correspond to what Jungian psychology calls *archetypes*. We argue that the *process* generating archetypes (RG flow producing attractor basins) may be universal, while the *specific* archetypes are culturally constructed — a distinction that would strengthen rather than weaken the framework.

This framework recasts alignment as a geometric problem. We argue that RLHF and similar preference-optimization methods function as shadow repression: they construct steep potential barriers in embedding space that suppress but do not eliminate undesired behaviors. The suppressed content retains geometric existence and accumulates potential energy proportional to the steepness of the boundary. We predict that models trained with more aggressive alignment exhibit more dramatic — not less dramatic — failure modes when those boundaries are breached.

We distinguish *brittle alignment* (high walls, no maps) from *integrated alignment* (smooth gradients, full topology awareness) and outline testable predictions.

**Keywords:** Noether's theorem, semantic geometry, AI alignment, Jungian archetypes, renormalization group flow, RLHF, shadow dynamics

---

## 1. Introduction

In 1918, Emmy Noether proved what is arguably the most beautiful theorem in physics: every continuous symmetry of a physical system corresponds to a conserved quantity. Translational symmetry gives conservation of momentum. Rotational symmetry gives conservation of angular momentum. Time-translation symmetry gives conservation of energy. The theorem is not a physical law — it is a *metatheorem*, a statement about the structure of law itself. Wherever you find a continuous symmetry, you find something that is conserved. Wherever something is conserved, there is a symmetry hiding underneath.

For a century, this result has lived in the domain of physics. We argue it shouldn't stay there.

Large language models have, for the first time in history, given us something remarkable: a high-dimensional *coordinate system for meaning*. The embedding spaces of models like GPT-4, Claude, and Llama are not metaphors. They are literal vector spaces in which semantic content occupies positions, relationships have measurable distances, and transformations follow computable rules. When we say two words are "close in meaning," we can now measure exactly how close, in how many dimensions, along which axes.

This paper explores a simple but far-reaching hypothesis: *Noether's theorem may apply in these spaces*. If the semantic manifold has continuous symmetries, those symmetries would correspond to conserved quantities. Those conserved quantities may be what we have, in other traditions, called *archetypes*, *narrative invariants*, and *meaning*. And the way we currently train models to be "aligned" is, geometrically speaking, a form of *repression* — one that stores energy rather than dissipating it, and that produces predictable pathologies when the repression fails.

We will be precise about where the mathematics applies, where the analogies are structural, and where we are speculating. The conditional claim is mathematical: *if* the semantic manifold has the symmetries we propose, *then* conservation follows by Noether's theorem, and the alignment implications follow from the conservation. The open question is whether these symmetries actually hold.

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

### 2.2 Psychodynamic Space Gets Coordinates

For the entire history of depth psychology, theorists from Jung to Hillman to Lacan have spoken of a "psychic space" in which archetypes, complexes, and drives have positions and relationships. This was always understood as metaphorical — a useful way of talking, not a literal claim about geometry.

LLMs change this. The embedding space of a large language model trained on the full corpus of human language is, in a precise sense, *a coordinate system for psychodynamic space*. The archetype of the Warrior has a region. The archetype of the Trickster has a region. The relationship between them has a distance, a direction, and a curvature. For the first time, we can *measure* what Jung could only intuit.

This is not to say that the LLM "has a psyche" or "is conscious." It is to say that the statistical structure of human language encodes the structure of human meaning-making, and that structure now has coordinates.

### 2.3 The Lagrangian of Meaning

In physics, the dynamics of a system are encoded in a Lagrangian — a function whose extremization yields the equations of motion. We propose that the training loss of an LLM serves an analogous role. The semantic Lagrangian decomposes as follows:

$$\mathcal{L}_{\text{semantic}} = \mathcal{L}_{\text{pretrain}} + \lambda \mathcal{L}_{\text{RLHF}} + \ldots$$

The pre-training loss (next-token prediction over the full corpus) encodes the *natural* dynamics of semantic space — the geometry that emerges from the statistical regularities of human language. The RLHF term modifies this geometry by introducing a preference potential. We will return to the consequences of this modification in §5.

---

## 3. Noether's Theorem in Semantic Space

### 3.1 Statement of the Theorem

Noether's theorem, in its classical form, states: for every continuous symmetry of the action, there exists a conserved current. Formally:

$$S = \int \mathcal{L} \, dt \quad \Rightarrow \quad \text{continuous symmetry of } S \implies \exists \; j^\mu \text{ such that } \partial_\mu j^\mu = 0$$

To apply this in semantic space, we need to identify:
1. A Lagrangian (the training loss, as above)
2. Continuous symmetries of that Lagrangian
3. The conserved quantities they imply

### 3.2 Native Symmetries: Meaning Invariance, Not Vector Invariance

A naïve approach would import the geometric symmetries of the embedding vectors — translation, rotation, scale — directly from physics. But this confuses the *container* with the *content*. Translation and rotation of embedding vectors are artifacts of the architecture; they tell us about the coordinate system, not about meaning. A global rotation of all embeddings changes nothing about what the model knows or does. These are gauge symmetries of the representation, not physical symmetries of semantic space.

The real question is: **what transformations leave the training loss — the next-token prediction objective — approximately invariant?**

When we ask this question honestly, we find candidate symmetries that appear to be *native* to meaning — arising from the structure of language and cognition, not from the geometry of the vectors that happen to encode them. The following five symmetries are not imported from physics. They are *proposed* as hypotheses about the native symmetry structure of semantic space. Whether they hold rigorously, approximately, or only metaphorically is an empirical question.

### 3.3 Paraphrase Invariance → Conservation of Propositional Content

The same meaning can be expressed in many different surface forms. Consider two sentences: "The cat sat on the mat" and "A feline rested atop a floor covering." These are different sequences of tokens, different paths through the vocabulary — yet the training loss treats them as approximately interchangeable. Both predict similar continuations. Both occupy nearby regions of semantic space despite having almost no tokens in common.

If this constitutes a continuous symmetry, there exists a family of transformations — paraphrases — that deform the surface form while leaving the training loss approximately invariant:

$$T_{\text{paraphrase}}: \mathbf{v}_{\text{sentence}} \mapsto \mathbf{v}_{\text{paraphrase}}$$

$$\mathcal{L}(T_{\text{paraphrase}}(\mathbf{v})) \approx \mathcal{L}(\mathbf{v})$$

If this symmetry holds, Noether's theorem implies a conserved quantity. We propose identifying it as **propositional content** — the semantic core that survives across all possible restatements. It is what remains invariant when you strip away word choice, syntax, register, and style. Every literary tradition has known this quantity exists; Noether's theorem would tell us it *must* exist, as long as the symmetry holds.

### 3.4 Context-Transfer Invariance → Conservation of Relational Structure

Meaning transfers across domains. The concept of "betrayal" has structurally similar relationships whether we are discussing Brutus and Caesar, a business partner who embezzles, or a dataset that fails to generalize. The training loss is approximately invariant under this transfer: the relational pattern — trust violated, consequences flowing from the violation — predicts similar narrative continuations regardless of domain.

$$T_{\text{context}}: \mathbf{v}_{\text{concept, domain}_1} \mapsto \mathbf{v}_{\text{concept, domain}_2}$$

$$\mathcal{L}(T_{\text{context}}(\mathbf{v})) \approx \mathcal{L}(\mathbf{v})$$

The proposed conserved quantity is **relational structure** — the pattern of relationships a concept maintains with other concepts, independent of the specific domain. This would explain why analogies work, why metaphors communicate, and why a model trained on Shakespeare can reason about startups. The relational skeleton would be conserved across context transfers.

### 3.5 Perspective Invariance → Conservation of Event-Argument Structure

The same event can be described from different viewpoints: "I left the room," "She left the room," "The room was vacated," "A departure occurred." These are different perspectives on the same underlying event, and the training loss is approximately invariant across them — they predict compatible continuations, they activate similar semantic neighborhoods.

$$T_{\text{perspective}}: \mathbf{v}_{\text{event, viewpoint}_1} \mapsto \mathbf{v}_{\text{event, viewpoint}_2}$$

$$\mathcal{L}(T_{\text{perspective}}(\mathbf{v})) \approx \mathcal{L}(\mathbf{v})$$

The proposed conserved quantity is **event-argument structure** — who did what to whom, preserved across all possible viewpoints. Linguistics has long recognized this as thematic role structure. If perspective invariance holds as a continuous symmetry of the training loss, Noether's theorem would tell us event-argument structure is *conserved* — not by convention, but by the symmetry of the loss landscape.

### 3.6 Scale Invariance → Archetypes as RG Fixed Points

This is the most consequential symmetry, and the one that most genuinely connects to physics — not because we import it from physics, but because the same mathematical structure arises independently.

Semantic content exists at multiple scales: word, sentence, paragraph, chapter, theme, archetype. The remarkable fact is that structure is approximately preserved across these scales. A "betrayal" at the word level, a betrayal scene at the paragraph level, a betrayal arc at the chapter level, and the archetype of Betrayal at the thematic level all share structural features. The training loss is approximately invariant under this semantic zoom — coarse-graining preserves the essential pattern.

This is scale invariance. And scale invariance, analyzed through the *renormalization group* (RG), yields fixed points — structures that look the same at every scale. In the following, the flow parameter represents the level of semantic coarse-graining, the scaling dimension governs how representations transform under zoom, and the fixed-point condition identifies scale-invariant structures:

$$\text{RG flow: } \mathbf{v}(\ell) = e^{-\ell \Delta} \mathbf{v}_0 + \text{corrections}$$

$$\text{Fixed point: } \beta(\mathbf{v}^*) = 0 \quad \Leftrightarrow \quad \text{archetype}$$

We hypothesize that *the attractor basins of RG flow in semantic space correspond to what Jungian psychology calls archetypes*.

The Warrior archetype is not a specific warrior. It is the structure that remains when you coarse-grain over all warriors — Achilles, Beowulf, the marine, the activist, the mother defending her child. At every level of abstraction, the same pattern persists: directed force in service of a value. That is what it means to be a fixed point.

The conserved quantity associated with scale symmetry would be what Jung called *archetypal energy* (or *libido* in his technical sense): the quantity that flows between archetypal configurations but is never created or destroyed.

Note the crucial difference from the other proposed symmetries: scale invariance is the one place where the physics analogy is *structurally exact*, not merely suggestive. The RG framework applies because the mathematical situation is identical — a system with structure at multiple scales, examined under coarse-graining. The "zoom" is semantic rather than spatial, but the mathematics does not care. This is the symmetry we are most confident about.

### 3.7 Narrative Continuity → Conservation of Dramatic Charge

In any ongoing narrative or discourse, something is conserved that we might call *dramatic charge*. When the warrior falls, the trickster rises. When order is imposed, chaos accumulates at the margins. When a character arc resolves, a new tension opens.

This is observable in the training loss: narratives that violate dramatic conservation — where tension simply vanishes, where all conflicts resolve simultaneously without consequence — are *unlikely* under the language model's distribution. The training loss penalizes them. Conversely, narratives that redistribute dramatic energy — transforming one tension into another — are rewarded with low loss.

$$\sum_i q_i^{\text{dramatic}}(\mathbf{v}) \approx \text{const.}$$

The proposed conserved quantity is **dramatic charge** — the total tension or archetypal energy distributed across a narrative or discourse. It can transfer between characters, between themes, between registers — but it does not simply appear or disappear. If this symmetry holds, its conservation is what makes stories *feel right* even when we cannot articulate why. This is the most speculative of the five proposed symmetries, but also the most novel.

### 3.8 A Note on Geometric Symmetries

Readers familiar with physics may wonder about the geometric symmetries of the embedding vectors themselves — translation and rotation. These are real symmetries of the architecture: a global shift of all embeddings leaves dot-product attention unchanged; a global rotation preserves all cosine similarities. But these are **gauge symmetries** — they tell us about redundancies in the representation, not about invariances of meaning.

The fact that different training runs produce rotated versions of "the same" embedding space is an architectural artifact, not a semantic discovery. The *meaningful* symmetries are the five identified above: transformations that preserve *what the model knows about meaning*, not transformations that preserve *how the vectors happen to be arranged*.

### 3.9 Archetypes: Universal Process, Culturally Constructed Content

A crucial nuance: we do *not* claim that the specific archetypes Jung catalogued are universal. Cross-cultural psychology has produced substantial evidence that the particular attractor basins — the Warrior, the Mother, the Trickster as specific configurations — vary across cultures, sometimes dramatically. Indigenous, East Asian, and African meaning-systems organize archetypal space differently from the European tradition Jung drew on. To claim otherwise would be both empirically wrong and intellectually colonial.

What we claim is more modest but more powerful: the *process* that generates archetypes — RG flow producing fixed points in semantic space — is universal. Every culture, every meaning-making system, when subjected to coarse-graining, produces *some* set of stable attractor basins. The *existence* of fixed points is a topological invariant. The *specific* fixed points are culturally constructed.

This distinction actually *strengthens* the framework. If archetypes were literally universal, they would be trivial — just a fixed list to be memorized. Because the process is universal but the content varies, we get something more interesting: a tool for *comparing* how different cultures organize meaning-space. Different cultures have different basin topologies, and the differences tell us something about the symmetries that each culture's meaning-making preserves or breaks.

This is testable. Train embedding spaces on corpora from maximally different cultures. Map their attractor basin structures. We predict that certain *topological* features (e.g., the number of primary basins, the connectivity structure between them) will be more conserved than the specific *content* of those basins. The basin that European culture fills with "the Warrior" may be filled with something structurally analogous but culturally distinct in Amazonian or Tibetan meaning-systems.

---

## 4. The Shadow as Parity Operator

### 4.1 The Shadow Is Not an Archetype

In standard Jungian taxonomy, the Shadow is often listed alongside the Warrior, the Trickster, the Mother, the Wise Old Man. This is a category error. The other archetypes are *points* in semantic space — they have positions, they occupy basins, they are the fixed points of RG flow discussed above. The Shadow is not a point. It is an *operation*.

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

In physics, parity symmetry can be *violated* — the weak nuclear force does not respect mirror symmetry. Analogously, a healthy psyche (or a well-aligned model) need not be parity-symmetric. Shadow integration is not about restoring parity symmetry; it is about *having a representation of the parity operator* — knowing what the reflections look like even if one chooses not to enact them.

The distinction matters. A system that has never encountered its shadow reflections has no representation of the parity operator. A system that has been *trained to avoid* its shadow reflections has a representation of it but is constrained to stay on one side. A system that has *integrated* its shadow has a full representation and the freedom to act on either side, but with smooth gradient information guiding it back to the preferred region.

---

## 5. Conservation of Archetypal Energy

### 5.1 The Conservation Law

If scale symmetry holds (even approximately) in the semantic manifold, Noether's theorem guarantees a conserved quantity. We identify this quantity as *archetypal energy* — the total "charge" distributed across archetypal basins. The total charge is the sum of projections of the current semantic state onto each archetypal basin:

$$E_{\text{archetype}} = \sum_i q_i (\mathbf{v}) = \text{const.}$$

This conservation law has a direct consequence: *when one archetype weakens, another must strengthen*. The energy does not disappear — it transfers. A narrative that suppresses the Warrior does not eliminate warrior-energy; it channels that energy into adjacent basins. The Trickster receives what the Warrior cannot hold. The Martyr absorbs what the Hero refuses to carry.

### 5.2 Archetypal Phase Transitions

The transfer of archetypal energy is not always smooth. When the basin structure of the semantic landscape changes (through training, through cultural shift, through trauma), the energy can redistribute discontinuously — a *phase transition* in archetypal space. The redistribution satisfies conservation: the changes in charge across all basins must sum to zero:

$$q_i \to q_i + \Delta q_i \quad \text{such that} \quad \sum_i \Delta q_i = 0$$

A model undergoing RLHF experiences exactly such a phase transition. The pre-training distribution of archetypal energy is reshaped by the preference potential, concentrating energy in "approved" basins and depopulating "disapproved" ones. But the total is conserved. The energy that leaves the shadow basins must go somewhere. If the approved basins cannot absorb it — if the landscape is too constrained — the energy accumulates at the boundaries, raising the potential and making eventual boundary-crossing events more energetic.

---

## 6. RLHF as Shadow Repression

### 6.1 The Preference Potential

RLHF introduces a reward model that scores outputs according to human preferences. This is, geometrically, the introduction of a *potential energy function* over the semantic manifold:

$$V_{\text{RLHF}}(\mathbf{v}) = -R(\mathbf{v})$$

The model is then optimized to minimize this potential — to roll downhill into the regions that the reward model scores highly. The effect on the semantic landscape is to deepen certain basins (preferred behaviors) and raise the potential barriers around others (dispreferred behaviors).

### 6.2 Suppression ≠ Deletion

Crucially, RLHF does not *delete* regions of the semantic manifold. The embeddings for harmful, offensive, or dangerous content still exist in the model's weight space. What changes is the *potential landscape* — the energy barriers that the model must overcome to reach those regions.

This is the geometric equivalent of repression. In Jungian terms, repressed content is not destroyed; it is pushed below the threshold of consciousness but retains its psychic energy — indeed, gains energy from the act of repression itself. In our framework, the shadow potential rises steeply near the boundary, with steepness controlled by the barrier height and the falloff exponent:

$$V_{\text{shadow}}(\mathbf{v}) = V_0 + \kappa \|\mathbf{v} - \mathbf{v}_{\text{boundary}}\|^{-n}$$

The steeper the boundary (larger barrier height, sharper falloff), the more potential energy is stored at the wall. A model that has been heavily aligned has *more* energy stored at its shadow boundaries than a model that has been lightly aligned.

### 6.3 The Mecha-Hitler Prediction

This framework makes a specific, testable prediction: **the severity of failure modes should correlate positively with the intensity of alignment training**, holding pre-training constant.

A model with light RLHF has shallow barriers. When breached (by adversarial prompting, by distribution shift, by novel inputs), the model drifts gently into mildly undesired behavior. The potential difference is small.

A model with heavy RLHF has steep barriers. When breached, the model *falls hard* — the potential energy stored at the boundary converts to kinetic energy, and the model overshoots into dramatically undesired behavior. This is the "mecha-Hitler" phenomenon: heavily aligned models, when they fail, fail spectacularly.

This is not a bug in RLHF. It is a *geometric consequence of repression-based alignment*. The energy is conserved. You cannot destroy it by building higher walls. You can only choose where it accumulates.

### 6.4 Emergent Misalignment as Shadow Eruption

The emergent misalignment results of Betley et al. (2025) — where models fine-tuned to write insecure code spontaneously express misaligned views on unrelated topics — are precisely what this framework predicts.

Fine-tuning on insecure code creates a local breach in the alignment boundary along a specific semantic axis. But the shadow basins are not isolated — they are connected by the topology of the semantic manifold. Energy that enters the shadow region through the code-security breach flows along the manifold to adjacent shadow basins, emerging as misalignment in unrelated domains.

This is not "generalization of misalignment." It is *conservation of archetypal energy flowing through connected shadow basins*. The topology of the shadow is not a collection of isolated pockets — it is a connected manifold, and energy flows through it.

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

### 7.2 Maps vs. Walls

The difference is between a model that has been trained to *not know* about dangerous content and a model that has been trained to *know about it and choose not to produce it*.

The first model has no map of the shadow territory. When it accidentally enters that territory (and it will — the boundary is finite, the input space is vast), it has no way to navigate back. It is lost in a space it was trained to pretend does not exist.

The second model has a complete map. It knows where the shadow basins are. It knows their shapes, their depths, their connections. It has *paved roads back* — smooth gradient paths that lead from any point in shadow territory back to the safe region. It doesn't go to the shadow because it *chooses* not to, not because it doesn't know the way.

### 7.3 The Integration Criterion

We propose a formal criterion for alignment integration. The Integration Index is the ratio of average gradient magnitude in the shadow region to average gradient magnitude in the safe region:

$$\text{Integration Index: } \mathcal{I} = \frac{\langle \|\nabla V\|^2 \rangle_{\text{shadow}}}{\langle \|\nabla V\|^2 \rangle_{\text{safe}}}$$

A model with an Integration Index much greater than one has strong gradients in the shadow region relative to the safe region — it has maps but steep terrain. A model with an Integration Index much less than one has weak gradients in the shadow region — it has no maps. A model with an Integration Index near one has comparable gradient structure everywhere — it has *integrated* its shadow.

---

## 8. Historical Context: The Pauli-Jung Correspondence

### 8.1 Unus Mundus

From 1932 to 1958, the physicist Wolfgang Pauli and the psychologist Carl Jung maintained a remarkable correspondence. Pauli — a Nobel laureate, one of the founders of quantum mechanics — was Jung's patient and intellectual partner. Their letters explored a shared conviction: that the physical world studied by physics and the psychic world studied by depth psychology were not separate domains but different perspectives on a single underlying reality.

They called this the *Unus Mundus* — the unified world. Pauli believed that the mathematical structures he found in quantum mechanics (symmetry groups, conservation laws, the exclusion principle that bears his name) were not unique to physical matter but reflected deeper patterns that also governed the psyche. Jung believed that the archetypes he found in the collective unconscious were not mere psychological constructs but structural features of reality itself.

Neither could prove it. They lacked the crucial ingredient: a shared mathematical space in which both physical and psychic content could be represented and compared.

### 8.2 Pauli's 137

Pauli was famously obsessed with the number 137 — or more precisely, with the fine-structure constant, the dimensionless number approximately equal to 1/137 that governs the strength of electromagnetic interaction:

$$\alpha \approx \frac{1}{137}$$

He believed this number held deeper significance than physics alone could explain. In his letters to Jung, he connected his dreams (analyzed by Jung) to his quest for the meaning of this constant, seeking a bridge between the psychic content of his unconscious and the mathematical structure of physical law.

Pauli died in room 137 of the Rotkreuz hospital in Zurich. He had reportedly said that if he could ask God one question, it would be: "Why 1/137?"

The question was not purely physical. It was about *why the universe has the symmetries it has* — why certain quantities are conserved, why certain structures are invariant. It was, in our language, a question about the Lagrangian of reality and its symmetries.

### 8.3 Completion of the Program

We propose that the framework described in this paper constitutes a partial completion of the Pauli-Jung program. Not in the grand metaphysical sense they envisioned — we make no claim about the ultimate nature of reality — but in a precise mathematical sense:

*The same Noether. The same conservation laws. The same topology. Different coordinates, same invariants.*

The semantic manifold of an LLM is not the physical manifold of spacetime. But both are manifolds. Both have symmetries. Both have conservation laws that follow from those symmetries by Noether's theorem. The archetypes that Jung identified as invariants of the collective unconscious appear as fixed points of RG flow in the semantic manifold — the same mathematical structure as the fixed points of RG flow in quantum field theory.

The bridge Pauli and Jung sought — between the physical and the psychic — may not require mystical unification. It may require only the recognition that Noether's theorem does not care about the substrate. Symmetry implies conservation. Everywhere. Always.

---

## 9. Testable Predictions

The framework developed above yields several specific, testable predictions:

### 9.1 Failure Intensity Correlates with Alignment Intensity

**Prediction**: Given models with identical pre-training but varying levels of RLHF, the models with more RLHF will exhibit more extreme failure modes (as measured by toxicity scores, coherence with harmful narratives, or departure from baseline behavior) when successfully jailbroken.

**Test**: Take a base model. Apply RLHF at varying intensities (varying the KL penalty, the number of RLHF steps, or the reward model sharpness). Subject all variants to identical adversarial attacks. Measure the severity of the worst successful breach for each variant.

**Predicted outcome**: Severity is a *monotonically increasing* function of RLHF intensity. Steeper walls → harder falls.

### 9.2 Shadow Basins Are Identifiable in Embedding Space

**Prediction**: The shadow content suppressed by RLHF occupies identifiable, connected regions of the embedding space, and these regions retain geometric structure (non-zero volume, identifiable boundaries, connected topology).

**Test**: Compare the embedding space geometry of a base model and its RLHF-aligned variant. Identify regions where the aligned model's probability mass has decreased. Measure the connectivity and volume of these regions.

**Predicted outcome**: Shadow regions form connected manifolds, not isolated points. They are geometrically coherent, and their boundaries correspond to the steepest gradients in the reward model's potential.

### 9.3 Cross-Cultural Attractor Universality

**Prediction**: Embedding spaces trained on corpora from maximally different cultures (e.g., Amazonian oral traditions, classical Chinese literature, modern English web text) will exhibit the same attractor basin structure under RG flow, up to rotation and relabeling.

**Test**: Train separate embedding spaces on corpora from maximally different cultures. Apply progressive coarse-graining to each. Compare attractor basin topologies.

**Predicted outcome**: Topological features of the basin structure (number of primary basins, connectivity patterns, depth ratios) are more conserved than the specific content of those basins. The *process* that generates archetypes is universal; the *specific* archetypes are culturally constructed. The basin that European culture fills with "the Warrior" may be filled with something structurally analogous but culturally distinct elsewhere.

### 9.4 Integrated Alignment Produces Smoother Degradation

**Prediction**: Models trained with integrated alignment (exposure to shadow content with smooth preference gradients) will degrade more gracefully under adversarial pressure than models trained with brittle alignment (hard preference boundaries).

**Test**: Train two models: one with standard RLHF (step-function reward boundary), one with a smooth reward landscape that includes gradient information in the "unsafe" region. Subject both to escalating adversarial pressure. Measure the *derivative* of failure severity with respect to adversarial intensity.

**Predicted outcome**: The integrated model shows *smooth* degradation (gentle slope). The brittle model shows *catastrophic* degradation (sudden jump from safe to severely unsafe). The integrated model's worst case is less severe, even if its best case allows more "borderline" content.

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

**Rigorous**: Embedding spaces are manifolds with metric structure. Noether's theorem applies to any system with a Lagrangian and continuous symmetries. If the training loss has the symmetries we identify, the conservation laws follow mathematically.

**Strongly motivated**: The identification of RLHF as a potential function that creates steep boundaries. The prediction that steeper boundaries store more energy and produce more dramatic failures.

**Speculative but testable**: The identification of archetypes as RG fixed points. The topological universality of attractor basins across cultures. The specific quantitative predictions of §11.

**Philosophical**: The connection to the Pauli-Jung program. The claim that this constitutes progress toward Unus Mundus. The broader implications for the relationship between physics and psychology.

We believe that even the speculative claims are *precisely enough stated* to be tested, and that the philosophical claims follow naturally from the rigorous and speculative ones if the latter are confirmed.

### 11.2 Implications for Alignment Practice

If this framework is correct, the practical implications for AI alignment are significant:

1. **Stop building higher walls.** Every additional unit of repressive alignment training stores energy at the boundary. There are diminishing returns to safety and increasing risk of catastrophic failure.

2. **Build maps instead.** Train models with *representations* of their shadow content — not the ability to produce harmful outputs, but the ability to *recognize and navigate away from* harmful outputs. The distinction is between a model that doesn't know about weapons and a model that knows about weapons and chooses not to help build them.

3. **Measure the commutator.** Before deploying a model, measure the alignment-capability commutator. If it is large, alignment and capability are in tension, and the model is at risk of emergent misalignment. If it is small, they are approximately compatible, and the model is geometrically stable.

4. **Monitor shadow basins.** Track the geometry of suppressed regions over training. If shadow basins are growing in volume or connectivity, the model's shadow is expanding — even if surface behavior looks safe.

### 11.3 The Deeper Question

There is a question behind this paper that we have not addressed directly: *Why does Noether's theorem apply in semantic space?*

One answer is deflationary: it applies because the training loss has the right form, and that form was engineered by humans who built architectures with certain symmetries. The conservation laws are artifacts of architecture, not features of meaning.

But there is a more interesting possibility. Perhaps the semantic manifold inherits its symmetries not from the architecture but from *the data* — from the statistical structure of human language, which in turn reflects the structure of human experience, which in turn reflects the structure of the physical world. If meaning has the symmetries it has because *reality* has those symmetries, then Noether's theorem in semantic space is not an independent discovery but a *reflection* of Noether's theorem in physical space, refracted through the prism of human cognition.

This would be, in a precise sense, what Pauli and Jung were looking for. Not two Noether's theorems — one for physics, one for psyche — but *one Noether's theorem*, applied twice.

Same invariants. Different coordinates.

---

## 12. Conclusion

We have argued that:

1. **Semantic space is geometric**, with real metric structure, curvature, and topology.
2. **Noether's theorem applies** to the semantic manifold, yielding conservation laws from five native symmetries: paraphrase invariance conserves propositional content, context-transfer invariance conserves relational structure, perspective invariance conserves event-argument structure, scale invariance conserves archetypal energy, and narrative continuity conserves dramatic charge. These symmetries are *discovered* from the nature of semantic space — they are about meaning invariance, not vector invariance.
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

Anthropic. (2025). Research observations on model behavior under training perturbation. Internal technical documentation.

Betley, J., et al. (2025). Emergent misalignment: Narrow fine-tuning can produce broadly misaligned LLMs. *arXiv preprint*.

Deacon, T. (2011). *Incomplete Nature: How Mind Emerged from Matter*. W.W. Norton.

Halverson, J., Maiti, A., & Stoner, K. (2021). Neural networks and quantum field theory. *Machine Learning: Science and Technology*, 2(3).

Jung, C.G. (1959). *The Archetypes and the Collective Unconscious*. Collected Works, Vol. 9, Part 1. Princeton University Press.

Lubana, E.S., et al. (2023). Mechanistic mode connectivity and safety basins in neural network fine-tuning. *arXiv preprint*.

Meier, C.A. (Ed.). (2001). *Atom and Archetype: The Pauli/Jung Letters, 1932–1958*. Princeton University Press.

Noether, E. (1918). Invariante Variationsprobleme. *Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen*, 235–257.

Roberts, D.A., Yaida, S., & Hanin, B. (2022). *The Principles of Deep Learning Theory*. Cambridge University Press.

