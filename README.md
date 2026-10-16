# Energy Semiotics of Molecules (ESM)
### A field-based search for recurring units and rules in chemistry: a testable hypothesis about an executable 3D language

---

## Prologue: what is left if we start not from the atom, but from the field?

Imagine that in front of us is not a structural formula but a large archive of three-dimensional molecular fields. We do not yet know where the boundaries of atoms run in them, what should count as a bond, or which regions matter more than others. The question of ESM is simple in form and radical in its consequences: **can the recurring units and organizing rules of chemistry be recovered if they are sought directly in the fields, without giving the algorithm an atomic markup?**

Chemistry is accustomed to starting from the atom, and this descriptive language has been extraordinarily fruitful. ESM does not declare atoms a mistake and does not deny known chemical regularities. It tests whether the atomic picture is one of the stable ways of decomposing a more primary spatial organization. Perhaps a blind search will return atomic basins. Perhaps it will find parts of basins, structures spanning several atoms, or combinations of field features with the shape of voids. And perhaps different levels will turn out to be valid at the same time. None of these answers can be assigned in advance.

By "field" we mean here not a vague image and not a single universal energy function, but a quantity defined at every point of three-dimensional space. Examples: electron density, a measure of how electronic charge is distributed in space; electrostatic potential, a characteristic of the electrical influence at each point; the electron localization function (ELF), a field that helps to describe regions of electron localization. Only fields with an explicitly stated method of calculation and preserved data are used. Local energy density, if it becomes necessary, requires a separate definition and a representation convention. The word "energy" in the project's name does not mean that all these quantities are one and the same energy. The source fields may be computed quantum-chemically; the abandonment of atoms refers to the **logic of pattern discovery**, not to a denial that atomic nuclei participate in constructing a physical system.

Why speak of decipherment, then? Not because a molecule is literally written in letters. We borrow the discipline of the question: when the internal structure of a record is unknown, one first establishes which units recur, where and with what they occur, which variants are contextual and which are different units; only then are function and rules of combination tested. In ESM the analogue of a text becomes a corpus of fields, a "letter" is a yet-unknown stable object of the field, a "word" is a reproducible motif, and "grammar" is the regularities of composition and transition. These are working names for the levels of the search, not a claim that the corresponding objects have already been found.

There is one more principled step here. A chemical pattern may not only describe a finished molecule: under suitable conditions it may take part in the very change of the field. A static dictionary is therefore only the beginning. The question of "execution" requires ordered states and trajectories: how the field changes during a reaction, binding, collision, or a change of environment or conformation. Obtaining such sequences may become a separate, especially resource-intensive part of the project: the fields of the initial and final states alone are not enough if it is important to understand the course of the rearrangement. At the same time, it cannot be decided in advance whether transition fields will turn out to be evidence of composition rules, of variability of word forms, a refinement of the letters themselves, or the grammar of execution; the data may change the understanding of several levels at once. Without such data one can study stable states and their combinations, but one cannot pass off a map of static similarities as a discovered dynamic grammar.

This program does not ask the reader to believe a beautiful analogy. It proposes a sequence of checks. First, blind extraction of field features, for example regions where its local shape changes; then comparison arranged so that a rotation or shift of the whole molecule is not in itself counted as a new pattern; next, a search for repeatability, combinations, and contextual variants; after that, a check of the link to known properties and, where trajectories are available, to transition rules. Here topology is a way of describing how field features are connected with one another; statistics helps to distinguish stable repetition from chance; information theory provides a measure of how far recurring units shorten the description of the corpus; dynamics is needed to study changes in time. At the first stage a neural network must not replace the discovery itself: explicit mathematical procedures make it possible to verify where each found unit came from. Larger models may be used later, as a tool and not as a predetermined answer.

**An empty output is not counted as a discovery about the absence of chemical organization.** If the analysis returns no structures, no stable relations, and no patterns at all, this means that the current logic, representation, resolution, or settings have not yet extracted the known organization of the field. The search must be broadened or the procedure revised: perhaps it is looking for the wrong class of objects, loses the form of the field, or fails to take into account voids, context, or the necessary scale. A different and quite meaningful case is when atomic basins are not recovered, but another structure is found and independently reproduced. In that case the absence of an atomic answer is not an error: it is a possible result of a field-based search. A single predetermined form of the "letter" is assigned neither to the algorithm nor to nature.

The success criteria are also arranged in levels. The minimum is not "exactly atoms were found" but the appearance of a reproducible field inventory, stable under reasonable changes of the analysis, or of some other explicit structure that can be tested on new data. Stronger: the structure predicts combinations, properties, or transitions that were not used in extracting it. Stronger still: the rules generalize compactly and withstand independent chemical, physical, or experimental checks. The failure of an individual hypothesis, for example that a letter coincides with an atomic basin, that the alphabet is finite, or that the grammar is more compact than standard physics, is not equivalent to the failure of the whole program. But the absence of any detectable order whatsoever cannot be declared a scientific negative result without revising the very way of observation.

Why did this path not become a generally accepted program earlier? Not because no one studied fields. The quantum theory of atoms in molecules (QTAIM) and the topology of the electron localization function (ELF) have a long literature; there are also works in which electron density or three-dimensional electronic-structure data are analyzed with machine learning. These publications confirm the soundness of **parts of the methodological foundation**, but they do not confirm ESM as a whole: they do not establish a blindly discovered corpus alphabet, its "words," and an executable grammar as a single result. SMILES is a string notation of the molecular graph, useful for conveying composition and connectivity, but not a complete representation of the spatial field. The distinctiveness of ESM should be sought not in the claim "no one saw the fields before us," but in a testable combination of field-based search, repeatability statistics, contextual identity, dynamics, and criteria formulated in advance.

Hence the value of any outcome. If an independent search converges on atoms, this will support the atomic picture as a stable field decomposition, but will not automatically prove all of chemistry. If supra-atomic or inter-basin units are found, they may provide new levels of representation. If recurring "words" appear while the rules remain standard physics, we will still obtain testable field units for artificial intelligence (AI) and an interpretable corpus. If the dictionary turns out to be partially open, a stable core can be separated from a rare tail. If specific predictions fail a deferred check, we will mark the boundary of applicability rather than pass similarity off as a law. The practical hope, to model and design structures at a higher level and to reduce part of the expensive calculations, remains conditional until accuracy, transferability, and error bounds are confirmed.

The practical program does not end with the recovery of the alphabet. It asks in sequence: which field units recur; which rules govern their use and variation; which combinations form stable syllables and words; which grammar describes the construction and change of whole structures. The search for the field remainder, invariants, and contextual variants are internal tasks of the first level, not a substitute for the subsequent levels. In the current code, the main completed loop is concentrated on the search for candidate "letters" and their initial grouping; the pilot checks that are present do not mean that the levels of rules, words, and grammar are already complete. Each stage can produce an informative result of its own. No result, whether atomic, supra-atomic, partial, negative for an individual hypothesis, or so far ambiguous, closes the following questions by itself or makes the research a failure. The subsequent sections unfold this program and set out the methods, checks, and criteria for each level.

## Methodological frame: why linguistics, and the status of the hypothesis

In classical chemistry the atom is a powerful and repeatedly verified unit of description. ESM tests a narrower but fundamental question: can the atomic decomposition be obtained independently from the field, and does it exhaust all the stable units that the field allows us to detect? Here the atom is not an initial label but one of the possible results of a blind analysis; the field organization may turn out to be atomic, supra-atomic, subatomic, or mixed (§3.1).

Linguistics is used here not as a model of nature but as a source of methods for analyzing complex systems with unknown internal structure. If chemistry is regarded as an unknown system for recording and executing information, its investigation naturally reduces to a problem of decipherment. Solving this problem requires methods from several disciplines, each of which is responsible for its own layer of the theory and its own aspect of the system's structure, rather than competing with the others for the same explanation:

- **Linguistics** answers not the question "what is a field?" but the question "how does one look for structure in an unknown writing system?": methods of decipherment (§6), the distinction between syntax and semantics (§10), and morphology (§4.4–4.6).
- **Quantum chemistry** answers the question "what exactly are we measuring?": density functional theory (DFT), the quantum theory of atoms in molecules (QTAIM), the electron localization function (ELF), and conceptual DFT (§2) provide ways to obtain fields and to relate them to physical quantities.
- **Topology** answers the question "which objects are stable?": for example, whether critical points and the connections between regions of the field are preserved under a small change in its description; Morse homology and other invariants formalize some of these properties (§3, §15).
- **Information theory** answers the question "what is the minimal unit of information?": a criterion of compressibility and reusability that separates the fundamental from the derived (§12).
- **Statistics and machine learning** answer the question "how can repeatability be assessed at the scale of a corpus?": explicit algorithms of comparison, clustering, and distributional tests (§11, §15); a neural network does not set the ontology of letters and does not replace the first verifiable stage.
- **Dynamical systems theory** answers the question "by what rules do they change?": it analyzes paths instead of snapshots; this includes Markov state models (MSM), which describe transition probabilities, and Takens' theorem on reconstructing dynamics from a time series (§8, §9, §17).

Decipherment therefore describes, above all, the **discovery stage**: we do not assign atomic boundaries or meanings to the field in advance. Later, if the inventory withstands the checks, the same results may become the basis for models, generation, and simulation. The transfer of linguistic methods to chemical fields is itself not considered proven in advance; it is tested together with the physical assumptions of ESM.

### Why it has been hard to unite these lines

It would be incorrect to claim that before ESM no one investigated electronic fields or applied machine learning to them. Such works exist and form an important part of the project's foundation. What remains open is not the fact that density can be analyzed topologically or fed into a model, but a broader question: can stable recurring units, their relations, contextual variants, and rules of change be recovered from a large corpus without atomic hints?

Concrete difficulties have hindered, and continue to hinder, this integration. Fields are expensive to obtain and store; their discretization and the level of the quantum-chemical calculation affect the visible details; comparison requires invariance to translation and rotation; statistics of rare motifs requires a diverse corpus. Transition trajectories are even more expensive than static structures: a single reaction may require numerous consistent fields along the path, as well as repeated calculations for different environments, mechanisms, and initial states. Besides cost, there is also a methodological question: a transitional pattern may belong to letter variability, to word form, or to the rules of grammar, and this cannot be assigned before the analysis. Moreover, atom-centered representations are convenient and efficient, whereas a field-based program must redefine which objects it compares. This is not a story in which earlier chemistry "looked in the wrong place," but the reason why uniting the existing directions into a common decipherment program remains, for now, a separate testable task.

The literature confirms individual supporting points, but not the whole ESM theory. The topological extraction of atoms and bonds from density is studied in quantum topology [Bader et al. (1980)](https://doi.org/10.1002/ijch.198000003); electron localization and the topology of ELF in [Becke and Edgecombe (1990)](https://doi.org/10.1063/1.458517); machine learning of electron density in [Grisafi et al. (2019)](https://doi.org/10.1021/acscentsci.8b00551) and [Brockherde et al. (2017)](https://doi.org/10.1038/s41467-017-00839-3); the direct use of three-dimensional electronic structure for property prediction is shown, for example, by [Casey et al. (2020)](https://doi.org/10.1021/acs.jcim.0c00259). These results make field analysis and field-based learning reasonable research directions. They **do not confirm** that a universal energy alphabet has been found, that the words and rules of ESM exist in the stated sense, or that they are capable of replacing calculations. Such a conclusion would require open protocols of our own and independent verification.

### Why different disciplines are needed here
Each discipline covers its own type of error. Quantum chemistry sets the origin of the fields and the physical level of the calculation; topology looks for stable objects in them; statistics tests repeatability; information theory measures reusability; decipherment methods force us to separate distribution from a meaning ascribed in advance; dynamics is needed for transitions. Their combination does not make a conclusion automatically correct, but it makes it possible not to confuse a numerical pattern with a physical structure, or a static correlation with a rule of execution.

A voxel grid without topology remains an array of numbers; topology without an assessment of stability may take numerical artifacts for structures; a search for repeatability without information theory does not measure whether the dictionary shortens the description; automatic clustering without a decipherment methodology risks grouping the data by a random feature or by a label introduced in advance. For the first stage, what is needed here is not necessarily neural networks but reproducible calculations, controlled comparisons, and statistics. Finally, a static description does not show how a system passes between states: this requires dynamic methods and trajectories (§8–10). Each discipline adds a separate check; none of them alone proves ESM.

### The "implicit grammar" hypothesis: the analogy with language models and its limits
In this program the analogy with language models is limited and does not serve as proof. We do not assume that a neural network will itself "learn chemistry," and we do not use its hidden representations as evidence that a grammar exists. The working question is more modest and more testable: does a diverse corpus of fields contain distributional regularities and recurring structures that can be extracted by explicit procedures and then tested on independent data?

Physical verification here is not an "oracle" that automatically settles the dispute in favor of one method. Computational models have their own approximations; experiments have their own errors and conditions. Therefore the structures found are compared with independent sources, and the origin of each label is stated explicitly (§16, §22.10). A field pattern becomes evidence not because it looks beautiful or coincided with a known descriptor in the same data, but because a previously unused prediction withstood an independent check.

---

## 1. How the ESM hypothesis is formulated

Standard chemistry successfully describes molecules through atoms, bonds, and their representations, from structural formulas to SMILES. ESM proposes to test whether this picture is one of the projections of a richer organization of three-dimensional fields: electron density, potentials, ELF, and other explicitly defined quantities. At the search stage the atom is declared neither a mistake nor a minimal unit: it remains one of the testable interpretations of the discovered field structures. Perhaps repeatability is determined not only by local form but also by relations, the topology of the surroundings, and the change of the field.

If the dynamic part of the hypothesis is confirmed, the difference from written language will be fundamental: a chemical structure does not merely describe a state, it is a physical state; a reaction is not a sentence about a transformation but the transformation of the field itself. ESM therefore asks not only "which forms recur?" but also "which conditions force them to pass into another state?". This is the testable meaning of the word "performativity" in the project, not an already established property of the discovered alphabet.

---

## 2. The ontological layer: field, not atom

### 2.1 Precedents in existing theory
The field-based approach has support in existing science, although the ontological hypothesis of ESM itself remains open. In particular, quantum chemistry provides several fields and ways of analyzing them:

- **DFT, density functional theory (the Hohenberg–Kohn theorem)**: in the exact theory the ground-state energy is expressed through the electron density ρ(r), and not through the coordinates of individual electrons. The nuclei set the external electrical influence; this does not mean that practical calculations do without atomic coordinates or approximations.
- **QTAIM (Bader's theory)**: the atom is derived not as an initial postulate but as a topological object, a basin of attraction of the gradient flow of ρ(r). A bond is not an abstract line but a critical point (bond critical point) in the field. Both types of objects are obtained with the same topological apparatus, but which of them corresponds to a "letter" and which to a "word form" (a relation between letters) is an open question of the theory, see §3.1.
- **ELF (Electron Localization Function)**: identifies regions of localization of electron pairs (bonding, lone pairs) as topological features of a scalar field, again without a priori knowledge of which atoms "belong" to which bond.
- **Conceptual DFT**: introduces reactivity descriptors (chemical potential, hardness, the Fukui function) as derivatives of the density with respect to the number of electrons; that is, it characterizes not an "atom" but the *reactivity of a region of the field* in a particular context.

### 2.2 The limitation of the precedents
It is important to distinguish a physical field from the way it is computed. Many fields are obtained from a calculation in which nuclei and atom-centered basis functions already take part in constructing the wave function; ESM can then analyze the saved field without using nuclear coordinates as hints for identifying patterns. Such a search is blind at the discovery stage, but this does not mean that the original calculation was atom-free. Real-space grid methods offer another option, with their own costs and numerical limitations. The calculation method, basis, grid, and accuracy must be preserved as part of the provenance of the data.

There is also a terminological boundary. Electron density, electrostatic potential, and ELF are specific fields; the total energy of a system is a scalar quantity, and local "energy densities" depend on the chosen definition or representation. The ESM claim is therefore tested for specific fields and formulas, not for an unspecified "energy field" in general. Different fields may yield consistent or different alphabets; this is a separate result, not a reason to hide the choice of representation.

### 2.3 Void as part of the field, not its absence
Critical points are one way of reading the field: they indicate where its local shape changes. But even a complete list of maxima, minima, and saddle points does not describe the entire pattern between them. Also important are the regions, the sequences, and the mutual position of features, and the empty space that they surround or leave open. As in the shape of the letter "o," the opening can be part of a recognizable form rather than a background.

For this reason the program includes a separate branch of low-density-region topology, and not only an analysis of singular points. It may describe the number of connected "empty" zones, closed cavities, and through channels, topological characteristics related to Betti numbers. A cage critical point gives one local signal about a cavity, but it does not replace a description of its whole shape, volume, and connectivity. This is a concrete computable hypothesis that must be tested on the same field, with control of the grid and thresholds.

For intuition, local features can be compared to strokes and voids to openings; this is merely a guide, not physical proof. The methodological conclusion is stricter: if one looks only for extrema, one can exclude in advance precisely those units that are defined by regions of low density or by relations among features.

Practical consequence: the topology of the void can be investigated even if the field does not give a stable partition into atomic basins. This is not an alternative to hypotheses A–D (§3.1) but one more branch of the search, kept and tested in parallel with point-based, regional, and relational structures.

---

## 3. The hierarchy of the language

The table below shows possible **scales of description**, not objects that have already been found, and not the order in which the work is carried out. The order of the research stages (the search for letters, rules for them, syllables/words, and grammar) is described separately in §15. The names "letter," "word," and others for now denote testable roles in this scheme.

| Level | Analogue | Physical carrier |
|---|---|---|
| Pixel | stroke of a hieroglyph | raw field sample at a grid point (carries no meaning on its own; NOT an atom, see §3.1) |
| Letter | symbol/operator | candidate for an independent recurring unit of the field: a point feature, a region, or a relation (the exact type is open, §3.1) |
| Word | morpheme | candidate for a reproducible motif or combination of units across different molecules (the boundary between letter and word is open) |
| Phrase | word combination | combination of motifs working together (reaction domain, active site) |
| Sentence | function | the integral active site of a molecule |
| Program | text, discourse | reaction networks, metabolic cascades, catalytic cycles |

Meaning at each level arises through the interaction of the lower levels and is not reducible to their sum (emergence, as in the catalytic sites of enzymes, where activity is determined by the geometry of the whole pocket and not by the sum of the properties of individual residues).

### 3.1 An open question: what physically corresponds to a "letter" — competing hypotheses, not an established fact
Assigning a specific type of object to the role of "letter" is a hypothesis, not a settled question. Critical points and basins provide natural starting candidates, but they do not exhaust the search space: the unit may be a distributed form of the field, a sequence of maxima and minima, a pattern in a ring, or a relation among several features and voids. Competing versions are given below, but this list is open; no form is assigned as a unit merely because it is easy to compute.

- **Hypothesis A (flat alphabet)**: different types of local field features (nuclear, bond, ring, and cage critical points) may be units of one level. Here a "topological feature" means a form of the field defined by the behavior of its values and derivatives in the neighborhood of a point; the specific set of such objects remains part of the choice to be tested.
- **Hypothesis B (letter = atomic basin, relation = larger unit)**: the letter is the basin of attraction around a nuclear attractor (a QTAIM atom), the region of space from which the gradient-flow lines of the electron density converge on a single attractor. Its charge, volume, and dipole may depend on the molecule and the environment, so the letter is not identified with an unchanging classical "ball." In this hypothesis, bond, ring, and cage critical points are not independent letters but possible relations between basins; it is precisely this role that the table above conditionally assigns to larger units, without setting it for all variants of ESM. A historical analogy is the syllabic signs of Linear B: in the decipherment it turned out that a sign could encode a combination of consonant and vowel rather than a separate phoneme. The analogy illustrates the possibility of being wrong about the scale of the unit, but by itself does not confirm Hypothesis B.
- **Hypothesis C (subatomic or inter-basin unit)**: the minimal stable object may be part of one basin, may span parts of several basins, or may have a distributed internal form, for example an alternation of minima and maxima that cannot be reduced to a single extremum. In this case the classical boundary of a basin need not coincide with the boundary of a unit.
- **Hypothesis D (relation or contextual pattern)**: the letter may be not a local feature in itself but its position among other patterns: a configuration, a symmetry, a sequence, the topology of the surrounding cavity, or a combination of field features in a finite region. Maxima in a ring, for example, may describe background or surroundings, while a minimum, or a joint pattern inside the ring, turns out to be the transferable unit. This is not a claim that it is so, but an alternative that a point-based search for extrema alone cannot test.

There are no decisive data between these hypotheses (and intermediate variants, including the possibility that different classes of units act in different contexts). Accepting any version before testing would repeat the error against which §6.1 warns: ascribing meaning to a unit before its status has been established. Therefore each procedure below must report which classes of objects it could detect at all.

A separate, easily confused pair of statements: "atoms are a reliably confirmed structure of chemistry" and "atoms are letters of the ESM alphabet" differ in strength and subject. The first does not require confirmation by this theory; the second remains open among hypotheses A–D. A check that the pipeline is able to match what it finds with known atomic structures (`atom_recovery_sanity_check`) is a useful diagnostic, but not a right to impose on atoms the status of letters: the same algorithm may successfully detect atoms as one layer and find other units as another.

### 3.2 How the hypotheses are falsifiable
For some of the starting hypotheses, discrimination is feasible with the distributional test of §6.1: if bond, ring, or cage critical points are independent units, their combinations should behave differently than if they are obligatory relations between atomic basins. This tests variants A/B within the family of critical points, but does not close C/D. Distributed forms and relations need their own previously described units of observation and signs of repeatability: for example, whether a pattern of maxima and minima inside a ring persists, whether the relative geometry of a group of features is rediscovered, and whether a void configuration predicts an independent property.

Practical consequence for the verification program (§15): candidates are first grouped by the type of observed object (critical points, regions/forms of the field, relations between patterns, void topology), and within the point-based branch also by the type of critical point. Incomparable definitions of a unit must not be mixed in a single clustering; but the alphabet also must not be restricted to only those candidates that admit a point label.

### 3.3 Which outcomes are meaningful — and which outcome requires revising the method

The test of §3.2 must not be designed so that only a non-atomic answer is considered interesting. If a blind analysis of the field stably recovers nuclear attractors and atomic basins, this will be important independent confirmation that the familiar atomic decomposition has a field basis. But such a result will confirm precisely the reproducibility of the atomic level; it will not prove that atoms are the only units, and it will not close the questions about words, context, and rules.

If atomic basins do not turn out to be the minimal repeatable units, this does not mean that the analysis found nothing. The field may show stable parts of basins, inter-basin forms, voids, relations, or other scales of organization. In that case the hypothesis of the letter-atom has not withstood the test, but the field structure may turn out to be richer. The comparison with the atomic markup takes place after the blind discovery (§3.4), so as not to pass off a label introduced in advance as a discovery.

There is, however, an outcome that cannot be entered in the column "theory refuted": the complete absence of any structure, repeatability, and relations. For chemical fields this means that the current procedure does not yet resolve their organization, because of insufficient completeness of the search, the chosen type of field, the grid, the scale, the similarity feature, or another error of formulation. Hypotheses must be broadened and the logic revised before any conclusion about nature is drawn. This is a requirement on the research instrument, not a promise that any particular implementation with any parameters will necessarily find atoms.

This does not rule out the possibility of a negative scientific result. An individual hypothesis, for example that a letter always coincides with an atomic basin, may fail a previously specified test on suitable positive and negative controls. A different matter is an empty output of a procedure that has not shown it is capable of detecting even known or synthetically specified structures: it characterizes, for now, the limits of that procedure, not the absence of organization in general.

Thus, atomic, non-atomic, and mixed answers are admissible scientific findings; specific ideas about the alphabet and grammar may fail the test; a completely empty result is a signal to continue diagnosis, not a standalone conclusion that chemistry has no patterns. This order makes the program open to an unexpected structure of the field, but does not allow a beautiful interpretation to cover for a faulty search.

### 3.4 How not to substitute a pre-supplied hint for the test
The practical implementation of the test of §3.2 carries a risk that is worth stating directly: if the search for nuclear attractors (candidates for a "letter" under Hypothesis B) technically relies on already known nuclear coordinates as search seeds, and all the more so as a forced classification when numerical convergence fails, then the test result partly presupposes its own answer rather than deriving it independently. A test that answers "yes, the letter is an atom" partly because the letter was told in advance where to look for the atom proves nothing.

A practical check requires two conditions: (1) where possible, compute the field on a uniform spatial grid (a voxel is the three-dimensional analogue of a pixel), for example using grid-based DFT in the GPAW package; (2) after computing the field, carry out a separate "blind" search for critical points, without using atomic coordinates as starting points or labels. Its result is then compared with the usual, "hinted" search. If the position of an atomic attractor is recovered by the blind pass, this supports Hypothesis B; if it is missed, then in this test there is no independent topological confirmation, but this by itself does not refute the hypothesis; if a stable attractor is found without a corresponding atom, it is a candidate for a unit of another type, not automatic proof of Hypothesis C.

Operational rule: a "blind" search does not mean a sparse arbitrary grid of starting points. For a cube file, one first finds the local maxima and minima of the field values on the voxels, then refines their coordinates by solving `∇f=0` (the gradient equals zero), and classifies them by the matrix of second derivatives. Only after that can the found attractors serve as starting points for the search for saddle points. Atomic coordinates from the header of the cube file are not used in these steps. Even if the source field was computed with an atom-centered basis, this restriction refers to its provenance and does not permit mixing atomic coordinates into the subsequent discovery, description, or grouping.

A controlled synthetic test with two overlapping Gaussians shows something important for the implementation: in a sum of smooth fields, the extrema may be displaced relative to the centers of the components. This demonstrates why a found point cannot be forcibly assigned an atomic label merely by its proximity to a given center. But a synthetic example is not a counterexample to the QTAIM theorem for the electron density of a real molecule, and by itself does not show that nuclear attractors are displaced in chemical systems. This requires separate calculations of a physically correct density, with control of accuracy near the nucleus, and a comparison with the standard topology. A synthetic test checks the honesty of the algorithm on a composite field; a chemical claim requires a chemical field.

### 3.5 The correspondence of a hierarchy level to a physical scale is not fixed
The hierarchy table at the beginning of the section sets out a ladder of levels, but does not specify to which physical scale (number of atoms, size of the molecule) each level corresponds, and this is not an oversight but a deliberate limitation. For strongly delocalized systems (for example, aromatic rings and fluxional clusters, §4.1(3)), the very partition into discrete basins may be physically incorrect: if the field has no clear boundaries, decomposition into "letters" and "words" is already a methodological error and not a technical difficulty.

This implies a possibility already outlined in §4.1(3): for a small, fully delocalized molecule, **the whole molecule may turn out to be the minimal indecomposable unit**, not because it is structurally simple but because it is structurally indivisible. In that case it is not a "sentence" consisting of "words" but itself acts as a "letter" or a "syllable," and which one is determined by the degree of delocalization of the particular system, and not by a fixed correspondence "one molecule = one level of the hierarchy." Symmetrically, a large, weakly interacting system (for example, a supramolecular complex with a weak hydrogen bond between independently stable fragments) may turn out to be a "story," a multi-level text within a single formal molecular record, rather than a single "sentence." The correspondence between a level of the hierarchy and a physical object (atom/group/molecule) is therefore not a table of fixed equivalences but a variable determined by the topological decomposability of each particular case, and it must be measured (for example, through the stability of basin boundaries under varying conditions, §4.3) and not postulated from the number of atoms in the formula.

### 3.6 What makes two occurrences the same letter: three competing definitions of identity
Hypotheses A–D (§3.1) ask *which type of field object or pattern* may be a letter. A separate question, logically independent of them, is by what criterion two occurrences found in different molecules are considered occurrences of *one and the same* letter. The answer to the second does not follow from the answer to the first, and there are at least three options here as well, all of them physically meaningful:

- **Identity by local field shape.** Two features belong to one letter if the field around them is arranged identically: the distribution of values coincides in a neighborhood taken at the characteristic scale of the region. This is a direct reading of §2: the meaning is carried by the shape of the field itself.
- **Identity by surroundings.** Two features belong to one letter if their position among *other* features of the field is arranged identically: how many features of each type are at what distance, and how they are positioned relative to one another in terms of directions. The feature's own amplitude is not taken into account at all. This is a reading of §11.2 (Saussure): a sign means through contrast with what could occupy the neighboring positions, and not through its own internal properties.
- **A mixed definition**, in which both descriptions are brought to a comparable scale and used jointly. It is necessary at least because the first two are not nested in one another: a feature may be indistinguishable in local shape from many others and yet uniquely positioned, and vice versa.

These are not three implementations of one idea and not a question of descriptive accuracy: they are three different answers to the question "what is counted as a feature of a letter at all," giving in the general case different alphabets on the same corpus. Choosing one in advance would mean the same error against which §6.1 warns: fixing a significant feature before it has been established that it is significant.

Practical consequence: each definition runs as a separate, fully labeled variant of the analysis, and its inventory is not mixed with the inventories of the others. They are compared not by the "plausibility" of the resulting alphabet but by the criteria of §19: which definition gives an inventory that saturates, is reproducible on held-out data, compresses the corpus better, and gives more stable classes that survive the check of §16. An outcome is also possible in which different definitions turn out to be correct for different types of critical points; this is a meaningful result, not a failure of choice.

It is worth noting separately: the definition by surroundings in principle does not require the local shape of the field to be informative, and therefore remains applicable exactly where §3.5 and §4.1(3) predict the failure of a local description: in strongly delocalized systems without clear boundaries. This makes it not a fallback in case the first one fails, but a test of an independent hypothesis about the carrier of identity.


---

## 4. The nature of meaning: the correspondence of form and function is nonlinear

### 4.1 Three types of mismatch between form and meaning
By "meaning" we understand here not verbal definitions but measurable or computable properties of a molecule, for example acidity, binding, or catalytic activity. In chemistry the correspondence between the atomic form of notation and such properties is not required to be one-to-one or to be composed of independent contributions:

1. **Different structures, a similar property** (*many-to-one*, isosterism): some chemically different groups can perform a similar function in a particular molecular environment. For example, the carboxyl group (`-COOH`) and the tetrazole ring are used in a number of contexts as bioisosteric replacements: their atomic composition and geometry differ, although individual properties may be close. This is not universal interchangeability; it is tested for a specific property and conditions. In ESM terms these are possible different "spellings" of a similar function, not proof of a common field unit.

2. **One-to-many (contextual polysemy)**: one and the same local atomic group performs different functions depending on the global context of the molecule (induction, resonance, coordination environment). Here the meaning is not given locally; it is the result of the global self-consistency of the whole system (a solution of the Schrödinger equation for the whole, and not a lookup on a local fragment). This is the key difference from operator dispatch in programming languages: there, type resolution is a cheap local operation; here it is the emergent result of the relaxation of the whole system to an energy minimum.

3. **Uncertainty of the unit itself (delocalization)**: in a number of cases the very notion of a discrete "letter" is physically undefined. Aromatic delocalization (a benzene ring with a bond order of 1.5) is not a superposition of two separate "words," the resonance structures, but a single indecomposable physical object; an attempt to split it into additive units is already a methodological error and not a technical inaccuracy. An even sharper case is **fluxional (nonrigid) molecules** (carborane clusters, some metallocene complexes), in which atoms exchange positions faster than a separate structure can be fixed. This is the analogue of an "inseparable ligature spelling" in writing: decomposition into letters there contradicts the physics of the object.

### 4.2 Practical consequence
The functional groups of traditional chemistry (`-OH`, `-COOH`, `-NH2`) are not a mistake but a working first-order approximation: a compression that is correct where the contextual correction is small relative to the base signal. The approximation systematically breaks down where this correction becomes comparable to the base signal or exceeds it: catalytic centers, organometallic complexes, delocalized and fluxional systems.

### 4.3 Indexicality of context: a systematic, not a random modulation of meaning
So as not to call every dependence of the field on the environment a "change of meaning," ESM proposes a working distinction. A smooth and predictable change of one unit when the context changes is hereafter called an indexical response; several stable, functionally different readings are called possible polysemy (§4.1(2)). This is a transfer of terms, not a claim that the chemical case is already proven to coincide with the linguistic category. In formal semantics (D. Kaplan), indexicality describes a rule whose content depends on context; a simple example is the pronoun "I": it points to the speaker, and the particular speaker changes depending on the situation.

Therefore the solvent and the protein environment are treated as possible contextual parameters whose influence should be measured, not as sources of new meaning known in advance. If the properties of the field change smoothly and predictably, this will support the hypothesis of a change within one unit. A sharp jump may indicate a change of mechanism or another regime, but by itself does not yet prove the appearance of a separate "word"; for that, stability and reproducibility across independent examples are needed.

### 4.4 Paradigmatic (inflection) vs lexical (synonymy) variability
The formulation of §4.1(1) ("one and the same word written in different fonts") mixes two linguistic phenomena that are different in nature and are worth separating explicitly. Morphology distinguishes a lexeme (a dictionary unit, the analogue of the "character" of §4.3) and its word forms: systematic, regular variations of one lexeme by grammatical category (number, possessive case, tense: *table/tables/table's*, *run/ran/running*), which linguistically remain "the same word" but are not identical in meaning. This differs from synonymy: different lexemes with the same or similar meaning ("big"/"large"), not connected by a common root or a regular rule of transition from one to the other.

The transfer to ESM accordingly distinguishes two types of connection between classes or realizations:
- **Synonymy (lexical variability)** is already described in §4.1(1): structurally unrelated atomic ensembles (`-COOH`/tetrazole) implementing one function. An idiosyncratic, pairwise structural coincidence that does not generalize to other lexemes of the corpus.
- **Inflection (paradigmatic variability)** is not described earlier and requires a separate mechanism: one and the same "lexeme," under the action of a specific, regular parameter (protonation state, isotopic substitution, oxidation state of a coordination center, conformer), gives a systematically different but related class: not an arbitrarily different function, but a different "grammatical form" of the same function, predictably derived by a common rule.

This is a direct generalization of §4.3: indexicality there described a shift of content *within* one class under the action of the environment; inflection is the same mechanism, only strong enough to cross the boundary between stable clusters (§15, step 3) and to give rise to a systematically linked pair of different classes, rather than a modulation of one. Isotopic substitution (§11.5) is a good example of a borderline case: whether it remains an axis within a class (an allophonic detail, as in most contexts) or gives rise to a separate, systematically shifted class (as with a strong primary kinetic isotope effect) is an empirical question, settled by the same cluster analysis, and not a category given in advance.

### 4.5 An operational criterion of distinction
The distinction does not require new machinery; it is a direct extension of the criterion of fundamentality through reusability (§12) from the level of "dictionary units" to the level of "rules of transition between units":

- If the transition from class A to class B under a change of a parameter is described by one and the same transformation rule, which also applies to many other, structurally unrelated pairs of classes of the corpus under the same change of parameter, this is a sign of inflection: the rule is reusable, like a case ending in a language, and is not specific to one pair.
- If the connection between two functionally equivalent classes does not generalize to any other pair in the corpus (an idiosyncratic structural coincidence), this is synonymy and not inflection, regardless of how functionally close the two classes are.

A reusable transformation rule is fundamental in the same sense as a reusable basis element of the dictionary (§12); an idiosyncratic correspondence is derived or specific to the pair, even if both classes are individually stable (§15, step 3). Stability is assessed by rebuilding the classes on slightly different samples of the corpus (a bootstrap check).

### 4.6 Consequence for the registry of realizations (§13)
The flat structure of the registry of realizations (§13.3), in which all atomic realizations of a class are listed as equal instances without internal structure, is insufficient for inflection: the word forms of one lexeme are not interchangeable "pronunciations" of one meaning, but units that differ in content, linked by a common rule of meaning and a predictable transformation. The registry needs an additional grouping level, the **paradigm**: a set of class identifiers (`class_id`) linked by a common transformation rule and a common axis of variation (for example, charge, isotope, or oxidation state). This is a separate link between classes, and not one more list of instances within a single class.

### 4.7 Caution
Like the other distinctions in the theory, the division into synonymy and inflection is a hypothesis that requires testing on a large corpus (§15, step 6), not an established fact: a rule that seems generalizable on a small calibration sample may not hold on a larger corpus, and the threshold (how many pairs must confirm the rule for it to be considered a "paradigm" and not a coincidence on a few examples) itself requires empirical calibration, and not an arbitrary choice made in advance.

---

## 5. Existing partial implementations of a "form = function" language

Besides the methods for obtaining and topologically analyzing fields, there are approaches that compare spatial distributions or functional motifs of molecules. These are partial methodological predecessors, not already finished versions of ESM:

- **CoMFA / 3D-QSAR** (*Comparative Molecular Field Analysis*; quantitative models linking three-dimensional structure and activity): they align molecules and compare steric and electrostatic fields at the points of a common 3D grid. This is an example of using field-based features, but atomic structures and a predefined alignment remain essential in them.
- **Pharmacophore modeling**: describes the spatial combination of properties needed for interaction (for example, a hydrogen-bond donor/acceptor or a hydrophobic region), and not a required specific atomic composition. Different chemical groups can realize a similar pharmacophore; this is an example of many ways of realizing a function, but not proof that the ESM field unit coincides with a pharmacophore.
- **Equivariant graph neural network models**: models that take into account how the atomic structure transforms under rotation and translation of the molecule; some of them pass features between neighboring atoms. They show how context can influence a prediction, but their initial unit usually remains the atomic graph, and not a blindly found field motif.

These examples substantiate that spatial fields and abstract functional motifs are worth analyzing. But not all of them abandoned the atom in the same way, and they do not confirm the thesis of a chemical language: it is precisely the transition from such partial methods to an independent search for the units and rules of the field that remains the question of ESM.

---

## 6. Decipherment methodology: how to look for the grammar of an unknown language

What is useful here is not a literal identity of chemistry and writing but a similarity of the research situation: the internal units are not yet known, so one first studies their repeatability and distribution, rather than assigning them meaning. The history of decipherment provides a set of methods for this and, especially importantly, warnings about errors.

### 6.1 Structure before meaning (the method of Ventris, Linear B)
Early attempts at decipherment sought the meaning of a sign through visual similarity to already known writing systems, and systematically erred. The breakthrough method (Michael Ventris) was the opposite: first a purely distributional model is built, a positional grid showing which signs occur in the same positions (and therefore encode a common parameter, for example a vowel sound), which signs are interchangeable, and which never occur adjacent to each other. The grammar is derived **before** the semantics is known.

Transfer to ESM: do not try to ascribe physical meaning to a cluster of patterns at once. First comes distributional statistics, that is, a comparison of where motifs occur in the corpus and what they neighbor: which motifs appear together, which almost never occur together, where in the structure they are located, and how their form changes when the context changes. In this way the organization can be tested before an interpretation of it has been proposed.

### 6.2 Bilinguals as anchors (the method of the Rosetta Stone)
Champollion's decipherment of hieroglyphs became possible thanks to a place where both form and meaning are known at once (three versions of one text in different scripts). A small set of such "bilinguals" makes it possible to extend the interpretation to the whole corpus.

Transfer to ESM: a set of molecules with reliably measured physical or biological properties (for example, a binding constant, an acidity index pKa, catalytic activity, or activation energy), points where the field and the property are known at the same time. These "anchor" examples provide an independent test of hypotheses about a motif; their interpretation can then be tested on the rest of the corpus, rather than derived from it in a circle.

### 6.3 The criterion of consistency of a hypothesis
In decipherment, a hypothesis about the meaning of a sign is accepted only when it works consistently in all contexts of its occurrence in the corpus, and not in one convenient case. Transfer: if the function ascribed to an energy motif requires exceptions more often than it works, the probable cause is a wrongly chosen invariant or a wrong alignment (the "font" and the "letter" have been confused), and not that "physics violates the rule."

### 6.4 Separating code from noise (the scribe's orthography vs genuine meaning)
In decipherment a separate task is to distinguish meaningful structural variation from scribal errors/orthographic variants of writing one and the same sign. Transfer: thermal/conformational fluctuations must be methodologically separated from a genuine change of function; otherwise one and the same "motif" will be erroneously split into many different "words" simply because of noise in the geometry.

### 6.5 What is specific to ESM that breaks a direct transfer of decipherment methods
Classical decipherment deals with a **fixed code** (the meaning of a sign is constant within the language, polysemy is a rare exception). ESM is arranged differently: polysemy is **built into the physics**; the contextual modulation of meaning is not random and not an exception but a systematic mechanism (see §4.1). Hence the decipherment of ESM is closer not to translating a dead text with a fixed key but to analyzing a protocol in which the meaning of a symbol is generated anew in every act of use through interaction with the context (an analogue of systems in which interpretation depends on the runtime state rather than on a static symbol table). This requires supplementing the methods of language decipherment with methods of analyzing dynamical systems (see §8).

---

## 7. Language as execution: performativity

### 7.1 The precise linguistic frame
In speech act theory (J. Austin), some utterances do not merely report an event but perform an action if the conditions for its validity are met: for example, words about concluding a marriage have that status only in the appropriate situation and when spoken by an authorized person (English *felicity conditions*). We transfer this idea to chemistry as a testable question, not as an established fact: can a particular configuration of the field not only describe a state but also take part in its physical change? In this conditional sense the phrase "in the language of ESM we do not describe a transformation, we produce it" denotes a hypothesis about performativity.

### 7.2 The conditionality of execution: an important clarification
If such an analogy is confirmed, it will be conditional: the presence of suitable reagents and geometry does not yet guarantee a reaction. A surmountable energy barrier, a catalyst, and a certain medium may be required. Therefore the "conditions of execution" here are physically testable parameters of the reaction, not an additional chemical entity; a correctly constructed combination may turn out to be inaccessible under the given conditions.

### 7.3 Grammatical categories responsible for the conditions of execution
If the performative frame is adopted, the language must have categories encoding not "meaning" but "conditions of execution," and computable physical analogues already exist for them:
- the energy barrier of a transition is an analogue of modality ("possible/impossible" execution);
- a path on the potential energy surface is an analogue of the physical trajectory of the execution of an act in time;
- the topology of the transition state is an analogue of mood: a path that could have led to the product but did not because of a competing lower barrier is a "subjunctive," unfinished action.

---

## 8. Language as a dynamical system, not a static dictionary

### 8.1 Why a static dictionary is not enough
The analogy with loop variables, recursion, and conditional statements points to a specific class of already known chemical systems, and is not a metaphor:

- **Oscillating reactions** (the Belousov–Zhabotinsky reaction) are systems with state variables that change cyclically in time under the control of feedback; a direct analogue of a loop variable.
- **Autocatalysis**: a reaction product simultaneously serves as a catalyst of its own generation; a direct analogue of recursion (self-reference).
- **Allosteric regulation of enzymes**: ligand binding at one site switches behavior at another, remote site through a conformational change; a form of conditional branching (an if-branch) encoded in geometry and not in a separate control block.

These examples show why a single list of equilibrium motifs may not be enough to describe processes. They motivate, but do not prove for ESM, the need for a specific architecture: the testable question is whether a model with state and feedback improves the prediction of transitions compared with a static dictionary.

### 8.2 The dimensionality of the field description and responses to the environment
It is important here not to confuse two things: an ordinary field is given in three spatial coordinates, whereas the set of its responses to different conditions can be described by a larger number of features. This is an additional **dimensionality of the data description**, not a claim about additional physical dimensions of space. For example, to the electron density ρ(x,y,z) one can add measurable or computable responses to environmental parameters:
- the polarizability tensor (and hyperpolarizability), a response to an external electric field;
- the Fukui function f(r) = ∂ρ/∂N (conceptual DFT), a response to a change in the number of electrons, a standard descriptor of reactivity.

One can estimate the intrinsic dimensionality of the cloud of such responses, the number of independent directions of data variation, for example by methods of correlation dimension or maximum likelihood. But this by itself does not prove the existence of a "hidden dimension": the result depends on the representation, the range of conditions, noise, and sample size. Takens' theorem applies to a different, narrower formulation: to the reconstruction of a dynamical attractor from an ordered time series of observations when its premises are satisfied; the number 2d+1 refers to the dimension of the embedding space, not to the number of independent molecules or samples ([Takens, 1981](https://www.math.ucdavis.edu/~saito/data/synchrosqueezing/takens_detect-strange-attractors-turbulence.pdf)). Therefore for ESM it is correct to speak of two separate checks: an estimate of the dimensionality of the description of a family of field responses and, if time trajectories are available, a check of the reconstruction of the dynamics. High dimensionality may support the need for a richer description, but does not automatically confirm the ontological hypothesis about the field.

---

## 9. The word as execution, not as record: the result vs the process of obtaining it

### 9.1 Static structure: the state after a process
The equilibrium geometry of a molecule or the folded structure of a protein shows the state of the system, but not the whole history that led to it. This distinction is important for ESM: a snapshot may help to find stable motifs, but from it alone one cannot reliably reconstruct the sequence of transitions and the conditions of the reaction. An analogy is an attempt to understand the operation of a program from its saved final state: some traces are visible, but the order of the operations performed is usually not recorded in the result itself. Static data are therefore suitable for finding recurring states, while the grammar of transitions requires time trajectories or other data about the process.

A practical consequence follows: databases of static structures (for example, PDB or equilibrium geometries from PubChem) mainly provide examples of states, and not the complete course of their formation. They can be used to build a dictionary of stable states (§15), but conclusions about transition rules will remain preliminary until they are tested on data about dynamics.

### 9.2 Existing apparatus describing the path, not only the result
- **The theory of the folding energy landscape (funnel theory)**: the native structure of a protein is not a "written-down answer" but an attractor of a funnel-shaped landscape to which many different trajectories converge. The meaning (function) is encoded in the shape of the funnel as a whole, and not in the coordinates of its bottom.
- **Levinthal's paradox**: folding cannot be an enumeration of ready-made conformations (that would take longer than the universe has existed); hence the path is guided by local physics at every step, that is, the very process of execution is determined by the grammar of the landscape, and not by a random enumeration of "words."
- **Markov state models (MSM)**: a molecular dynamics trajectory is divided into a set of metastable states (the alphabet) plus a matrix of transition probabilities between them (the grammar). This is a direct operational analogue of what needs to be sought: not clustering of static snapshots, but clustering of states along a trajectory plus an explicit model of transitions between them.
- **Transition path sampling / intrinsic reaction coordinate (IRC)**: methods that compute not the structure of the product but the whole minimum-energy path from the reactant through the transition state to the product; a direct analogue of recording a sentence as it is being uttered, and not of reading what has already been said.
- **Anfinsen's experiment**: all the information for folding is already contained in the one-dimensional amino acid sequence, while the folding into 3D itself occurs deterministically under given environmental conditions. This gives a natural precedent for a source/runtime separation: the 1D sequence is closer to the source code of a program, and the 3D structure to the result of its execution by an interpreter (the environment). The difference from ESM is that the "source code" here is one-dimensional and not three-dimensional; that is, not every chemical language has such an explicit linear record before execution.

---

## 10. Two-level grammaticality: combinatorially possible vs physically executable

### 10.1 A distinction by analogy with syntax and semantics
In natural language there is a difference between a syntactically incorrect string (the language does not generate it at all) and a syntactically correct but semantically anomalous phrase (Chomsky's classic example: "Colorless green ideas sleep furiously": the grammar is impeccable, the meaning is impossible). In ESM this distinction acquires a physical, not an abstract, status:

- **Combinatorially incorrect**: the structure does not exist even formally (a violation of valence, an impossible bond graph).
- **Combinatorially correct but physically unexecutable**: the bond graph is formally correct (the valences add up), but the required geometric realization in 3D demands an energy unattainable under the conditions considered. An example is inverted adamantane: the connectivity is formally meaningful, but it requires a configuration of the central carbon atom that is incompatible with tetrahedral sp3 geometry without extreme angular strain.

Unlike natural language, here physics itself acts as the censor: an impossible "sentence" does not merely sound strange; it is not physically realized or immediately relaxes away from the attempt at writing (§10.3).

### 10.2 Existing chemical apparatus of physical prohibitions
- **Ring strain and Bredt's rule**: anti-Bredt alkenes (a double bond at the bridgehead atom of a small bicyclic ring) are forbidden not combinatorially but geometrically, by the same mechanism as inverted adamantane.
- **The Woodward–Hoffmann rules (conservation of orbital symmetry)**: show a prohibition on the *path* separately, and not only on the result: some reactions are impossible not because the product is unstable but because the transition process itself violates orbital symmetry and requires an insurmountable barrier. That is, not only the final result but also the very act of "pronouncing" must be orbitally allowed.
- **Thermodynamic vs kinetic control**: it is important to distinguish an absolute prohibition from a prohibition specific to the conditions of the medium. Some structures are unattainable not absolutely but only at a given temperature/pressure/catalyst; this is not an orthographic error but a dialectal restriction: the word exists in the language but not in this "dialect" of conditions.

### 10.3 Self-correction as a built-in mechanism of the language, not post-processing
- **Proofreading in DNA replication** (the exonuclease proofreading activity of polymerase): a literal physical mechanism of checking and rejecting an error directly in the process of the act of writing, and not after its completion.
- **Relaxation after perturbation**: some unstable configurations do indeed pass into more stable states, but not necessarily into the nearest or chemically "correct" state; barriers, decay, and several competing outcomes are possible. Therefore this is not a universal mechanism of self-correction of the language but a testable class of processes: one must establish when a failed configuration is blocked, when it relaxes, and which products are formed.

---

## 11. Distinctive vs silent parameters: not every physical parameter carries function

### 11.1 The problem
A separate danger for the theory is to implicitly ascribe functional significance to a parameter merely because it is "chemically noticeable" or intuitively seems important. The opposite error is also possible: failing to notice that a parameter considered secondary turns out to be decisive in a narrow context. An operational, not an intuitive, criterion of separation is needed.

### 11.2 The linguistic frame: arbitrariness and differentiality of the sign (Saussure)
Saussure separates two theses that are important here in combination. Between the signifier (form) and the signified (meaning) there is no natural connection; it is a convention (the arbitrariness of the sign). A subtler consequence: the value of a sign is not substantial but **differential**: a sign means something not by virtue of its own internal properties but only through contrast with the alternatives that could occupy the same position but do not. As applied to the theory: some parameters (a specific electronegativity, the mass of a nucleus) may carry not meaning in themselves but may simply fill a position in the system; and the functional load is encoded not by the magnitude of the parameter but by whether its change distinguishes the outcome.

### 11.3 An operational method: the commutation test (the Prague School, Trubetzkoy)
A method developed in phonology precisely for this task: to determine whether a feature is distinctive (phonemic) or a random variation (allophonic), a minimal pair is taken: exactly one parameter is changed while the others are held fixed, and it is checked whether the meaning changes. A classic example: aspiration of consonants is phonemically irrelevant in English (an allophone, a "silent" feature) but distinctive in Hindi (the same physical feature, different words). Conclusion: one and the same parameter can be silent in one system/context and significant in another; the status of a parameter is not absolute but system- and context-dependent.

### 11.4 Three independent implementations of this test that already exist in chemistry and biochemistry
1. **Degeneracy of the genetic code (silent mutations)**: the third position of a codon (the "wobble position") can often change without changing the amino acid; direct, literal proof that a parameter (the specific nucleotide in this position) physically varies but carries no functional load in most contexts.
2. **Alanine scanning and deep mutational scanning**: the systematic replacement of each residue of a protein in turn by alanine (a minimally perturbing substituent) with measurement of the change in function; an operational commutation test that arose independently in biochemistry to separate positions that carry a functional load from placeholder positions, where the occupancy of the position matters and not the specific chemical identity of the substituent.
3. **Matched Molecular Pair Analysis (MMPA)** in medicinal chemistry: the systematic collection of pairs of compounds that differ in exactly one fragment while the rest of the structure is held fixed, with measurement of the change in activity. Such analysis makes it possible to compare the consequences of substitutions quantitatively; specific conclusions require a sufficient number of comparable pairs and control of conditions.

### 11.5 Caution: "silent" does not mean "absolutely silent"
Isotopic substitution is an instructive borderline case. The mass of a nucleus in the overwhelming majority of chemical contexts does not affect reactivity or geometry: "sound without meaning" in its pure form. But in a narrow context, the kinetic isotope effect (KIE), when the breaking/forming of a bond involving the given atom is part of the rate-limiting step of the reaction, the same parameter becomes decisive for the rate of the process. Hence a methodological rule: the conclusion of a commutation test must be formulated as "the parameter is silent in the tested range of contexts," and not "the parameter is absolutely silent." The error here is symmetric to the error of §4.2 (functional groups work while the contextual correction is small and break down in rare cases), only turned inside out: a parameter that seems unimportant almost everywhere may hold up the entire function in one narrow structural context.

### 11.6 Relation to the criterion of fundamentality (§12)
The commutation test and the criterion of compressibility/reusability (§12) solve complementary tasks: the criterion of compressibility looks for which *units* are reused across the whole corpus (what may be a letter and what a derived combination); the commutation test checks which *axes or parameters within an already found unit* affect function and which can be varied without a noticeable change in outcome under the tested conditions.

### 11.7 Categorical vs continuous significance: a parameter as a consequence, not an axiom
Another danger in interpreting the commutation test is the tacit assumption that every parameter is an independent variable of the system, whereas some parameters are in fact derived, computable as a function of more fundamental ones (the same boundary as in §12 between a dictionary unit and its decomposition, only applied not to a whole letter but to an individual axis within it). Hence a refinement to §11.5: a low average effect of a parameter under variation (a continuous metric: the change of outcome averaged over the corpus) does not mean low significance if the parameter in fact encodes not a continuous contribution but a discrete class boundary, by analogy with the phonological division into classes (vowel/consonant/sign, §6.1). Such a parameter by construction gives an almost zero effect almost everywhere and a sharply nonzero effect only at rare transitions that are nevertheless structurally defining for the class; a commutation metric averaged over the corpus systematically underestimates precisely this type of significance, masking the categorical role of the parameter as continuous insignificance.

Practical consequence: the significance of a parameter must be assessed along two independent axes, and not by one aggregate score: continuous sensitivity (the mean change of outcome when the parameter is varied over random minimal pairs, as in §11.3) and categorical sensitivity (the change of outcome specifically on deliberately constructed pairs that cross the supposed class boundary, and not on randomly chosen ones). A parameter that is silent by the first metric and significant by the second is not a contradiction but a sign that it marks out a class and does not vary a magnitude within a class; such a parameter must not be discarded by the compressibility criterion (§12) solely on the basis of low continuous significance.

### 11.8 Two different commutation tests: by physical parameter and by axis of description
The method of §11.3 admits two nonequivalent transfers that are easily taken for one, since both reduce to the formula "change exactly one thing, hold the rest fixed."

- **The test by physical parameter.** A property of the object itself is changed (a substituent, an isotope, the protonation state, the oxidation state), and the change in function is measured. This is what alanine scanning and MMPA do independently (§11.4), and it is a direct answer to the question of §11.1.
- **The test by axis of description.** One coordinate of the invariant description of a feature is changed while the rest are held fixed, and it is checked whether *membership in a unit* changes: whether this ceases to be the same letter. Function is not measured here at all.

The second test is available earlier than the first: it requires neither measured properties nor the synthesizability of the changed object, and is therefore applicable immediately after the units have been identified, before any semantics. In this it corresponds exactly to the original phonological test, which asks *which phoneme resulted*, and not what the word means. But it also answers only this narrower question: which coordinates of the description carry the distinction between units. One cannot conclude from its result that the corresponding physical parameter is silent: this is a statement about the structure of the description, and not about physics.

Two limitations of the second test are worth keeping in mind separately. First, the coordinates of an invariant description are in general algebraically linked: they are built from one and the same decomposition of the field, so independent variation of one of them may correspond to a physically unrealizable field. An "axis of description" here is not a synonym for a "parameter" in the sense of §11.7, and the categorical/continuous fork of that section is formulated for parameters and is not automatically transferred to coordinates. Second, the significance of an axis makes sense only relative to a null model: how strongly the outcome changes under a shift of the same magnitude in an arbitrary direction. Without it, a "significant axis" would mean only "the axis with the largest numerical range," that is, a property of the units of measurement.

Hence the rule for recording conclusions: the result of a test by axis of description is formulated as a statement about the representation, and the result of a test by physical parameter as a statement about the subject. Mixing them produces exactly the error against which §16 is written: a property of the instrument is declared a property of nature.


---

## 12. The criterion of "fundamental vs derived"

Not every measurable parameter of the field is equally important. One candidate may repeatedly describe the data of different molecules, another may require a new combination for each case. Therefore **reusability and the gain in compactness across the whole corpus** are considered here as operational criteria for candidates for basic units, and not as a previously proven claim about their physical fundamentality.

The principle of sparse dictionary decomposition (methods of dictionary learning, sparse coding, and independent component analysis, ICA): data are represented through a small common set of reusable components. As an operational criterion, a candidate for a basic unit must give a stable and compact description of different molecules; if each case requires a new special combination, the unit may be derived. This is a testable criterion for choosing a representation, and not proof that the chosen components are physically fundamental.

At the same time, the largest amplitude is not equal to the greatest explanatory value: a strong signal may reflect a common background or an artifact of scale, while a weak one may reflect a stable recurring detail. Therefore the contribution of a component must be compared with how much it recurs across molecules, is preserved under a reasonable change of the conditions of analysis, and improves the description of held-out data.

---

## 13. The phonetic realization layer: from classification to generation

### 13.1 The problem
Clustering of energy patterns (§11, §12) gives a classification: the ability to recognize that two different atomic ensembles realize one "word." Classification is not the same as command of a language: a dictionary that stores only abstract equivalence classes makes it possible to recognize what is already given but not to build something new. A generative capacity (in particular, for inverse design through direct tensor matching of a target pattern, rather than gradient descent over topology) requires a separate layer linking an abstract class with the set of its concrete physical realizations.

### 13.2 The linguistic frame: phonology and phonetics
A phoneme is an abstract equivalence class (for example, /t/) that unites different concrete articulatory realizations, allophones (aspirated, unreleased, etc.), which are determined by the phonetic context (position in the word, neighboring sounds) and are not random. The dictionary of a language is nevertheless obliged to store concrete transcriptions of pronunciation for practical command of the language, and not only the statement "this is one phoneme." A pattern descriptor invariant to the atomic realization (§11, §12) is phonology: an abstract class. A parallel registry of concrete "pronunciations," atomic realizations that physically embody the same class, is necessary; this is the phonetics of the theory.

### 13.3 The content of the realization layer
For each found equivalence class ("word"), the registry stores a set of known atomic realizations together with: a concrete atomic/structural record; the context of the medium in which this realization was observed (an indexical parameter, §4.3); a measure of typicality (the distance to the class centroid in the invariant space); and, where known, an independently measured stability/energetic accessibility of the realization (decay barrier, synthetic accessibility). The last is directly related to two-level grammaticality (§10): not every atomic record that formally belongs to a class is physically executable under the given conditions, and this must be recorded as part of the entry and not discarded.

### 13.4 Conditional realization as a testable hypothesis
By analogy with the conditioning of allophones by the phonetic environment, the choice of a particular atomic realization within a class is probably not random but systematically related to the context of the medium (§4.3): for example, if a less polar realization of a class is systematically preferred in contexts requiring greater lipophilicity, this is not a declaration but a testable correlation between the content of the registry of realizations and the indexical parameter of the medium. Its empirical check is a natural extension of the distributional analysis (§6.1) to the realization layer, and not only to the layer of abstract classes.

---

## 14. From decipherment to production: what is missing

The entire apparatus described up to this point (detection of critical points, invariant descriptors, clustering into letters and words, commutation testing, isosteric calibration, expectation profiles with explicit falsification (§15)) is an apparatus of **recognition**. In terms of language acquisition this corresponds to a receptive vocabulary: the ability to recognize a familiar word and understand its approximate meaning in context. In children the receptive vocabulary systematically and significantly outpaces the productive one (comprehension exceeds production by an order of magnitude), and this is not a random asymmetry but a structural one: comprehension requires only matching against an already stored pattern, whereas speaking requires generating a new, previously unencountered combination and immediately checking whether it worked. This section records this gap as a separate, later stage of the program, not because it is less important but because an attempt to close it prematurely would mean building a production mechanism on an alphabet whose stability has itself not yet been confirmed (§15).

### 14.1 There is no generative mechanism in principle
The `realization_registry` (§13.3) is a passive list of already known realizations of a word. In no part of the scheme is a procedure described that would sample a new, previously unencountered atomic configuration from `paradigm.transformation_rule` or from the typical composition of `constituent_letters`. This is the same gap that §13 pointed to from the very beginning ("classification is not the same as command of a language"), but the verification program (§15) does not solve exactly this task: the listed steps analyze or compare existing data, and do not generate and test new structures.

### 14.2 There is no cheap, fast feedback
A child who has said a wrong word receives a correction within seconds and immediately tries again. The only procedures in this theory capable of checking whether "the utterance worked" are a full DFT-level calculation, topological analysis, commutation testing, and the Mantel test on a calibration set (§15, step 5), which take hours or days per attempt. The cycle of trial and error on which language acquisition is built requires feedback that is orders of magnitude faster.

### 14.3 Error is not built in as an informative signal
Overgeneralization in children (for example, "goed" instead of "went," formed by analogy) is not a failure but direct evidence that the child has extracted a productive rule and has not simply memorized the forms. In the current scheme, `grammaticality_status: "physically_forbidden"` (§13.3) is a terminal failure label, cataloged in the `forbidden_map` (§10) after the fact, and not a signal that immediately refines the rule during generation. A productive system must be able to propose grammatically plausible but physically unexecutable combinations and learn from this at once.

### 14.4 The confidence requirements are incompatible with the pace of acquisition
In psycholinguistics, *fast mapping* is the ability of a child to approximately acquire a new word from one or several examples. The scientific verification of ESM is arranged differently: it requires assessing stability on repeated samples, prediction on held-out data not used in construction, and statistical significance (§11–§12, §15). This raises reliability but does not give the system a fast mode of generalization from a single example; such a mode is not part of the current program.

### 14.5 Even the receptive side is not complete
The grammar proper (§8–10, §17: transition models, the prohibition map, bifurcation points) so far exists as a scheme of analysis: work with trajectories and transitions (§15, steps 7–9) remains outside the current scope of the practical implementation. In other words, the analysis of levels above the word is not complete: an integral functional domain, for example the active center of an enzyme, is so far described as a research task, and not as a ready method.

### 14.6 The status of the "cheap surrogate validator"
From the gap listed (§14.2) it follows that production needs a cheap, fast surrogate for full verification: a rough estimate that "this combination is probably executable" that is orders of magnitude cheaper than a DFT calculation followed by topological analysis. It is important to state this precisely: such a validator is not the next item in the program that can already be designed now, but a hypothetical result whose very existence is conditional on the outcome of an earlier stage. Before designing a surrogate, one must pass through the stage of deciphering the alphabet (§15) and obtain an empirical answer to the question: does a finite, stable, compressible core of units exist at all (in the spirit of the criterion of §12, reusability across the corpus, extended from individual units to the rate at which new ones appear), or does each new compound continue, in essence, to require a new analysis? If the alphabet does not converge (the number of new, previously unencountered units does not decrease as the corpus grows, and classes found on one part of the data are poorly rediscovered on a held-out part), a cheap surrogate is impossible in principle: there would be nothing to train it on, since a surrogate model must generalize on stable, reusable regularities, and if such regularities do not exist physically, no validator architecture will solve this problem. Thus the question "how to build a surrogate validator" is logically not posed before the answer to the question "is there anything to reuse at all" is obtained: an attempt to design it now would mean solving an engineering task before the fact that makes it solvable in principle has been established.

### 14.7 A three-level verification architecture
If the alphabet converges (§14.6), the "cheap surrogate validator" is not a single monolithic component but a pipeline of increasing cost and increasing accuracy, arranged on the model of how a compiler stages its checks: not everything is done at one stage with one and the same tool.

- **Level 0: the preprocessor.** A purely symbolic, statistical check that recomputes nothing: a reference lookup of the proposed combination against structures already mined from the corpus: the `forbidden_map` (§10, known prohibitions by type: combinatorial/geometric/orbital/conditionally kinetic), `obligatory_pairing_score` (§3.2, the absence of a statistically obligatory neighbor), membership in an already confirmed `paradigm`/`transformation_rule` (§4.4–4.6, a known transition rule applied to an attested member: a strong prior signal), and the position in `position_sensitivity` (§17, hot spot/bifurcation point: not a rejection but an escalation to the next level). The cheapness here is not quantitative (faster than DFT) but qualitative: Level 0 performs no computation over the field at all, only a table lookup against what has already been calculated and stored: an O(1)/O(log n) operation, and not an O(field computation).
- **Level 1: the cheap surrogate** (§14.6 in the narrow sense): a semi-empirical calculation or an ML potential, for what passed Level 0 without an unambiguous verdict.
- **Level 2: full DFT**: only for final confirmation or for the targeted extension of the very grammar that Level 0 uses.

Without feedback from Level 2 to Level 0 (an error found at the expensive level must update the `forbidden_map`/`paradigm`), the architecture does not learn by itself but merely exploits a grammar mined once; this is a direct application of the principle of §14.3 (error as an informative signal, not a terminal label) to a concrete engineering scheme, and not an abstract wish.

### 14.8 The preprocessor is a linter, not a type checker
Here the direct analogy with the compiler stops, and this must be recorded explicitly and not left implied. The grammar of a programming language is precisely and completely specified by its creators: a syntax error is a fact, not a hypothesis. The grammar of chemistry in this theory is statistically derived from a finite corpus (§6, §15); it is not given in advance and cannot be given in advance, since the very object of decipherment is a structure that is not yet known in full. The preprocessor is therefore by its nature not a strict, reliable type checker but a linter with heuristic rules mined from the corpus, closer to modern static code analyzers that look for "code smells" by patterns than to a formal type checker.

Hence there are two types of errors that must be tracked explicitly and separately, and the preprocessor must not be considered infallible merely because it is cheap:
- **A false rejection**: a valid structure is discarded because the corpus at the time of rule mining simply never saw it. This is an error of corpus completeness, not an error of the method, and its frequency must fall as the corpus grows in the same way as `novel_letter_rate` in §19.7; if it does not fall, this is a symptom of the same incomplete convergence of the alphabet.
- **A false acceptance**: an invalid structure is accepted because it accidentally resembles an allowed pattern along those axes of the descriptor that the commutation test (§11) has not yet checked in this particular region of the space.

Both quantities are not an optional metric for after-the-fact reporting but a mandatory part of the definition of the preprocessor itself: the statement "Level 0 works," without current estimates of both types of error accompanying it, has no testable content.

### 14.9 The order of dependence: nothing works before its turn
The architecture of §14.7 consumes artifacts ("letters" in §3, "syllables"/"words" in §4, transition rules in §4.4–4.6, the prohibition map in §10, positional sensitivity in §17) that at the current stage of the program do not yet exist in a finished, saved form. The first-priority task is not building the preprocessor but the sequential establishment and **preservation of the results of the grammar analysis** precisely in a form that Level 0 can use as a ready table: first the candidates and types of units (§3.1–3.2, including hypotheses A–D and the alternatives yet to be added), then syllables/words and the rules of their composition (§4, §9–10), and only after that the assembly of the `forbidden_map`/`paradigm`/`position_sensitivity` into a single, versioned source from which the preprocessor does a table lookup and does not recompute the grammar on the fly at every call. An attempt to assemble Level 0 before these artifacts have stabilized would produce a linter trained on a demonstrably incomplete and, worse, not yet convergence-tested (§19.7) body of rules: that is, a tool whose false-positive and false-negative rate in both directions (§14.8) is unknown and cannot be reliably estimated, because the corpus from which the rules were derived has not yet settled.

---

## 15. From hypotheses to calculation: a program of empirical verification

Up to this point we have asked what may turn out to be a unit, how to distinguish it from context, and why the analogy with language is appropriate at all. Now this is translated into a sequence of calculations. At each step a testable result is indicated; static structures are considered separately from the process of their formation, which requires time trajectories (§9). An empty output of the whole chain is not counted as a negative discovery: it means that observation has not yet resolved the subject (§§3.3, 22.9).

### Four levels of decipherment: from "letters" to the rules of the language

These are four levels of questions to the corpus of fields, and not four previously proven objects and not rigid gates through which the research must pass only on one desired answer. Results accumulate and are versioned; new runs supplement the processed molecules and will continue unfinished computations. A change in the definition of the field, the level of theory, or the method of search creates a separate branch of the analysis, so that differences of representation are not mixed imperceptibly. Each level can give a useful result even when the next remains open.

1. **The search for "letters": the first and currently the main implemented level.** The algorithm looks in three-dimensional fields for recurring features and builds preliminary similarity classes. The search must be broad and permits a redundant result: one motif may give rise to several candidates because of scale, context, discretization, or incompleteness of the descriptor. Within this level are tasks that should not be confused with the subsequent levels of the language: filtering of undescribed parts of the field; the search for missed units and their fragments; the study of transitional forms, if trajectories are available; the identification of contextual "letterforms"; the search for invariants, variants, and field classes. The remainder is not declared a letter automatically: it may be noise, an artifact, or a signal that the representation needs to be extended.

   This is the most expensive stage, since for each molecule a suitable field must be obtained (often by carrying out a quantum-chemical calculation, building a dense grid, and writing/reading large `.cube` files) and extraction with quality control must be carried out. A corpus of millions of molecules means significant expenditures of computing time, disk space, and input-output. Therefore caching of fields and features, batch/parallel processing, limiting simultaneous RAM consumption, and the ability to resume are part of the reproducible method, not a cosmetic optimization. A GPU is useful where the specific calculation or batch operation is supported by the chosen library; its presence by itself does not speed up every DFT calculation and the whole pipeline.

   In `esm_master_analytics.py`, pilot downstream diagnostics are already present, for example estimates of compressibility, distributions, stability, and relation to known properties. They are useful as separate checks, but they do not mean that a systematic analysis of the entire remainder of the field, a check of all contextual variants, or a complete matching of units across environments is complete. The current main output is the candidates found and their preliminary grouping, and not a completed alphabet.

2. **The search for rules for the "letters."** Once even preliminary candidates have appeared, their local behavior and usage can be investigated: which changes of the descriptor preserve the unit and which give rise to variants; in which field and environmental contexts it occurs; which combinations of features are stable and which are forbidden or rare; how the unit changes under a controlled perturbation. These are rules of the letter level: regularities of identification, variation, and local surroundings. They are tested on independent molecules and controls, and clustering thresholds are not accepted as ready-made laws. The result may confirm a familiar atomic regularity, reveal another context, or show that a particular supposed rule does not transfer.

3. **The search for "syllables" and "words" and the testing of their meaning.** Next, recurring combinations of letters and of the relations between them are sought, which form stable field motifs of larger scale. Unlike the decipherment of an unknown script, the meaning of a word does not have to be guessed blindly: the corresponding chemical function or property may already be known from measurements and calculations, or may be computed by standard chemical methods. The question is then formulated in a testable way: does the field combination single out this meaning, does it transfer to new molecules, and does it give a representation different from a simple repetition of an atomic label? A known function becomes an independent check, and not a circular definition of the word. At the same time the possibility is preserved of discovering stable words whose meaning is not yet described by known properties.

4. **The search for the grammar and rules of the language.** At this level it is established how letters, syllables, and words form larger structures and how their admissible combinations depend on context. For static corpora these are rules of composition, constraints, and positional dependence; for the performative side, regularities of transitions between states in a reaction, binding, collision, or a change of medium or conformation. Such a grammar requires not only snapshots of finished molecules but also comparable sequences of states, trajectories, and conditions. The rules may turn out to be compact, partial, scale-dependent, or no more economical than standard physics; all these outcomes report where exactly the explanatory power of the field language lies.

   **Modeling transitional fields for "words" is a separate, high-cost, and cross-cutting task.** To study the change of the field in the course of a reaction, it is not enough to save the `.cube` of the initial and final molecule: comparable fields are needed for a sequence of intermediate states and, depending on the question, transition states, alternative paths, conformations, and conditions of the medium. For each frame one has to obtain or verify the electronic structure, save volumetric data, consistently match coordinates and grids, and track how the patterns found arise, vanish, and transform. The number of calculations and the volume of storage may exceed the cost of static fields for the same set of molecules many times over.

   At the same time we do not know in advance to which level of the language the discovered dynamics should be assigned. A stable part of a transitional pattern may clarify what to count as a "true letter"; dependence on the medium may turn out to be a variant of its letterform or a word form; a sequence of transformations may reveal the rules of spelling a word; the constraints and directionality of transitions may reveal the grammar of execution. It is also possible that one and the same process clarifies several levels at once. Therefore dynamic fields should first be saved and described as sequences of observations, and only then should competing interpretations be tested. This is a cross-cutting line that can refine all four levels, and not a category assigned in advance solely to the "grammar." Every result, including the absence of a stable transitional pattern or a dependence on specific conditions, remains informative.

None of the four levels is an exam whose failure nullifies the research. Supra-atomic motifs found at the first stage do not prevent seeking rules for them; the failure of a particular rule does not devalue the inventory; the absence of a ready semantics does not prevent the discovery of a repeatable word; the absence of trajectories limits the conclusion about a dynamic grammar but does not erase the structural results. A null or ambiguous result is information about the limits of the current method and a reason to refine the search, the data, or the scale, and not an automatic conclusion that a field language is absent. Results can return to the previous levels with the provenance and version preserved: new "letters" refine words, new words help to discover rules, and the regularities found suggest which remainders of the field to check. This is an iterative program, not a promise of a monotonic decrease of the alphabet or of a guaranteed compactness of the grammar.

### Practical precautions of the extraction pipeline
Three systematic sources of artifacts require explicit elimination before proceeding to clustering:

- **Separating the level of theory for geometry and for field topology.** Cheap semi-empirical methods (a minimal valence basis) are suitable for geometry optimization, conformational sampling, and rough energetic screening of candidates, but the density matrix at this level is a poor approximation of the true ρ(r): first of all, the polarization and diffuse effects are lost, which determine precisely the fine features (weak noncovalent critical points, secondary ELF features at lone pairs). The topological analysis of the field (critical points, basins) must be counted as a separate, higher level of theory (with polarization functions and a dispersion correction) on top of the cheap geometry, and not at the same level at which the geometry was obtained. An additional check of confidence in a found feature is the stability of its type, index, and position on passing between basis sets of different quality; features that are not stable under such a transition are marked as an artifact of the method, and not as a physical "letter."
- **Soft, not hard, region boundaries.** Quantum-chemical fields, especially delocalized ones, do not have sharp boundaries; they flow smoothly into one another. A hard cutoff of the field at the topological boundary of a basin creates edge artifacts, especially sensitive in the subsequent spectral decomposition. The basin boundary is used only to determine the centroid and the characteristic scale of the region, and not as a binary mask: the cut-out region is a volume of fixed radius around the centroid with a weight that decays smoothly toward the edge, so that the contribution of neighboring regions weakens gradually and does not cut off abruptly.
- **Invariants that do not depend on the choice of a local coordinate system.** For axially or spherically symmetric fragments (linear groups, symmetric aromatic systems, spherical ions), the tensor of second moments of the field is degenerate, and an explicit choice of local axes on its basis is physically undefined; it is chosen by numerical noise, because of which physically identical patterns receive representations randomly rotated relative to one another and do not coincide on comparison. The solution is to use descriptors invariant to rotation by construction (contractions of the coefficients of the expansion of the field in spherical harmonics over all components at a fixed order of expansion), and not descriptors that require a preliminary choice of a frame.
- **Discretization of the field is as much a choice of representation as the level of theory.** The field in a calculation is given not as a function but as a set of samples on a grid plus a rule for restoring values between them. Both the grid step and the order of this rule enter the result: the classification of a feature rests on the second derivatives of the field, and for a low-order reconstruction they are determined by the distance at which they are computed, and not only by the field itself. The stability check formulated above for the basis set must therefore extend to this axis as well: a feature that changes type, index, or position on refining the grid or changing the reconstruction rule is marked in the same way as one that did not survive a change of the level of theory. The default "the grid is fine enough" imperceptibly turns a calculation parameter into a statement about the nature of the object.
- **The set of features found is a lower estimate, not a complete list.** Any practical search for zeros of the gradient is multi-start: it finds what at least one start converged to. A guarantee of completeness is given only by procedures with a proof of the absence of missed solutions in the volume, which are substantially more expensive. Until they are available, the inventory of the corpus is a lower bound, and this directly affects the reading of the curves of §19.7: reaching a plateau may mean either that the alphabet has been found or that the ability of the procedure to find anything new has been exhausted. These two cases are distinguished only by an independent change of the search procedure itself with the corpus unchanged: if the inventory grows as a result, the plateau was a property of the search and not a property of chemistry.

**The operational plan: calculations and checks within the four levels:**

The four levels above describe *what* we are looking for, from letters to rules. The following numbered list describes *which data and checks* we will use; it is not one more hierarchy of the language and not a strict calendar schedule. Items 1–6 relate mainly to static fields, 7–9 to trajectories and constraints on transitions, and 10–11 are additional cross-cutting checks of representation and dimensionality.

1. **An open search for candidates for "letters"**: several parallel branches, and not one object chosen in advance. The point branch looks for critical points and basins of different fields; the regional one for local forms and distributions of values, including joint patterns of maxima, minima, and saddle regions; the relational one for the mutual position and topology of groups of features; the void branch for connectivity, cavities, and channels (§2.3). Candidates may be, for example, features of the electrostatic potential (ESP), ELF basins, groups of regions with stable connectivity, or multipole characteristics of the field distribution. These are examples of search objects, and not a complete list of possible letters.
2. **An invariant representation**: construction of a region descriptor invariant to rotation by construction (see the precautions above), giving a "fingerprint" comparable between different molecules without the need to align the whole molecule or to choose a local coordinate system.
3. **Comparison of repeatability within each branch**: candidates of one representation family are first compared among themselves and are not forcibly mixed with objects of another type. For point features the type of critical point is taken into account; for regions, relations, and voids the corresponding invariants of form and context are applied. Clustering is one way of initial grouping, not a definition of a letter in itself. Each class receives a bootstrap estimate of stability and then undergoes sensitivity to resolution, thresholds, and the level of calculation; the matching of classes between branches is carried out separately.
4. **The contrast-pair check (commutation test)**: for each feature of a found unit, two maximally similar records that differ only in this feature are compared (by analogy with alanine scanning and MMPA, §11.4). In this way it is checked whether a change of the feature is associated with a change of the unit or of function; the other parameters are held constant where possible.
5. **The test on isosteric pairs: a controlled design, not a single pair**: one positive pair (`-COOH`/tetrazole) is not enough for proof; the design must cover all four combinations of bond-graph similarity and function similarity (by analogy with the multitrait–multimethod matrix and the methods of Campbell and Fiske): dissimilar graph/similar function is the actual test of the hypothesis (the positive case); similar graph/dissimilar function is a mandatory negative control (for example, an acid and a structurally close amide must give different, not converged, invariants; otherwise the method picks up only trivial graph similarity and not physics); similar graph/similar function and dissimilar graph/dissimilar function are additional calibration points. With a small number of examples, significance is assessed with the Mantel test: a comparison of the matrix of independently known functional similarity (measured pKa, binding constants, literature MMPA data) with the matrix of distances in the invariant space, with significance estimated by exact enumeration of permutations and not by asymptotic statistics. The success criteria are fixed before the calculations in order to avoid post hoc fitting of the interpretation. An important caveat: the reproduction of an already known, textbook isosterism is a necessary calibration of the pipeline, not proof that the theory sees something inaccessible to the classical view; a substantive test requires applying the calibrated invariant to a group that was not part of the training sample, with an explicit quantitative prediction later checked against independent data (see §16 on the epistemology of such a check).
6. **Distributional analysis on a large corpus**: applying the method of Ventris (§6.1), positional/contextual statistics of the occurrence of motifs over a large number of known molecules, before any attempt to ascribe meaning.

**The void topology branch (§2.3).** Regions of low density are analyzed on a par with the features of the field: their shape, connectivity, cavities, and channels receive their own descriptions, stability checks, and reuse statistics. This branch is not derived from the number of critical points and must not be lost in constructing the alphabet; the results of the different branches are compared only after each has been described at a level suitable for it.

**Data about the process (transitions, not only final states):**

7. **Building a corpus of trajectories instead of a corpus of snapshots**: supplementing static databases with data from single-molecule experiments (for example, FRET, energy transfer between fluorescent labels, or optical tweezers), molecular dynamics simulations, and computed IRC reaction paths (the trajectory of the minimum-energy path from reactant to product; §9.2). It is also important to vary the conditions of the medium, for example the polarity of the solvent or the environment of a protein pocket (§4.3), in order to test the response of the field to context.
8. **Clustering in the MSM formalism**: metastable states along trajectories as the alphabet, the matrix of transitions between them as a first-order grammar, instead of clustering isolated points in the space of descriptors.
9. **Building the prohibition map**: the systematic collection of not only successful but also unsuccessful paths: reversible reactions, decayed intermediates, rejected synthesis attempts (data usually discarded in standard databases oriented only toward finally stable structures). Classification of prohibitions by type (combinatorial, geometric (ring strain, anti-Bredt), orbital (Woodward–Hoffmann), conditionally kinetic (§10.2)) as a separate source of grammatical rules, no less informative than a list of allowed words.

**Cross-cutting checks of representation:**

10. **Sparse decomposition onto a common basis**: applying the criterion of §12 to candidate descriptors in order to check whether different molecules can be described compactly by reusable components. "Fundamentality" here means the result of such a comparison, and not a previously established property of the object.
11. **Estimating the dimensionality of the description**: for a calibrated set of molecules, calculating the Fukui function and the polarizability tensor as additional coordinates and estimating the intrinsic dimensionality of the resulting set of features by the methods indicated in §8.2. This is a separate task from the reconstruction of dynamics by Takens' theorem.

---

## 16. The epistemological problem of validation criteria

### 16.1 Formulation of the problem
Any check of a field description against previously chosen properties (for example, pKa or a binding constant) risks answering only the question of whether it reproduces already known chemical distinctions. In other words, the choice of the target metric of the check limits in advance what we are able to notice. This cannot be avoided completely, but the risk can be reduced by combining a check against known properties with independent structural tests and a search for stable unexplained classes.

### 16.2 Checking without predefined labels
Besides checking the relation of the description to known properties, one can check whether it gives a more compact description (§12) than a control built on shuffled or randomly chosen regions of the field of the same scale. Such a check does not require naming the classes found in advance, although it does not by itself prove that they are physically significant.

### 16.3 The search for "orphan" classes
An "orphan" class is here conventionally a stable class, reproducible under resampling, whose relation has so far not been found with any of the previously known properties tested. This does not mean that it necessarily has a new function: the cause may also be incompleteness of labels, a lack of data, or an artifact of the descriptor. Such classes are candidates for subsequent investigation. If, after the checks are extended, all stable classes are explained by known properties, this will constrain the hypothesis of an additional hidden functional level, but will not refute the field description as such.

### 16.4 Anomaly as the engine of discovery
In practice this means: not inventing a meaning for such a class from its form, but measuring a broad set of independent properties for it and establishing which of them are reproducibly associated with membership in the class. The analogy with the discovery of Neptune is useful only in this limited sense: an unexplained deviation may indicate where to look next; it does not guarantee that a new entity is hiding behind it. The result of such screening is a testable association, which then needs to be confirmed on independent data.

### 16.5 Requirements for anchors: independence of source and matching of level
Section §6.2 introduces anchors as places where form and function are known at the same time, but does not specify two requirements without which an anchor does not work as a check and only looks like one.

**Independence of source.** An anchor must come from a measurement that is not derived from the same field on which the description is built. A quantity computed from the same electron density and at the same level of theory correlates with the description partly by construction: the check then shows that two functions of one and the same calculation are related to each other, and says nothing about whether the description captures function. This is not a prohibition on using computed properties (at early stages they are often the only thing available in the required volume) but a requirement to declare the degree of their dependence and not to count the resulting correlation as an answer to §16.1. A full-fledged anchor is an independently measured quantity, even if there are orders of magnitude fewer such points than computed ones.

**Matching of the hierarchy level.** A property measured for a whole molecule belongs to the level of the "sentence" (§3). To compare it directly with the description of one local feature means to test not the letter but an unspoken hypothesis that this particular letter determines the property of the whole. There are two correct paths here: either an anchor attributable to the region itself (a quantity defined for a region of the field and not for the molecule), or an explicit composition model in which the contribution of each unit to the property of the whole is written out and is itself subject to testing, and then it is precisely this model that is tested, together with §19.6 (the gap between the additive prediction and the calculation of the assembled whole). The tacit identification "property of the molecule ≈ property of its most conspicuous letter" is not a coarsening but a substitution of the subject of the test.


---

## 17. Robustness of meaning to late perturbation: an analysis available only after partial decipherment

This section describes not the initial stage of decipherment (§15) but a subsequent analytical layer, applicable only when part of the dictionary has already been decoded (§11–§13): the definition of "position in a sentence" and "order of adding words," used below, itself requires that the units of the alphabet have already been identified. The section formalizes an observation: the context accumulated by the preceding "words" is not neutral with respect to the next one; most of it creates inertia that is robust to perturbation, but in individual positions the same system is sharply sensitive to a small change.

### 17.1 Known examples of unequal sensitivity
- **Sharp differences in activity between close compounds** (*activity cliffs*): in studies of quantitative structure–activity relationships (QSAR), pairs of compounds with similar structure but sharply different biological activity are observed (cf. controlled pairs, §15, step 5). Such cases show that a local change sometimes has a large effect; this is an analogy for testing ESM, and not a confirmation of its model.
- **Key contact positions of a protein** (*hot spots*): alanine scanning of protein interfaces shows that the replacement of some amino acids sharply changes the strength of binding, while that of many others hardly changes it at all. This motivates looking for unequal sensitivity to structural changes; transferring the conclusion to field "words" will require measurements of our own.

### 17.2 A possible model of context accumulation, and canalization
A Bayesian analogy helps to formulate how new information changes the estimate of the already accumulated context: usually a single piece of evidence only refines the earlier picture, and for a sharp revision it must noticeably change the ratio of likelihoods. In ESM this is for now a way of posing a question, not a physical explanation of a reaction. What must be tested is whether the later addition of a particular motif really changes the prediction of a property or reaction path sharply, and whether such an effect is reproduced in independent examples.

In Waddington's theory of development, "canalization" means the robustness of a trajectory to some perturbations; a similar robustness can also be sought in the energy landscape (§9.2), but it cannot be derived from an analogy. If most controlled changes have little effect on the outcome while a small group of positions or stages changes it noticeably, this will be an observable profile of robustness and sensitivity. The cause of such a profile remains to be established separately.

### 17.3 Possible switching points in the course of a process
A hot spot is an element of an already formed structure whose change strongly affects function. A separate possibility is a point on the reaction path where a small change of conditions switches the system between competing mechanisms. A distant linguistic analogy is a sentence that at first admits one reading but later requires revision; the chemical version must be confirmed by trajectories and barrier calculations, not by similarity of names. Distinguishing these scenarios requires data about the course of the process, and not only a comparison of final molecules.

### 17.4 Methodological status: extending the commutation test to a trajectory
A stable background, individual sensitive positions, and switching points in the course of a process cannot be distinguished by a simple comparison of final structures. The commutation test (§11.3) changes one factor in a pair of comparable cases; for the analysis of the order of changes it must be extended: related trajectories are compared and it is measured how each subsequent step changes the prediction relative to the already accumulated context. Such an analysis is possible only after the units have been found and ways of matching positions have been established; its result will be a sensitivity map, and not a new letter in itself.

---

## 18. Explicit limitations of the theory

- Even methods that then analyze the field without atomic markup (QTAIM, ELF) often obtain their source data from a calculation with atom-centered basis functions. A calculation on a spatial grid without such functions is possible but is usually more expensive.
- A continuous 3D field has no natural canonical frame of reference; the task of finding a recurring motif in unaligned field data is orders of magnitude harder than a subgraph search on atomic graphs, where the problem of canonicalization was solved long ago.
- The functional groups of traditional chemistry are not abolished by the theory; they remain a working approximation where the contextual correction is small. The theory predicts concrete limits of applicability of this approximation (§4.2), and does not declare it wholly untenable.
- Polysemy built into the physics (§6.5), rather than being a rare exception, means that a direct transfer of the methods of deciphering natural languages is insufficient: it requires supplementation with methods of analyzing dynamical systems and the theory of performative acts, and not only distributional linguistics.
- The analysis of static fields (steps 1–6) does not by itself recover the grammar of transitions (§9.1); it yields a dictionary and preliminary hypotheses about the links between words. Conclusions about dynamics require testing on a corpus of trajectories (§15, steps 7–9).
- Standard chemical databases are biased toward synthesized and stable structures and away from failed attempts; therefore building the prohibition map (§15, step 9) will require a targeted collection of data that are rarely available in ready form.
- The status of any parameter as "distinctive" or "silent" (§11) is not absolute but is determined by the range of contexts tested; a negative result of a commutation test does not exclude the existence of a narrow context in which the same parameter turns out to be decisive (an analogue of the kinetic isotope effect, §11.5).
- The distinction between indexicality (§4.3) and genuine polysemy (§4.1(2)) itself requires empirical testing over a sufficiently broad range of environmental conditions: with a small number of conditions studied, a smooth indexical shift and a jump-like polysemy may be statistically indistinguishable, and the conclusion about the character of the dependence remains preliminary until data are accumulated at diverse values of the environmental parameter.
- The layer of concrete realizations (§13) requires data of larger scale than the minimal controlled test (§15, step 5): a reliable check of the dependence of the choice of realization on the medium (§13.4) needs a diverse corpus, and not only calibration examples.
- The practical precautions of the extraction pipeline (separation of the level of theory, soft boundaries, frame-independent invariants, §15) reduce but do not completely eliminate the risk of systematic error of the method: the stability of the findings on passing between levels of theory is a necessary but not a sufficient check of their physical reality.
- The limitation of the epistemology of validation (§16) applies to the program as a whole: even the presence of stable, reproducible "orphan" clusters (§16.3) demonstrates the existence of an unexplained structure, but does not guarantee that this structure corresponds to a meaningful chemical function rather than to an artifact of the chosen set of descriptors; distinguishing these two outcomes requires independent empirical verification (§16.4), and not the stability of the cluster alone.
- The analysis of the robustness of meaning to late perturbation (§17) is methodologically secondary with respect to the rest of the program: it cannot be carried out at the stage of constructing the alphabet (§15), since the very posing of the question about "position in a sequence" and "order of context accumulation" presupposes an already partially decoded dictionary. An attempt to apply it earlier will lead to confusion between segmentation noise (§6.4) and a genuine bifurcation point (§17.3).
- The basic assignment of a specific type of critical point to a "letter" versus a "word form" (§3.1) has not been established and may turn out to be wrong in both of the proposed variants; the whole subsequent hierarchy (§3), including the extraction program (§15, step 1), must be regarded as conditional on this choice until the distributional test (§3.2) has been passed, and not as an agreed fact of the theory.
- The distinction between synonymy and inflection (§4.4–4.6) does not guarantee in advance that both classes actually exist in the corpus or that the boundary between them is sharp: an outcome is possible in which almost all of the observed variability turns out to be of one type and the other is statistically empty; this is an open empirical question, and not a structural assumption of the theory.
- The section on the transition from decipherment to production (§14) describes a gap, not a solution: in particular, a cheap surrogate validator (§14.6) is not being designed within the current program; its feasibility and the very possibility of its existence depend on the outcome of the deciphering of the alphabet (§15) and remain unknown until this result is obtained.
- The inventory of features extracted from the corpus is a lower estimate of completeness (§15): the leveling off of the alphabet discovery curves (§19.7) onto a plateau does not by itself distinguish "the alphabet has been found" from "the search procedure has exhausted its capabilities," and this distinction requires a separate variation of the procedure itself.
- A significant part of the anchors available in the required volume is computed from the same field on which the description is built (§16.5); early positive results of correlation checks are therefore systematically inflated relative to what independently measured quantities would give.
- A prohibition map built from the absence of observations in a static corpus loses informativeness as the corpus grows, and does not gain it (§21.4), and does not replace a targeted collection of failed paths (§15, step 9).
- The topology of the void (§2.3) is included in the program as a parallel layer (§15) but has not yet been run on any corpus: all statements about the alphabet at the moment refer to the layer of singular points and are not automatically extended to the void layer.
- The response to an environmental parameter (§4.3, §13.4) requires a corpus that deliberately varies the medium; a corpus collected under one fixed condition makes the corresponding checks undetermined, and not negative.
- Conclusions obtained on a corpus of debugging scale (§21.1) are statements about the workability of the pipeline, not about chemistry, regardless of how meaningful the numbers obtained there look.

---

## 19. Target data: expected profiles as a criterion of falsification

The verification program (§15) describes what needs to be computed. Here, for each level, it is specified in advance which result would support the corresponding hypothesis and which would force its revision. This is not a prophecy about the form of all chemical corpora: the profiles below are testable expectations and reference points. Their exact criteria are fixed before the analysis of a particular corpus; the failure of one profile limits the corresponding hypothesis but does not prove the absence of any structure and does not automatically refute the other levels of ESM. The blocks in monospaced font are proposed names of report fields and expected checks, and not results already obtained or a mandatory implementation format. Complete emptiness requires first returning to the adequacy of observation (§§3.3, 22.9).

### 19.1 A reusable pattern for checking an expectation
Each item below is not a bare metric but a comparison of two quantities with two different kinds of uncertainty, which cannot be reduced to one number without loss of meaning:

```text
ExpectationCheck
{
    expected_shape      # parametric or qualitative shape predicted by the theory
    observed_shape      # shape actually measured on the current corpus
    test_method         # a suitable test of distribution shape or a permutation check
    p_value             # agreement of observed with expected: about the HYPOTHESIS
    confidence          # reliability of the estimate of observed_shape across repeated samples: about the MEASUREMENT
    corpus_version
}
```

`p_value` answers "does the observed resemble the predicted"; `confidence` answers "how stably was the observed itself measured," regardless of whether it coincides with the prediction. An estimate may be precise but not correspond to the theory, or it may correspond to the theory but be based on too small a sample; mixing these two questions into one number loses precisely the distinction for the sake of which this whole section exists.

### 19.2 The profile of the letter corpus
```text
ExpectedLetterCorpusProfile
{
    alphabet_size_curve: ExpectationCheck
        # expected: sublinear, saturating growth of the number of unique stable
        # classes with corpus size (an analogue of Heaps' curve for the
        # vocabulary of a language).
        # Failure: linear growth without saturation: the "letters" are unique to
        # each molecule, there is no reusable alphabet.

    cluster_stability_distribution: ExpectationCheck
        # expected: distribution of stability under resampling (§15, step 3):
        # a cluster near 1.0 (real letters) separate from a cluster near 0
        # (segmentation noise). Failure: a smooth
        # unimodal continuum without a gap: the "letter/not a letter" boundary
        # does not exist physically, it is an artifact of the clustering threshold.

    occurrence_frequency: ExpectationCheck
        # expected: a power-law (Zipfian) frequency distribution: a small number
        # of very frequent letters, a long tail of rare ones. Failure: a uniform
        # distribution: there is no hierarchy of basic and specialized units.

    intrinsic_dimension: ExpectationCheck
        # expected (§12): the true dimensionality of the manifold of invariant
        # vectors << the raw dimensionality of bispectrum_vector. Failure:
        # intrinsic_dimension ≈ raw_dimension: the invariant does not capture any
        # hidden structure, it simply stores raw data without compression.

    entropy_per_letter: ExpectationCheck
        # expected: low within-class entropy relative to between-class entropy.
        # High within-class entropy while low bootstrap
        # instability is retained is a signal that the class merges heterogeneous
        # phenomena (§15, step 3, a partitioning that is too coarse).
}
```

### 19.3 The profile of the word corpus
```text
ExpectedWordCorpusProfile
{
    word_length_distribution: ExpectationCheck
        # expected: a meaningful modal scale of the number of letters per word.
        # Failure: arbitrary, unbounded length without a typical scale:
        # there is no natural unit "word" between the letter and the whole molecule.

    constituent_letters.occupancy_frequency: ExpectationCheck
        # expected: for most words, high occupancy of the role by one letter
        # class, but with a meaningful share of words showing low occupancy
        # while function is preserved (synonymy of a letter within a word, §4.4).
        # Failure: occupancy is either always ~100% (there is no variability,
        # the theory is redundant), or always random (there is no stable
        # composition at all).

    connectivity_topology.topology_variance: ExpectationCheck
        # expected: low for real words: the bond graph is reproducible
        # between instances at a fixed class center. High variability with
        # a small distance in the invariant space is a sign that two different
        # words have been glued together under one class_id.
}
```

### 19.4 The profile of the grammar
```text
ExpectedGrammarProfile
{
    co_occurrence_matrix.ppmi_distribution: ExpectationCheck
        # PPMI is a measure of co-occurrence beyond random expectation.
        # expected: an uneven, clustered structure of the word graph: hub
        # nodes and a periphery (an analogue of the syntactic graphs of natural
        # languages). Failure: uniform connectivity: there is no grammar, any
        # word combines with any other.

    forbidden_map.coverage: ExpectationCheck
        # expected: a meaningful but not extreme share of combinatorially valid
        # word pairs (§10) is physically forbidden: not ~0% (the grammar is
        # trivial) and not ~100% (the language is degenerate, one trajectory
        # is allowed).

    transition_model.sparsity: ExpectationCheck
        # expected (§9.2, MSM): a sparse transition matrix: from each state a
        # small number of next states are accessible with high probability,
        # and not a nearly uniform distribution over all states.
}
```

### 19.5 The profile of word meanings
```text
ExpectedWordMeaningProfile
{
    named_correlates.coverage_fraction: ExpectationCheck
        # expected: PARTIAL coverage of stable words by a significant
        # correlation with at least one known property (§16.3). 100% coverage
        # is suspicious (overfitting to already known chemistry, the theory
        # says nothing new); 0% coverage is also a bad sign (the invariant does
        # not track anything chemically meaningful). A healthy result
        # contains both poles: some words with named_correlates, some orphans.

    descriptors: ExpectationCheck
        # expected: meaning as a distribution over several independent axes
        # (acidity, nucleophilicity, redox potential, etc.) shows a
        # structure of variation not reducible to one hidden parameter. Complete
        # collinearity of all descriptors is a signal that the multidimensionality
        # of meaning is illusory (in itself a different, but also meaningful result).

    paradigm_correlated_shift: ExpectationCheck
        # expected (§4.4–4.6): members of one paradigm show a consistent
        # shift of descriptors along the axis of variation of the paradigm. This is
        # an operational check that inflection really is inflection, and not
        # unrelated words that happen to be grouped together.
}
```

### 19.6 The profile of sentence meaning
Before introducing `emergence_gap`, it is important not to pass off similarity as identity. In quantum chemistry, correlation energy is defined through the difference between the Hartree–Fock energy and a fuller correlated description; this shows that an independent one-particle approximation does not exhaust a many-particle system. But this quantity is not a ready-made measure of the "non-additivity of words" and does not prove the ESM analogy. For field motifs, `emergence_gap` must be defined separately: compare a previously specified prediction of the whole from its components with an independent result for the whole, using one and the same target quantity and held-out data. Correlation energy can serve as a methodological analogy and a reminder of non-additivity, but not as a direct quantitative replacement for this check.

```text
ExpectedSentenceMeaningProfile
{
    emergence_gap: ExpectationCheck
        # Checks whether a previously specified prediction of a property of the
        # whole from its components is worse than an independent prediction
        # for the whole.
        # The numerical metric must be defined separately on one target quantity
        # and held-out data; correlation energy in quantum chemistry can be
        # mentioned only as an analogy of non-additivity, and not as this metric.
        # If the compositional model is not inferior to the model of the whole,
        # the specific hypothesis about the added value of the "word/sentence"
        # level is not supported.

    execution_success_rate_by_grammaticality_status: ExpectationCheck
        # expected (§10): "attested" >> "combinatorially_valid_untested" >>
        # "physically_forbidden" by frequency of actual execution: the three
        # statuses must really be separated by outcome, and not be a decorative label.

    execution_vs_sentence_meaning: ExpectationCheck
        # Match the expected property of the whole structure with the outcome of
        # the specific process of its formation; the conditions of the medium,
        # barriers, and competing reaction paths are taken into account
        # separately (§7.2). This is a proposed separation of indicators,
        # and not an already established "semantics" of the sentence.
}
```

### 19.7 The profile of the alphabet discovery process
This is a separate, stronger question than the static distributions of §19.2–19.6: not "what does the alphabet look like," but "does the process of discovering it itself come to a stop," directly connected with the status of the surrogate validator (§14.6):

```text
ExpectedAlphabetDiscoveryProfile
{
    candidate_count_curve: ExpectationCheck
        # For the critical-point branch, expected: growth roughly proportional
        # to corpus size, since the number of raw points per molecule is finite.
        # The regional and relational branches have their own candidate-count
        # curves; they must not be combined into one indicator.
        # Saturation here would mean that the molecules of the corpus are too
        # similar to one another, and not that the alphabet has been found.

    merge_rate: ExpectationCheck
        # expected: decreasing to zero as the corpus grows. If it does not
        # decrease, the segmentation threshold is unstable, the letter
        # boundaries "float" and are not being discovered.

    split_rate: ExpectationCheck
        # expected: also decreasing, but from the opposite side: initially
        # coarse classes are split as the commutation test (§11.7)
        # reveals hidden categorical subparameters. Joint convergence of
        # merge_rate and split_rate to zero is a strong sign: the alphabet
        # has stabilized from both sides at once.

    stable_core_fraction: ExpectationCheck
        # expected: a Pareto-like shape: a small core of classes stabilizes
        # almost immediately, the long tail of rare classes is revised longest.

    novel_letter_rate: ExpectationCheck
        # expected: a decreasing function of corpus size (an analogue of Heaps'
        # curve). Even in natural languages this curve rarely reaches exactly zero
        # on a finite corpus; the success criterion is convergence of the rate
        # of decrease, and not a literal zero.

    rediscovery_rate: ExpectationCheck
        # The decisive item: the share of critical points from held-out (not used
        # in construction) molecules that fall into already known classes and
        # do not require new ones. expected: high, close to the corpus_coverage
        # of the training corpus. A low rediscovery_rate with an outwardly
        # decreasing novel_letter_rate on the training data is a sign of
        # overfitting of the alphabet to a specific corpus, and not of the discovery
        # of universal units; this criterion is stricter than the others, since it
        # does not depend on how the curve is computed on the same data that were
        # used for training.

    corpus_coverage: ExpectationCheck
        # expected: growth with a leveling off onto a plateau, but not at 100%:
        # some critical points remain uncovered (the Zipfian tail of §19.2).
        # 100% coverage is suspicious in the same way as 100% coverage of
        # named_correlates for words: a probable sign that noise is
        # silently pulled to the nearest cluster, and not left unclustered.

    compression_gain: ExpectationCheck
        # A direct extension of §12 into dynamics: the total description length
        # of the corpus (in bits) through the dictionary of letters versus storing
        # the raw fields separately (the minimum description length criterion,
        # MDL). Expectation: growth with a leveling off onto a plateau as the
        # corpus grows, a quantitative and not a metaphorical version of
        # "the alphabet compresses."
}
```

Strong evidence of stabilization is not one favorable number but a consistent result: new units on held-out data become rarer, already found classes are rediscovered, and merges and splits decrease under an honest repeated reanalysis; at the same time the gain in compression is confirmed by an independent test. These curves need not all reach a literal zero or one, especially with a rare long tail. A realistic result is a stable core of frequent units with a continuing growth of the tail of rare ones (§21.4). This is neither "the whole alphabet has been found" nor "there is no alphabet," but a partial map with an explicitly indicated region of confidence.

If the procedure does not identify candidates at all or all stages return empty sets, this is not evidence of a closed alphabet and not a zero point of the discovery curve: such a run has not resolved the structure of the field and requires diagnosis or an extension of the class of objects sought. If, however, candidates exist but do not form stable repeatable classes on a held-out corpus, this is a narrower and already meaningful conclusion: the chosen representation, corpus, and identity criterion do not yet confirm the hypothesis of a transferable alphabet. Distinguishing these cases is a mandatory part of the report.

One caveat to this list is procedural rather than statistical in nature: joint convergence of the merge and split rates is measurable only in the mode of repeated reanalysis of already accumulated descriptions (§21.2). In the mode of continuous replenishment of the corpus, the observed split rate is zero by the very design of the procedure, and not because the letter boundaries have settled, and mistaking one for the other would mean declaring a property of the implementation to be a result.


### 19.8 The profile of synonyms
```text
ExpectedSynonymProfile
{
    synonym_clusters: [{cluster_id, member_class_ids, confidence}]

    realization_diversity: ExpectationCheck
        # expected: high structural heterogeneity of atomic realizations
        # among the members of a cluster, higher than the typical within-class
        # heterogeneity of one ordinary word (what distinguishes synonymy
        # from allophonic variability within one class, §13.2).

    graph_distance: ExpectationCheck
        # expected: a large distance in the space of the atomic graph
        # (for example, Tanimoto distance on Morgan fingerprints, a reference point
        # already used in the isosteric calibration of §15, step 5) between the members of
        # a cluster. A small graph_distance would mean ordinary compositional
        # variability within one word, and not synonymy.

    function_distance: ExpectationCheck
        # expected: a small distance in named functional properties
        # (pKa, affinity), the very basis for calling the cluster synonyms,
        # and not merely objects that are similar by invariant but
        # functionally different.

    invariant_distance: ExpectationCheck
        # expected: a small distance in the space of the ESM invariant, a testable
        # claim of the theory, generalized from a single calibration pair (§15, step 5)
        # to statistics over the whole corpus.
}
```

The decisive signal is not any one of the four numbers separately but the joint pattern: a negative or absent correlation graph_distance ↔ invariant_distance together with a positive correlation invariant_distance ↔ function_distance across the whole set of synonym_clusters. The first without the second means that the invariant simply does not correlate with the graph (which could be chance, not necessarily synonymy); the second without the first means that the invariant merely rediscovers graph similarity, which is exactly what the negative control of §15, step 5 explicitly warns against.

### 19.9 The profile of contextual variation: one class or several
Here it is checked whether one descriptor merges different cases under a common label. In the working distinction of ESM, a smooth and predictable response of one unit to the environment is contextual variation; several stable functions systematically linked with different contexts may be a candidate for polysemy; unrelated functions with no dependence on context detected point rather to an insufficient descriptor or an erroneous merge. None of these interpretations follows from clustering alone.

```text
ExpectedPolysemyProfile
{
    realization_similarity: ExpectationCheck
        # expected: high (unlike synonyms): this is one and the same
        # class_id, and not several; the check is needed to make sure that
        # what we have before us is not an artifact of merging two different
        # classes (§15, step 3).

    function_divergence: ExpectationCheck
        # expected: meaningfully large in the meaning descriptors (§19.5)
        # in different contexts, despite high closeness of realization:
        # the defining sign of the phenomenon itself.

    context_dependence: ExpectationCheck
        # The share of function_divergence explainable by the environmental
        # parameter (for example, R² or mutual information between
        # environment_context and the descriptor).
        # The decisive fork of interpretation:
        #   high context_dependence + smooth predictable change
        #       → a contextual response within the unit (§4.3);
        #   high context_dependence + several stable functions
        #   separated by context
        #       → a candidate for polysemy, requiring independent verification;
        #   low context_dependence + large function_divergence
        #       → a probable merge of different phenomena or a missed context;
        #         revise the descriptor and the search branches (§15).

    environment_switch_rate: ExpectationCheck
        # The share of classes for which a sharp context-linked
        # transition between regimes has been detected. This is a signal for
        # analysis, not a ready label "polysemy": a transition may mean a
        # change of mechanism, a split of the class, or an insufficient
        # sample of environmental conditions.
}
```

### 19.10 The status of the section
The set of profiles above is not a guarantee that chemical data will be distributed in just this way, but a list of expectations that can be checked in advance. A mismatch between the observed profile and the forecast is informative only when the measurements themselves and the input data have passed an audit. A complete failure of all stages at once does not become a valid confirmation or refutation: it requires returning to the representation, the algorithms, and the settings (§§3.3, 22.9). Some of the expectations logically precede the others: until reproducible units have been identified, the tests of words, grammar, and meanings remain undetermined: not false, but not yet applicable.


---

## 20. Freedom of representation: why the method has no single correct set of settings

### 20.1 The source of the problem: the field has no canonical record
Classical chemistry works in a ready-made representation: atoms and bonds are given before any analysis, and the dispute concerns only what to compute from them. A continuous field has no such representation (§18): no canonical frame of reference, no natural unit, and no previously given boundary between object and background. All this has to be specified by us, and each such specification is a substantive assumption and not a technical detail.

Hence a property of the method that from the outside looks like an excessive number of settings: there are many of them not because the method is raw, but because an honest enumeration of the assumptions made is longer here than in a method where the representation is given in advance and is therefore not discussed. Reducing this list to a single "recommended" set would not make the method more rigorous; it would hide the assumptions while leaving them in effect.

### 20.2 Where exactly the choice arises
The degrees of freedom are located along the extraction chain, and each answers its own question:

- **Which field is read.** Electron density, electrostatic potential, the electron localization function are different scalar fields of one and the same system, with different topology and different features. A "letter" is defined relative to the chosen field, and the coincidence of inventories between fields is a result, not a premise.
- **At what level of theory and with what discretization.** See the precautions of §15: both the approximation by which the field was obtained and the grid on which it is recorded enter the answer.
- **What counts as a candidate.** The type of topological object (§3.1) and, separately and in parallel, the void layer (§2.3).
- **Which neighborhood, and of what scale, is described.** The size of the region around a feature and the way it is bounded determine what counts as the context of a letter and what is already a neighboring letter. A neighborhood that is too small describes numerical noise, one that is too large erases the difference between neighbors.
- **How finely the shape is described.** The detail of the angular decomposition is literally a decision about which differences of shape count as differences between letters. A description that is too coarse merges different phenomena into one unit (homonymy as an artifact, §19.9), one that is too fine splits one phenomenon into many units (thermal and conformational noise as different words, §6.4).
- **By what criterion two occurrences are counted as one letter.** The three competing definitions of §3.6.
- **Where the boundary between units lies.** The threshold beyond which two realizations are counted as different letters. It has no universal value: it depends on all the previous items and on the amplitude of the particular field, so it is determined from the data themselves and must be varied, not assigned as a constant.
- **What counts as an anchor and at what level of the hierarchy it is measured.** See §16.5.

None of these items is a performance setting. Each is a fork that answers in a different way the question "what exactly are we looking for?"

### 20.3 Two different kinds of settings that must not be confused
A distinction that determines what to do with each setting:

- **Representational.** They change the very subject of measurement: a different choice gives a different inventory, and the results of two choices are not comparable element by element. They cannot be treated as hyperparameters tuned for quality: tuning here would mean choosing the ontology that gives the more pleasing answer, which is precisely the problem of §16 applied to one's own hands. The correct handling is to run them as parallel variants, label them fully, and compare them by the independent criteria of §19.
- **Budgetary.** They change only the precision of the estimate while the subject remains unchanged: how many examples take part in resampling, how many permutations are carried out in a significance test, how large the subsample is on which a statistic is estimated. Their choice does not change *what* is being estimated, and therefore does not require parallel runs, but it must be recorded, since the second of the two quantities of §19.1 depends on it: the reliability of the measurement itself.

Mixing these two kinds is the cheapest possible error here: a change of a budgetary setting presented as a scientific result, or a change of a representational one carried out as routine tuning.

### 20.4 What makes a finding a finding
From this follows a criterion that complements §19: robustness to variation of representation. A genuine finding must behave differently with respect to the two classes of settings:

- be **invariant** to those choices that by their meaning should not affect it: to the budgetary settings as a whole, to refinement of the grid, to a reasonable extension of the level of theory, to a moderate shift of the threshold between units;
- change **predictably** under those choices that by their meaning should affect it: an increase in the detail of the description must split units in an understandable direction and not shuffle them; a change of field must give a comparable, and not a random, picture.

A unit that exists only under one combination of settings is an artifact of that combination, regardless of how statistically stable it is within it. Stability under resampling of the corpus and stability to a change of representation are different things, and the first does not imply the second: resampling varies the data with the representation fixed and by construction cannot detect an error built into the representation itself. A map of sensitivity to representational choices is therefore not an appendix to the report but part of the result on a par with the profiles of §19.

### 20.5 Comparability of results between runs
Since representational settings define the subject, any unit has meaning only together with the full set of choices under which it was obtained. Practical consequence: the corpus is grown under one fixed set at a time, and results obtained under different sets are compared at the level of the final profiles (§19) and are never mixed into one inventory. This is not bookkeeping but a condition of the meaningfulness of the very statement "this unit occurs in so many molecules."

---

## 21. Scale regime: what changes on moving to a corpus of hundreds of millions of molecules

### 21.1 Three qualitatively different scales
The program of §15 is formulated independently of corpus size, but the methods available at each scale differ qualitatively, and not quantitatively:

- **Debugging (hundreds to thousands of molecules).** What is tested is the workability of the extraction machine, and not the properties of chemistry. Any number obtained here is a statement about the pipeline: that the search finds features, that the descriptions are reproducible, that the checks run to the end. The inventory of this scale is not a basis for chemical conclusions in any form, and still less is it an alphabet.
- **Calibration (tens to hundreds of thousands).** The scale at which expensive but rigorous procedures are still available: resampling, permutation tests, the controlled designs of §15, step 5, a check on a held-out part, the sensitivity map of §20.4. Here it is established what to trust, and here the thresholds are also estimated.
- **Discovery (a guideline of hundreds of millions and more).** A large corpus makes it possible to investigate more reliably the saturation of the inventory and the rate of appearance of new units; this is not a universal threshold but a guideline for questions related to the behavior of the corpus under a many-fold growth. Here the check of reusability (§12) becomes substantially more convincing than on a small calibration sample.

These scales are practical planning regimes, and not a law of nature or a guarantee that any dataset of such a size is sufficient. Conclusions are not transferred automatically: a small corpus tests the pipeline and provisionally calibrates the criteria; a large one gives a stronger test of transferability and of the frequency of rare units.

### 21.2 Incremental discovery changes the status of a unit
Rebuilding the partition of the whole corpus from scratch at each replenishment is impossible at such a scale. The consequence here is not an engineering one but a conceptual one: a unit ceases to be a globally optimal partition of all the data and becomes **historical**: that to which an occurrence was assigned at the moment of its appearance, with a subsequent possibility of revision.

Revision here is asymmetric, and the asymmetry is physically unavoidable. Merging units is cheap: it is enough to compare their summary descriptions and declare them one unit, keeping the former names as references to the new one. Splitting is never cheap: to divide a unit one must return to the descriptions of its occurrences, that is, to the stored array and not to the summary. Therefore the stream of replenishment by its nature knows only how to coarsen, while splitting is a separate, planned operation of repeated reanalysis of what has been accumulated.

Hence two requirements without which the incremental mode ceases to be scientifically meaningful. The first: the descriptions of occurrences are preserved in full and are not replaced by summaries, otherwise splitting becomes impossible forever and not merely expensive. The second: the former names of units are not rewritten retroactively; the history of assignments is preserved together with an indication of what it was later merged into, otherwise no statement about the past state of the inventory is verifiable. The direct consequence for §19.7 is recorded at the end of that section.

### 21.3 Exact estimates become unavailable, and this is part of the result
Many quantities require comparing every occurrence with every other: for example, all pairwise distances, the full co-occurrence matrix, or permutation tests on the whole corpus. Their cost grows quadratically; at the target scale an exact calculation may become impractical within the available budget, even if technically it remains a finite operation.

The answer is not to give them up but to move to volume-bounded estimates: random subsamples for distributions, compact summaries for counting the number of distinct carriers, repeated reanalysis instead of full recomputation. It is essential that such a replacement changes the precision but not the subject of the estimate, and therefore belongs to the budgetary class of settings (§20.3). It is precisely for this reason that §19.1 requires keeping two quantities separate: the agreement of the observed with the predicted, and the reliability of the observation itself. At the target scale the second ceases to be a formality and becomes the main limit on the strength of a conclusion: the statement "the curve has reached a plateau" is meaningful only together with the precision with which each of its points was obtained.

### 21.4 Rarity becomes the norm, and the absence of observations becomes ever less informative
A Zipf-type distribution (§19.2), with many very rare units and a small number of frequent ones, implies that the absolute number of rare units may increase as the corpus grows. Therefore, for a significant part of the inventory there may be too few data for a reliable estimate. It is reasonable to set the status of a unit on a scale, from "insufficient observations" to "stably reused and has passed independent verification," and not only as "letter/not a letter." Such a graded result is provided for in §19.7.

The other side of the same growth concerns prohibitions and works against intuition. The number of possible combinations of units grows quadratically with the size of the inventory, while the number of combinations actually observed grows at best linearly with the size of the corpus. Therefore the statement "these two units have never been seen together" becomes ever less informative as the corpus grows: the overwhelming majority of pairs have not been seen simply because there are incomparably more pairs than observations. A prohibition map built from the absence of observations in a static corpus does not improve with scale but degenerates, which turns the requirement of §15, step 9 (a targeted collection of failed paths) from a desirable addition into the only working source of prohibitions.

### 21.5 Held-out verification as the main criterion
All curves taken on the same corpus on which the inventory was built share one vulnerability: they may reach a plateau not because the alphabet is complete but because the corpus is homogeneous: the molecules in it are too similar to one another or were generated by the same method of selection. At the target scale this is not a hypothetical risk: large corpora are assembled automatically and inherit the biases of their sources (§18).

Therefore the decisive check remains the one on a part set aside in advance and, where possible, arranged differently: what share of the features of previously unseen molecules falls into already known units and what share requires new ones. This is the only one of the criteria of §19.7 that does not depend on whether the curve is computed on the same data, and the only one whose result cannot be improved simply by continuing to grow a homogeneous corpus.

## 22. What outcomes ESM can yield — and what they will make possible

ESM does not have a single binary outcome, "found a language" or "did not find one." The program has several independent levels: the description of the field, recurring units, their combinations, contextual meanings, and the rules of transition between states. A result at one level does not automatically prove a result at the next: the discovery of stable units does not yet mean that a grammar has been found, and the compactness of a dictionary does not in itself prove the predictive power of the rules. Therefore every conclusion must be accompanied by a scope of applicability, an estimate of reliability, and a check on held-out data (§§16, 19, 21.5).

### 22.1. A corpus of fields as a new material for training

Even if a stable alphabet or compact rules have not yet been established, an independent result remains: a corpus of molecules described not only by SMILES strings or bond graphs but by three-dimensional energy fields and their topology (electron density, potential, ELF, and derived representations), with context, data provenance, and known properties. Such a corpus makes it possible to train and test models directly on spatial-energetic structure, to compare it with traditional representations, and to investigate what information those lose or retain.

This is a valuable dataset and set of representations for AI, but by itself it does not yet confirm the hypothesis of a language, does not guarantee an improvement in predictions, and does not mean that the fields were obtained without quantum-chemical calculations. Its scientific value increases if the source fields and the occurrences of patterns, the calculation method, the conditions, the uncertainty, and independent outcome labels are preserved: then the corpus can be reanalyzed and different ways of decoding compared, without substituting later summaries for the source data.

### 22.2. Field decomposition independently returns atoms

One possible outcome: a blind analysis of the field that does not assume atoms as letters stably identifies units that coincide with nuclear basins or another atomic decomposition, and recurring groups of such units form larger patterns ("words"). If this result is reproduced on different datasets, calculation methods, and held-out molecules, it will become independent evidence in favor of atomic structure as a fundamental level of organization and will show that the atomic path of chemistry was productive.

Two statements must be distinguished here: "atoms are stable objects of the field" and "atoms exhaust the ESM alphabet." The first does not prove the second. Even with atoms as letters, ESM may discover stable supra-atomic units, contextual variants, and rules of assembly. For AI, an atomically consistent outcome offers the possibility of learning from reusable field "words" and their energy realizations, and not only from atomic tokens and SMILES. This may give a more meaningful three-dimensional representation and better transferability between contexts; the advantage must be measured on independent tasks and not assumed in advance.

Such a result would support the fundamental orientation of atomic chemistry, but would not prove the infallibility of all historical models, approximations, and practices. Convergence to atoms is a separate testable conclusion, and not an automatic confirmation of the whole of chemical theory.

### 22.3. Non-atomic or mixed field units are found

Another outcome: parts of atomic basins, structures crossing several basins, or several types of units at once turn out to be stable. Some units may appear as a distributed pattern of maxima and minima, a shape of a surface, or a relation between local features, and not as a separate extremum. If such patterns are reproducible and predict independent properties or transitions better than alternative partitions, ESM will provide a language for describing the organization of the field that the familiar atomic markup may fail to single out.

The practical application is the formulation and testing of new hypotheses about recurring structural motifs, the search for analogies between chemically dissimilar systems, and the detection of features related to the function of a material, a reaction region, or an active center. Such conclusions do not abolish atoms and do not mean that the units found already constitute an explanation of causality: control of representation artifacts, comparison with known models, and independent checks are necessary.

### 22.4. A large dictionary and compact executable rules

The strongest scenario: the corpus contains very many recognizable field "words," but their combinations, meanings, and changes are described by a comparatively small set of stable rules. For example, a million distinguishable realizations may be organized by a compact grammar, if these rules really reproduce the observed structures and transitions and do not merely compress the training data. This would not be merely a catalog of molecules but a testable model of how local field units form larger configurations and how configurations are executed when conditions change.

With sufficient coverage and successful testing on new systems, such a grammar could become the basis for:

- generation and inverse design of molecular and material structures from given field features or properties;
- prediction of the compatibility of units, stable configurations, and probable transitions;
- modeling of reactions, binding, active centers, conformational changes and, where suitable data are available, biological cascades;
- training AI models that operate on reusable field units and rules, and not only on sequences of atomic symbols;
- selection of the most uncertain or critical cases for expensive calculation, experiment, or more detailed modeling.

Since the language here is performative (§§7–9), a static dictionary is insufficient for modeling chemical execution. Transition rules must be tested on ordered trajectories and under explicitly specified conditions: medium, temperature, pressure, collision, binding, and other perturbations. Only then can one speak of predicting not only the result but also the process of obtaining it.

If the rules confirm accuracy, scope of applicability, and robustness on independent data, they could potentially replace part of the repeated expensive calculations or give more accurate predictions than general empirical approximations in the corresponding area. This is not a promise of a universal replacement of quantum chemistry and not a consequence of compactness alone: comparisons with experiment and with strong independent models, calibration of uncertainty, error analysis, and explicit limits of applicability are required. Conversely, a field model should not be considered less fundamental merely because its reference source used DFT: what is compared is the quality of the predictions and the agreement with observations, and not the prestige of the name of the method.

### 22.5. Recurring "words" are found, but the rules have not become more compact

ESM may reveal atom-letters and recurring field fragments corresponding to parts of molecules, while the rules of their formation ultimately coincide with standard atomic physics and give no substantial compression of the description. This is a meaningful result and not a failure: it shows that the field has a useful reusable structure, but that in the tested area no more compact dynamic grammar has been found.

Such "words" can serve as fundamental building blocks for AI trained on real three-dimensional energy structures and not exclusively on SMILES. A model will be able to use the repeated use of motifs, context, and field features as an additional level of representation, while retaining standard physics as the rules. Possible benefits are the quality of representation, transfer between molecules, the search for analogues, and interpretability; savings in computation are not guaranteed and must be assessed separately. This outcome would support the direction of atomic chemistry and at the same time add to it empirically tested levels of organization.

### 22.6. A stable core, an open tail, and contextual variants

A realistic intermediate result is a small reproducible set of frequently occurring units and a large long tail of rare, so far undetermined cases (§21.4). Synonyms are also possible, different descriptions of one unit, as well as contextual forms of one unit that a naive clustering would mistakenly count as different letters. Then the correct result will be not an artificially closed alphabet but a versioned map: a stable core, candidates, rare observations, variants, and the conditions of their discrimination.

Such a result already makes it possible to build models with explicit uncertainty: to apply reliably known rules directly, to send borderline cases for additional analysis, and to use new data to refine historical assignments without destroying the original occurrences (§21.2). It also shows which datasets, fields, and trajectories are most important to collect next.

### 22.7. A specific hypothesis was not confirmed in the tested area

A scientifically negative result is possible at each specific level: for example, field candidates are found but are not rediscovered on held-out molecules; stable units exist but do not give predictable combinations; the rules do not generalize; or the new rules do not surpass standard physical models. Then the corresponding hypothesis is not confirmed in the stated area, and a practical application of precisely this level is not justified.

This is not the same as a complete zero at all stages. If the analysis returned no organization whatsoever, this is not yet a valid refutation of ESM: the chosen procedure may have missed objects of another type, scale, or context (§§3.3, 22.9). And if stable structures exist but the new rules give no compactness or predictive benefit, it is precisely this limited conclusion that may be a strong and honest negative result. The report must state which hypothesis the data reject, for which fields and corpus, and which alternatives remain untested.

### 22.8. How to assess the practical value of a result

The practical level of application is determined not by how convincing a visualization of the alphabet looks but by the totality of checks: rediscovery on held-out and differently collected data; robustness to admissible choices of representation; the relation of units to independent properties and transitions; comparison with atomic and other baseline models; assessment of errors and uncertainty; a gain in accuracy, data, or computational cost on a task specified in advance. For generation, the physical admissibility and the successful execution of the proposed structures are additionally tested, and for dynamics the quality of trajectories and of transition predictions.

The outcomes may combine several scenarios: for example, atoms may turn out to be stable letters, some supra-atomic words contextual, the dictionary partially open, and the rules compact only for individual classes of transformations. Therefore an ESM report must communicate not one general label but a matrix of results by level: what was discovered, where it is reproduced, what it predicts, what remains unknown, and what application is already justified. Precisely such a separate conclusion makes it possible to recognize the independent value of a field-based AI representation without prematurely claiming a discovered universal grammar or a replacement of all of chemistry.

### 22.9. Complete emptiness is a reason to broaden the search, not to declare the absence of a language

For ESM a simple but fundamental criterion applies: **if the entire pipeline detects no structures, repeatability, dependencies, or relations at all, this result is not accepted as scientific evidence that chemical fields lack organization**. It means that the current way of reading the field has not yet coped with the task. The cause may be in the software (an error in extraction, transfer, or saving of data); methodological (unsuitable invariants, too coarse a grid, an excessive threshold, or an inadequate similarity); or theoretical: the search is limited to expected classes of objects and does not include the form in which the field is actually organized.

Diagnosis must therefore proceed in two directions at once. On the one hand, the correctness of the inputs, each stage of the pipeline, and robustness to rotation, translation, resolution, level of calculation, and rerunning are checked. On the other, the space of hypotheses is broadened: not only nuclear attractors and basins are examined, but also other critical points, sections of the field, forms of extrema, ridges and relations, regions of low density, cavities, channels, and multiscale and contextual patterns. If the familiar atomic basins are not found, this in itself does not mean an error: perhaps the minimal stable unit is arranged differently. An error of inference arises only when the absence of atoms is at once declared proof of the absence of structure.

This is how an internal test is formulated: not for "obligatory coincidence with atoms," but for the ability of ESM to continue the search until an explainable and reproducible structure is reached. A positive control may include known fields and atomic basins for comparison, but must not force the blind algorithm to take the atomic markup for the answer. If the search remains empty, the logic is changed or supplemented, the previous result is preserved, and the audit is repeated; only after a stable structure appears can one assess whether it supports the atomic, non-atomic, or mixed hypothesis.

### 22.10. Checking what has been found with independent chemical and physical calculations

After the search has yielded a reproducible field structure, a stricter check is applied: the units found, their boundaries, combinations, and supposed rules are compared with chemical and physical results that were not used to tune or train the algorithm. Depending on the hypothesis, these may be independent calculations of electronic structure, topological characteristics of the field, energies and barriers, stability of configurations, response to a change of medium, reaction trajectories, and also experimentally measured quantities. What is checked is not only visual coincidence but also whether the objects found help to predict differences between molecules, properties, or transitions on held-out examples.

To avoid circular confirmation, the provenance of each check must be stated explicitly. If letters are identified from a field computed by one method, a simple re-measurement of a feature from the same calculation is a consistency check but not independent proof. A stronger check uses a different computational protocol, independently prepared data, or experimental observations. None of these sources is declared an unconditional truth: their areas of applicability, uncertainties, and errors are compared. Especially significant is the case when field patterns give a stable forecast that was not contained in the criteria of their extraction and is confirmed by an independent physical calculation or experiment.

Thus the verification of ESM has two different levels. The **internal test** answers whether the field procedure extracts any reproducible organization at all and whether it withstands control of the representation, the code, and the parameters. If the entire result is empty, the search must be continued or revised; if atoms are not detected but a different organization of the field is reliably found, this is an admissible result. The **external physical check** answers whether the specific units and rules are correct and useful. The first level prevents an empty output from being mistakenly declared proof of the absence of structure; the second does not allow any repetition to be declared a physically significant discovery.

---

## Conclusion: not to assign the answer, but to learn how to find it

ESM begins not with a denial of the atom but with a question that atomic notation usually leaves off-screen: which stable forms and relations are already contained in the three-dimensional field of a molecule? To answer it, we turn to decipherment as a discipline for searching for unknown structure, to topology as a language of stable forms, to statistics as a check of repeatability, to information theory as a measure of reuse, and to dynamics as a way of investigating execution.

The path is deliberately open. A letter may turn out to be an atomic basin, a part of one, a structure spanning several basins, a pattern of void, or a combination of these levels. Meaning is not assigned by external similarity: it must emerge in independent properties or predictions. A static map is not passed off as a grammar of transitions; that requires trajectories. No attractive visualization replaces checks on new data.

This openness has a strict condition: complete emptiness is not declared a victory of skepticism and is not passed off as the absence of chemical order. If the procedure sees nothing, we check what exactly it could see, what it lost, and which other classes of structures should be sought. If it sees a stable structure, we check its origin, reproducibility, independent physical connection, and predictive power. Only after that can one say what has been found: an atomic foundation, a new level of field organization, a partial dictionary, compact rules, or a boundary beyond which the rules give no gain.

That is why the result is important under any meaningful outcome: an atomic answer will strengthen the link of the field with familiar chemistry; a new answer will extend the language of description; an intermediate one will give a reliable corpus and a map of the unknown; the failure of a particular grammar will outline its area of applicability. The goal is not to confirm a metaphor chosen in advance, but to build an analysis in which the nature of the field can surprise us, and the data can make us change our minds.

---

## Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
