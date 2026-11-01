# Report wording review

This is an editorial review, not an AI-authorship diagnosis. Repetitive phrasing, generic conclusions or grammatical accuracy do not prove who wrote a passage.

The review covers the current report source, including its expanded dataset and judge-prompt appendices. Tables, figures, equations, citations and verbatim prompt blocks were kept unchanged.

The requested rough draft now uses mainly grammar and punctuation slips in explanatory prose: 124 grammar changes, 112 punctuation changes and exactly 20 spelling slips in more difficult words. The abstract contains no introduced spelling mistakes. Headings, captions, quoted prompts and other protected material are excluded. These slips are an explicit style choice, not evidence of human authorship.

## Main areas revised

- Abstract and introduction: generic opening claims, repeated framing and an overbroad gate result.
- Related work and methodology: novelty claims, metaphors, vague model descriptions and broad software claims.
- Evaluation and results: absolute conclusions where the evidence supports estimates or uncertainty.
- Memory and ablations: rhetorical headings, speculative explanations and a statement of equal scores that did not match the table.
- Final system and discussion: repeated summaries and language that treated an estimate below the risk limit as a guarantee.
- Conclusion: promotional or circular wording in place of specific findings.

## Passage-by-passage changes

The revisions below show the readable wording before the requested rough-language pass.

### 1. Abstract

**Issue:** Replaced a generic problem statement and promotional claim with the specific reason for the project.

**Before:** People in distress often do not ask for help, and when a support system starts the conversation they reply briefly and indirectly. COCOON showed that active listening finds such hidden needs better than direct questioning. However, its corpus is Chinese, single-session and unreleased, and its single-model supporter cannot check a reply before sending it. We rebuild COCOON in English with open models and improve it in four ways:

**Revised wording:** An emotional-support system has little to work with when a person gives short answers and does not ask for help. COCOON approaches this problem through active listening. We used that approach as our starting point, but needed an English pipeline that could follow a person across sessions and check replies before sending them. We built COCOON_EN with open models and made four additions:

### 2. Abstract

**Issue:** Distinguished simulated elapsed time from actual time spent waiting.

**Before:** 2,369 linked conversations in which simulated people resist, change and come back after real time gaps.

**Revised wording:** 2,369 linked conversations with guarded simulated users, changing situations and sampled time gaps.

### 3. Abstract

**Issue:** Replaced an unqualified causal claim with the measured scope.

**Before:** New metrics show which supporter turn caused resistance.

**Revised wording:** Turn-level scores measure intrusiveness and estimate the resistance associated with a supporter reply.

### 4. Abstract

**Issue:** Scoped the gate result to the tested configuration and replaced the stock closing sentence.

**Before:** Going beyond our proposal, we also test activation steering and preference alignment as cheaper ways to make the supporter gentler. On held-out simulated users, the gate cuts intrusiveness by 17% and keeps intrusive replies to 4.6% of turns, under its 10% limit. Preference alignment gives the largest drop in intrusiveness against its own baseline. The need-state memory ties for the best score on the ES-MemEval benchmark. Together, these results show which design choices make a proactive supporter safer, and they provide an open English pipeline to build on.

**Revised wording:** We also tested activation steering and preference alignment after the work in the proposal. In the untuned four-agent test, the gate reduced IP by about 17%; 4.6% of scored turns were both released and intrusive, below the 10% budget. Preference alignment produced the largest IP reduction against its own control. Need-state memory tied for the best overall score among the memory designs we tested on ES-MemEval. These experiments let us examine the contributions of training, memory and reply checking separately.

### 5. Introduction

**Issue:** Removed a sweeping assumption about users being happy to answer questions and a repetitive three-part opening.

**Before:** Most emotional-support chat systems assume that the user asks for help: the user starts the conversation, names the problem and is happy to answer questions. Many people in distress do not act this way. They may not know what they need, may not want to say it, or may push back when someone tries to read them.

**Revised wording:** Emotional-support chat systems usually begin with a user asking for help. In our setting, the system starts the conversation instead. The person may be unsure what they need, reluctant to discuss it, or uncomfortable with the system making guesses.

### 6. Introduction

**Issue:** Used a direct transition tied to the proposal.

**Before:** Our proposal pointed out three more gaps:

**Revised wording:** Our proposal focused on three gaps in this setup:

### 7. Introduction

**Issue:** Led with the work completed while retaining the comparison constraint.

**Before:** Contributions. We do not claim to beat COCOON: the language, models, simulated users and judges all differ, so the numbers cannot be compared directly. Our contributions are:

**Revised wording:** Contributions. Our work extends COCOON through an English implementation, linked sessions and explicit reply checks. The different language, models, users and judges prevent a direct score comparison with the original paper. We contribute:

### 8. Related Work

**Issue:** Removed unsupported field-wide novelty claims and described the actual comparison.

**Before:** Splitting a supporter into several agents is not new. We use the split for a different purpose: it creates a point where a critic can check each draft against clear rules before it is sent. Psychological reactance~\cite{dillard2005} has so far been measured on whole conversations. Our metrics score the supporter's own turn, not only the user's reaction, and estimate cause by replaying the user's reply.

**Revised wording:** We use separate agent roles so that analysis, planning and reply checking can be inspected before a message is sent. Alongside the conversation-level scales, we measure intrusiveness and reactance~\cite{dillard2005} at individual turns. Replaying the simulated user's reply after a neutral alternative gives a counterfactual estimate of the supporter's effect.

### 9. Methodology: overview

**Issue:** Replaced repetitive naming and the overbroad claim that two models do all the work.

**Before:** COCOON_EN is the name of our English corpus and the system built on it. It is not the COCOON corpus, which is Chinese and unreleased, and no COCOON data was used. The work has four stages: building the data, training the models, generating supporter turns with the full pipeline (Fig.~\ref{fig:pipeline}), and evaluating. Two open models from different families do all the work:

**Revised wording:** COCOON_EN includes the English corpus and the system trained and tested with it. We built the corpus independently because the original COCOON data was unavailable. The workflow covers data construction, fine-tuning, response generation and evaluation (Fig.~\ref{fig:pipeline}). The core pipeline uses two open models from different families:

### 10. Methodology: overview

**Issue:** Limited the claim to the experiments instead of all development activities.

**Before:** No paid or closed model is used anywhere.

**Revised wording:** The project experiments use open-weight models.

### 11. Methodology: profiles

**Issue:** Replaced a vague quality label with the actual selection process.

**Before:** Three checks reject weak profiles:

**Revised wording:** We kept a profile only if it passed three checks:

### 12. Methodology: dialogue generation

**Issue:** Used standard sampling terminology and a concrete hardware explanation.

**Before:** a warmer sampling temperature for the user. Only one model needs to be loaded, so everything fits on a single GPU.

**Revised wording:** a higher sampling temperature for the user. Reusing the loaded model for both roles allowed dialogue generation on a single GPU.

### 13. Methodology: annotation

**Issue:** Replaced a qualitative claim with the existing saved count.

**Before:** The labels are not repetitive: almost every strategy phrase is different.

**Revised wording:** The annotations contain 21,354 distinct strategy phrases.

### 14. Methodology: agents

**Issue:** Removed anthropomorphic wording and made model sharing explicit.

**Before:** COCOON's supporter thinks and writes in one step. We split it into four agents (Fig.~\ref{fig:pipeline}). All four are the same loaded model with different instructions; only the Generator can use the fine-tuned adapter. Every agent sees the full conversation word for word, plus a short memory summary.

**Revised wording:** We separated the supporter into four roles: Analyzer, Strategist, Generator and Critic (Fig.~\ref{fig:pipeline}). They share the loaded base model but receive different instructions. Only the Generator can use the fine-tuned adapter. Each role receives the conversation and a short memory summary.

### 15. Methodology: gate

**Issue:** Separated the calibration target from a universal guarantee.

**Before:** The threshold is not hand-tuned. It is calibrated on held-out conversations so that the share of replies that are both sent and intrusive stays within an error budget of 10%. The untuned and the fine-tuned Generator are calibrated separately.

**Revised wording:** We selected the threshold on held-out conversations with a target budget of 10% for turns that are both released and intrusive. Calibration was done separately for the untuned and fine-tuned Generators. Held-out tests then checked the observed rate under each configuration.

### 16. Methodology: metrics

**Issue:** Explained the procedure without presenting simulated counterfactual scores as direct human causal evidence.

**Before:** Counterfactual PRI separates resistance the supporter caused from resistance the user already had. We swap the real supporter turn for a neutral reflective reply and regenerate the user's answer from the same point, with the same random seed. Counterfactual PRI is PRI(real) minus PRI(neutral).

**Revised wording:** Counterfactual PRI compares the simulated user's reaction to the actual supporter turn with their reaction to a neutral reflective reply. We regenerate the answer from the same conversation state and random seed. Counterfactual PRI is PRI(real) minus PRI(neutral), giving an estimate of the change associated with that supporter turn.

### 17. Methodology: memory

**Issue:** Replaced a slogan-like contrast with ownership and update details.

**Before:** Our proposal suggested remembering the supporter's beliefs about the person's needs, not a list of facts. The memory is one object per person; the Analyzer writes to it and every agent reads from it (Fig.~\ref{fig:pipeline}).

**Revised wording:** We designed need-state memory to keep track of the supporter's current beliefs about a person's needs. Each person has a separate memory object. The Analyzer updates it, and all four agents receive its summary (Fig.~\ref{fig:pipeline}).

### 18. Methodology: memory

**Issue:** Replaced an informal metaphor with the graph relationship.

**Before:** The surface feeling becomes a root node, and each need hangs under the shallower need it explains.

**Revised wording:** The surface feeling forms the root node. Deeper needs link to the shallower needs they explain.

### 19. Methodology: steering and DPO

**Issue:** Removed the casual aside and stated the reason for the extension.

**Before:** Steering and alignment were not part of our proposal. We added them anyway to test whether a gentler supporter can be obtained more cheaply than with the gate.

**Revised wording:** After the proposed experiments, we tested steering and DPO as two additional ways to reduce intrusiveness without adding inference calls.

### 20. Experimental Setup: hardware

**Issue:** Corrected product spelling and avoided an unsupported general claim about another backend.

**Before:** Anything that needs a LoRA adapter or a steering hook runs on HuggingFace transformers (4-bit), because Ollama cannot load either.

**Revised wording:** Runs requiring a LoRA adapter or steering hook use Hugging Face Transformers with 4-bit weights. This was the backend used for those features in our setup.

### 21. Experimental Setup: scoring

**Issue:** Kept the useful explanation without emphatic conversational formatting.

**Before:** So 0.415 means an average rating of about 3.5 out of 7, not a 41.5% success rate.

**Revised wording:** For example, 0.415 corresponds to an average rating of about 3.5 out of 7; it is not a 41.5% session success rate.

### 22. Results: fine-tuned supporters

**Issue:** Replaced a sweeping explanation with the displayed values.

**Before:** The basic metrics are almost the same for every model, because our judge saturates at about 83. Only Success separates the models.

**Revised wording:** Most basic ratings are close across the three models, with several near 83. Success varies from 0.383 to 0.415, so we report it alongside the turn-level metrics when comparing the systems.

### 23. Results: reactive baselines

**Issue:** Replaced an abrupt verdict with the comparison and its supported finding.

**Before:** Reactive baselines. Our proposal aimed to beat reactive baselines on Success, and we tested this in two ways. Neither met that aim.

**Revised wording:** Reactive baselines. We used two comparisons to test the proposal's target of improving Success over reactive baselines. The results did not establish that improvement, but they show how performance changes with the user setting and training corpus.

### 24. Results: memory

**Issue:** Replaced a rhetorical heading and conversational judgement with the actual checks.

**Before:** Are the stored beliefs right? Of the 785 stored needs, 90% matched one of the person's true needs, and confirmed needs were right 91% of the time. Re-fitting the memory's confidence numbers from evidence the memory already stores (number of quotes, status, sessions seen) nearly halved the calibration error.

**Revised wording:** Checking stored needs. We compared 785 stored needs with the reference needs. Overall, 90% matched a reference need; among confirmed needs, the match rate was 91%. Re-fitting confidence from the stored quote counts, status and session history nearly halved the calibration error.

### 25. Results: ES-MemEval

**Issue:** Distinguished measured scores from a possible explanation.

**Before:** It was weak at detecting conflicts, because it stores beliefs about needs, not the factual statements those questions test.

**Revised wording:** Conflict detection remained harder: need-state memory scored 0.292, compared with 0.532 for word-match retrieval. Storing needs rather than all past factual statements may contribute to this gap.

### 26. Results: gate

**Issue:** Fixed the implication that every blocked draft was judged intrusive.

**Before:** Every caught draft is an intrusive reply that the user never sees.

**Revised wording:** A blocked draft is revised or replaced before release. The two percentages distinguish how often the gate intervenes from how many intrusive drafts it catches.

### 27. Results: turn-level metrics

**Issue:** Replaced the claim of removing all user effects with the reference-based estimate.

**Before:** PRI mixes the supporter's effect with the user's mood and style. For example, the corrective user pushes back more by design. Counterfactual PRI removes this effect by comparing each real turn with a neutral reply in the same place.

**Revised wording:** Observed PRI reflects both the supporter's reply and the simulated user's behaviour. The corrective user, for example, is instructed to push back against wrong guesses. Counterfactual PRI uses a neutral reply at the same point as a reference for estimating the supporter's contribution.

### 28. Results: turn-level metrics

**Issue:** Replaced a broad causal summary with the observed result.

**Before:** So the methods that lower IP, namely the gate and DPO, also remove measurable resistance.

**Revised wording:** The gated untuned pipeline and DPO also had counterfactual PRI estimates close to zero.

### 29. Results: turn-level metrics

**Issue:** Removed an absolute claim and kept the comparison.

**Before:** These turn-level scores separated the agents from the gate when Success could not.

**Revised wording:** These scores showed differences between the agent and gate configurations that were less clear in Success.

### 30. Results: DPO versus full system

**Issue:** Avoided interpreting overlapping uncertainty as proof of equality.

**Before:** so the two are about equal.

**Revised wording:** so this comparison does not establish a Success difference.

### 31. Results: DPO versus full system

**Issue:** Removed repetitive contrast and kept the proposed combination clearly in future work.

**Before:** So DPO reaches the same Success as the full system while being gentler. The full system offers something different: it checks each reply before sending it, keeps a memory of the person, and records why a reply was stopped. DPO changes the Generator and the gate checks its output, so the two can be combined (Section~\ref{sec:conc}).

**Revised wording:** DPO had a similar Success estimate and lower IP than the full system in these runs. The full pipeline also provides persistent memory and checks each candidate reply before release. Since DPO changes the Generator while the gate checks its output, combining them is a practical next experiment (Section~\ref{sec:conc}).

### 32. Ablation: training

**Issue:** Explained the checkpoint choice and the scope of the loss comparison.

**Before:** Validation loss goes down as the model gets bigger (Table~\ref{tab:loss}). The third epoch overfits in every run, so we kept epoch 2.

**Revised wording:** Validation loss. Within each target format, validation loss decreased with model size (Table~\ref{tab:loss}). The best checkpoint was from epoch 2; validation loss increased in epoch 3. Loss values across the two target formats cover different tokens and should not be compared directly.

### 33. Ablation: PVI

**Issue:** Replaced a long rhetorical heading with a direct one.

**Before:** How much do the thoughts tell the model about the reply?

**Revised wording:** Information in the thoughts.

### 34. Ablation: PVI

**Issue:** Replaced vague praise with the measured trend.

**Before:** The thoughts carry a lot of useful information at every size, and less as the model grows

**Revised wording:** Mean PVI is positive at all three sizes and decreases with model size

### 35. Ablation: model size

**Issue:** Removed an unsupported explanation of internal model behaviour.

**Before:** A small model trained on the analysis seems to act on its guesses too early.

**Revised wording:** This result suggests that the benefit of training on thoughts depends on model size; the scores alone do not show why the smaller model became more intrusive.

### 36. Ablation: agents and gate

**Issue:** Replaced an emphatic opener with the experimental design.

**Before:** This is the central test from our proposal. It crosses one call against four agents, and gate off against gate on, using the untuned 7B so that only the design changes (Table~\ref{tab:2x2}).

**Revised wording:** We tested one call and four agents, each with and without the gate (Table~\ref{tab:2x2}). All four conditions use the untuned 7B. This lets us compare the agent split and the gate within the same model setup.

### 37. Ablation: agents and gate

**Issue:** Removed an unobserved claim about the model keeping guesses to itself.

**Before:** With an explicit Analyzer and Strategist, the untuned model acts on guesses it would otherwise keep to itself.

**Revised wording:** The measured IP increase shows why the split needs to be evaluated together with a reply check.

### 38. Ablation: agents and gate

**Issue:** Matched the wording to confidence intervals that include zero.

**Before:** No factor changes Success.

**Revised wording:** The Success differences remain uncertain.

### 39. Ablation: agents and gate

**Issue:** Removed conversational history and repeated framing.

**Before:** Our proposal said: ``If the gate carries the effect while decomposition alone contributes nothing, that is what we would report.'' The gate carries the safety effect. The least intrusive cell is one call with the gate.

**Revised wording:** The gate reduced IP in both architectures. One call with the gate had the lowest IP among the four tested conditions.

### 40. Ablation: memory

**Issue:** Corrected the claim of equal session-1 scores and replaced the metaphor with the displayed comparison.

**Before:** The gain appears exactly where memory has something to carry: in session 1 the two are equal, and from session 2 on the memory version is ahead for every gap length and every kind of change. Intrusiveness is unchanged. We read this as a small, steady benefit in later sessions, obtained without any memory-aware training.

**Revised wording:** The session averages in Table~\ref{tab:ftmem} place the memory version ahead in sessions 2--4, while its session-1 estimate is lower. IP is close at 0.090 without memory and 0.092 with memory. These results support a modest later-session benefit even though the model did not receive memory summaries during training.

### 41. Final system: gate output

**Issue:** Removed the equation of non-fallback with a real or helpful answer and retained the uncertainty.

**Before:** So in the full system the gate rarely needs the safe reply, and the person almost always gets a real answer. In the full system, 9.1% of scored turns [7.4%, 11.2%] were both sent and intrusive, with the estimate under the 10% limit.

**Revised wording:** The full system therefore used a generated reply on most scored turns. Its released-and-intrusive share was 9.1% [7.4%, 11.2%]. The estimate is below the 10% budget, although the interval crosses it. The release percentage does not measure how helpful the reply was.

### 42. Discussion: calibration

**Issue:** Replaced guarantee-like wording with the measured estimates and intervals.

**Before:** The gate's threshold is set from held-out drafts, not by hand, so we can choose how many intrusive replies are allowed through. On new test people, the four-agent system sent intrusive replies on only 4.6% of turns and the full system on 9.1%, both under the 10% limit. With ExTES people, the gate stayed safe by using the safe reply more often.

**Revised wording:** We calibrated the threshold on held-out drafts using a 10% risk target. The released-and-intrusive shares were 4.6% for the untuned four-agent system and 9.1% for the full system. The latter confidence interval extends above 10%, so its point estimate alone does not establish the bound. ExTES tests used the reflective fallback more often.

### 43. Discussion: memory

**Issue:** Replaced conversational praise with the reported match rate.

**Before:** Its confirmed beliefs are usually right.

**Revised wording:** Confirmed needs matched the reference needs 91% of the time.

### 44. Discussion: metrics

**Issue:** Replaced dramatic contrast and exact-cause language with the scope of the measurements.

**Before:** Success barely separated the systems. Under a strict reading, a rating of at least 5 out of 7, every system said the real need in only a small share of sessions. Intrusiveness and counterfactual reactance did separate the systems, and they pointed to the exact turns that caused resistance.

**Revised wording:** Success differences were often small relative to their uncertainty. Using a threshold of at least 5 out of 7 also produced low session-level hit rates. IP and counterfactual PRI supplied additional detail by scoring individual replies and the simulated reactions that followed them.

### 45. Conclusion

**Issue:** Replaced evaluative adjectives with the observed benefit.

**Before:** preference alignment is a cheap and strong way to reduce intrusiveness;

**Revised wording:** DPO reduced IP without adding inference calls;

### 46. Conclusion

**Issue:** Removed circular wording and named the evidence.

**Before:** the need-state memory is competitive as a memory;

**Revised wording:** need-state memory tied for the best overall ES-MemEval score among the tested designs;
