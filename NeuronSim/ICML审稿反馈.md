# LLM Feedback by Program Chairs

LLM Feedbackby Program Chairs18 Jan 2026, 18:53Conference, Authors

**Feedback:**

Hello!

You requested a review of your paper submitted to ICML using the Google Paper Assistant Tool (PAT). The resulting AI Feedback can be found below. Note that this feedback is posted automatically, and is only visible to authors. Importantly, the feedback will **not** be used in the review process. Reviewers, area chairs, and program committee members will **not** have access to the PAT feedback.

Disclaimer: Please note that the models used by the PAT pipeline are not infallible; they may hallucinate and make mistakes. Authors should treat the generated feedback with the same critical eye they would apply to a human review.

PAT Feedback Model: MODEL_A

PATLibraryRunPipeline:

# HIGH LEVEL SUMMARY

------

# Paper Summary

The paper "LASER: A High-Fidelity Spike Representation SNN Framework With Surrogate-Free Training" introduces a framework aimed at minimizing the accuracy gap between Spiking Neural Networks (SNNs) and Artificial Neural Networks (ANNs). LASER is composed of three main elements. First, Bit Spike Encoding (BSE) is proposed to provide a high-fidelity, bidirectional mapping between quantized values and spike sequences for linear computations. Second, the Adaptive Spiking Neural Codec (ASNC) approximates nonlinear functions via adaptive piecewise fitting, designed to localize approximation errors. Third, the framework employs the Straight-Through Estimator (STE) for training, arguing that the high fidelity of the forward pass makes this approach principled. Experiments demonstrate near-machine precision for BSE compared to baseline coding schemes, minimal performance degradation on large language models (up to LLaMA-2 70B), and significant energy reduction for the nonlinear component when benchmarked on Loihi 2 versus a GPU.

# Key Issues Roadmap

- **[1. Introduction and Background]**: The characterization of the training method as "Surrogate-Free" requires justification, as the Straight-Through Estimator (STE) is conventionally defined as a surrogate gradient technique.
- **[1. Introduction and Background] & [2. Methodology: BSE]**: Claims that BSE provides an "exact correspondence" with continuous values and preserves original machine precision require clarification, as the method involves a quantization step.
- **[2. Methodology: BSE]**: Critical details required for reproducibility are missing, notably the mathematical definition of the Bit Weights ($W_{bit}(t)$) and the specification of the quantization parameters (Set $S$).) and the specification of the quantization parameters (Set ).
- **[2. Methodology: BSE]**: The efficiency implications of the "decode–compute–re-encode" strategy, which uses dense ANN operations for activation-activation multiplications, require discussion.
- **[3. Methodology: ASNC and STE Training]**: The description of the ASNC module lacks sufficient mathematical detail for reproduction, particularly regarding the definitions of the adaptive splitting criteria (Eqs. 7-9) and the specific training procedure for the module.
- **[4. Experimental Evaluation and Conclusion]**: The generalization of overall energy efficiency gains (200x) appears to be based solely on a micro-benchmark of the nonlinear activation component, potentially omitting the energy costs of linear operations and the dense operations used in the hybrid computation strategy.
- **[4. Experimental Evaluation and Conclusion]**: There is a potential contradiction between the claim that linear operations are lossless under BSE and the experimental results (Table 5) showing that SNN-izing linear projections (QKV) increases perplexity.

# DETAILED SEGMENT REVIEWS

------

## [1] SEGMENT: 1. Introduction and Background

## **PAGES: [[1, 3]]**

**1. Summary**

The reviewed segment (Abstract, Section 1: Introduction, and Section 2: Related Works) motivates and introduces the LASER framework. It identifies the primary challenges preventing Spiking Neural Networks (SNNs) from matching Artificial Neural Network (ANN) accuracy: errors in encoding continuous values, difficulties in representing nonlinear functions, and training complications due to non-differentiability. LASER proposes a three-component solution: Bit Spike Encoding (BSE) for high-fidelity linear computation, Adaptive Spiking Neural Codec (ASNC) for localized nonlinear approximation, and training using the Straight-Through Estimator (STE). The Related Works section positions LASER against existing encoding methods (Rate Coding, TTFS) and techniques for handling nonlinearity, arguing that LASER offers superior fidelity and structural consistency. Table 1 provides initial empirical evidence comparing the reconstruction fidelity of these encoding schemes.

**2. Potential Mistakes and Improvements**

1. **Correctness/Clarity (Mischaracterization of "Surrogate-Free Training"):** The title, Abstract (L020), and Introduction (L066) claim the framework utilizes "Surrogate-Free Training." However, the methodology explicitly employs the Straight-Through Estimator (STE) for backpropagation (L026, L096, Section 3.3). In SNN literature, STE is conventionally defined as a surrogate gradient technique, as it uses the identity function as a substitute derivative for the non-differentiable spiking mechanism. While the authors argue that this choice is principled rather than heuristic due to the high fidelity of their forward pass (P5), labeling the training as "Surrogate-Free" contradicts the established definition of STE.
2. **Clarity (Precision of "Exact Correspondence" claims for BSE):** The introduction describes BSE as an "exact correspondence scheme" (L76) and a "precise bidirectional mapping" (L68) between continuous values and spikes. However, the description of the mechanism states it involves transforming floating-point numbers into integers through quantization (L78-79). Since quantization from floating-point to integer representations is typically lossy, the mapping is not exact relative to the original continuous values. The exactness appears to apply only to the mapping between the quantized integer representation and the spike sequence. The claims regarding "exact correspondence" with continuous values should be clarified to reflect that the precision is bounded by the quantization level.
3. **Clarity (Table 1 Methodology and Interpretation):** Table 1 reports reconstruction fidelity (MSE) based on 10,000 random values. The reported MSE values for the baselines, Rate Coding and TTFS, are extremely high ($10^4$–$10^5$). The distribution and dynamic range of the input values are not specified. Without this context, the significance of the MSE values is difficult to interpret (as the relative error cannot be determined) and the experiment lacks sufficient detail for reproduction.–). The distribution and dynamic range of the input values are not specified. Without this context, the significance of the MSE values is difficult to interpret (as the relative error cannot be determined) and the experiment lacks sufficient detail for reproduction.
4. **Clarity (Positioning BSE in Related Work):** Section 2 (Linear encoding) primarily compares BSE with Rate Coding and Temporal Coding (TTFS). BSE is described as mapping integers to spike sequences consistent with their binary representation (L83). The related work section does not adequately address prior SNN research that utilizes binary encoding schemes. The novelty of BSE relative to existing binary encoding approaches is therefore not clearly established in this segment.

**3. Minor Corrections and Typos**

- L081: Missing reference "Appendix ??".
- Table 1 Caption: The caption states "BSE attains...", but the corresponding rows in the table are labeled "Ours". Consistency in terminology is needed.

------

## [2] SEGMENT: 2. Methodology: Bit Spike Encoding (BSE)

## **PAGES: [[3, 4]]**

**1. Summary**

The reviewed segment (Section 3 introduction and Section 3.1, Pages 3-4) outlines the methodology of the LASER framework and details its first core component: Bit Spike Encoding (BSE). Section 3 presents the roadmap involving BSE for precise linear computation, ASNC for nonlinear approximation, and STE for training. Section 3.1 describes BSE as a bidirectional mapping between continuous values and spike sequences. It involves a Soma (Eq. 3) that decodes input spikes, performs a linear operation, and applies uniform affine quantization to the result ($Q_y$). An encoder (Eq. 4) then converts $Q_y$ into output spikes $S_y(t)$ using Integrate-and-Fire dynamics with a dynamic threshold. The framework handles weight-activation multiplication in the spike domain but uses a "decode–compute–re-encode" strategy for activation-activation multiplications.). An encoder (Eq. 4) then converts into output spikes using Integrate-and-Fire dynamics with a dynamic threshold. The framework handles weight-activation multiplication in the spike domain but uses a "decode–compute–re-encode" strategy for activation-activation multiplications.

**2. Potential Mistakes and Improvements**

The following points identify areas where the description of the BSE methodology lacks clarity, completeness, or sufficient justification.

- **Clarity and Reproducibility: Missing Definition of Bit Weights ($W_{bit}(t)$).).** The BSE mechanism fundamentally relies on the "bit weight" $W_{bit}(t)$, introduced in Equation 2. It is used for decoding input spikes (calculating $Q_a$ in Eq. 3) and defining the dynamic threshold for encoding ($\Theta(t)$ in Eq. 4). Although the text implies a binary representation (Lines 82-83), suggesting $W_{bit}(t)$ are powers of two, the precise mathematical definition of this variable (including the temporal ordering) is missing. This omission prevents the reproduction of the encoding scheme., introduced in Equation 2. It is used for decoding input spikes (calculating in Eq. 3) and defining the dynamic threshold for encoding ( in Eq. 4). Although the text implies a binary representation (Lines 82-83), suggesting are powers of two, the precise mathematical definition of this variable (including the temporal ordering) is missing. This omission prevents the reproduction of the encoding scheme.
- **Correctness and Clarity: Overstatement of Precision and Entropy Preservation.** Lines 78-80 claim that BSE involves transforming floating-point numbers into integers while "preserving the same information entropy and machine precision as the original values." Equation 3 describes a uniform affine quantization process (involving `round` and `clip`). Uniformly quantizing a floating-point distribution into an integer representation is inherently lossy and does not preserve the original machine precision or entropy. While the mapping between the quantized integer $Q_y$ and the spike sequence $S_y(t)$ is exact, the initial conversion from the continuous value introduces error. The claims regarding precision preservation should be revised to clarify that BSE precisely maps the operations of a and the spike sequence is exact, the initial conversion from the continuous value introduces error. The claims regarding precision preservation should be revised to clarify that BSE precisely maps the operations of a *quantized* ANN into the spiking domain, rather than losslessly preserving the original continuous values.
- **Clarity and Reproducibility: Underspecified Quantization Parameters (Set S).** In Equation 3, the quantization parameters ($\lambda_{out}, \mu_{out}$) depend on $S$, defined as "the set of observed Y values used for normalization" (Lines 161-162, Page 3). The methodology does not specify how $S$ is determined. It is unclear if the quantization is static (e.g., using statistics from post-training calibration) or dynamic (e.g., using moving averages during training). These details are necessary for reproducibility.) depend on , defined as "the set of observed Y values used for normalization" (Lines 161-162, Page 3). The methodology does not specify how is determined. It is unclear if the quantization is static (e.g., using statistics from post-training calibration) or dynamic (e.g., using moving averages during training). These details are necessary for reproducibility.
- **Design Rationale: Discussion of Efficiency Trade-offs for Hybrid Computation.** For activation-activation multiplications (e.g., QK attention), the method employs a "decode–compute–re-encode" strategy (Lines 186-197). This involves converting spikes back to dense numerical values and performing standard ANN Multiply-Accumulate (MAC) operations. While this preserves accuracy, it deviates from the event-driven computation paradigm that motivates SNNs for energy efficiency (Lines 43-48). The methodology should discuss the impact of these dense operations on the overall efficiency and latency goals, particularly when considering deployment on neuromorphic hardware.
- **Clarity: Ambiguity of Sparsity Claims.** Lines 211-212 state that under BSE, "ANN activations can be rendered as sparse and ordered spike trains." If BSE is implemented as a standard binary representation spread over time, the encoding is not inherently sparse (e.g., a value might require spikes at all N time steps). Clarification is needed regarding whether the achieved sparsity is solely due to the input data distribution (e.g., zero activations) or if BSE actively enforces sparsity.
- **Clarity: Presentation of Equations 3 and 4.** The structure defining the computation and encoding is confusing. Equation 3 defines the Soma forward pass (computation and quantization). Equation 4 is intended to define the encoding (spike generation). However, the first line of Eq. 4 (L169) summarizes the Soma operation from Eq. 3, while the subsequent lines define the IF dynamics for encoding. The presentation should clearly separate the layer computation (Soma/Eq. 3) from the subsequent spike generation (Encoding/Eq. 4).

**3. Minor Corrections and Typos**

- Lines 81, 215: References to "Appendix ??" are unresolved placeholders.
- Line 169 (Page 4): The syntax for the Soma definition appears to have mismatched or unusual delimiters: `(Qy, λout, µout) = Soma({Sa(t)}N−1 t=0 ; W, λ, µ, Wbit, β)`.

------

## [3] SEGMENT: 3. Methodology: ASNC and STE Training

## **PAGES: [[4, 5]]**

**1. Summary** The reviewed segment (Sections 3.2 and 3.3, Pages 4-5) details the Adaptive Spiking Neural Codec (ASNC) and the Straight-Through Estimator (STE) training approach of the LASER framework. Section 3.2 introduces ASNC, a method for approximating nonlinear functions via piecewise fitting, designed to localize errors by maintaining the BSE format at inputs and outputs. It features an adaptive mechanism (Eqs. 7-9) for splitting input segments based on difficulty. Section 3.3 describes the use of a hard STE (identity gradient) for training, justifying this by the high fidelity of the forward pass established by BSE and ASNC, drawing parallels to Quantization-Aware Training (QAT).

**2. Potential Mistakes and Improvements**

The primary concerns relate to the clarity and reproducibility of the ASNC methodology (Section 3.2).

- **[Clarity/Reproducibility] Insufficient definition of the ASNC Adaptive Splitting Mechanism (Sec 3.2, Eqs. 7-9).** The adaptive splitting mechanism is central to how ASNC balances fidelity and complexity but lacks the mathematical detail required for reproduction. Numerous critical terms in the governing equations are undefined:
  - In Eq. 8 (Multi-criteria score): The components $E_{sep}$ (error separation), $C_{cons}$ (internal consistency), $B_{bal}$ (data balance), and $Y_{sep}$ (target separation) are only described conceptually (L238-240) and lack mathematical definitions. The determination of the trade-off weights $\alpha_i$ is also unspecified. (error separation), (internal consistency), (data balance), and (target separation) are only described conceptually (L238-240) and lack mathematical definitions. The determination of the trade-off weights is also unspecified.
  - In Eq. 9 (Importance weighting): The terms $I_{func}(i)$ (functional importance), $I_{error}(i)$ (error profile), and $D_i$ (data density) are similarly undefined (L247-248). (functional importance), (error profile), and (data density) are similarly undefined (L247-248).
  - In Eq. 7: The update mechanism for the adaptive threshold $\tau_{\text{adaptive}}$ is missing (L232). is missing (L232).
- **[Clarity/Reproducibility] Ambiguity in ASNC computation flow and missing training details (Sec 3.2, Eqs. 5-6).** The operational workflow, mathematical definitions, and training procedure for ASNC require clarification.
  - **Training Procedure:** The methodology does not specify how the ASNC module is trained. The specific loss function used to optimize the trainable parameters in Equation 5 ($s_i, b_i, \theta_i(t), w_i(t)$) and the timing of this training (e.g., pre-training vs. concurrent with end-to-end training) are not provided (L188).) and the timing of this training (e.g., pre-training vs. concurrent with end-to-end training) are not provided (L188).
  - **Operational Workflow and Input Format:** In Equation 5, an affine transformation is applied to the input $x$ ($x s_i+b_i$), implying $x$ is a continuous numerical value. It should be clarified how $x$ is obtained from the input (presumably a BSE spike train), indicating if a decode-compute-re-encode strategy is used internally. (), implying is a continuous numerical value. It should be clarified how is obtained from the input (presumably a BSE spike train), indicating if a decode-compute-re-encode strategy is used internally.
  - **Undefined Operations:** The function `LIFquantize` in Equation 5 is not defined.
  - **Missing Re-encoding Step:** The mathematical link between the analog estimate $C_i(x)$ (output of Eq. 5) and the spike train $S^{BSE}_i(t)$ (used in Eq. 6) is not defined. The re-encoding process (L258) needs to be explicit. (output of Eq. 5) and the spike train (used in Eq. 6) is not defined. The re-encoding process (L258) needs to be explicit.

**3. Minor Corrections and Typos**

- Page 4, L215 and L249: References to "Appendix ??" are placeholders. These must be linked to the supporting proofs (e.g., entropy preservation of BSE, tighter error bound of ASNC) to allow verification of the theoretical claims.
- Page 5, L220: The variable $N$ is defined as the total number of segments. This conflicts with the use of $N$ in Section 3.1 (e.g., Eq. 3) where it denotes the BSE time window/bit width. Distinct notation should be used. is defined as the total number of segments. This conflicts with the use of in Section 3.1 (e.g., Eq. 3) where it denotes the BSE time window/bit width. Distinct notation should be used.

------

## [4] SEGMENT: 4. Experimental Evaluation and Conclusion

## **PAGES: [[5, 11]]**

**1. Summary** The segment under review comprises Section 4 (Experiments) and Section 5 (Conclusion). Section 4 aims to validate the core claims of the LASER framework: high-fidelity representation via BSE, error localization via ASNC, scalability, and energy efficiency.

Section 4.1 compares BSE's reconstruction fidelity against Rate Coding and TTFS (Table 1). Section 4.2 provides ablation studies on LLaMA-2 7B (Tables 2-6), analyzing component-level errors and layer-wise sensitivity to demonstrate error localization. It also includes a comparison with the SpikeLLM baseline (Table 7). Section 4.3 evaluates end-to-end performance on various LLMs up to 70B parameters (Table 8). Section 4.4 analyzes the energy efficiency of the ASNC component on Loihi 2 versus an A100 GPU (Table 9). Section 5 concludes the paper.

**2. Potential Mistakes and Improvements**

1. **Validity: Generalization of Energy Efficiency Claims (Sec 4.4, Sec 5).** The energy efficiency analysis (Table 9) is based solely on a micro-benchmark comparing the nonlinear activation component (ASNC on Loihi 2 vs. SiLU on A100). The paper extrapolates this finding to claim a "200× improvement in energy efficiency" (L436) overall. This generalization is potentially unsupported for the following reasons:
   - LLM inference is dominated by linear operations, whose energy consumption under BSE on Loihi 2 is not reported.
   - The methodology employs a "decode–compute–re-encode" strategy (L186-197) for activation-activation multiplications (e.g., QK attention), relying on "standard ANN multiply–accumulate (MAC) operations" (L194). The energy cost of this hybrid approach, including the dense MAC operations and the encoding/decoding overhead, is omitted. The claims regarding overall energy efficiency should be qualified to reflect the limited scope of the micro-benchmark.
2. **Correctness: Incorrect Calculation of Improvement Factor (Sec 4.2.4).** Lines 380-382 state: "Overall, compared to the prior baseline, our method reduces degradation by more than a factor of $2.1$–$2.5$". This calculation is incorrect. Degradation is the increase in PPL from the ANN baseline (5.12). The degradation for LASER is +0.46. The degradation for the prior baseline is +6.7 to +9.0 (L379). The reduction factor in degradation is therefore approximately 14.6x to 19.6x. The reported factors (2.1–2.5) correspond to the ratio of the absolute PPL values. The description should be corrected.–". This calculation is incorrect. Degradation is the increase in PPL from the ANN baseline (5.12). The degradation for LASER is +0.46. The degradation for the prior baseline is +6.7 to +9.0 (L379). The reduction factor in degradation is therefore approximately 14.6x to 19.6x. The reported factors (2.1–2.5) correspond to the ratio of the absolute PPL values. The description should be corrected.
3. **Validity/Clarity: Inconsistency Regarding Fidelity of Linear Operations (Sec 4.2.1 vs 4.2.3).** The paper claims that BSE ensures linear computations are "strictly equivalent" (L217) and that the linear path is "lossless" (L329), strictly localizing error to the nonlinear module (ASNC). However, the fine-grained ablation in Table 5 shows that SNN-izing the QKV projections (linear operations) increases PPL by +0.16, a significant portion of the total +0.46 degradation. This contradicts the claims that linear operations are lossless and that error is strictly confined to ASNC. This discrepancy requires clarification.
4. **Clarity/Fairness: Contextualization of Baseline Comparison (Sec 4.2.4, Table 7).** The comparison with SpikeLLM shows LASER achieving much lower PPL. However, the configurations differ substantially: LASER uses FP16 precision and 16 time steps, while SpikeLLM uses aggressive quantization (W4A4/W2A16) and low latency (2 or 4 steps). The discussion should acknowledge that these methods are evaluated at different operating points in the accuracy/efficiency trade-off space, as the PPL difference is influenced by both the methodology and the precision/latency budgets.
5. **Clarity: Missing Details on ASNC Complexity (Sec 4).** Section 3.2 describes ASNC using an adaptive splitting mechanism (Eq. 7). The experimental evaluation does not report the resulting complexity of the ASNC modules (e.g., the typical number of segments 'N' from Eq. 6) required to achieve the reported fidelity in the LLM experiments. This information is necessary to assess the overhead introduced by ASNC and for reproducibility.
6. **Clarity: Reproducibility of Encoding Fidelity (Sec 4.1.1, Table 1).** The evaluation of reconstruction fidelity (Table 1) is based on "10,000 random values" (L292). The distribution and range of these values are not specified, which hinders reproducibility.

**3. Minor Corrections and Typos**

- L368: "The gap is only +0.09 PPL (5.18 vs. 5.21)". The difference between 5.21 and 5.18 is 0.03, not 0.09.
- Inconsistency in PPL reporting: Table 3 (L350) reports "Full SNN" PPL as 5.58 ± 0.04. Table 4 (L361) reports the PPL for the same configuration as 5.61 ± 0.04.
- L383: "LEASER" should be "LASER".
- L409 (Table 7 footnote b): The placeholder "[Citation]" remains.
- L442 (Table 8 caption): "LEASER" should be "LASER".