\subsubsection*{Linear Model for Joint Effect Estimation}

To quantify the joint effects of wild-type (WT) variants and clinical variants on variant effect predictor (VEP) scores, we used regularized linear regression via \texttt{sklearn.linear\_model.Ridge} (L2 regularization). For each gene, we constructed a binary matrix $\mathbf{X}$ (\#haplotypes $\times$ \#WT variants), representing the presence or absence of each WT variant in each haplotype, and a continuous matrix $\mathbf{Y}$ (\#haplotypes $\times$ \#sites) containing VEP scores for each haplotype-site pair. We then fit a multi-target Ridge regression model:

\[
\mathbf{Y} = \mathbf{X}\mathbf{\beta} + \varepsilon
\]

Here, $\mathbf{\beta}$ (a WT variants $\times$ sites coefficient matrix, as returned by \texttt{Ridge.coef\_}) is fit by minimizing squared error plus a regularization penalty established by the strength parameter $\alpha$. The absolute value of each coefficient, $|\beta_{ij}|$, measures the magnitude of the joint effect between WT variant $i$ and site $j$, and the sign of $\beta_{ij}$ encodes the direction of the effect (negative values indicating more pathogenic, positive indicating more benign).

For each WT variant-site pair, we computed:
\begin{enumerate}
    \item Absolute joint effect magnitude $|\beta_{ij}|$
    \item Signed joint effect $\beta_{ij}$
    \item The number of haplotypes containing the WT variant (computed by summing the corresponding binary indicator column)
    \item Weighted joint effect: $\sqrt{|\beta_{ij}| \times n_\mathrm{haplotypes}}$ (to balance effect size and sample count)
\end{enumerate}
Model quality was evaluated using standard metrics from \texttt{sklearn.metrics}: $R^2$, explained variance, mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE).

\subsubsection*{Epistasis Testing}

To identify truly epistatic (non-additive) interactions (as opposed to additive effects), we implemented a statistical testing procedure that directly compares observed joint effects to the sum of their separate effects.

For each WT variant $i$ and clinical variant $j$ pair, we first computed:

\begin{enumerate}
    \item \textbf{Individual WT variant effect:} The mean VEP difference between haplotypes where WT variant $i$ is present (WT=1) vs. absent (WT=0), averaging across all clinical variants:
    \[
    \text{Effect}_{\mathrm{WT},\,i} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( \overline{\mathrm{VEP}}(\mathrm{WT}_i{=}1,\, s) - \overline{\mathrm{VEP}}(\mathrm{WT}_i{=}0,\, s) \right)
    \]
    where $\mathcal{S}$ is the set of all clinical variant sites.

    \item \textbf{Individual clinical variant effect:} The average VEP for clinical variant $j$ relative to the overall VEP baseline (mean across all haplotypes and all clinical variants):
    \[
    \text{Effect}_{\mathrm{Clinical},\,j} = \overline{\mathrm{VEP}}(\mathrm{Clinical}_j) - \overline{\mathrm{VEP}}_\mathrm{baseline}
    \]
\end{enumerate}

The \textbf{expected additive effect} is then:
\[
\text{Effect}_\mathrm{additive} = \text{Effect}_{\mathrm{WT},\,i} + \text{Effect}_{\mathrm{Clinical},\,j}
\]

The \textbf{observed joint effect} is given by the corresponding $\beta_{ij}$ coefficient from the main \texttt{Ridge} model, which reflects the effect when both WT variant $i$ and clinical variant $j$ are present.

To statistically assess whether this joint effect is epistatic, we compared two nested models, both fit via \texttt{sklearn.linear\_model.Ridge}:

\begin{itemize}
\item \textbf{Additive model:}
\[
\mathrm{VEP} = \beta_0 + \beta_1 \cdot \mathrm{WT}
\]
where the WT variant's effect is constant (clinical variant is absorbed in the intercept).

\item \textbf{Interaction model:}
\[
\mathrm{VEP} = \beta_0 + \beta_1 \cdot \mathrm{WT} + \beta_2 \cdot \left(\mathrm{WT} \times \Delta_\mathrm{deviation}\right)
\]
where $\Delta_\mathrm{deviation} = \text{Effect}_\mathrm{additive} - \text{Effect}_{\mathrm{WT},\,i}$, capturing deviation from pure additivity, with $\beta_2$ representing the epistatic interaction strength.
\end{itemize}

Model fits for both cases were obtained using \texttt{Ridge} (with the same regularization setup), and the null hypothesis ($\beta_2 = 0$, i.e., no interaction/epistasis) was tested via an F-test comparing the residual sum of squares ($\mathrm{RSS}$):

\[
F = \frac{(RSS_\mathrm{additive} - RSS_\mathrm{interaction}) / (df_\mathrm{additive} - df_\mathrm{interaction})}{RSS_\mathrm{interaction} / df_\mathrm{interaction}}
\]

where $df_\mathrm{additive} = n - 2$ and $df_\mathrm{interaction} = n - 3$ ($n =$ number of haplotypes). P-values were calculated using \texttt{scipy.stats.f.cdf} (F-distribution, $1$ and $df_\mathrm{interaction}$ degrees of freedom). Since \texttt{Ridge} is a regularized model, degrees of freedom are approximate, but the F-test remains valid for comparing nested models.

A pair was classified as epistatic if the interaction model fit led to a statistically significant improvement ($p < 0.05$ by default, controlled via the \texttt{epistasis\_pvalue\_threshold} parameter) over the additive-only model. For each test, we recorded:

\begin{enumerate}
    \item Epistasis $p$-value and F-statistic
    \item $R^2$ and MSE for both the additive and interaction models
    \item $R^2$ improvement ($\Delta R^2 = R^2_\mathrm{interaction} - R^2_\mathrm{additive}$)
    \item Interaction coefficient $\beta_2$ (strength of epistasis)
    \item The joint effect $\beta_{ij}$, expected additive effect, and their deviation
    \item Individual WT variant and clinical variant effects
\end{enumerate}

To ensure robust inference, we restricted epistasis tests to variant pairs meeting the following criteria:

\begin{enumerate}
    \item At least 10 haplotypes with complete, non-missing data for both the WT variant and the clinical variant site
    \item Sufficient variation in WT variant status (std $> 10^{-10}$)
    \item Both WT=0 and WT=1 represented in the dataset
\end{enumerate}
These filters prevent false positives from insufficient or uninformative data.

% ============================================================================
% SEPARATOR: Multi-target approach (above) vs. Window-based approach (below)
% ============================================================================

\subsubsection*{Window-Based Epistasis Testing for Site-Centered Models}

For analyses where clinical variants are analyzed individually with site-centered genomic windows (e.g., SpliceAI predictions), we implemented an alternative epistasis testing approach that accounts for the fact that each clinical variant is modeled separately with a distinct set of WT variants within its genomic window.

\paragraph*{Model Training Strategy}

Rather than fitting a single multi-target model across all clinical variants simultaneously, we fit separate Ridge regression models for each clinical variant $j$:

\[
\mathbf{Y}_j = \mathbf{X}_j\mathbf{\beta}_j + \varepsilon_j
\]

where $\mathbf{X}_j$ is a binary matrix (\#haplotypes $\times$ \#WT variants within window $j$) containing only WT variants within a genomic window centered on clinical variant $j$ (typically $\pm 5$ kb), and $\mathbf{Y}_j$ is a vector (\#haplotypes $\times$ 1) containing VEP scores for clinical variant $j$ only. This window-based approach ensures that each model focuses on WT variants in spatial proximity to the clinical variant of interest, which is particularly relevant for splicing models where local sequence context matters.

\paragraph*{Cross-Model WT Effect Estimation}

A key difference from the multi-target approach is how individual WT variant effects are computed. Since each model includes a different set of WT variants (determined by the genomic window), we cannot compute WT individual effects within a single model context. Instead, we compute the individual effect of each WT variant $i$ by averaging its effect across \textit{all} clinical variant models where it appears:

\[
\text{Effect}_{\mathrm{WT},\,i} = \frac{1}{|\mathcal{M}_i|} \sum_{m \in \mathcal{M}_i} \left( \overline{\mathrm{VEP}}(\mathrm{WT}_i{=}1,\, m) - \overline{\mathrm{VEP}}(\mathrm{WT}_i{=}0,\, m) \right)
\]

where $\mathcal{M}_i$ is the set of all clinical variant models (indexed by $m$) that include WT variant $i$ in their genomic window. This cross-model averaging provides a robust estimate of the WT variant's average effect across different genomic contexts, which serves as the expected additive effect when testing epistasis.

\paragraph*{Epistasis Testing Procedure}

For each WT variant $i$ and clinical variant $j$ pair, epistasis testing proceeds as follows:

\begin{enumerate}
    \item \textbf{Expected additive effect:} Since we are testing one clinical variant at a time, the clinical variant effect is constant across all haplotypes (absorbed into the intercept). Therefore, the expected additive effect is simply:
    \[
    \text{Effect}_\mathrm{additive} = \text{Effect}_{\mathrm{WT},\,i}
    \]
    where $\text{Effect}_{\mathrm{WT},\,i}$ is computed across all models as described above.
    
    \item \textbf{Observed joint effect:} The coefficient $\beta_{ij}$ from the model for clinical variant $j$ represents the observed joint effect.
    
    \item \textbf{Model comparison:} We fit the same additive and interaction models as in the multi-target approach:
    \begin{itemize}
        \item \textbf{Additive model:} $\mathrm{VEP}_j = \beta_0 + \beta_1 \cdot \mathrm{WT}_i$
        \item \textbf{Interaction model:} $\mathrm{VEP}_j = \beta_0 + \beta_1 \cdot \mathrm{WT}_i + \beta_2 \cdot (\mathrm{WT}_i \times \Delta_\mathrm{deviation})$
    \end{itemize}
    where $\Delta_\mathrm{deviation} = \beta_{ij} - \text{Effect}_{\mathrm{WT},\,i}$.
    
    \item \textbf{Statistical testing:} The F-test and p-value calculation proceed identically to the multi-target approach.
\end{enumerate}

\paragraph*{Implementation Details and Computational Workflow}

All analyses were conducted in Python 3.12.9, and all core methods have been implemented as custom functions available in the public GitHub repository \href{https://github.com/bschilder/VEP_DNA}{\texttt{https://github.com/bschilder/VEP_DNA}}. Linear regression models for both additive effects and epistasis testing use \texttt{sklearn.linear\_model.Ridge}, statistical hypothesis tests rely on \texttt{scipy.stats.f}, and data manipulation leverages \texttt{pandas}.

The workflow is structured in a flexible two-stage procedure that works for both the multi-target and window-based approaches:
\begin{enumerate}
    \item \textbf{Model Training:} First, the main linear models are fit for either all clinical variants simultaneously (multi-target mode) or for each clinical variant individually in their respective genomic windows (window-based mode). This is done using the \texttt{wtvariants\_to\_vep\_linear\_model} function, which supports Ridge regression, user-specified regularization, and (if desired) integrated epistasis testing via the \texttt{test\_epistasis} flag.
    \item \textbf{Epistasis Testing:} After all models are trained, epistasis is assessed using the \texttt{test\_epistasis\_across\_models} function. This computes WT individual effects by aggregating across all relevant clinical variant models—enabling the calculation of expected additive effects even for window-based approaches where each model contains a different set of WT variants. Batch epistasis testing is efficiently performed across all possible model-variant pairs, handling scenarios where variant sets differ due to genomic windowing.
\end{enumerate}

The above design enables:
\begin{itemize}
    \item Aggregation and estimation of WT individual effects across all models, ensuring robust expected effect calculations for use in epistasis testing.
    \item Scalable, efficient batch processing of epistasis tests, regardless of whether a multi-target or window-based modeling approach is used.
    \item Generation of a combined dataframe (\texttt{epistasis\_df}) that contains both the initial interaction results and all epistasis annotations and statistics. Updated model dictionaries are returned with integrated epistasis results.
\end{itemize}