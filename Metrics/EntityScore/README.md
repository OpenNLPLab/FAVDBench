# EntityScore

**EntityScore** measures the extent to which consistently referred words or series of words, known as entities and often manifested as nouns, in the predicted text match those in the annotated text. We use an off-the-shelf Natural Language Toolkit library to extract nouns as entities. The mathematical expression of the EntityScore $\mathbb{ES}$ is as follows:

```math
\begin{aligned}
R(\mathbf p,\mathbf r)=\frac{\#\{ \mathbf p \cap \mathbf r \}}{\#\{\mathbf r\}}, \\
C(\mathbf p,\mathbf r)=\frac{\cos(\mathrm{T5}(\mathbf p), \mathrm{T5}(\mathbf r))+1} 2, \\
\mathbb{ES}(\mathbf p,\mathbf r)= \frac{2 R(\mathbf p,\mathbf r) C(\mathbf p, \mathbf r)}{R(\mathbf p,\mathbf r) + C(\mathbf p, \mathbf r)}
\end{aligned}
```
