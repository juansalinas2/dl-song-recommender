# Model Architecture and Experimental Progression

Music recommendation is usually modeled through collaborative filtering (or using sloppy features, such as valence). We take an embedding approach of 10 second song clips that aims to respect human notions of song similarity. 

The aim of notebooks 4-7 is to learn an audio embedding from song clips that respects those informal musical relationships well enough to support retrieval.

Across these notebooks, the backbone remains nearly unchanged. What changes is the way the model is asked to understand musical similarity: first through genre tags alone, then through increasingly audio-centered forms of regularization and supervision.

## Shared Encoder

All four notebooks use the same late-fusion architecture. Each song is represented by:

- one full-mix spectrogram
- four stem spectrograms

A shared `ResNet18` encodes the mix and each stem. The stem outputs are then grouped into two musical branches:

- harmonic branch: bass, other, vocals
- drum branch: drums

These pieces are fused into a single retrieval embedding:

$$
z = \mathrm{normalize}(m + \alpha_h h + \alpha_d d)
$$

where $m$ is the mix embedding, $h$ is the harmonic embedding, and $d$ is the drum embedding. The learned weights $\alpha_h$ and $\alpha_d$ determine how much the harmonic and drum branches contribute to the final representation.

This vector $z$ is the embedding used for cosine nearest-neighbor retrieval. A small projection head is also used during training for contrastive losses, but retrieval always uses the fused song embedding.

![Model architecture](docs/diagrams/model_1.png)

## Progression at a Glance

| Notebook | Main supervision | Central question |
| --- | --- | --- |
| 4 | Genre-tag teacher | Can audio recover the neighborhood structure implied by genre tags? |
| 5 | Genre-tag teacher + light InfoNCE | Does weak same-song contrastive structure improve local consistency? |
| 6 | Genre-tag teacher + stronger audio regularization | Does stronger audio grounding improve the tag-based baseline? |
| 7 | Blended audio and tag teacher | Should the teacher itself be partly defined by audio? |


## Training Logic

The training objective has three components:

$$
L = \lambda_{\mathrm{cross}}L_{\mathrm{cross}} + \lambda_{\mathrm{view}}L_{\mathrm{view}} + \lambda_{\mathrm{inst}}L_{\mathrm{inst}}
$$

The cross-song term is the main retrieval loss. It shapes the geometry of the embedding space by requiring pairwise cosine similarity in the learned audio embedding to match a frozen teacher.

$$
L_{\mathrm{cross}} = \mathrm{mean}_{i<j}\left(\cos(z_i, z_j) - T_{ij}\right)^2
$$

For notebooks 4-6, the teacher $T_{ij}$ is defined by genre-tag similarity. In notebook 7, it is replaced by a blended audio-tag teacher.

The view-alignment term keeps the fused embedding close to the mix, harmonic, and drum branches that produced it:

$$
L_{\mathrm{view}} = \tfrac{1}{2}\left((1-\cos(z,m)) + (1-\cos(z,h))\right) + w_d(1-\cos(z,d))
$$

The instance term $L_{\mathrm{inst}}$ is an auxiliary InfoNCE loss used in notebooks 5-7. It encourages different views of the same song to remain close, but it does not define the global retrieval structure.

## Notebook 4: Tag-Based Model

Notebook 4 is the clearest expression of the central idea. A genre-tag teacher (from notebook 2) defines which songs should be near one another, and the audio model is trained to reproduce that geometry directly from spectrograms.

The important point is that the model is not predicting tags. Instead, the genre-tag space acts as a geometric reference. The model learns an audio embedding whose pairwise structure mirrors the pairwise structure induced by the tags. In that sense, notebook 4 is a metric learning model guided by genre tags.

## Notebook 5: Adding Same-Song Contrastive Structure

Notebook 5 keeps the same encoder and the same tag teacher, but adds a small symmetric InfoNCE term between different views of the same song.

The purpose is to improve local consistency without changing the global tag-defined geometry. If notebook 4 already captures broad genre-tag neighborhoods, notebook 5 asks whether a weak same-song contrastive term can preserve more of each track's individual sonic identity.

## Notebook 6: Pushing Harder on Audio Grounding

Notebook 6 keeps the same tag teacher, but gives more weight to the audio-side objectives. The model is still organized primarily by genre tags, yet it is pushed more strongly to preserve audio consistency across views and to remain stable under the late-fusion setup.

This notebook asks whether the notebook 4 embedding is too strongly shaped by the tag teacher alone. It therefore increases audio-side regularization without abandoning the original tag-defined target.

## Notebook 7: Blending Tag and Audio Teachers

Notebook 7 changes the supervision more substantially. Instead of using only the genre-tag teacher, it defines the cross-song target by blending frozen audio similarity and frozen tag similarity:

$$
T_{ij} = 0.50\,S_{\mathrm{audio}}(A_i, A_j) + 0.50\,S_{\mathrm{tag}}(E_i, E_j)
$$

Here the teacher itself is no longer purely tag-defined; audio similarity contributes directly to the cross-song target.

## Interpretation

Taken together, notebooks 4-7 form the following progression:

- Notebook 4 asks whether genre-tag structure can be distilled into audio embeddings.
- Notebook 5 asks whether a little same-song contrastive pressure improves that embedding.
- Notebook 6 asks whether stronger audio regularization improves the balance between tags and sound.
- Notebook 7 asks whether the teacher itself should be partly audio-defined.
